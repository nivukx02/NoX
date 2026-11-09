package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Aligns learned body-space landmarks to calibrated metric reconstruction without replacing
 * calibrated multi-view geometry.
 *
 * The learned body-space coordinate frame is not assumed to match NoX world coordinates. The engine
 * therefore fits a robust weighted similarity transform from jointly observed anatomical anchors.
 * A limited reflection hypothesis is evaluated for coordinate-handness differences.
 *
 * The aligned prior is intentionally conservative: its confidence is capped below authoritative
 * triangulation, its uncertainty has a hard floor, and poor fits are rejected. This makes the prior
 * useful for weakly observed joints and temporal continuity without converting model output into a
 * fabricated metrology claim.
 */
class BodySpacePriorAlignmentEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        val minimumAnchorCount: Int = 3,
        val preferredAnchorCount: Int = 5,
        val minimumAnchorConfidence: Double = 0.18,
        val minimumAnchorSpanMeters: Double = 0.04,
        val maximumScaleRatio: Double = 2.50,
        val minimumScaleRatio: Double = 0.40,
        val maximumAnchorResidualMeters: Double = 0.22,
        val robustHuberDeltaMeters: Double = 0.025,
        val maximumIterations: Int = 6,
        val priorConfidenceCap: Double = 0.34,
        val priorUncertaintyFloorMeters: Double = 0.025,
        val priorUncertaintyScale: Double = 1.80,
        val minimumFitQuality: Double = 0.20,
    ) {
        init {
            require(minimumAnchorCount >= 3)
            require(preferredAnchorCount >= minimumAnchorCount)
            require(minimumAnchorConfidence in 0.0..1.0)
            require(minimumAnchorSpanMeters > 0.0)
            require(maximumScaleRatio > minimumScaleRatio)
            require(minimumScaleRatio > 0.0)
            require(maximumAnchorResidualMeters > 0.0)
            require(robustHuberDeltaMeters > 0.0)
            require(maximumIterations in 1..32)
            require(priorConfidenceCap in 0.0..1.0)
            require(priorUncertaintyFloorMeters > 0.0)
            require(priorUncertaintyScale >= 1.0)
            require(minimumFitQuality in 0.0..1.0)
        }
    }

    data class Prior(
        val landmarkId: Int,
        val positionMeters: Vec3,
        val confidence: Double,
    ) {
        init {
            require(landmarkId >= 0)
            require(positionMeters.x.isFinite() && positionMeters.y.isFinite() && positionMeters.z.isFinite())
            require(confidence in 0.0..1.0)
        }
    }

    data class MetricAnchor(
        val landmarkId: Int,
        val positionMeters: Vec3,
        val confidence: Double,
        val uncertaintyMeters: Double = 0.0,
    ) {
        init {
            require(landmarkId >= 0)
            require(positionMeters.x.isFinite() && positionMeters.y.isFinite() && positionMeters.z.isFinite())
            require(confidence in 0.0..1.0)
            require(uncertaintyMeters.isFinite() && uncertaintyMeters >= 0.0)
        }
    }

    data class AlignedPrior(
        val landmarkId: Int,
        val positionMeters: Vec3,
        val confidence: Double,
        val uncertaintyMeters: Double,
        val anchorResidualMeters: Double,
        val transformedByReflectionHypothesis: Boolean,
    )

    data class Result(
        val accepted: Boolean,
        val aligned: List<AlignedPrior>,
        val anchorCount: Int,
        val robustRmsResidualMeters: Double,
        val maximumResidualMeters: Double,
        val scale: Double,
        val fitQuality: Double,
        val priorContributionConfidence: Double,
        val transformedByReflectionHypothesis: Boolean,
        val warnings: List<String>,
    ) {
        companion object {
            fun rejected(warning: String): Result =
                Result(
                    accepted = false,
                    aligned = emptyList(),
                    anchorCount = 0,
                    robustRmsResidualMeters = Double.POSITIVE_INFINITY,
                    maximumResidualMeters = Double.POSITIVE_INFINITY,
                    scale = 1.0,
                    fitQuality = 0.0,
                    priorContributionConfidence = 0.0,
                    transformedByReflectionHypothesis = false,
                    warnings = listOf(warning),
                )
        }
    }

    private data class WeightedPair(
        val prior: Vec3,
        val metric: Vec3,
        val baseWeight: Double,
        val landmarkId: Int,
    )

    private data class Transform(
        val rotation: QuatLike,
        val translation: Vec3,
        val scale: Double,
        val reflectionHypothesis: Boolean,
    )

    private data class Fit(
        val transform: Transform,
        val residuals: List<Double>,
        val weightedRms: Double,
    )

    private data class QuatLike(
        val w: Double,
        val x: Double,
        val y: Double,
        val z: Double,
    ) {
        fun normalized(): QuatLike {
            val n = kotlin.math.sqrt(w * w + x * x + y * y + z * z)
            if (!n.isFinite() || n < 1e-12) {
                return QuatLike(1.0, 0.0, 0.0, 0.0)
            }
            return QuatLike(
                w / n,
                x / n,
                y / n,
                z / n,
            )
        }

        fun rotate(v: Vec3): Vec3 {
            val q = normalized()
            val uv = Vec3(
                q.y * v.z - q.z * v.y,
                q.z * v.x - q.x * v.z,
                q.x * v.y - q.y * v.x,
            )
            val uuv = Vec3(
                q.y * uv.z - q.z * uv.y,
                q.z * uv.x - q.x * uv.z,
                q.x * uv.y - q.y * uv.x,
            )
            return v +
                uv * (2.0 * q.w) +
                uuv * 2.0
        }
    }

    @Volatile
    private var closed = false

    @Synchronized
    fun align(
        priors: List<Prior>,
        metricAnchors: List<MetricAnchor>,
    ): Result {
        check(!closed) {
            "BodySpacePriorAlignmentEngine is closed"
        }

        if (priors.isEmpty()) {
            return Result.rejected("BODY_PRIOR_EMPTY")
        }

        val priorById =
            priors.asSequence()
                .filter {
                    it.confidence >=
                        config.minimumAnchorConfidence
                }
                .distinctBy {
                    it.landmarkId
                }
                .associateBy {
                    it.landmarkId
                }

        val metricById =
            metricAnchors.asSequence()
                .filter {
                    it.confidence >=
                        config.minimumAnchorConfidence
                }
                .distinctBy {
                    it.landmarkId
                }
                .associateBy {
                    it.landmarkId
                }

        val anchorIds =
            preferredAnchorIds()
                .filter {
                    priorById[it] != null &&
                        metricById[it] != null
                }

        val pairs =
            anchorIds.mapNotNull { id ->
                val prior =
                    priorById[id]
                        ?: return@mapNotNull null
                val metric =
                    metricById[id]
                        ?: return@mapNotNull null

                WeightedPair(
                    prior = prior.positionMeters,
                    metric = metric.positionMeters,
                    baseWeight =
                        (
                            min(
                                prior.confidence,
                                metric.confidence,
                            ) *
                                (
                                    1.0 /
                                        (
                                            1.0 +
                                                metric.uncertaintyMeters *
                                                    20.0
                                        )
                                )
                        )
                            .coerceAtLeast(0.10),
                    landmarkId = id,
                )
            }

        if (pairs.size < config.minimumAnchorCount) {
            return Result.rejected(
                "BODY_PRIOR_INSUFFICIENT_METRIC_ANCHORS",
            )
        }

        val priorSpan =
            maximumPairDistance(
                pairs.map {
                    it.prior
                },
            )

        val metricSpan =
            maximumPairDistance(
                pairs.map {
                    it.metric
                },
            )

        if (
            priorSpan <
                config.minimumAnchorSpanMeters ||
            metricSpan <
                config.minimumAnchorSpanMeters
        ) {
            return Result.rejected(
                "BODY_PRIOR_ANCHOR_SPAN_TOO_SMALL",
            )
        }

        val candidates =
            listOf(false, true)
                .mapNotNull { reflection ->
                    val transformedPairs =
                        pairs.map { pair ->
                            pair.copy(
                                prior =
                                    applyReflectionHypothesis(
                                        pair.prior,
                                        reflection,
                                    ),
                            )
                        }

                    fitRobust(
                        pairs = transformedPairs,
                        reflectionHypothesis = reflection,
                    )
                }

        val best =
            candidates.minByOrNull {
                it.weightedRms
            }
                ?: return Result.rejected(
                    "BODY_PRIOR_FIT_FAILED",
                )

        val transform = best.transform

        if (
            transform.scale <
                config.minimumScaleRatio ||
            transform.scale >
                config.maximumScaleRatio
        ) {
            return Result.rejected(
                "BODY_PRIOR_SCALE_OUTSIDE_SAFE_ENVELOPE",
            )
        }

        if (
            best.residuals.any {
                it > config.maximumAnchorResidualMeters
            }
        ) {
            return Result.rejected(
                "BODY_PRIOR_ANCHOR_RESIDUAL_TOO_LARGE",
            )
        }

        val residualQuality =
            expNegative(
                best.weightedRms /
                    config.robustHuberDeltaMeters
                        .coerceAtLeast(1e-6),
            )

        val supportQuality =
            (
                pairs.size.toDouble() /
                    config.preferredAnchorCount.toDouble()
            )
                .coerceIn(0.0, 1.0)

        val fitQuality =
            (
                residualQuality * 0.76 +
                    supportQuality * 0.24
            )
                .coerceIn(0.0, 1.0)

        if (fitQuality < config.minimumFitQuality) {
            return Result.rejected(
                "BODY_PRIOR_FIT_QUALITY_LOW",
            )
        }

        val metricUncertainty =
            metricAnchors
                .map {
                    it.uncertaintyMeters
                }
                .average()
                .coerceAtLeast(0.0)

        val priorConfidenceScale =
            (
                fitQuality * 0.68
            ).coerceIn(
                0.08,
                config.priorConfidenceCap,
            )

        val priorUncertainty =
            max(
                config.priorUncertaintyFloorMeters,
                max(
                    metricUncertainty,
                    best.weightedRms *
                        config.priorUncertaintyScale,
                ),
            )

        val aligned =
            priors.mapNotNull { prior ->
                val position =
                    applyTransform(
                        transform,
                        prior.positionMeters,
                    )

                if (!position.finite()) {
                    return@mapNotNull null
                }

                val anchorResidual =
                    best.residuals
                        .getOrNull(
                            pairs.indexOfFirst {
                                it.landmarkId ==
                                    prior.landmarkId
                            },
                        )
                        ?: best.weightedRms

                AlignedPrior(
                    landmarkId = prior.landmarkId,
                    positionMeters = position,
                    confidence =
                        (
                            prior.confidence *
                                priorConfidenceScale
                        )
                            .coerceIn(
                                0.0,
                                config.priorConfidenceCap,
                            ),
                    uncertaintyMeters =
                        max(
                            config.priorUncertaintyFloorMeters,
                            priorUncertainty +
                                anchorResidual * 0.80,
                        ),
                    anchorResidualMeters =
                        anchorResidual.coerceAtLeast(0.0),
                    transformedByReflectionHypothesis =
                        transform.reflectionHypothesis,
                )
            }

        return Result(
            accepted = aligned.isNotEmpty(),
            aligned = aligned,
            anchorCount = pairs.size,
            robustRmsResidualMeters = best.weightedRms,
            maximumResidualMeters =
                best.residuals.maxOrNull()
                    ?: 0.0,
            scale = transform.scale,
            fitQuality = fitQuality,
            priorContributionConfidence =
                priorConfidenceScale,
            transformedByReflectionHypothesis =
                transform.reflectionHypothesis,
            warnings = buildList {
                if (
                    pairs.size <
                        config.preferredAnchorCount
                ) {
                    add(
                        "BODY_PRIOR_ANCHOR_SUPPORT_REDUCED",
                    )
                }

                if (
                    best.weightedRms >
                        config.robustHuberDeltaMeters
                ) {
                    add(
                        "BODY_PRIOR_ROBUST_FIT_RESIDUAL_ELEVATED",
                    )
                }

                if (transform.reflectionHypothesis) {
                    add(
                        "BODY_PRIOR_COORDINATE_HANDNESS_ADJUSTED",
                    )
                }

                if (
                    metricUncertainty >
                        config.priorUncertaintyFloorMeters
                ) {
                    add(
                        "BODY_PRIOR_INHERITS_METRIC_ANCHOR_UNCERTAINTY",
                    )
                }
            }.distinct(),
        )
    }

    @Synchronized
    fun reset() {
        check(!closed) {
            "BodySpacePriorAlignmentEngine is closed"
        }
    }

    override fun close() {
        closed = true
    }

    private fun preferredAnchorIds(): List<Int> =
        listOf(
            0,
            11,
            12,
            23,
            24,
        )

    private fun applyReflectionHypothesis(
        value: Vec3,
        enabled: Boolean,
    ): Vec3 =
        if (!enabled) {
            value
        } else {
            Vec3(
                value.x,
                value.y,
                -value.z,
            )
        }

    private fun fitRobust(
        pairs: List<WeightedPair>,
        reflectionHypothesis: Boolean,
    ): Fit? {
        if (pairs.size < config.minimumAnchorCount) {
            return null
        }

        var weights =
            pairs.map {
                it.baseWeight
                    .coerceAtLeast(1e-5)
            }

        var transform: Transform? = null

        repeat(config.maximumIterations) {
            val candidate =
                fitWeighted(
                    pairs = pairs,
                    weights = weights,
                    reflectionHypothesis = reflectionHypothesis,
                )
                    ?: return@repeat

            transform = candidate

            val residuals =
                pairs.map { pair ->
                    distance(
                        applyTransform(
                            candidate,
                            pair.prior,
                        ),
                        pair.metric,
                    )
                }

            weights =
                pairs.indices.map { index ->
                    pairs[index]
                        .baseWeight
                        .coerceAtLeast(1e-5) *
                        huberWeight(
                            residuals[index],
                        )
                }
        }

        val finalTransform =
            transform
                ?: return null

        val residuals =
            pairs.map { pair ->
                distance(
                    applyTransform(
                        finalTransform,
                        pair.prior,
                    ),
                    pair.metric,
                )
            }

        val weightSum =
            weights.sum()
                .coerceAtLeast(1e-9)

        val weightedRms =
            kotlin.math.sqrt(
                pairs.indices.sumOf { index ->
                    weights[index] *
                        residuals[index] *
                        residuals[index]
                } /
                    weightSum,
            )

        if (!weightedRms.isFinite()) {
            return null
        }

        return Fit(
            transform = finalTransform,
            residuals = residuals,
            weightedRms = weightedRms,
        )
    }

    private fun fitWeighted(
        pairs: List<WeightedPair>,
        weights: List<Double>,
        reflectionHypothesis: Boolean,
    ): Transform? {
        if (weights.size != pairs.size) {
            return null
        }

        val weightSum =
            weights.sum()
                .coerceAtLeast(1e-9)

        val priorCenter =
            weightedCenter(
                points = pairs.map {
                    it.prior
                },
                weights = weights,
                weightSum = weightSum,
            )

        val metricCenter =
            weightedCenter(
                points = pairs.map {
                    it.metric
                },
                weights = weights,
                weightSum = weightSum,
            )

        var h00 = 0.0
        var h01 = 0.0
        var h02 = 0.0
        var h10 = 0.0
        var h11 = 0.0
        var h12 = 0.0
        var h20 = 0.0
        var h21 = 0.0
        var h22 = 0.0
        var priorEnergy = 0.0

        pairs.indices.forEach { index ->
            val p =
                pairs[index].prior -
                    priorCenter
            val qMetric =
                pairs[index].metric -
                    metricCenter
            val w = weights[index]

            h00 +=
                w * p.x * qMetric.x
            h01 +=
                w * p.x * qMetric.y
            h02 +=
                w * p.x * qMetric.z
            h10 +=
                w * p.y * qMetric.x
            h11 +=
                w * p.y * qMetric.y
            h12 +=
                w * p.y * qMetric.z
            h20 +=
                w * p.z * qMetric.x
            h21 +=
                w * p.z * qMetric.y
            h22 +=
                w * p.z * qMetric.z

            priorEnergy +=
                w * p.dot(p)
        }

        if (
            priorEnergy <= 1e-12 ||
            !priorEnergy.isFinite()
        ) {
            return null
        }

        val sigma =
            h00 +
                h11 +
                h22

        val zx =
            h21 -
                h12
        val zy =
            h02 -
                h20
        val zz =
            h10 -
                h01

        val k =
            arrayOf(
                doubleArrayOf(
                    sigma,
                    zx,
                    zy,
                    zz,
                ),
                doubleArrayOf(
                    zx,
                    h00 - h11 - h22,
                    h01 + h10,
                    h02 + h20,
                ),
                doubleArrayOf(
                    zy,
                    h01 + h10,
                    -h00 + h11 - h22,
                    h12 + h21,
                ),
                doubleArrayOf(
                    zz,
                    h02 + h20,
                    h12 + h21,
                    -h00 - h11 + h22,
                ),
            )

        var q =
            doubleArrayOf(
                1.0,
                0.0,
                0.0,
                0.0,
            )

        repeat(24) {
            val next =
                DoubleArray(4)

            for (row in 0..3) {
                for (column in 0..3) {
                    next[row] +=
                        k[row][column] *
                            q[column]
                }
            }

            val norm =
                kotlin.math.sqrt(
                    next.sumOf {
                        it * it
                    },
                )

            if (
                !norm.isFinite() ||
                norm < 1e-12
            ) {
                return@repeat
            }

            for (index in 0..3) {
                next[index] /=
                    norm
            }

            q = next
        }

        val rotation =
            QuatLike(
                q[0],
                q[1],
                q[2],
                q[3],
            ).normalized()

        val numerator =
            pairs.indices.sumOf { index ->
                val p =
                    pairs[index].prior -
                        priorCenter
                val metric =
                    pairs[index].metric -
                        metricCenter

                weights[index] *
                    metric.dot(
                        rotation.rotate(p),
                    )
            }

        val scale =
            (
                numerator /
                    priorEnergy
            )
                .coerceIn(
                    config.minimumScaleRatio * 0.50,
                    config.maximumScaleRatio * 1.50,
                )

        if (
            !scale.isFinite() ||
            scale <= 0.0
        ) {
            return null
        }

        val translation =
            metricCenter -
                rotation.rotate(
                    priorCenter,
                ) *
                scale

        if (!translation.finite()) {
            return null
        }

        return Transform(
            rotation = rotation,
            translation = translation,
            scale = scale,
            reflectionHypothesis = reflectionHypothesis,
        )
    }

    private fun weightedCenter(
        points: List<Vec3>,
        weights: List<Double>,
        weightSum: Double,
    ): Vec3 {
        var x = 0.0
        var y = 0.0
        var z = 0.0

        points.indices.forEach { index ->
            val point =
                points[index]
            val weight =
                weights[index]

            x += point.x * weight
            y += point.y * weight
            z += point.z * weight
        }

        return Vec3(
            x / weightSum,
            y / weightSum,
            z / weightSum,
        )
    }

    private fun applyTransform(
        transform: Transform,
        value: Vec3,
    ): Vec3 {
        val reflected =
            applyReflectionHypothesis(
                value,
                transform.reflectionHypothesis,
            )

        return transform.rotation.rotate(
            reflected,
        ) *
            transform.scale +
            transform.translation
    }

    private fun huberWeight(
        residual: Double,
    ): Double {
        val delta =
            config.robustHuberDeltaMeters

        if (residual <= delta) {
            return 1.0
        }

        return delta /
            residual.coerceAtLeast(1e-9)
    }

    private fun maximumPairDistance(
        points: List<Vec3>,
    ): Double {
        if (points.size < 2) {
            return 0.0
        }

        var maximum = 0.0

        for (i in points.indices) {
            for (j in i + 1 until points.size) {
                maximum =
                    max(
                        maximum,
                        distance(
                            points[i],
                            points[j],
                        ),
                    )
            }
        }

        return maximum
    }

    private fun distance(
        a: Vec3,
        b: Vec3,
    ): Double =
        (a - b).norm()

    private fun expNegative(
        value: Double,
    ): Double =
        kotlin.math.exp(
            -value.coerceAtLeast(0.0),
        )

    private fun Vec3.finite(): Boolean =
        x.isFinite() &&
            y.isFinite() &&
            z.isFinite()
}

package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Shape-space temporal consistency for multi-view facial geometry.
 *
 * Facial meshes move with the head, so comparing raw world coordinates frame-to-frame is a poor
 * stability test. This engine removes global translation through centroid normalization and tracks
 * semantic contour edges and radial distances. Rigid head motion therefore survives while sudden
 * topology/shape jumps are penalized.
 */
class FaceMeshTemporalConsistencyEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val edgeRatioMinimum: Double = 0.55,
        val edgeRatioMaximum: Double = 1.80,
        val radialRatioMinimum: Double = 0.70,
        val radialRatioMaximum: Double = 1.45,
        val maximumShapeVelocityMetersPerSecond: Double = 1.8,
        val maximumShapeAccelerationMetersPerSecondSquared: Double = 30.0,
        val maximumGapSeconds: Double = 0.20,
        val minimumPoints: Int = 10,
        val minimumConfidence: Double = 0.30,
        val historyAlpha: Double = 0.20,
        val warningScore: Double = 0.72,
        val rejectScore: Double = 0.48,
        val persistentWarningFrames: Int = 4,
        val persistentRejectFrames: Int = 8,
    ) {
        init {
            require(edgeRatioMinimum > 0.0)
            require(edgeRatioMaximum > edgeRatioMinimum)
            require(radialRatioMinimum > 0.0)
            require(radialRatioMaximum > radialRatioMinimum)
            require(maximumShapeVelocityMetersPerSecond > 0.0)
            require(maximumShapeAccelerationMetersPerSecondSquared > 0.0)
            require(maximumGapSeconds > 0.0)
            require(minimumPoints >= 3)
            require(minimumConfidence in 0.0..1.0)
            require(historyAlpha in 0.0..1.0)
            require(warningScore in 0.0..1.0)
            require(rejectScore in 0.0..warningScore)
            require(persistentWarningFrames >= 1)
            require(persistentRejectFrames >= persistentWarningFrames)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val pointsBySemanticId: Map<Int, Vec3>,
        val baseConfidence: Double,
        val meanReprojectionErrorPx: Double,
    )

    data class Result(
        val timestampNanos: Long,
        val accepted: Boolean,
        val confidence: Double,
        val rigidMotionMeters: Double,
        val shapeVelocityMetersPerSecond: Double,
        val shapeAccelerationMetersPerSecondSquared: Double,
        val medianEdgeRatio: Double,
        val medianRadialRatio: Double,
        val edgeInlierRatio: Double,
        val radialInlierRatio: Double,
        val persistentWarningFrames: Int,
        val persistentRejectFrames: Int,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private data class History(
        var timestampNanos: Long,
        var centroid: Vec3,
        var normalized: Map<Int, Vec3>,
        var previousEdgeLengths: Map<PairKey, Double>,
        var shapeVelocityMetersPerSecond: Double,
        var shapeAccelerationMetersPerSecondSquared: Double,
        var confidenceEma: Double,
        var warningFrames: Int,
        var rejectFrames: Int,
        var baselineNormalized: Map<Int, Vec3> = emptyMap(),
    )

    private data class PairKey(
        val first: Int,
        val second: Int,
    )

    private var history: History? = null

    @Synchronized
    fun evaluate(input: Input): Result {
        val clean = input.pointsBySemanticId.filterValues {
            it.x.isFinite() &&
                it.y.isFinite() &&
                it.z.isFinite()
        }

        if (clean.size < config.minimumPoints) {
            return Result(
                timestampNanos = input.timestampNanos,
                accepted = false,
                confidence = 0.0,
                rigidMotionMeters = 0.0,
                shapeVelocityMetersPerSecond = 0.0,
                shapeAccelerationMetersPerSecondSquared = 0.0,
                medianEdgeRatio = 1.0,
                medianRadialRatio = 1.0,
                edgeInlierRatio = 0.0,
                radialInlierRatio = 0.0,
                persistentWarningFrames = history?.warningFrames ?: 0,
                persistentRejectFrames = history?.rejectFrames ?: 0,
                warnings = listOf("face-mesh-observation-count-below-production-minimum"),
                recommendations = listOf(
                    "Hold facial geometry and reacquire at least " +
                        config.minimumPoints +
                        " stable semantic points.",
                ),
            )
        }

        val centroid = centroid(clean.values.toList())
        val normalized = clean.mapValues { (_, point) ->
            point - centroid
        }

        val previous = history
        if (previous == null) {
            history = History(
                timestampNanos = input.timestampNanos,
                centroid = centroid,
                normalized = normalized,
                previousEdgeLengths = buildEdgeMap(normalized),
                shapeVelocityMetersPerSecond = 0.0,
                shapeAccelerationMetersPerSecondSquared = 0.0,
                confidenceEma = input.baseConfidence.coerceIn(0.0, 1.0),
                warningFrames = 0,
                rejectFrames = 0,
                baselineNormalized = normalized,
            )

            return Result(
                timestampNanos = input.timestampNanos,
                accepted = input.baseConfidence >= config.minimumConfidence &&
                    input.meanReprojectionErrorPx <= 8.0,
                confidence = (
                    0.82 * input.baseConfidence.coerceIn(0.0, 1.0) +
                        0.18 * reprojectionScore(
                            input.meanReprojectionErrorPx,
                        )
                    ).coerceIn(0.0, 1.0),
                rigidMotionMeters = 0.0,
                shapeVelocityMetersPerSecond = 0.0,
                shapeAccelerationMetersPerSecondSquared = 0.0,
                medianEdgeRatio = 1.0,
                medianRadialRatio = 1.0,
                edgeInlierRatio = 1.0,
                radialInlierRatio = 1.0,
                persistentWarningFrames = 0,
                persistentRejectFrames = 0,
                warnings = emptyList(),
                recommendations = emptyList(),
            )
        }

        val dt = (
            (input.timestampNanos - previous.timestampNanos)
                .coerceAtLeast(1L) / 1e9
        )

        if (dt > config.maximumGapSeconds) {
            history = History(
                timestampNanos = input.timestampNanos,
                centroid = centroid,
                normalized = normalized,
                previousEdgeLengths = buildEdgeMap(normalized),
                shapeVelocityMetersPerSecond = 0.0,
                shapeAccelerationMetersPerSecondSquared = 0.0,
                confidenceEma = input.baseConfidence.coerceIn(0.0, 1.0),
                warningFrames = 0,
                rejectFrames = 0,
                baselineNormalized = normalized,
            )

            return Result(
                timestampNanos = input.timestampNanos,
                accepted = input.baseConfidence >= config.minimumConfidence,
                confidence = input.baseConfidence.coerceIn(0.0, 1.0) *
                    0.88,
                rigidMotionMeters =
                    centroid.distanceTo(previous.centroid),
                shapeVelocityMetersPerSecond = 0.0,
                shapeAccelerationMetersPerSecondSquared = 0.0,
                medianEdgeRatio = 1.0,
                medianRadialRatio = 1.0,
                edgeInlierRatio = 1.0,
                radialInlierRatio = 1.0,
                persistentWarningFrames = 0,
                persistentRejectFrames = 0,
                warnings = listOf("face-mesh-temporal-history-reset-after-gap"),
                recommendations = listOf(
                    "Keep face-camera timing continuous for dense temporal geometry.",
                ),
            )
        }

        val oldNormalized = previous.normalized
        val sharedIds = normalized.keys.intersect(oldNormalized.keys)
        val edgeRatios = ArrayList<Double>()
        val radialRatios = ArrayList<Double>()
        var edgeInliers = 0
        var edgeTotal = 0
        var radialInliers = 0
        var radialTotal = 0

        val currentEdges = buildEdgeMap(normalized)
        val currentRadial = normalized.mapValues { (_, value) ->
            value.norm()
        }
        val previousRadial = oldNormalized.mapValues { (_, value) ->
            value.norm()
        }

        currentEdges.forEach { (key, currentLength) ->
            val previousLength = previous.previousEdgeLengths[key]
                ?: return@forEach
            if (previousLength < 1e-5 || currentLength < 1e-5) return@forEach
            val ratio = (
                currentLength / previousLength
            ).coerceIn(0.05, 20.0)
            edgeRatios += ratio
            edgeTotal += 1
            if (
                ratio in
                    config.edgeRatioMinimum..config.edgeRatioMaximum
            ) {
                edgeInliers += 1
            }
        }

        sharedIds.forEach { id ->
            val currentRadius = currentRadial[id] ?: return@forEach
            val previousRadius = previousRadial[id] ?: return@forEach
            if (previousRadius < 1e-5 || currentRadius < 1e-5) return@forEach
            val ratio = (
                currentRadius / previousRadius
            ).coerceIn(0.05, 20.0)
            radialRatios += ratio
            radialTotal += 1
            if (
                ratio in
                    config.radialRatioMinimum..config.radialRatioMaximum
            ) {
                radialInliers += 1
            }
        }

        val medianEdgeRatio = median(edgeRatios, 1.0)
        val medianRadialRatio = median(radialRatios, 1.0)
        val edgeInlierRatio =
            if (edgeTotal == 0) 0.0
            else edgeInliers.toDouble() / edgeTotal
        val radialInlierRatio =
            if (radialTotal == 0) 0.0
            else radialInliers.toDouble() / radialTotal

        val edgeChanges = currentEdges.mapNotNull { (key, currentLength) ->
            val previousLength =
                previous.previousEdgeLengths[key]
                    ?: return@mapNotNull null
            abs(currentLength - previousLength)
        }

        val radialChanges = sharedIds.mapNotNull { id ->
            val currentRadius = currentRadial[id] ?: return@mapNotNull null
            val previousRadius = previousRadial[id] ?: return@mapNotNull null
            abs(currentRadius - previousRadius)
        }

        val rotationInvariantShapeChange = median(
            edgeChanges + radialChanges,
            0.0,
        )

        val shapeVelocity = rotationInvariantShapeChange / dt

        val shapeAcceleration = abs(
            shapeVelocity -
                previous.shapeVelocityMetersPerSecond,
        ) / dt

        val edgeShapeScore = ratioScore(
            medianEdgeRatio,
            config.edgeRatioMinimum,
            config.edgeRatioMaximum,
        )

        val radialShapeScore = ratioScore(
            medianRadialRatio,
            config.radialRatioMinimum,
            config.radialRatioMaximum,
        )

        val velocityScore = (
            1.0 -
                (
                    shapeVelocity /
                        config.maximumShapeVelocityMetersPerSecond
                )
        ).coerceIn(0.0, 1.0)

        val accelerationScore = (
            1.0 -
                (
                    shapeAcceleration /
                        config.maximumShapeAccelerationMetersPerSecondSquared
                )
        ).coerceIn(0.0, 1.0)

        val reprojectionScore =
            reprojectionScore(input.meanReprojectionErrorPx)

        val confidence = (
            0.24 * edgeShapeScore +
                0.18 * radialShapeScore +
                0.18 * edgeInlierRatio +
                0.12 * radialInlierRatio +
                0.12 * velocityScore +
                0.08 * accelerationScore +
                0.08 * reprojectionScore
        ).coerceIn(0.0, 1.0)

        val rigidMotionMeters =
            centroid.distanceTo(previous.centroid)

        previous.shapeAccelerationMetersPerSecondSquared =
            previous.shapeAccelerationMetersPerSecondSquared * 0.70 +
                shapeAcceleration * 0.30
        previous.shapeVelocityMetersPerSecond =
            previous.shapeVelocityMetersPerSecond * 0.72 +
                shapeVelocity * 0.28
        previous.confidenceEma =
            previous.confidenceEma * (1.0 - config.historyAlpha) +
                confidence * config.historyAlpha

        val baselineScaleRisk =
            if (previous.baselineNormalized.isEmpty()) {
                0.0
            } else {
                previous.baselineNormalized.keys.intersect(normalized.keys)
                    .mapNotNull { id ->
                        val base = previous.baselineNormalized[id] ?: return@mapNotNull null
                        val current = normalized[id] ?: return@mapNotNull null
                        val baseNorm = base.norm()
                        val currentNorm = current.norm()
                        if (baseNorm < 1e-8 || currentNorm < 1e-8) {
                            null
                        } else {
                            val ratio =
                                (currentNorm / baseNorm)
                                    .coerceIn(0.05, 20.0)
                            val symmetryRatio =
                                if (ratio <= 1.0) {
                                    ratio
                                } else {
                                    1.0 / ratio
                                }
                            (1.0 - symmetryRatio).coerceIn(0.0, 1.0)
                        }
                    }.let { values ->
                        if (values.isEmpty()) 0.0 else values.average()
                    }
            }

        val warning =
            confidence < config.warningScore ||
                baselineScaleRisk >= 0.25
        val reject = confidence < config.rejectScore ||
            baselineScaleRisk >= 0.40 ||
            edgeInlierRatio < 0.35 ||
            radialInlierRatio < 0.35

        if (warning) {
            previous.warningFrames += 1
        } else {
            previous.warningFrames =
                max(0, previous.warningFrames - 1)
        }

        if (reject) {
            previous.rejectFrames += 1
        } else {
            previous.rejectFrames =
                max(0, previous.rejectFrames - 1)
        }

        previous.timestampNanos = input.timestampNanos
        previous.centroid = centroid
        previous.normalized = normalized
        previous.previousEdgeLengths = currentEdges

        val warnings = buildWarnings(
            confidence = confidence,
            edgeInlierRatio = edgeInlierRatio,
            radialInlierRatio = radialInlierRatio,
            shapeVelocity = shapeVelocity,
            shapeAcceleration = shapeAcceleration,
            meanReprojectionErrorPx = input.meanReprojectionErrorPx,
            warningFrames = previous.warningFrames,
            rejectFrames = previous.rejectFrames,
        )

        val recommendations = buildRecommendations(
            confidence = confidence,
            edgeInlierRatio = edgeInlierRatio,
            radialInlierRatio = radialInlierRatio,
            shapeVelocity = shapeVelocity,
            shapeAcceleration = shapeAcceleration,
            meanReprojectionErrorPx = input.meanReprojectionErrorPx,
        )

        val accepted = previous.rejectFrames < config.persistentRejectFrames &&
            confidence >= config.rejectScore &&
            input.baseConfidence >= config.minimumConfidence

        return Result(
            timestampNanos = input.timestampNanos,
            accepted = accepted,
            confidence = (
                previous.confidenceEma * 0.45 +
                    confidence * 0.55
            ).coerceIn(0.0, 1.0),
            rigidMotionMeters = rigidMotionMeters,
            shapeVelocityMetersPerSecond = shapeVelocity,
            shapeAccelerationMetersPerSecondSquared = shapeAcceleration,
            medianEdgeRatio = medianEdgeRatio,
            medianRadialRatio = medianRadialRatio,
            edgeInlierRatio = edgeInlierRatio,
            radialInlierRatio = radialInlierRatio,
            persistentWarningFrames = previous.warningFrames,
            persistentRejectFrames = previous.rejectFrames,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    @Synchronized
    fun reset() {
        history = null
    }

    private fun buildEdgeMap(
        points: Map<Int, Vec3>,
    ): Map<PairKey, Double> {
        val grouped = points.entries
            .sortedBy { it.key }
            .groupBy { it.key / 1000 }

        val result = HashMap<PairKey, Double>()
        grouped.values.forEach { contourPoints ->
            val sorted = contourPoints.sortedBy { it.key % 1000 }
            for (index in 0 until sorted.lastIndex) {
                val a = sorted[index]
                val b = sorted[index + 1]
                val key = PairKey(
                    minOf(a.key, b.key),
                    maxOf(a.key, b.key),
                )
                result[key] = a.value.distanceTo(b.value)
            }
        }
        return result
    }

    private fun centroid(points: List<Vec3>): Vec3 =
        if (points.isEmpty()) {
            Vec3(0.0, 0.0, 0.0)
        } else {
            Vec3(
                points.sumOf { it.x } / points.size,
                points.sumOf { it.y } / points.size,
                points.sumOf { it.z } / points.size,
            )
        }

    private fun ratioScore(
        ratio: Double,
        minimum: Double,
        maximum: Double,
    ): Double {
        if (!ratio.isFinite()) return 0.0
        if (ratio in minimum..maximum) return 1.0

        return when {
            ratio < minimum ->
                (ratio / minimum).coerceIn(0.0, 1.0)
            else ->
                (maximum / ratio).coerceIn(0.0, 1.0)
        }
    }

    private fun reprojectionScore(errorPx: Double): Double =
        if (!errorPx.isFinite()) {
            0.0
        } else {
            exp(-errorPx / 4.0).coerceIn(0.0, 1.0)
        }

    private fun median(
        values: List<Double>,
        fallback: Double,
    ): Double {
        val sorted = values.filter { it.isFinite() }.sorted()
        if (sorted.isEmpty()) return fallback
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }

    private fun buildWarnings(
        confidence: Double,
        edgeInlierRatio: Double,
        radialInlierRatio: Double,
        shapeVelocity: Double,
        shapeAcceleration: Double,
        meanReprojectionErrorPx: Double,
        warningFrames: Int,
        rejectFrames: Int,
    ): List<String> = buildList {
        if (confidence < config.warningScore) {
            add("face-mesh-temporal-consistency-degraded")
        }
        if (edgeInlierRatio < 0.60) {
            add("facial-contour-edge-length-consistency-degraded")
        }
        if (radialInlierRatio < 0.60) {
            add("facial-shape-radial-consistency-degraded")
        }
        if (shapeVelocity > config.maximumShapeVelocityMetersPerSecond) {
            add("facial-shape-velocity-exceeds-production-envelope")
        }
        if (
            shapeAcceleration >
                config.maximumShapeAccelerationMetersPerSecondSquared
        ) {
            add("facial-shape-acceleration-exceeds-production-envelope")
        }
        if (meanReprojectionErrorPx > 6.0) {
            add("facial-reprojection-error-is-elevated")
        }
        if (warningFrames >= config.persistentWarningFrames) {
            add("persistent-facial-temporal-instability")
        }
        if (rejectFrames >= config.persistentRejectFrames) {
            add("persistent-facial-geometry-instability")
        }
    }.distinct()

    private fun buildRecommendations(
        confidence: Double,
        edgeInlierRatio: Double,
        radialInlierRatio: Double,
        shapeVelocity: Double,
        shapeAcceleration: Double,
        meanReprojectionErrorPx: Double,
    ): List<String> = buildList {
        if (edgeInlierRatio < 0.60) {
            add("reduce facial occlusion and verify contour tracking around eyes, nose and lips")
        }
        if (radialInlierRatio < 0.60) {
            add("keep facial scale stable and verify that the face detector is not switching tracks")
        }
        if (shapeVelocity > config.maximumShapeVelocityMetersPerSecond) {
            add("verify timestamp alignment before treating rapid facial deformation as real motion")
        }
        if (
            shapeAcceleration >
                config.maximumShapeAccelerationMetersPerSecondSquared
        ) {
            add("reduce intermittent contour loss and reacquire the same face track")
        }
        if (meanReprojectionErrorPx > 6.0) {
            add("verify face-camera calibration and lens distortion parameters")
        }
        if (confidence < config.rejectScore) {
            add("hold the latest facial mesh until temporal geometry recovers")
        }
    }.distinct()
}

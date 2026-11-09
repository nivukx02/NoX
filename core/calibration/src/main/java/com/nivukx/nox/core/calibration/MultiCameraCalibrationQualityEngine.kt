
package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Evaluates whether a multi-camera rig is geometrically strong enough for production capture.
 *
 * Calibration is separated from reconstruction. A weak multi-view rig must not silently provide
 * metric truth when monocular RGB can still operate safely.
 */
class MultiCameraCalibrationQualityEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumCameraCount: Int = 2,
        val minimumCoverage: Double = 0.72,
        val maximumReprojectionErrorPx: Double = 3.5,
        val maximumClockSkewMillis: Double = 4.0,
        val minimumBaselineMeters: Double = 0.08,
        val maximumBaselineMeters: Double = 18.0,
        val minimumPairConfidence: Double = 0.54,
        val minimumTriangulationAngleDegrees: Double = 4.0,
        val preferredTriangulationAngleDegrees: Double = 12.0,
    )

    data class Camera(
        val id: String,
        val positionMeters: Vec3,
        val forward: Vec3,
        val reprojectionErrorPx: Double,
        val clockOffsetNanos: Long,
        val confidence: Double,
    )

    data class Pair(
        val a: String,
        val b: String,
        val baselineMeters: Double,
        val triangulationAngleDegrees: Double,
        val reprojectionErrorPx: Double,
        val clockSkewMillis: Double,
        val confidence: Double,
    )

    data class Result(
        val ready: Boolean,
        val cameraCount: Int,
        val pairCount: Int,
        val coverageScore: Double,
        val reprojectionScore: Double,
        val timingScore: Double,
        val baselineScore: Double,
        val triangulationScore: Double,
        val pairwiseRedundancyScore: Double = 0.0,
        val baselineDiversityScore: Double = 0.0,
        val angularDiversityScore: Double = 0.0,
        val rigScore: Double,
        val recommendations: List<String>,
        val diagnostics: List<String>,
    )

    fun evaluate(
        cameras: List<Camera>,
        observedCoverage: Double,
    ): Result =
        evaluate(
            cameras = cameras,
            observedCoverage = observedCoverage,
            drift = null,
        )

    fun evaluate(
        cameras: List<Camera>,
        observedCoverage: Double,
        drift: MultiCameraCalibrationDriftEngine.Snapshot?,
    ): Result {
        val pairs = buildPairs(cameras)
        val coverage = observedCoverage.coerceIn(0.0, 1.0)

        val reprojection = if (cameras.isEmpty()) {
            0.0
        } else {
            val mean = cameras
                .map { it.reprojectionErrorPx.coerceAtLeast(0.0) }
                .average()

            expScore(
                value = mean,
                good = config.maximumReprojectionErrorPx * 0.45,
                fail = config.maximumReprojectionErrorPx,
            )
        }

        val timing = if (cameras.isEmpty()) {
            0.0
        } else {
            val worstMillis = cameras.maxOf {
                abs(it.clockOffsetNanos).toDouble() / 1e6
            }

            expScore(
                value = worstMillis,
                good = config.maximumClockSkewMillis * 0.35,
                fail = config.maximumClockSkewMillis,
            )
        }

        val baseline = if (pairs.isEmpty()) {
            0.0
        } else {
            pairs.map { pair ->
                when {
                    pair.baselineMeters < config.minimumBaselineMeters ->
                        0.15
                    pair.baselineMeters > config.maximumBaselineMeters ->
                        0.45
                    else ->
                        1.0
                }
            }.average()
        }

        val triangulation = if (pairs.isEmpty()) {
            0.0
        } else {
            pairs.map { pair ->
                when {
                    pair.triangulationAngleDegrees <
                        config.minimumTriangulationAngleDegrees ->
                        0.15
                    pair.triangulationAngleDegrees <
                        config.preferredTriangulationAngleDegrees ->
                        0.62
                    else ->
                        1.0
                } * pair.confidence.coerceIn(0.0, 1.0)
            }.average()
        }

        val pairCoverage = pairs
            .map { it.confidence }
            .averageOrZero()

        val pairwiseRedundancyScore =
            if (cameras.size < 3) {
                if (pairs.any { it.confidence >= config.minimumPairConfidence }) {
                    0.45
                } else {
                    0.0
                }
            } else {
                val threshold =
                    config.minimumPairConfidence.coerceIn(0.0, 1.0)
                val trustedPairs =
                    pairs.count { it.confidence >= threshold }
                val possiblePairs =
                    cameras.size * (cameras.size - 1) / 2
                (
                    trustedPairs.toDouble() /
                        possiblePairs.coerceAtLeast(1).toDouble()
                ).coerceIn(0.0, 1.0)
            }

        val baselineDiversityScore =
            if (pairs.size < 2) {
                0.0
            } else {
                val baselines =
                    pairs.map { it.baselineMeters }
                        .filter { it.isFinite() && it > 0.0 }
                if (baselines.size < 2) {
                    0.0
                } else {
                    val mean = baselines.average().coerceAtLeast(1e-6)
                    val coefficientOfVariation =
                        (
                            baselines
                                .map { abs(it - mean) / mean }
                                .average()
                            ).coerceIn(0.0, 1.0)
                    (
                        1.0 -
                            coefficientOfVariation * 0.75
                        ).coerceIn(0.0, 1.0)
                }
            }

        val angularDiversityScore =
            if (pairs.isEmpty()) {
                0.0
            } else {
                val usefulAngles =
                    pairs
                        .map { it.triangulationAngleDegrees }
                        .filter { it.isFinite() }
                if (usefulAngles.isEmpty()) {
                    0.0
                } else {
                    val preferred =
                        config.preferredTriangulationAngleDegrees
                            .coerceAtLeast(
                                config.minimumTriangulationAngleDegrees,
                            )
                    (
                        usefulAngles
                            .map {
                                (
                                    it /
                                        preferred
                                ).coerceIn(0.0, 1.0)
                            }
                            .average()
                    ).coerceIn(0.0, 1.0)
                }
            }

        val cameraCountScore = (
            cameras.size.toDouble() /
                config.minimumCameraCount.toDouble()
            ).coerceIn(0.0, 1.0)

        val driftTrust = drift?.rigTrust ?: 1.0

        val rigScore = (
            cameraCountScore * 0.06 +
                coverage * 0.20 +
                reprojection * 0.18 +
                timing * 0.12 +
                baseline * 0.10 +
                triangulation * 0.16 +
                pairCoverage * 0.05 +
                pairwiseRedundancyScore * 0.05 +
                baselineDiversityScore * 0.04 +
                angularDiversityScore * 0.04
            ).coerceIn(0.0, 1.0)

        val ready =
            cameras.size >= config.minimumCameraCount &&
                coverage >= config.minimumCoverage &&
                reprojection >= 0.50 &&
                timing >= 0.50 &&
                baseline >= 0.50 &&
                triangulation >= 0.45 &&
                pairwiseRedundancyScore >=
                    (if (cameras.size >= 4) 0.40 else 0.20) &&
                angularDiversityScore >= 0.45 &&
                driftTrust >= 0.45 &&
                rigScore >= 0.60 &&
                drift?.hardFailure != true

        val recommendations = buildList {
            if (cameras.size < config.minimumCameraCount) {
                add("Connect at least " + config.minimumCameraCount + " calibrated cameras.")
            }
            if (coverage < config.minimumCoverage) {
                add("Increase subject-space camera coverage before relying on multi-view metric scale.")
            }
            if (reprojection < 0.60) {
                add("Recalibrate intrinsics/extrinsics and inspect lens distortion.")
            }
            if (timing < 0.60) {
                add("Improve peer clock synchronization before fast-motion capture.")
            }
            if (baseline < 0.50) {
                add("Adjust camera spacing to produce a useful stereo baseline.")
            }
            if (triangulation < 0.55) {
                add("Increase angular separation while keeping cameras on overlapping subject space.")
            }
            if (pairs.any { it.confidence < config.minimumPairConfidence }) {
                add("Remove or recalibrate low-confidence camera pairs.")
            }
            if (pairwiseRedundancyScore < 0.40 && cameras.size >= 4) {
                add("Increase camera redundancy so multiple independent pairs support each region.")
            }
            if (baselineDiversityScore < 0.45) {
                add("Reposition cameras to avoid a rig dominated by one narrow baseline.")
            }
            if (angularDiversityScore < 0.45) {
                add("Increase angular diversity around the subject while preserving overlapping visibility.")
            }
            drift?.let {
                if (it.hardFailure) {
                    add("Hold multi-view metric output until rig drift is recalibrated.")
                } else if (it.recalibrationRecommended) {
                    add("Run a live drift-aware calibration pass before high-speed capture.")
                }
            }
            if (!ready) {
                add("Use monocular RGB mode as the metric-reconstruction fallback.")
            }
        }

        val diagnostics = buildList {
            add("cameraCount=" + cameras.size)
            add("pairCount=" + pairs.size)
            add("coverage=" + format(coverage))
            add("reprojectionScore=" + format(reprojection))
            add("timingScore=" + format(timing))
            add("baselineScore=" + format(baseline))
            add("triangulationScore=" + format(triangulation))
            add("pairCoverage=" + format(pairCoverage))
            add("pairwiseRedundancyScore=" + format(pairwiseRedundancyScore))
            add("baselineDiversityScore=" + format(baselineDiversityScore))
            add("angularDiversityScore=" + format(angularDiversityScore))
            add("driftTrust=" + format(driftTrust))
            add("rigScore=" + format(rigScore))
        }

        return Result(
            ready = ready,
            cameraCount = cameras.size,
            pairCount = pairs.size,
            coverageScore = coverage,
            reprojectionScore = reprojection,
            timingScore = timing,
            baselineScore = baseline,
            triangulationScore = triangulation,
            pairwiseRedundancyScore = pairwiseRedundancyScore,
            baselineDiversityScore = baselineDiversityScore,
            angularDiversityScore = angularDiversityScore,
            rigScore = rigScore,
            recommendations = recommendations.distinct(),
            diagnostics = diagnostics,
        )
    }

    private fun buildPairs(
        cameras: List<Camera>,
    ): List<Pair> {
        if (cameras.size < 2) return emptyList()

        val result = ArrayList<Pair>(
            cameras.size * (cameras.size - 1) / 2,
        )

        for (i in 0 until cameras.lastIndex) {
            for (j in i + 1 until cameras.size) {
                val a = cameras[i]
                val b = cameras[j]

                val baseline = magnitude(
                    b.positionMeters - a.positionMeters,
                )

                val angle = angleBetween(
                    a.forward,
                    b.forward * -1.0,
                )

                val reprojection = (
                    a.reprojectionErrorPx +
                        b.reprojectionErrorPx
                    ) * 0.5

                val clock = (
                    abs(a.clockOffsetNanos - b.clockOffsetNanos)
                    ).toDouble() / 1e6

                val reprojectionScore = expScore(
                    reprojection,
                    config.maximumReprojectionErrorPx * 0.45,
                    config.maximumReprojectionErrorPx,
                )

                val clockScore = expScore(
                    clock,
                    config.maximumClockSkewMillis * 0.35,
                    config.maximumClockSkewMillis,
                )

                val angleScore = when {
                    angle < config.minimumTriangulationAngleDegrees ->
                        0.15
                    angle < config.preferredTriangulationAngleDegrees ->
                        0.70
                    else ->
                        1.0
                }

                val confidence = (
                    a.confidence.coerceIn(0.0, 1.0) *
                        b.confidence.coerceIn(0.0, 1.0) *
                        reprojectionScore *
                        clockScore *
                        angleScore
                    ).coerceIn(0.0, 1.0)

                result += Pair(
                    a = a.id,
                    b = b.id,
                    baselineMeters = baseline,
                    triangulationAngleDegrees = angle,
                    reprojectionErrorPx = reprojection,
                    clockSkewMillis = clock,
                    confidence = confidence,
                )
            }
        }

        return result
    }

    private fun expScore(
        value: Double,
        good: Double,
        fail: Double,
    ): Double {
        if (!value.isFinite()) return 0.0
        if (value <= good) return 1.0
        if (value >= fail) return 0.15

        val span = (fail - good).coerceAtLeast(1e-6)
        val normalized = (value - good) / span

        return (
            1.0 -
                normalized * 0.85
            ).coerceIn(0.15, 1.0)
    }

    private fun angleBetween(
        a: Vec3,
        b: Vec3,
    ): Double {
        val ma = magnitude(a)
        val mb = magnitude(b)
        if (ma < 1e-8 || mb < 1e-8) return 0.0

        val dot = (
            a.x * b.x +
                a.y * b.y +
                a.z * b.z
            ) / (ma * mb)

        return Math.toDegrees(
            kotlin.math.acos(
                dot.coerceIn(-1.0, 1.0),
            ),
        )
    }

    private fun magnitude(value: Vec3): Double =
        sqrt(
            value.x * value.x +
                value.y * value.y +
                value.z * value.z,
        )

    private fun format(value: Double): String =
        if (!value.isFinite()) {
            "nan"
        } else {
            "%.4f".format(java.util.Locale.US, value)
        }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}

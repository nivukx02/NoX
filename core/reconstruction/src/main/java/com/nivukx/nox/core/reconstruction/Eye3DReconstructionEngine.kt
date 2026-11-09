package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.EyeObservation
import com.nivukx.nox.core.model.EyeSide
import com.nivukx.nox.core.model.EyeTrackingFrame
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max

/**
 * Multi-phone 3D eye center and gaze fusion.
 *
 * Eye centers are triangulated from synchronized camera observations. Gaze directions are rotated
 * into the calibrated rig frame and fused by confidence-weighted spherical averaging. The angular
 * disagreement is reported so a bad face camera cannot masquerade as a precise gaze estimate.
 */
class Eye3DReconstructionEngine(
    private val historyPerDevice: Int = 16,
    private val maxTimestampSkewNanos: Long = 35_000_000L,
    private val maximumDevices: Int = 128,
) {
    init {
        require(historyPerDevice in 2..256)
        require(maxTimestampSkewNanos > 0L)
        require(maximumDevices in 16..128)
    }
    data class Result(
        val eye: EyeSide,
        val centerWorld: Vec3,
        val gazeDirectionWorld: Vec3,
        val confidence: Double,
        val angularSpreadDegrees: Double,
        val observations: Int,
        val centerUncertaintyMeters: Double = 0.0,
        val uncertainty95MajorMeters: Double = 0.0,
        val uncertaintyMajorMeters: Double = 0.0,
        val uncertaintyMiddleMeters: Double = 0.0,
        val uncertaintyMinorMeters: Double = 0.0,
        val uncertaintyDepthMeters: Double = 0.0,
        val uncertaintyAnisotropyRatio: Double = 1.0,
        val uncertaintyDepthDominance: Double = 0.0,
        val uncertaintyConfidence: Double = 1.0,
        val conditionScore: Double = 0.0,
        val cheiralityRatio: Double = 1.0,
        val robustReprojectionP95Px: Double = Double.POSITIVE_INFINITY,
        val viewDiversityScore: Double = 0.0,
        val gazeStability: Double = 0.0,
        val angularVelocityDegreesPerSecond: Double = 0.0,
        val parallaxDegrees: Double = 0.0,
        val symmetricEpipolarConsistencyScore: Double = 1.0,
        val symmetricEpipolarEvaluatedPairCount: Int = 0,
        val metricEvidenceAuthority: Double = 0.0,
        val symmetricEpipolarWarnings: List<String> = emptyList(),
        val meanIrisRadiusNormalized: Double = 0.0,
        val meanIrisStructureConfidence: Double = 0.0,
        val meanEyelidOcclusion: Double = 0.0,
    )

    private data class DeviceFrame(
        val deviceId: DeviceId,
        val frame: EyeTrackingFrame,
    )

    private val history = HashMap<DeviceId, ArrayDeque<EyeTrackingFrame>>()

    private data class TemporalGazeState(
        var direction: Vec3,
        var timestampNanos: Long,
        var angularVelocityDegreesPerSecond: Double,
        var stability: Double,
    )

    private val temporalGaze =
        java.util.EnumMap<EyeSide, TemporalGazeState>(
            EyeSide::class.java,
        )
    private val triangulator = TriangulationEngine(
        minParallaxDegrees = 0.25,
        huberDeltaPx = 2.5,
    )

    @Synchronized
    fun push(deviceId: DeviceId, frame: EyeTrackingFrame) {
        val queue = history.getOrPut(deviceId) { ArrayDeque() }
        while (queue.size >= historyPerDevice) queue.removeFirst()
        queue.addLast(frame)
        enforceDeviceCapacity()
    }

    @Synchronized
    private fun enforceDeviceCapacity() {
        while (history.size > maximumDevices) {
            val victim = history.minByOrNull { (_, queue) ->
                queue.lastOrNull()?.timestampNanos ?: Long.MIN_VALUE
            }?.key ?: return
            history.remove(victim)
        }
    }

    @Synchronized
    fun reconstruct(
        timestampNanos: Long,
        calibration: CalibrationSnapshot,
        eye: EyeSide,
    ): Result? {
        val selected = history.mapNotNull { (deviceId, queue) ->
            val frame = queue.minByOrNull { abs(it.timestampNanos - timestampNanos) }
                ?: return@mapNotNull null
            if (abs(frame.timestampNanos - timestampNanos) > maxTimestampSkewNanos) {
                return@mapNotNull null
            }
            val observation = if (eye == EyeSide.LEFT) frame.left else frame.right
            val intrinsics = calibration.intrinsics[deviceId]
            val extrinsics = calibration.extrinsics[deviceId]
            if (observation == null || intrinsics == null || extrinsics == null) {
                return@mapNotNull null
            }
            Triple(deviceId, frame, observation)
        }

        if (selected.size < 2) return null

        val observations = selected.map { (deviceId, _, observation) ->
            val intrinsics = calibration.intrinsics.getValue(deviceId)
            val pose = calibration.extrinsics.getValue(deviceId).pose
            val sensorPoint =
                observation.irisCenter
                    ?: observation.center

            TriangulationEngine.Observation(
                cameraPose = pose,
                intrinsics = intrinsics,
                cameraId = deviceId.value,
                u = sensorPoint.x,
                v = sensorPoint.y,
                weight = observation.confidence,
                visibility =
                    observation.confidence
                        .coerceIn(0.0, 1.0),
                sourceQuality =
                    frame.captureTimingConfidence
                        .coerceIn(0.0, 1.0),
                temporalAlignmentConfidence =
                    frame.captureTimingConfidence
                        .coerceIn(0.0, 1.0),
                temporalUncertaintyNanos =
                    frame.frameDurationNanos
                        .coerceAtLeast(0L),
                calibrationUncertainty =
                    calibration.uncertaintyProfiles[deviceId]
                        ?: com.nivukx.nox.core.model.CalibrationUncertaintyProfile(),
                motionBlurRisk =
                    frame.estimatedMotionBlurRisk,
            )
        }

        val triangulated = triangulator.robust(observations) ?: return null

        data class GazeVector(
            val direction: Vec3,
            val baseWeight: Double,
        )

        val gazeVectors = selected.mapNotNull { (deviceId, _, observation) ->
            if (observation.confidence <= 0.01) return@mapNotNull null
            val cameraPose =
                calibration.extrinsics[deviceId]?.pose
                    ?: return@mapNotNull null
            val directionWorld =
                cameraPose.rotation
                    .rotate(observation.gazeDirection)
                    .normalized()
            if (directionWorld.norm() < 1e-6) return@mapNotNull null

            val opennessWeight =
                (
                    0.35 +
                        0.65 *
                            observation.openness
                                .coerceIn(0.0, 1.0)
                    )
            val blinkWeight =
                if (observation.blink) 0.35 else 1.0

            GazeVector(
                direction = directionWorld,
                baseWeight =
                    observation.confidence
                        .coerceIn(0.01, 1.0) *
                        opennessWeight *
                        blinkWeight,
            )
        }

        if (gazeVectors.size < 2) return null

        var averaged =
            gazeVectors
                .fold(Vec3(0.0, 0.0, 0.0)) { sum, vector ->
                    sum + vector.direction * vector.baseWeight
                }
                .normalized()

        repeat(robustGazeIterations) {
            val robustSum =
                gazeVectors.fold(Vec3(0.0, 0.0, 0.0)) { sum, vector ->
                    val angleDegrees =
                        Math.toDegrees(
                            acos(
                                vector.direction
                                    .dot(averaged)
                                    .coerceIn(-1.0, 1.0),
                            ),
                        )

                    val normalizedResidual =
                        angleDegrees /
                            gazeOutlierScaleDegrees
                                .coerceAtLeast(1e-6)

                    val cauchyWeight =
                        1.0 /
                            (
                                1.0 +
                                    normalizedResidual *
                                        normalizedResidual
                            )

                    sum +
                        vector.direction *
                            (
                                vector.baseWeight *
                                    cauchyWeight
                            )
                }

            val refined = robustSum.normalized()
            if (refined.norm() >= 1e-6) {
                averaged = refined
            }
        }

        val worldDirections = gazeVectors.map { it.direction }

        val angularResiduals =
            worldDirections.map { d ->
                Math.toDegrees(
                    acos(
                        d.dot(averaged)
                            .coerceIn(-1.0, 1.0),
                    ),
                )
            }

        val angularSpread = angularResiduals.average()

        val gazeInlierRatio =
            angularResiduals.count {
                it <= gazeOutlierScaleDegrees * 2.5
            }.toDouble() /
                angularResiduals.size.coerceAtLeast(1).toDouble()

        val geometricConfidence = (
            triangulated.inliers.toDouble() /
                max(triangulated.observations, 1)
            ).coerceIn(0.0, 1.0)

        val uncertaintyConfidencePrior =
            if (
                triangulated.uncertaintyMeters.isFinite()
            ) {
                kotlin.math.exp(
                    -triangulated.uncertaintyMeters / 0.02,
                ).coerceIn(0.0, 1.0)
            } else {
                0.0
            }

        val angularConfidence =
            (
                (1.0 - angularSpread / 20.0) * 0.72 +
                    gazeInlierRatio * 0.28
            ).coerceIn(0.0, 1.0)

        val parallaxConfidence =
            (
                1.0 -
                    kotlin.math.exp(
                        -triangulated.parallaxDegrees / 1.5,
                    )
                ).coerceIn(0.0, 1.0)

        val uncertaintyConfidence =
            triangulated.uncertaintyConfidence
                .coerceIn(0.0, 1.0)

        val geometryConfidence =
            (
                triangulated.inliers.toDouble() /
                    max(triangulated.observations, 1)
                ).coerceIn(0.0, 1.0) *
                triangulated.conditionScore
                    .coerceIn(0.0, 1.0) *
                triangulated.viewDiversityScore
                    .coerceIn(0.0, 1.0)

        val observationFreshness =
            selected
                .map {
                    (
                        1.0 -
                            abs(
                                it.second.timestampNanos -
                                    timestampNanos,
                            ).toDouble() /
                            maxTimestampSkewNanos.toDouble()
                        )
                    .coerceIn(0.0, 1.0)
                }
                .average()

        val metricEvidenceAuthority =
            triangulated.metricEvidenceAuthority
                .coerceIn(0.0, 1.0)

        val confidence = (
            geometryConfidence * 0.24 +
                uncertaintyConfidence * 0.11 +
                uncertaintyConfidencePrior * 0.09 +
                angularConfidence * 0.09 +
                directionConsistencyPreview(
                    angularSpread,
                ) * 0.08 +
                triangulated.symmetricEpipolarConsistencyScore
                    .coerceIn(0.0, 1.0) * 0.08 +
                (
                    1.0 -
                        triangulated.uncertaintyMeters / 0.08
                    ).coerceIn(0.0, 1.0) * 0.08 +
                (0.55 + 0.45 * parallaxConfidence) * 0.07 +
                observationFreshness * 0.05 +
                selected.map { it.third.confidence }.average() * 0.05 +
                gazeInlierRatio * 0.02 +
                metricEvidenceAuthority * 0.04
            ).coerceIn(0.0, 1.0)

        val previousTemporal = temporalGaze[eye]

        val angularVelocity =
            if (
                previousTemporal == null ||
                    timestampNanos <= previousTemporal.timestampNanos
            ) {
                0.0
            } else {
                val dt =
                    (
                        timestampNanos -
                            previousTemporal.timestampNanos
                        ).toDouble() / 1e9

                val delta =
                    Math.toDegrees(
                        acos(
                            previousTemporal.direction
                                .dot(averaged)
                                .coerceIn(-1.0, 1.0),
                        ),
                    )

                delta / dt.coerceAtLeast(1e-4)
            }

        val directionConsistency =
            (
                1.0 -
                    angularSpread /
                        18.0
                ).coerceIn(0.0, 1.0)

        val motionConsistency =
            (
                1.0 -
                    angularVelocity /
                        600.0
                ).coerceIn(0.0, 1.0)

        val temporalStability =
            if (previousTemporal == null) {
                directionConsistency * 0.72 +
                    selected.map { it.third.confidence }.average() * 0.28
            } else {
                (
                    previousTemporal.stability * 0.62 +
                        directionConsistency * 0.20 +
                        motionConsistency * 0.18
                    ).coerceIn(0.0, 1.0)
            }

        if (
            previousTemporal == null ||
            timestampNanos > previousTemporal.timestampNanos
        ) {
            temporalGaze[eye] =
                TemporalGazeState(
                    direction = averaged,
                    timestampNanos = timestampNanos,
                    angularVelocityDegreesPerSecond = angularVelocity,
                    stability = temporalStability,
                )
        }

        return Result(
            eye = eye,
            centerWorld = triangulated.point,
            gazeDirectionWorld = averaged,
            confidence = confidence,
            angularSpreadDegrees = angularSpread,
            observations = selected.size,
            centerUncertaintyMeters = triangulated.uncertaintyMeters,
            uncertainty95MajorMeters =
                triangulated.uncertainty95MajorMeters,
            uncertaintyMajorMeters =
                triangulated.uncertaintyMajorMeters,
            uncertaintyMiddleMeters =
                triangulated.uncertaintyMiddleMeters,
            uncertaintyMinorMeters =
                triangulated.uncertaintyMinorMeters,
            uncertaintyDepthMeters =
                triangulated.uncertaintyDepthMeters,
            uncertaintyAnisotropyRatio =
                triangulated.uncertaintyAnisotropyRatio,
            uncertaintyDepthDominance =
                triangulated.uncertaintyDepthDominance,
            uncertaintyConfidence =
                triangulated.uncertaintyConfidence,
            conditionScore =
                triangulated.conditionScore,
            cheiralityRatio =
                triangulated.cheiralityRatio,
            robustReprojectionP95Px =
                triangulated.rmsReprojectionPx,
            viewDiversityScore =
                triangulated.viewDiversityScore,
            gazeStability = temporalStability,
            parallaxDegrees = triangulated.parallaxDegrees,
            angularVelocityDegreesPerSecond = angularVelocity,
            symmetricEpipolarConsistencyScore =
                triangulated.symmetricEpipolarConsistencyScore,
            symmetricEpipolarEvaluatedPairCount =
                triangulated.symmetricEpipolarEvaluatedPairCount,
            metricEvidenceAuthority =
                metricEvidenceAuthority,
            symmetricEpipolarWarnings =
                triangulated.symmetricEpipolarWarnings,
            meanIrisRadiusNormalized =
                selected.map {
                    it.third.irisRadiusNormalized
                }.average().coerceIn(0.0, 1.0),
            meanIrisStructureConfidence =
                selected.map {
                    it.third.irisStructureConfidence
                }.average().coerceIn(0.0, 1.0),
            meanEyelidOcclusion =
                selected.map {
                    it.third.eyelidOcclusion
                }.average().coerceIn(0.0, 1.0),
        )
    }

    private fun directionConsistencyPreview(
        angularSpreadDegrees: Double,
    ): Double =
        (
            1.0 -
                angularSpreadDegrees /
                    25.0
            ).coerceIn(0.0, 1.0)

    @Synchronized
    fun clear() {
        history.clear()
        temporalGaze.clear()
    }
}

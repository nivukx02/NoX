package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.FaceMeshPoint3D
import com.nivukx.nox.core.model.FaceMeshTriangle
import com.nivukx.nox.core.model.Vec3
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Multi-camera dense face surface solver.
 *
 * This engine converts synchronized detector-neutral face mesh frames into a calibrated world-space
 * facial surface. It does not assume that a detector's Z coordinate is metric. Metric depth comes
 * from calibrated multi-view triangulation.
 *
 * Pipeline:
 *  1. maintain a short synchronized history per camera;
 *  2. select the temporally closest usable frame for each calibrated device;
 *  3. triangulate every common point index using the robust multi-view solver;
 *  4. reject points with weak parallax, poor cheirality, or excessive reprojection;
 *  5. derive a surface topology only from triangles whose three vertices survived;
 *  6. aggregate per-point uncertainty into a surface-level quality score.
 *
 * The solver is intentionally stateless with respect to detector identity. Subject association is
 * performed upstream by PerceptionPipeline, while this stage focuses on geometry.
 */
class FaceMeshMultiViewFusionEngine(
    private val historyPerDevice: Int = 12,
    private val maximumTimestampSkewNanos: Long = 45_000_000L,
    private val maximumDevices: Int = 128,
    private val minimumPointConfidence: Double = 0.18,
    private val minimumParallaxDegrees: Double = 0.45,
    private val maximumReprojectionErrorPx: Double = 6.0,
    private val maximumUncertaintyMeters: Double = 0.16,
    private val maximumKinematicPoints: Int = 2_048,
    private val maximumTemporalPoints: Int = 4_096,
    private val precisionMinimumFeasibility: Double = 0.50,
    private val precisionMaximumFirstOrderDepthErrorMillimeters: Double = 1.0,
    private val precisionMinimumTemporalStability: Double = 0.55,
) : AutoCloseable {

    init {
        require(historyPerDevice in 2..256)
        require(maximumTimestampSkewNanos > 0L)
        require(maximumDevices in 16..128)
        require(minimumPointConfidence in 0.0..1.0)
        require(minimumParallaxDegrees > 0.0)
        require(maximumReprojectionErrorPx > 0.0)
        require(maximumUncertaintyMeters > 0.0)
        require(precisionMinimumFeasibility in 0.0..1.0)
        require(precisionMaximumFirstOrderDepthErrorMillimeters > 0.0)
        require(precisionMinimumTemporalStability in 0.0..1.0)
        require(maximumKinematicPoints > 0)
        require(maximumTemporalPoints >= 64)
    }

    data class SurfacePoint(
        val index: Int,
        val positionWorld: Vec3,
        val confidence: Double,
        val uncertaintyMeters: Double,
        val reprojectionErrorPx: Double,
        val parallaxDegrees: Double,
        val sourceCount: Int,
        val conditionScore: Double,
        val pairwiseReprojectionConsensus: Double = 1.0,
        val temporalStability: Double = 1.0,
        val temporalInnovationMeters: Double = 0.0,
        val subMillimeterFeasibilityScore: Double = 0.0,
        val firstOrderDepthErrorMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val reprojectionP95Px: Double = Double.POSITIVE_INFINITY,
        val tailAdjustedFirstOrderDepthErrorMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val tailAdjustedSubMillimeterFeasibilityScore: Double = 0.0,
    )

    data class Result(
        val timestampNanos: Long,
        val points: List<SurfacePoint>,
        val triangles: List<FaceMeshTriangle>,
        val confidence: Double,
        val meanReprojectionErrorPx: Double,
        val p95ReprojectionErrorPx: Double,
        val meanUncertaintyMeters: Double,
        val acceptedPointRatio: Double,
        val sourceDevices: List<DeviceId>,
        val sourceFrameCount: Int,
        val meanTemporalStability: Double = 0.0,
        val meanPairwiseReprojectionConsensus: Double = 1.0,
        val temporalRejectedPointRatio: Double = 0.0,
        val meanSubMillimeterFeasibilityScore: Double = 0.0,
        val meanFirstOrderDepthErrorMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val p95ReprojectionTailAdjustedDepthErrorMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val meanTailAdjustedSubMillimeterFeasibilityScore: Double = 0.0,
        val warnings: List<String>,
    ) {

        val precisionQualified: Boolean
            get() =
                meanSubMillimeterFeasibilityScore >= 0.50 &&
                    meanFirstOrderDepthErrorMillimeters.isFinite() &&
                    meanFirstOrderDepthErrorMillimeters <= 1.0 &&
                    meanTemporalStability >= 0.55

        val pointCount: Int
            get() = points.size

        val triangleCount: Int
            get() = triangles.size

        val metric: Boolean
            get() = points.size >= 3 && sourceDevices.size >= 2
    }

    private data class TimedFrame(
        val deviceId: DeviceId,
        val frame: FaceMeshFrame,
    )

    private val history =
        LinkedHashMap<DeviceId, ArrayDeque<FaceMeshFrame>>()

    private data class TemporalPointState(
        var position: Vec3,
        var velocity: Vec3,
        var acceleration: Vec3,
        var timestampNanos: Long,
        var stability: Double,
    )

    private val temporalPoints =
        HashMap<Int, TemporalPointState>()

    private val kinematicPoints =
        LinkedHashMap<Int, KinematicStabilityEngine>()

    private val geometricConditioningEngine =
        PerObservationGeometricConditioningEngine()

    private val triangulator = TriangulationEngine(
        minParallaxDegrees = minimumParallaxDegrees,
        huberDeltaPx = 2.5,
        cauchyScalePx = 6.0,
        iterations = 10,
        damping = 1e-4,
    )

    @Synchronized
    fun push(
        deviceId: DeviceId,
        frame: FaceMeshFrame,
    ) {
        val queue =
            history.getOrPut(deviceId) {
                ArrayDeque()
            }

        while (queue.isNotEmpty() &&
            frame.timestampNanos <= queue.last().timestampNanos
        ) {
            queue.removeLast()
        }

        queue.addLast(frame)

        while (queue.size > historyPerDevice) {
            queue.removeFirst()
        }
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
    ): Result? {
        pruneTemporalPointState(timestampNanos)
        val frames = selectFrames(
            timestampNanos = timestampNanos,
            calibration = calibration,
        )

        if (frames.size < 2) {
            return null
        }

        val availablePointIds =
            commonPointIds(frames)

        if (availablePointIds.size < 3) {
            return null
        }

        val accepted = ArrayList<SurfacePoint>(
            availablePointIds.size,
        )
        var pointsDroppedAfterTemporalHistory = 0

        availablePointIds.forEach { pointId ->
            val surfacePoint =
                triangulatePoint(
                    pointId = pointId,
                    frames = frames,
                    calibration = calibration,
                    timestampNanos = timestampNanos,
                )

            if (surfacePoint != null) {
                accepted += surfacePoint
            } else if (temporalPoints.containsKey(pointId)) {
                pointsDroppedAfterTemporalHistory += 1
            }
        }

        if (accepted.size < 3) {
            return null
        }

        val referenceFrame =
            frames.maxByOrNull {
                frameQuality(it.frame)
            }?.frame

        val triangles =
            buildSurfaceTriangles(
                points = accepted,
                referenceFrame = referenceFrame,
            )

        val errors =
            accepted
                .map { it.reprojectionErrorPx }
                .filter(Double::isFinite)
                .sorted()

        val uncertainties =
            accepted
                .map { it.uncertaintyMeters }
                .filter(Double::isFinite)

        val acceptedRatio =
            (
                accepted.size.toDouble() /
                    availablePointIds.size.toDouble()
                ).coerceIn(0.0, 1.0)

        val meanError =
            if (errors.isEmpty()) {
                Double.POSITIVE_INFINITY
            } else {
                errors.average()
            }

        val p95Error =
            percentile(errors, 0.95)

        val meanUncertainty =
            if (uncertainties.isEmpty()) {
                Double.POSITIVE_INFINITY
            } else {
                uncertainties.average()
            }

        val geometryScore =
            accepted
                .map {
                    it.confidence *
                        it.conditionScore
                }
                .average()
                .coerceIn(0.0, 1.0)

        val errorScore =
            if (meanError.isFinite()) {
                exp(
                    -meanError /
                        maximumReprojectionErrorPx
                            .coerceAtLeast(0.25),
                ).coerceIn(0.0, 1.0)
            } else {
                0.0
            }

        val uncertaintyScore =
            if (meanUncertainty.isFinite()) {
                exp(
                    -meanUncertainty /
                        maximumUncertaintyMeters
                            .coerceAtLeast(0.01),
                ).coerceIn(0.0, 1.0)
            } else {
                0.0
            }

        val topologyScore =
            if (triangles.isEmpty()) {
                0.0
            } else {
                (
                    triangles.size.toDouble() /
                        max(
                            referenceFrame
                                ?.triangles
                                ?.size
                                ?: triangles.size,
                            1,
                        ).toDouble()
                    ).coerceIn(0.0, 1.0)
            }

        val sourceScore =
            (
                frames.size.toDouble() / 4.0
                ).coerceIn(0.0, 1.0)

        val meanTemporalStability =
            accepted
                .map { it.temporalStability }
                .ifEmpty { listOf(0.0) }
                .average()
                .coerceIn(0.0, 1.0)

        val meanPairwiseReprojectionConsensus =
            accepted
                .map { it.pairwiseReprojectionConsensus }
                .ifEmpty { listOf(0.0) }
                .average()
                .coerceIn(0.0, 1.0)

        val meanSubMillimeterFeasibility =
            accepted
                .map { it.subMillimeterFeasibilityScore }
                .ifEmpty { listOf(0.0) }
                .average()
                .coerceIn(0.0, 1.0)

        val meanFirstOrderDepthErrorMm =
            accepted
                .mapNotNull {
                    it.firstOrderDepthErrorMillimeters
                        .takeIf(Double::isFinite)
                }
                .ifEmpty { emptyList() }
                .averageOrZeroFace()

        val tailAdjustedDepthErrorP95Mm =
            percentile(
                accepted
                    .mapNotNull {
                        it.tailAdjustedFirstOrderDepthErrorMillimeters
                            .takeIf(Double::isFinite)
                    }
                    .sorted(),
                0.95,
            )

        val meanTailAdjustedFeasibility =
            accepted
                .map { it.tailAdjustedSubMillimeterFeasibilityScore }
                .averageOrZeroFace()

        val confidence =
            (
                geometryScore * 0.24 +
                meanPairwiseReprojectionConsensus * 0.10 +
                    acceptedRatio * 0.16 +
                    errorScore * 0.20 +
                    uncertaintyScore * 0.12 +
                    topologyScore * 0.08 +
                    sourceScore * 0.06 +
                    meanTemporalStability * 0.08 +
                    meanSubMillimeterFeasibility * 0.06
                ).coerceIn(0.0, 1.0)

        val warnings =
            buildWarnings(
                sourceFrameCount = frames.size,
                acceptedPointRatio = acceptedRatio,
                meanError = meanError,
                p95Error = p95Error,
                meanUncertainty = meanUncertainty,
                triangleCount = triangles.size,
                confidence = confidence,
            )

        return Result(
            timestampNanos = timestampNanos,
            points = accepted.sortedBy { it.index },
            triangles = triangles,
            confidence = confidence,
            meanReprojectionErrorPx = meanError,
            p95ReprojectionErrorPx = p95Error,
            meanUncertaintyMeters = meanUncertainty,
            acceptedPointRatio = acceptedRatio,
            sourceDevices = frames.map { it.deviceId }
                .distinct()
                .sortedBy { it.value },
            sourceFrameCount = frames.size,
            meanTemporalStability = meanTemporalStability,
            meanPairwiseReprojectionConsensus =
                meanPairwiseReprojectionConsensus,
            temporalRejectedPointRatio =
                (
                    pointsDroppedAfterTemporalHistory.toDouble() /
                        availablePointIds.size.coerceAtLeast(1).toDouble()
                    ).coerceIn(0.0, 1.0),
            meanSubMillimeterFeasibilityScore =
                meanSubMillimeterFeasibility,
            meanFirstOrderDepthErrorMillimeters =
                meanFirstOrderDepthErrorMm,
            p95ReprojectionTailAdjustedDepthErrorMillimeters =
                tailAdjustedDepthErrorP95Mm,
            meanTailAdjustedSubMillimeterFeasibilityScore =
                meanTailAdjustedFeasibility,
            warnings = (
                warnings +
                    accepted.flatMap {
                        if (it.temporalStability < 0.45) {
                            listOf("dense-face-mesh-temporal-stability-degraded")
                        } else {
                            emptyList()
                        }
                    }
                ).distinct().take(16),
        )
    }

    @Synchronized
    fun diagnostics(): Map<DeviceId, Int> =
        history.mapValues { (_, queue) ->
            queue.size
        }

    @Synchronized
    fun reset() {
        history.clear()
        temporalPoints.clear()
        kinematicPoints.values.forEach { it.close() }
        kinematicPoints.clear()
    }

    override fun close() {
        reset()
    }

    private fun kinematicEngineFor(
        pointId: Int,
    ): KinematicStabilityEngine {
        val existing = kinematicPoints[pointId]
        if (existing != null) return existing

        while (kinematicPoints.size >= maximumKinematicPoints) {
            val oldest = kinematicPoints.entries.firstOrNull()?.key
                ?: break
            kinematicPoints.remove(oldest)?.close()
        }

        return KinematicStabilityEngine(
            KinematicStabilityEngine.Config(
                maximumHistorySamples = 10,
                minimumRegressionSamples = 4,
                maximumRegressionWindowSeconds = 0.18,
                maximumPredictionHorizonSeconds = 0.06,
                maximumSpeedMetersPerSecond = 2.0,
                maximumAccelerationMetersPerSecond2 = 40.0,
                maximumJerkMetersPerSecond3 = 500.0,
                measurementSigmaMeters = 0.0009,
                huberScaleMeters = 0.0025,
            ),
        ).also {
            kinematicPoints[pointId] = it
        }
    }

    private fun selectFrames(
        timestampNanos: Long,
        calibration: CalibrationSnapshot,
    ): List<TimedFrame> {
        val frames = ArrayList<TimedFrame>()

        calibration.extrinsics.keys
            .sortedBy { it.value }
            .forEach { deviceId ->
                val queue =
                    history[deviceId]
                        ?: return@forEach

                val nearest =
                    queue.minByOrNull {
                        kotlin.math.abs(
                            it.timestampNanos -
                                timestampNanos,
                        )
                    } ?: return@forEach

                val skew =
                    kotlin.math.abs(
                        nearest.timestampNanos -
                            timestampNanos,
                    )

                if (skew > maximumTimestampSkewNanos) {
                    return@forEach
                }

                val intrinsics =
                    calibration.intrinsics[deviceId]
                        ?: return@forEach
                val extrinsics =
                    calibration.extrinsics[deviceId]
                        ?: return@forEach

                if (
                    intrinsics.fx <= 1e-6 ||
                    intrinsics.fy <= 1e-6
                ) {
                    return@forEach
                }

                if (
                    nearest.points.none {
                        it.position.x.isFinite() &&
                            it.position.y.isFinite()
                    }
                ) {
                    return@forEach
                }

                frames += TimedFrame(
                    deviceId = deviceId,
                    frame = nearest,
                )
            }

        return frames
    }

    private fun commonPointIds(
        frames: List<TimedFrame>,
    ): Set<Int> {
        if (frames.isEmpty()) return emptySet()

        val first =
            frames.first().frame.points
                .map { it.index }
                .toMutableSet()

        frames.drop(1).forEach { timed ->
            first.retainAll(
                timed.frame.points.map { it.index }.toSet(),
            )
        }

        return first
    }

    private fun triangulatePoint(
        pointId: Int,
        frames: List<TimedFrame>,
        calibration: CalibrationSnapshot,
        timestampNanos: Long,
    ): SurfacePoint? {
        val pointObservations =
            ArrayList<TriangulationEngine.Observation>()

        frames.forEach { timed ->
            val deviceId = timed.deviceId
            val point =
                timed.frame.points.firstOrNull {
                    it.index == pointId
                } ?: return@forEach

            if (
                !point.position.x.isFinite() ||
                    !point.position.y.isFinite()
            ) {
                return@forEach
            }

            val confidence =
                timed.frame.confidence.coerceIn(0.0, 1.0)

            if (confidence < minimumPointConfidence) {
                return@forEach
            }

            val intrinsics =
                calibration.intrinsics[deviceId]
                    ?: return@forEach
            val pose =
                calibration.extrinsics[deviceId]?.pose
                    ?: return@forEach

            val scaled =
                scaleObservationToCalibration(
                    point = point,
                    frame = timed.frame,
                    intrinsics = intrinsics,
                )

            val frameDurationNanos =
                timed.frame.frameDurationNanos.coerceAtLeast(0L)
            val rollingShutterSkewNanos =
                timed.frame.rollingShutterSkewNanos.coerceAtLeast(0L)
            val rowTimeOffsetNanos =
                if (timed.frame.imageHeight > 0) {
                    (
                        (
                            point.position.y /
                                timed.frame.imageHeight.toDouble()
                        ) - 0.5
                    ) *
                        rollingShutterSkewNanos.toDouble() *
                        timed.frame.rollingShutterDirectionSign
                } else {
                    0.0
                }

            val previousVelocity =
                temporalPoints[pointId]
                    ?.velocity
                    ?: Vec3(0.0, 0.0, 0.0)

            pointObservations +=
                TriangulationEngine.Observation(
                    cameraPose = pose,
                    intrinsics = intrinsics,
                    cameraId = deviceId.value,
                    u = scaled.x,
                    v = scaled.y,
                    weight = confidence,
                    rollingShutterFraction =
                        if (frameDurationNanos > 0L) {
                            (
                                rollingShutterSkewNanos.toDouble() /
                                    frameDurationNanos.toDouble()
                            ).coerceIn(0.0, 1.0)
                        } else {
                            0.0
                        },
                    rowTimeOffsetNanos = rowTimeOffsetNanos.toLong(),
                    temporalAlignmentConfidence =
                        timed.frame.captureTimingConfidence
                            .coerceIn(0.0, 1.0),
                    temporalUncertaintyNanos =
                        frameDurationNanos,
                    motionVelocityMetersPerSecond = previousVelocity,
                    calibrationUncertainty =
                        calibration.uncertaintyProfiles[deviceId]
                            ?: com.nivukx.nox.core.model.CalibrationUncertaintyProfile(),
                )
        }

        if (pointObservations.size < 2) {
            return null
        }

        val conditioning =
            geometricConditioningEngine.evaluate(
                pointObservations,
            )

        val conditioningByCamera =
            conditioning.scores.associate {
                it.cameraId to it.weightMultiplier
            }

        val conditionedObservations =
            pointObservations.map { observation ->
                observation.copy(
                    weight =
                        (
                            observation.weight *
                                (
                                    conditioningByCamera[
                                        observation.cameraId
                                    ] ?: 1.0
                                )
                            ).coerceIn(
                            1e-4,
                            1.0,
                        ),
                )
            }

        val result =
            triangulator.robust(
                conditionedObservations,
            ) ?: return null

        if (
            result.rmsReprojectionPx >
                maximumReprojectionErrorPx ||
            result.uncertaintyMeters >
                maximumUncertaintyMeters ||
            result.cheiralityRatio < 0.80
        ) {
            return null
        }

        val kinematic =
            kinematicEngineFor(pointId).observe(
                KinematicStabilityEngine.Observation(
                    timestampNanos = timestampNanos,
                    positionMeters = result.point,
                    confidence =
                        pointObservations.map { it.weight }
                            .average()
                            .coerceIn(0.0, 1.0),
                ),
            )

        val sourceConfidence =
            conditionedObservations
                .map {
                    it.weight
                }
                .average()
                .coerceIn(0.0, 1.0)

        val errorConfidence =
            exp(
                -result.rmsReprojectionPx /
                    maximumReprojectionErrorPx
                        .coerceAtLeast(0.25),
            ).coerceIn(0.0, 1.0)

        val uncertaintyConfidence =
            exp(
                -result.uncertaintyMeters /
                    maximumUncertaintyMeters
                        .coerceAtLeast(0.01),
            ).coerceIn(0.0, 1.0)

        val parallaxConfidence =
            (
                1.0 -
                    exp(
                        -result.parallaxDegrees /
                            1.5,
                    )
                ).coerceIn(0.0, 1.0)

        val pairwiseConsensusConfidence =
            result.pairwiseReprojectionConsensus
                .coerceIn(0.0, 1.0)

        val precisionScore =
            result.subMillimeterFeasibilityScore
                .coerceIn(0.0, 1.0)

        val conditionScore =
            (
                result.conditionScore * 0.36 +
                    result.cheiralityRatio * 0.17 +
                    parallaxConfidence * 0.16 +
                    errorConfidence * 0.08 +
                    uncertaintyConfidence * 0.05 +
                    pairwiseConsensusConfidence * 0.10 +
                    precisionScore * 0.08
                ).coerceIn(0.0, 1.0)

        val confidence =
            (
                sourceConfidence * 0.31 +
                    errorConfidence * 0.23 +
                    uncertaintyConfidence * 0.17 +
                    parallaxConfidence * 0.11 +
                    result.conditionScore * 0.07 +
                    pairwiseConsensusConfidence * 0.09 +
                    precisionScore * 0.08
                ).coerceIn(0.0, 1.0)

        val previous =
            temporalPoints[pointId]

        val temporalResidual =
            if (
                previous == null ||
                timestampNanos <= previous.timestampNanos
            ) {
                0.0
            } else {
                val dt =
                    (
                        timestampNanos -
                            previous.timestampNanos
                        ).toDouble() / 1e9

                val predictedDt =
                    dt.coerceIn(
                        0.0,
                        0.20,
                    )
                val predicted =
                    previous.position +
                        previous.velocity *
                            predictedDt +
                        previous.acceleration *
                            (
                                0.5 *
                                    predictedDt *
                                    predictedDt
                            )

                result.point.distanceTo(
                    predicted,
                )
            }

        val temporalGate =
            (
                0.012 +
                    (
                        previous?.velocity?.norm()
                            ?: 0.0
                    ) *
                    0.65 *
                    (
                        (
                            timestampNanos -
                                (
                                    previous?.timestampNanos
                                        ?: timestampNanos
                                )
                        ).coerceAtLeast(0L).toDouble() /
                            1e9
                    ).coerceIn(0.0, 0.20) +
                    result.uncertaintyMeters * 2.5
                ).coerceIn(
                    0.008,
                    0.10,
                )

        if (
            previous != null &&
            temporalResidual >
                temporalGate *
                    3.0
        ) {
            return null
        }

        val temporalInnovationScore =
            if (previous == null) {
                1.0
            } else {
                (
                    1.0 -
                        temporalResidual /
                            temporalGate
                    ).coerceIn(0.0, 1.0)
            }

        val temporalStability =
            if (previous == null) {
                0.72
            } else {
                (
                    previous.stability * 0.62 +
                        temporalInnovationScore * 0.25 +
                        confidence * 0.13
                    ).coerceIn(0.0, 1.0)
            }

        if (
            previous != null &&
            timestampNanos > previous.timestampNanos
        ) {
            val dt =
                (
                    timestampNanos -
                        previous.timestampNanos
                    ).toDouble() / 1e9

            val boundedDt =
                dt.coerceIn(
                    1.0 / 240.0,
                    0.20,
                )

            val measuredVelocity =
                (
                    result.point -
                        previous.position
                ) /
                    boundedDt

            val measuredAcceleration =
                (
                    measuredVelocity -
                        previous.velocity
                ) /
                    boundedDt

            val accelerationMagnitude =
                measuredAcceleration.norm()
            val boundedAcceleration =
                if (
                    accelerationMagnitude.isFinite() &&
                        accelerationMagnitude > 90.0
                ) {
                    measuredAcceleration *
                        (90.0 / accelerationMagnitude)
                } else {
                    measuredAcceleration
                }

            previous.acceleration =
                previous.acceleration.lerp(
                    boundedAcceleration,
                    (
                        0.10 +
                            confidence * 0.30
                        ).coerceIn(
                            0.10,
                            0.40,
                        ),
                )

            previous.velocity =
                previous.velocity.lerp(
                    measuredVelocity,
                    (
                        0.18 +
                            confidence * 0.54
                        ).coerceIn(
                            0.18,
                            0.72,
                        ),
                )
            previous.position = result.point
            previous.timestampNanos = timestampNanos
            previous.stability = temporalStability
        } else if (previous == null) {
            ensureTemporalPointCapacity(pointId)
            temporalPoints[pointId] =
                TemporalPointState(
                    position = result.point,
                    velocity = Vec3(
                        0.0,
                        0.0,
                        0.0,
                    ),
                    acceleration = Vec3(
                        0.0,
                        0.0,
                        0.0,
                    ),
                    timestampNanos = timestampNanos,
                    stability = temporalStability,
                )
        }

        val kinematicStability =
            (
                kinematic.motionStability * 0.55 +
                    kinematic.modelConfidence * 0.25 +
                    kinematic.innovationConfidence * 0.20
                ).coerceIn(0.0, 1.0)

        val temporalAwareConfidence =
            (
                confidence * 0.74 +
                    temporalStability * 0.14 +
                    kinematicStability * 0.12
                ).coerceIn(0.0, 1.0)

        return SurfacePoint(
            index = pointId,
            positionWorld = result.point,
            confidence = temporalAwareConfidence,
            uncertaintyMeters = result.uncertaintyMeters,
            reprojectionErrorPx = result.rmsReprojectionPx,
            parallaxDegrees = result.parallaxDegrees,
            sourceCount = result.observations,
            conditionScore = conditionScore,
            pairwiseReprojectionConsensus =
                pairwiseConsensusConfidence,
            temporalStability = temporalStability,
            temporalInnovationMeters = temporalResidual,
            subMillimeterFeasibilityScore = precisionScore,
            firstOrderDepthErrorMillimeters =
                result.firstOrderDepthErrorMillimeters,
            reprojectionP95Px = result.reprojectionP95Px,
            tailAdjustedFirstOrderDepthErrorMillimeters =
                result.tailAdjustedFirstOrderDepthErrorMillimeters,
            tailAdjustedSubMillimeterFeasibilityScore =
                result.tailAdjustedSubMillimeterFeasibilityScore,
        )
    }

    private fun ensureTemporalPointCapacity(pointId: Int) {
        if (temporalPoints.containsKey(pointId) ||
            temporalPoints.size < maximumTemporalPoints
        ) return

        val victim =
            temporalPoints.entries
                .minByOrNull { (_, state) -> state.stability }
                ?.key
        if (victim != null) temporalPoints.remove(victim)
    }

    private fun pruneTemporalPointState(nowTimestampNanos: Long) {
        val retentionNanos =
            maximumTimestampSkewNanos
                .coerceAtLeast(45_000_000L) * 8L
        val cutoff =
            nowTimestampNanos - retentionNanos
        val iterator = temporalPoints.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.value.timestampNanos < cutoff) {
                iterator.remove()
            }
        }
    }

    private fun buildSurfaceTriangles(
        points: List<SurfacePoint>,
        referenceFrame: FaceMeshFrame?,
    ): List<FaceMeshTriangle> {
        if (referenceFrame == null) {
            return emptyList()
        }

        val valid =
            points.map { it.index }.toHashSet()

        return referenceFrame.triangles
            .filter {
                it.a in valid &&
                    it.b in valid &&
                    it.c in valid
            }
            .distinct()
            .take(960)
    }

    private fun frameQuality(
        frame: FaceMeshFrame,
    ): Double =
        (
            frame.confidence * 0.52 +
                frame.pointCoverage * 0.28 +
                frame.triangleCoverage * 0.20
            ).coerceIn(0.0, 1.0)

    private fun scaleObservationToCalibration(
        point: FaceMeshPoint3D,
        frame: FaceMeshFrame,
        intrinsics: com.nivukx.nox.core.model.CameraIntrinsics,
    ): com.nivukx.nox.core.model.Vec2 {
        return com.nivukx.nox.core.model.Vec2(
            x = point.position.x,
            y = point.position.y,
        )
    }

    private fun buildWarnings(
        sourceFrameCount: Int,
        acceptedPointRatio: Double,
        meanError: Double,
        p95Error: Double,
        meanUncertainty: Double,
        triangleCount: Int,
        confidence: Double,
    ): List<String> {
        val warnings = ArrayList<String>()

        if (sourceFrameCount < 3) {
            warnings += "dense-face-mesh-low-view-count"
        }
        if (acceptedPointRatio < 0.65) {
            warnings += "dense-face-mesh-low-point-acceptance"
        }
        if (
            meanError.isFinite() &&
            meanError > maximumReprojectionErrorPx * 0.75
        ) {
            warnings += "dense-face-mesh-elevated-reprojection"
        }
        if (
            p95Error.isFinite() &&
            p95Error > maximumReprojectionErrorPx * 1.5
        ) {
            warnings += "dense-face-mesh-reprojection-outliers"
        }
        if (
            meanUncertainty.isFinite() &&
            meanUncertainty > maximumUncertaintyMeters * 0.75
        ) {
            warnings += "dense-face-mesh-elevated-uncertainty"
        }
        if (triangleCount < 350) {
            warnings += "dense-face-mesh-topology-fragmented"
        }
        if (confidence < 0.50) {
            warnings += "dense-face-mesh-quality-degraded"
        }
        if (meanError.isFinite() && confidence < 0.65) {
            warnings += "dense-face-mesh-cross-view-evidence-limited"
        }
        if (confidence < 0.70) {
            warnings += "dense-face-mesh-precision-evidence-limited"
        }

        return warnings.distinct().take(12)
    }

    private fun List<Double>.averageOrZeroFace(): Double =
        filter(Double::isFinite)
            .takeIf { it.isNotEmpty() }
            ?.average()
            ?.coerceAtLeast(0.0)
            ?: 0.0

    private fun percentile(
        values: List<Double>,
        percentile: Double,
    ): Double {
        if (values.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }

        if (values.size == 1) {
            return values.first()
        }

        val sorted = values.sorted()
        val position =
            percentile.coerceIn(0.0, 1.0) *
                (sorted.size - 1)
        val lower = position.toInt()
        val upper =
            min(
                sorted.lastIndex,
                lower + 1,
            )
        val factor =
            position - lower

        return sorted[lower] * (1.0 - factor) +
            sorted[upper] * factor
    }
}

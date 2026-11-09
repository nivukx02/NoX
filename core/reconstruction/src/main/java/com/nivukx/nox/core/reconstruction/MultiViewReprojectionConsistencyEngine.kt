package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Runtime multi-view reprojection consistency engine.
 *
 * The reconstruction stack can already triangulate joints, stabilize them over time and enforce
 * kinematic constraints. This stage adds an explicit measurement-space check: every reconstructed
 * 3D joint is projected back into every contributing camera and compared with the original 2D
 * observation.
 *
 * The engine deliberately does not move a joint merely because one camera disagrees. It computes a
 * robust consensus signal, identifies view-level outliers, updates confidence/uncertainty metadata,
 * and lets the existing continuity controller decide whether the frame should be displayed. This
 * prevents a single blurred, desynchronized, badly calibrated or occluded phone from poisoning the
 * entire rig.
 *
 * Design goals:
 * - calibrated camera geometry rather than heuristic screen-space averaging;
 * - radial and tangential distortion support;
 * - motion-aware temporal reprojection using the reconstructed joint velocity;
 * - robust Huber weighting with a hard residual safety gate;
 * - explicit per-joint diagnostics and operator-readable warnings;
 * - bounded influence from calibration error, clock skew and camera motion;
 * - stateful residual history for detecting persistent per-joint degradation;
 * - deterministic execution with no hidden network/model dependency.
 */
class MultiViewReprojectionConsistencyEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumViewsPerJoint: Int = 2,
        val residualScalePx: Double = 2.5,
        val huberTransitionPx: Double = 2.5,
        val hardRejectResidualPx: Double = 18.0,
        val fatalResidualPx: Double = 45.0,
        val maximumPredictionSeconds: Double = 0.055,
        val maximumHistoryAgeSeconds: Double = 0.75,
        val historyAlpha: Double = 0.18,
        val residualScoreFloor: Double = 0.02,
        val minimumViewWeight: Double = 0.04,
        val calibrationPenaltyScalePx: Double = 4.0,
        val clockSkewPenaltyMillis: Double = 4.0,
        val cameraMotionPenaltyScale: Double = 0.70,
        val rollingShutterPenaltyScale: Double = 0.45,
        val minimumTemporalAlignmentConfidence: Double = 0.30,
        val maximumTemporalUncertaintyNanos: Long = 40_000_000L,
        val spreadScalePx: Double = 5.0,
        val persistentResidualThresholdPx: Double = 6.0,
        val persistentResidualFrames: Int = 5,
        val globalAcceptanceScore: Double = 0.55,
        val globalSoftWarningScore: Double = 0.72,
    ) {
        init {
            require(minimumViewsPerJoint >= 1)
            require(residualScalePx > 0.0)
            require(huberTransitionPx > 0.0)
            require(hardRejectResidualPx > huberTransitionPx)
            require(fatalResidualPx > hardRejectResidualPx)
            require(maximumPredictionSeconds >= 0.0)
            require(maximumHistoryAgeSeconds > 0.0)
            require(historyAlpha in 0.0..1.0)
            require(residualScoreFloor in 0.0..1.0)
            require(minimumViewWeight in 0.0..1.0)
            require(calibrationPenaltyScalePx > 0.0)
            require(clockSkewPenaltyMillis > 0.0)
            require(cameraMotionPenaltyScale >= 0.0)
            require(rollingShutterPenaltyScale >= 0.0)
            require(minimumTemporalAlignmentConfidence in 0.0..1.0)
            require(maximumTemporalUncertaintyNanos > 0L)
            require(spreadScalePx > 0.0)
            require(persistentResidualThresholdPx > 0.0)
            require(persistentResidualFrames >= 2)
            require(globalAcceptanceScore in 0.0..1.0)
            require(globalSoftWarningScore >= globalAcceptanceScore)
        }
    }

    data class CameraFrame(
        val deviceId: DeviceId,
        val timestampNanos: Long,
        val observations: List<Observation>,
        val clockOffsetNanos: Long = 0L,
        val confidence: Double = 1.0,
    ) {
        init {
            require(timestampNanos > 0L)
            require(abs(clockOffsetNanos) <= 1_000_000_000L)
            require(confidence in 0.0..1.0)
        }
    }

    data class Observation(
        val landmarkId: Int,
        val camera: CameraObservation,
    ) {
        init {
            require(landmarkId >= 0)
            require(camera.point.x.isFinite())
            require(camera.point.y.isFinite())
            require(camera.confidence in 0.0..1.0)
            require(camera.rollingShutterFraction.isFinite())
        }
    }

    data class Input(
        val skeleton: SkeletonFrame,
        val cameraFrames: List<CameraFrame>,
        val targetTimestampNanos: Long,
        val calibrationReprojectionErrorPx: Double = 0.0,
        val synchronizationConfidence: Double = 1.0,
        val cameraMotionRisk: Double = 0.0,
    ) {
        init {
            require(targetTimestampNanos > 0L)
            require(calibrationReprojectionErrorPx >= 0.0)
            require(synchronizationConfidence in 0.0..1.0)
            require(cameraMotionRisk in 0.0..1.0)
        }
    }

    data class ViewDiagnostics(
        val jointId: Int,
        val deviceId: DeviceId,
        val reprojectionErrorPx: Double,
        val robustWeight: Double,
        val observationConfidence: Double,
        val timeOffsetMillis: Double,
        val projectedDepthMeters: Double,
        val inImage: Boolean,
        val rejected: Boolean,
        val rejectionReason: String?,
    )

    data class JointDiagnostics(
        val jointId: Int,
        val observedViews: Int,
        val acceptedViews: Int,
        val rejectedViews: Int,
        val inlierRatio: Double,
        val medianResidualPx: Double,
        val rmsResidualPx: Double,
        val p95ResidualPx: Double,
        val maximumResidualPx: Double,
        val residualSpreadPx: Double,
        val measurementScore: Double,
        val viewAgreementScore: Double,
        val temporalConsistencyScore: Double,
        val finalScore: Double,
        val persistentResidualFrames: Int,
        val accepted: Boolean,
        val views: List<ViewDiagnostics>,
    )

    data class Result(
        val skeleton: SkeletonFrame,
        val score: Double,
        val meanResidualPx: Double,
        val medianResidualPx: Double,
        val p95ResidualPx: Double,
        val maximumResidualPx: Double,
        val inlierRatio: Double,
        val viewAgreementScore: Double,
        val temporalConsistencyScore: Double,
        val acceptedJointCount: Int,
        val rejectedJointCount: Int,
        val diagnostics: List<JointDiagnostics>,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val accepted: Boolean
            get() =
                score >= 0.55 &&
                    acceptedJointCount > 0 &&
                    warnings.none { it.startsWith("fatal:") }

        val bottleneck: String
            get() = when {
                inlierRatio < 0.55 -> "VIEW_OUTLIERS"
                viewAgreementScore < 0.50 -> "VIEW_AGREEMENT"
                temporalConsistencyScore < 0.50 -> "TEMPORAL"
                medianResidualPx > 6.0 -> "REPROJECTION"
                score < 0.55 -> "GLOBAL"
                else -> "NONE"
            }
    }

    private data class HistoryState(
        var timestampNanos: Long,
        var residualEmaPx: Double,
        var scoreEma: Double,
        var persistentHighResidualFrames: Int,
        var sampleCount: Int,
    )

    private data class Projection(
        val pixel: Vec2,
        val depthMeters: Double,
        val normalizedX: Double,
        val normalizedY: Double,
    )

    private val history = HashMap<Int, HistoryState>()

    @Synchronized
    fun evaluate(input: Input): Result {
        val source = sanitizeSkeleton(input.skeleton)
        if (source.joints.isEmpty()) {
            return Result(
                skeleton = source.copy(globalConfidence = 0.0),
                score = 0.0,
                meanResidualPx = 0.0,
                medianResidualPx = 0.0,
                p95ResidualPx = 0.0,
                maximumResidualPx = 0.0,
                inlierRatio = 0.0,
                viewAgreementScore = 0.0,
                temporalConsistencyScore = 0.0,
                acceptedJointCount = 0,
                rejectedJointCount = 0,
                diagnostics = emptyList(),
                warnings = listOf("fatal: no reconstructed joints available"),
                recommendations = listOf("Hold the current frame and reacquire a valid multi-view skeleton."),
            )
        }

        if (input.cameraFrames.isEmpty()) {
            return Result(
                skeleton = source.copy(
                    globalConfidence = min(source.globalConfidence, 0.28),
                ),
                score = 0.18,
                meanResidualPx = Double.POSITIVE_INFINITY,
                medianResidualPx = Double.POSITIVE_INFINITY,
                p95ResidualPx = Double.POSITIVE_INFINITY,
                maximumResidualPx = Double.POSITIVE_INFINITY,
                inlierRatio = 0.0,
                viewAgreementScore = 0.0,
                temporalConsistencyScore = temporalBaseline(),
                acceptedJointCount = 0,
                rejectedJointCount = source.joints.size,
                diagnostics = emptyList(),
                warnings = listOf("fatal: no camera observations supplied"),
                recommendations = listOf("Keep at least two calibrated cameras active for multi-view reconstruction."),
            )
        }

        val cameraCoverage = buildObservationIndex(input.cameraFrames)
        val jointDiagnostics = ArrayList<JointDiagnostics>(source.joints.size)
        val adjustedJoints = ArrayList<ReconstructionJoint>(source.joints.size)
        val allResiduals = ArrayList<Double>()
        var acceptedJointCount = 0
        var rejectedJointCount = 0

        source.joints.forEach { joint ->
            val diagnostics = evaluateJoint(
                joint = joint,
                targetTimestampNanos = input.targetTimestampNanos,
                observations = cameraCoverage[joint.landmarkId].orEmpty(),
                calibrationErrorPx = input.calibrationReprojectionErrorPx,
                synchronizationConfidence = input.synchronizationConfidence,
                cameraMotionRisk = input.cameraMotionRisk,
            )
            jointDiagnostics += diagnostics

            diagnostics.views
                .filter { !it.rejected && it.reprojectionErrorPx.isFinite() }
                .forEach { allResiduals += it.reprojectionErrorPx }

            if (diagnostics.accepted) {
                acceptedJointCount += 1
            } else {
                rejectedJointCount += 1
            }

            val confidence = adjustedConfidence(
                original = joint.confidence,
                jointScore = diagnostics.finalScore,
                acceptedViews = diagnostics.acceptedViews,
                observedViews = diagnostics.observedViews,
                temporalScore = diagnostics.temporalConsistencyScore,
            )

            val uncertainty = adjustedUncertainty(
                originalMeters = joint.positionUncertaintyMeters,
                rmsResidualPx = diagnostics.rmsResidualPx,
                cameraCount = diagnostics.acceptedViews,
                score = diagnostics.finalScore,
            )

            adjustedJoints += joint.copy(
                confidence = confidence,
                reprojectionErrorPx =
                    if (diagnostics.medianResidualPx.isFinite()) {
                        diagnostics.medianResidualPx
                    } else {
                        joint.reprojectionErrorPx
                    },
                positionUncertaintyMeters = uncertainty,
            )

            updateHistory(
                jointId = joint.landmarkId,
                timestampNanos = input.targetTimestampNanos,
                diagnostics = diagnostics,
            )
        }

        trimHistory(input.targetTimestampNanos)

        val finiteResiduals = allResiduals.filter { it.isFinite() }.sorted()
        val mean = finiteResiduals.averageOrZero()
        val median = finiteResiduals.percentile(0.50)
        val p95 = finiteResiduals.percentile(0.95)
        val maximum = finiteResiduals.maxOrNull() ?: 0.0
        val inlierRatio = jointDiagnostics
            .map { it.inlierRatio }
            .averageOrZero()

        val viewAgreement = weightedAverage(
            values = jointDiagnostics.map { it.viewAgreementScore },
            weights = jointDiagnostics.map {
                max(1.0, it.acceptedViews.toDouble())
            },
        ).coerceIn(0.0, 1.0)

        val temporalConsistency = weightedAverage(
            values = jointDiagnostics.map { it.temporalConsistencyScore },
            weights = jointDiagnostics.map {
                max(1.0, it.observedViews.toDouble())
            },
        ).coerceIn(0.0, 1.0)

        val measurementScore = if (finiteResiduals.isEmpty()) {
            0.0
        } else {
            exp(-median / config.residualScalePx).coerceIn(0.0, 1.0)
        }

        val calibrationScore = exp(
            -input.calibrationReprojectionErrorPx /
                config.calibrationPenaltyScalePx,
        ).coerceIn(0.0, 1.0)

        val syncScore = input.synchronizationConfidence
        val cameraMotionScore = (
            1.0 -
                input.cameraMotionRisk *
                config.cameraMotionPenaltyScale
        ).coerceIn(0.0, 1.0)

        val score = (
            0.32 * measurementScore +
                0.18 * inlierRatio +
                0.18 * viewAgreement +
                0.14 * temporalConsistency +
                0.08 * calibrationScore +
                0.06 * syncScore +
                0.04 * cameraMotionScore
        ).coerceIn(0.0, 1.0)

        val warnings = buildWarnings(
            score = score,
            medianResidualPx = median,
            p95ResidualPx = p95,
            maximumResidualPx = maximum,
            inlierRatio = inlierRatio,
            viewAgreement = viewAgreement,
            temporalConsistency = temporalConsistency,
            acceptedJointCount = acceptedJointCount,
            rejectedJointCount = rejectedJointCount,
            calibrationErrorPx = input.calibrationReprojectionErrorPx,
            synchronizationConfidence = input.synchronizationConfidence,
            cameraMotionRisk = input.cameraMotionRisk,
            diagnostics = jointDiagnostics,
        )

        val recommendations = buildRecommendations(
            score = score,
            medianResidualPx = median,
            inlierRatio = inlierRatio,
            viewAgreement = viewAgreement,
            temporalConsistency = temporalConsistency,
            calibrationErrorPx = input.calibrationReprojectionErrorPx,
            synchronizationConfidence = input.synchronizationConfidence,
            cameraMotionRisk = input.cameraMotionRisk,
            diagnostics = jointDiagnostics,
        )

        val globalConfidence = (
            source.globalConfidence * 0.46 +
                score * 0.54
        ).coerceIn(0.0, 1.0)

        val output = source.copy(
            joints = adjustedJoints,
            globalConfidence = globalConfidence,
        )

        return Result(
            skeleton = output,
            score = score,
            meanResidualPx = mean,
            medianResidualPx = median,
            p95ResidualPx = p95,
            maximumResidualPx = maximum,
            inlierRatio = inlierRatio,
            viewAgreementScore = viewAgreement,
            temporalConsistencyScore = temporalConsistency,
            acceptedJointCount = acceptedJointCount,
            rejectedJointCount = rejectedJointCount,
            diagnostics = jointDiagnostics,
            warnings = warnings.distinct(),
            recommendations = recommendations.distinct(),
        )
    }

    @Synchronized
    fun reset() {
        history.clear()
    }

    @Synchronized
    fun historySize(): Int = history.size

    private fun evaluateJoint(
        joint: ReconstructionJoint,
        targetTimestampNanos: Long,
        observations: List<IndexedObservation>,
        calibrationErrorPx: Double,
        synchronizationConfidence: Double,
        cameraMotionRisk: Double,
    ): JointDiagnostics {
        if (observations.isEmpty()) {
            return JointDiagnostics(
                jointId = joint.landmarkId,
                observedViews = 0,
                acceptedViews = 0,
                rejectedViews = 0,
                inlierRatio = 0.0,
                medianResidualPx = Double.POSITIVE_INFINITY,
                rmsResidualPx = Double.POSITIVE_INFINITY,
                p95ResidualPx = Double.POSITIVE_INFINITY,
                maximumResidualPx = Double.POSITIVE_INFINITY,
                residualSpreadPx = Double.POSITIVE_INFINITY,
                measurementScore = 0.0,
                viewAgreementScore = 0.0,
                temporalConsistencyScore = temporalScoreForJoint(
                    jointId = joint.landmarkId,
                    residualPx = null,
                    timestampNanos = targetTimestampNanos,
                ),
                finalScore = 0.0,
                persistentResidualFrames = history[joint.landmarkId]?.persistentHighResidualFrames ?: 0,
                accepted = false,
                views = emptyList(),
            )
        }

        val diagnostics = observations.map { indexed ->
            evaluateView(
                joint = joint,
                targetTimestampNanos = targetTimestampNanos,
                observation = indexed,
                calibrationErrorPx = calibrationErrorPx,
                synchronizationConfidence = synchronizationConfidence,
                cameraMotionRisk = cameraMotionRisk,
            )
        }

        val usable = diagnostics.filter {
            !it.rejected &&
                it.reprojectionErrorPx.isFinite() &&
                it.robustWeight >= config.minimumViewWeight
        }

        val residuals = diagnostics
            .mapNotNull { it.reprojectionErrorPx.takeIf(Double::isFinite) }
            .sorted()

        val acceptedResiduals = usable
            .map { it.reprojectionErrorPx }
            .sorted()

        val observedViews = diagnostics.size
        val acceptedViews = usable.size
        val rejectedViews = observedViews - acceptedViews
        val inlierRatio =
            acceptedViews.toDouble() /
                max(1.0, observedViews.toDouble())

        val median = acceptedResiduals.percentile(0.50)
        val rms = rootMeanSquare(acceptedResiduals)
        val p95 = acceptedResiduals.percentile(0.95)
        val maximum = acceptedResiduals.maxOrNull() ?: Double.POSITIVE_INFINITY

        val spread = if (acceptedResiduals.size < 2) {
            if (median.isFinite()) 0.0 else Double.POSITIVE_INFINITY
        } else {
            acceptedResiduals.percentile(0.90) -
                acceptedResiduals.percentile(0.10)
        }

        val measurementScore = if (median.isFinite()) {
            exp(-median / config.residualScalePx)
                .coerceIn(config.residualScoreFloor, 1.0)
        } else {
            0.0
        }

        val spreadScore = if (spread.isFinite()) {
            exp(-spread / config.spreadScalePx)
        } else {
            0.0
        }

        val countScore = if (config.minimumViewsPerJoint <= 1) {
            1.0
        } else {
            (
                acceptedViews.toDouble() /
                    config.minimumViewsPerJoint.toDouble()
            ).coerceIn(0.0, 1.0)
        }

        val viewAgreement = (
            0.62 * measurementScore +
                0.25 * spreadScore +
                0.13 * countScore
        ).coerceIn(0.0, 1.0)

        val temporalScore = temporalScoreForJoint(
            jointId = joint.landmarkId,
            residualPx = median.takeIf(Double::isFinite),
            timestampNanos = targetTimestampNanos,
        )

        val cameraQuality = (
            (1.0 - calibrationErrorPx /
                config.calibrationPenaltyScalePx).coerceIn(0.0, 1.0) *
                0.55 +
                synchronizationConfidence * 0.25 +
                (1.0 - cameraMotionRisk *
                    config.cameraMotionPenaltyScale).coerceIn(0.0, 1.0) *
                    0.20
        ).coerceIn(0.0, 1.0)

        val finalScore = (
            0.36 * measurementScore +
                0.20 * inlierRatio +
                0.18 * viewAgreement +
                0.14 * temporalScore +
                0.12 * cameraQuality
        ).coerceIn(0.0, 1.0)

        val persistentFrames =
            history[joint.landmarkId]?.persistentHighResidualFrames ?: 0

        val accepted =
            acceptedViews >= config.minimumViewsPerJoint &&
                inlierRatio >= 0.50 &&
                finalScore >= config.globalAcceptanceScore &&
                maximum <= config.fatalResidualPx

        return JointDiagnostics(
            jointId = joint.landmarkId,
            observedViews = observedViews,
            acceptedViews = acceptedViews,
            rejectedViews = rejectedViews,
            inlierRatio = inlierRatio,
            medianResidualPx = median,
            rmsResidualPx = rms,
            p95ResidualPx = p95,
            maximumResidualPx = maximum,
            residualSpreadPx = spread,
            measurementScore = measurementScore,
            viewAgreementScore = viewAgreement,
            temporalConsistencyScore = temporalScore,
            finalScore = finalScore,
            persistentResidualFrames = persistentFrames,
            accepted = accepted,
            views = diagnostics,
        )
    }

    private fun evaluateView(
        joint: ReconstructionJoint,
        targetTimestampNanos: Long,
        observation: IndexedObservation,
        calibrationErrorPx: Double,
        synchronizationConfidence: Double,
        cameraMotionRisk: Double,
    ): ViewDiagnostics {
        val frame = observation.frame
        val camera = observation.observation.camera
        val timeOffsetNanos =
            frame.timestampNanos +
                frame.clockOffsetNanos +
                camera.rowTimeOffsetNanos -
                targetTimestampNanos
        val timeOffsetSeconds =
            timeOffsetNanos / 1e9
        val clampedPrediction =
            timeOffsetSeconds.coerceIn(
                -config.maximumPredictionSeconds,
                config.maximumPredictionSeconds,
            )

        val motionCompensatedPoint =
            joint.position +
                joint.velocity * clampedPrediction

        val projection =
            project(
                point = motionCompensatedPoint,
                intrinsics = camera.intrinsics,
                cameraPosition = camera.pose.position,
                cameraRotation = camera.pose.rotation,
            )

        if (projection == null) {
            return ViewDiagnostics(
                jointId = joint.landmarkId,
                deviceId = frame.deviceId,
                reprojectionErrorPx = Double.POSITIVE_INFINITY,
                robustWeight = 0.0,
                observationConfidence = camera.confidence,
                timeOffsetMillis = abs(timeOffsetNanos) / 1e6,
                projectedDepthMeters = 0.0,
                inImage = false,
                rejected = true,
                rejectionReason = "point-behind-camera-or-invalid-projection",
            )
        }

        val error = projection.pixel.distanceTo(camera.point)
        val inImage = isInsideImage(
            projection = projection,
            intrinsics = camera.intrinsics,
        )

        val robustResidualWeight = huberWeight(error)
        val timeWeight = exp(
            -abs(timeOffsetNanos) / 1e6 /
                config.clockSkewPenaltyMillis,
        ).coerceIn(0.0, 1.0)

        val calibrationWeight = exp(
            -calibrationErrorPx /
                config.calibrationPenaltyScalePx,
        ).coerceIn(0.0, 1.0)

        val syncWeight = synchronizationConfidence.coerceIn(0.0, 1.0)

        val temporalUncertaintyRatio =
            (
                camera.temporalUncertaintyNanos.toDouble() /
                    config.maximumTemporalUncertaintyNanos.toDouble()
                ).coerceIn(0.0, 1.0)

        val temporalAlignmentWeight =
            (
                camera.temporalAlignmentConfidence
                    .coerceIn(0.0, 1.0) *
                    0.70 +
                    (
                        1.0 -
                            temporalUncertaintyRatio
                    ).coerceIn(0.0, 1.0) *
                    0.30
            ).coerceIn(0.0, 1.0)

        val temporalUncertaintyWeight =
            (
                1.0 -
                    0.55 * temporalUncertaintyRatio
                ).coerceIn(0.35, 1.0)

        val cameraMotionWeight = (
            1.0 -
                cameraMotionRisk *
                config.cameraMotionPenaltyScale
        ).coerceIn(0.0, 1.0)

        val rollingShutterWeight = (
            1.0 -
                abs(camera.rollingShutterFraction)
                    .coerceIn(0.0, 1.0) *
                    config.rollingShutterPenaltyScale
        ).coerceIn(0.0, 1.0)

        val observationWeight = (
            camera.confidence *
                frame.confidence *
                robustResidualWeight *
                timeWeight *
                calibrationWeight *
                syncWeight *
                temporalAlignmentWeight *
                temporalUncertaintyWeight *
                cameraMotionWeight *
                rollingShutterWeight
        ).coerceIn(0.0, 1.0)

        val rejected =
            !inImage ||
                error > config.hardRejectResidualPx ||
                temporalAlignmentWeight <
                    max(
                        config.minimumTemporalAlignmentConfidence,
                        0.60,
                    ) ||
                (
                    camera.temporalUncertaintyNanos.toDouble() /
                        config.maximumTemporalUncertaintyNanos.toDouble()
                    ) > 0.70 ||
                observationWeight <
                    config.minimumViewWeight ||
                projection.depthMeters <= 1e-5

        val rejectionReason = when {
            !inImage -> "projected-point-outside-image"
            error > config.hardRejectResidualPx ->
                "hard-reprojection-residual"
            temporalAlignmentWeight <
                config.minimumTemporalAlignmentConfidence ->
                "low-temporal-alignment-confidence"
            camera.temporalUncertaintyNanos >
                config.maximumTemporalUncertaintyNanos ->
                "excessive-temporal-uncertainty"
            observationWeight < config.minimumViewWeight ->
                "low-composite-view-weight"
            projection.depthMeters <= 1e-5 ->
                "invalid-depth"
            else -> null
        }

        return ViewDiagnostics(
            jointId = joint.landmarkId,
            deviceId = frame.deviceId,
            reprojectionErrorPx = error,
            robustWeight = observationWeight,
            observationConfidence = camera.confidence,
            timeOffsetMillis = abs(timeOffsetNanos) / 1e6,
            projectedDepthMeters = projection.depthMeters,
            inImage = inImage,
            rejected = rejected,
            rejectionReason = rejectionReason,
        )
    }

    private fun updateHistory(
        jointId: Int,
        timestampNanos: Long,
        diagnostics: JointDiagnostics,
    ) {
        val currentResidual =
            diagnostics.medianResidualPx.takeIf { it.isFinite() }
                ?: config.fatalResidualPx

        val currentScore = diagnostics.finalScore
        val previous = history[jointId]

        if (previous == null) {
            history[jointId] = HistoryState(
                timestampNanos = timestampNanos,
                residualEmaPx = currentResidual,
                scoreEma = currentScore,
                persistentHighResidualFrames =
                    if (currentResidual >= config.persistentResidualThresholdPx) 1 else 0,
                sampleCount = 1,
            )
            return
        }

        val dt =
            (timestampNanos - previous.timestampNanos)
                .coerceAtLeast(1L) / 1e9

        if (dt > config.maximumHistoryAgeSeconds) {
            previous.timestampNanos = timestampNanos
            previous.residualEmaPx = currentResidual
            previous.scoreEma = currentScore
            previous.persistentHighResidualFrames =
                if (currentResidual >= config.persistentResidualThresholdPx) 1 else 0
            previous.sampleCount = 1
            return
        }

        val alpha = config.historyAlpha
        previous.timestampNanos = timestampNanos
        previous.residualEmaPx =
            previous.residualEmaPx * (1.0 - alpha) +
                currentResidual * alpha
        previous.scoreEma =
            previous.scoreEma * (1.0 - alpha) +
                currentScore * alpha
        previous.persistentHighResidualFrames =
            if (currentResidual >= config.persistentResidualThresholdPx) {
                previous.persistentHighResidualFrames + 1
            } else {
                max(0, previous.persistentHighResidualFrames - 1)
            }
        previous.sampleCount += 1
    }

    private fun temporalScoreForJoint(
        jointId: Int,
        residualPx: Double?,
        timestampNanos: Long,
    ): Double {
        val previous = history[jointId] ?: return 1.0
        val dt =
            (timestampNanos - previous.timestampNanos)
                .coerceAtLeast(1L) / 1e9

        if (dt > config.maximumHistoryAgeSeconds) return 1.0

        val residual = residualPx ?: previous.residualEmaPx
        val residualAgreement =
            exp(
                -abs(residual - previous.residualEmaPx) /
                    max(config.residualScalePx, 1e-6),
            )

        val scoreAgreement = (
            1.0 -
                abs(
                    (
                        residualScore(residual) -
                            previous.scoreEma
                    ),
                )
        ).coerceIn(0.0, 1.0)

        val ageWeight =
            exp(-dt / config.maximumHistoryAgeSeconds)

        return (
            0.65 * residualAgreement +
                0.25 * scoreAgreement +
                0.10 * ageWeight
        ).coerceIn(0.0, 1.0)
    }

    private fun residualScore(residualPx: Double): Double =
        if (!residualPx.isFinite()) {
            0.0
        } else {
            exp(-residualPx / config.residualScalePx)
                .coerceIn(0.0, 1.0)
        }

    private fun trimHistory(timestampNanos: Long) {
        val cutoff =
            timestampNanos -
                (config.maximumHistoryAgeSeconds * 4.0 * 1e9).toLong()
        history.entries.removeIf {
            it.value.timestampNanos < cutoff
        }
    }

    private fun adjustedConfidence(
        original: Double,
        jointScore: Double,
        acceptedViews: Int,
        observedViews: Int,
        temporalScore: Double,
    ): Double {
        val coverage =
            acceptedViews.toDouble() /
                max(1.0, observedViews.toDouble())
        return (
            original.coerceIn(0.0, 1.0) * 0.46 +
                jointScore * 0.34 +
                coverage * 0.12 +
                temporalScore * 0.08
        ).coerceIn(0.0, 1.0)
    }

    private fun adjustedUncertainty(
        originalMeters: Double,
        rmsResidualPx: Double,
        cameraCount: Int,
        score: Double,
    ): Double {
        val residualMeters =
            if (!rmsResidualPx.isFinite()) {
                0.12
            } else {
                (
                    rmsResidualPx /
                        1000.0
                ).coerceIn(0.0005, 0.25)
            }

        val coverageScale =
            if (cameraCount <= 1) 1.45
            else 1.0 / sqrt(cameraCount.toDouble().coerceAtLeast(1.0))

        val confidenceScale =
            (1.25 - score.coerceIn(0.0, 1.0) * 0.75)
                .coerceIn(0.40, 1.25)

        return max(
            originalMeters.coerceAtLeast(0.0005),
            residualMeters * coverageScale * confidenceScale,
        )
    }

    private fun buildWarnings(
        score: Double,
        medianResidualPx: Double,
        p95ResidualPx: Double,
        maximumResidualPx: Double,
        inlierRatio: Double,
        viewAgreement: Double,
        temporalConsistency: Double,
        acceptedJointCount: Int,
        rejectedJointCount: Int,
        calibrationErrorPx: Double,
        synchronizationConfidence: Double,
        cameraMotionRisk: Double,
        diagnostics: List<JointDiagnostics>,
    ): List<String> {
        val warnings = ArrayList<String>()

        if (score < config.globalAcceptanceScore) {
            warnings += "fatal: multi-view reprojection consistency below production acceptance"
        } else if (score < config.globalSoftWarningScore) {
            warnings += "multi-view reprojection consistency degraded"
        }

        if (medianResidualPx.isFinite() &&
            medianResidualPx > config.huberTransitionPx
        ) {
            warnings += "median reprojection residual elevated"
        }

        if (p95ResidualPx.isFinite() &&
            p95ResidualPx > config.hardRejectResidualPx * 0.55
        ) {
            warnings += "upper-tail reprojection residual elevated"
        }

        if (maximumResidualPx.isFinite() &&
            maximumResidualPx > config.hardRejectResidualPx
        ) {
            warnings += "one or more camera views exceed hard residual gate"
        }

        if (inlierRatio < 0.65) {
            warnings += "multi-view inlier coverage is low"
        }

        if (viewAgreement < 0.58) {
            warnings += "camera views disagree on reconstructed joint locations"
        }

        if (temporalConsistency < 0.58) {
            warnings += "reprojection residuals are temporally unstable"
        }

        if (calibrationErrorPx > config.calibrationPenaltyScalePx) {
            warnings += "calibration reprojection error is reducing view confidence"
        }

        if (synchronizationConfidence < 0.70) {
            warnings += "camera synchronization confidence is limiting reprojection trust"
        }

        if (cameraMotionRisk > 0.65) {
            warnings += "camera motion risk is reducing reprojection trust"
        }

        diagnostics
            .filter { it.persistentResidualFrames >= config.persistentResidualFrames }
            .take(6)
            .forEach {
                warnings +=
                    "joint " + it.jointId +
                        " has persistent reprojection degradation"
            }

        if (acceptedJointCount == 0 && rejectedJointCount > 0) {
            warnings += "no reconstructed joints passed the multi-view reprojection gate"
        }

        return warnings
    }

    private fun buildRecommendations(
        score: Double,
        medianResidualPx: Double,
        inlierRatio: Double,
        viewAgreement: Double,
        temporalConsistency: Double,
        calibrationErrorPx: Double,
        synchronizationConfidence: Double,
        cameraMotionRisk: Double,
        diagnostics: List<JointDiagnostics>,
    ): List<String> {
        val recommendations = ArrayList<String>()

        if (calibrationErrorPx >
            config.calibrationPenaltyScalePx
        ) {
            recommendations += "re-run rig calibration and verify fixed reference camera geometry"
        }

        if (synchronizationConfidence < 0.70) {
            recommendations += "tighten peer clock alignment before accepting the next capture segment"
        }

        if (cameraMotionRisk > 0.65) {
            recommendations += "reduce camera movement or enable a stronger camera-motion compensation profile"
        }

        if (inlierRatio < 0.65) {
            recommendations += "inspect occluded or blurred cameras and remove low-confidence views"
        }

        if (viewAgreement < 0.58) {
            recommendations += "increase camera baseline coverage around the subject and verify extrinsics"
        }

        if (temporalConsistency < 0.58) {
            recommendations += "check capture cadence, rolling-shutter exposure and per-device timestamp drift"
        }

        if (medianResidualPx.isFinite() &&
            medianResidualPx > config.residualScalePx * 1.5
        ) {
            recommendations += "reduce motion blur and verify that all cameras use the intended focus distance"
        }

        if (diagnostics.any {
                it.persistentResidualFrames >=
                    config.persistentResidualFrames
            }
        ) {
            recommendations += "inspect persistent per-joint residuals for occlusion or subject self-intersection"
        }

        if (score >= config.globalSoftWarningScore) {
            recommendations += "multi-view residuals are within the preferred production operating envelope"
        }

        return recommendations
    }

    private fun buildObservationIndex(
        frames: List<CameraFrame>,
    ): Map<Int, List<IndexedObservation>> {
        val index = HashMap<Int, MutableList<IndexedObservation>>()
        frames.forEach { frame ->
            frame.observations.forEach { observation ->
                index.getOrPut(
                    observation.landmarkId,
                    ::mutableListOf,
                ) += IndexedObservation(
                    frame = frame,
                    observation = observation,
                )
            }
        }
        return index
    }

    private data class IndexedObservation(
        val frame: CameraFrame,
        val observation: Observation,
    )

    private fun project(
        point: Vec3,
        intrinsics: CameraIntrinsics,
        cameraPosition: Vec3,
        cameraRotation: com.nivukx.nox.core.model.Quat,
    ): Projection? {
        val cameraPoint = cameraRotation
            .conjugate()
            .normalized()
            .rotate(point - cameraPosition)

        if (!cameraPoint.x.isFinite() ||
            !cameraPoint.y.isFinite() ||
            !cameraPoint.z.isFinite() ||
            cameraPoint.z <= 1e-8
        ) {
            return null
        }

        var x = cameraPoint.x / cameraPoint.z
        var y = cameraPoint.y / cameraPoint.z
        val r2 = x * x + y * y

        val radial =
            1.0 +
                intrinsics.k1 * r2 +
                intrinsics.k2 * r2 * r2 +
                intrinsics.k3 * r2 * r2 * r2

        val tangentialX =
            2.0 * intrinsics.p1 * x * y +
                intrinsics.p2 * (r2 + 2.0 * x * x)

        val tangentialY =
            intrinsics.p1 * (r2 + 2.0 * y * y) +
                2.0 * intrinsics.p2 * x * y

        x = x * radial + tangentialX
        y = y * radial + tangentialY

        val pixel = Vec2(
            x * intrinsics.fx + intrinsics.cx,
            y * intrinsics.fy + intrinsics.cy,
        )

        return Projection(
            pixel = pixel,
            depthMeters = cameraPoint.z,
            normalizedX = x,
            normalizedY = y,
        )
    }

    private fun isInsideImage(
        projection: Projection,
        intrinsics: CameraIntrinsics,
    ): Boolean {
        val width = intrinsics.imageWidth
        val height = intrinsics.imageHeight

        if (width <= 0 || height <= 0) {
            return projection.pixel.x.isFinite() &&
                projection.pixel.y.isFinite()
        }

        return projection.pixel.x >= -width * 0.10 &&
            projection.pixel.x <= width * 1.10 &&
            projection.pixel.y >= -height * 0.10 &&
            projection.pixel.y <= height * 1.10
    }

    private fun huberWeight(errorPx: Double): Double =
        if (!errorPx.isFinite()) {
            0.0
        } else if (errorPx <= config.huberTransitionPx) {
            1.0
        } else {
            config.huberTransitionPx /
                max(errorPx, config.huberTransitionPx)
        }

    private fun temporalBaseline(): Double = 1.0

    private fun sanitizeSkeleton(
        source: SkeletonFrame,
    ): SkeletonFrame {
        val finiteJoints = source.joints.filter {
            it.position.x.isFinite() &&
                it.position.y.isFinite() &&
                it.position.z.isFinite() &&
                it.velocity.x.isFinite() &&
                it.velocity.y.isFinite() &&
                it.velocity.z.isFinite()
        }
        return source.copy(
            joints = finiteJoints,
            globalConfidence = source.globalConfidence
                .coerceIn(0.0, 1.0),
        )
    }

    private fun weightedAverage(
        values: List<Double>,
        weights: List<Double>,
    ): Double {
        if (values.isEmpty()) return 0.0
        val usable = values.indices.filter {
            values[it].isFinite() &&
                weights.getOrElse(it) { 1.0 }.isFinite() &&
                weights.getOrElse(it) { 1.0 } > 0.0
        }
        if (usable.isEmpty()) return 0.0
        val totalWeight = usable.sumOf {
            weights.getOrElse(it) { 1.0 }.coerceAtLeast(0.0)
        }
        if (totalWeight <= 1e-12) return 0.0
        return usable.sumOf {
            values[it].coerceIn(0.0, 1.0) *
                weights.getOrElse(it) { 1.0 }.coerceAtLeast(0.0)
        } / totalWeight
    }

    private fun rootMeanSquare(values: List<Double>): Double {
        if (values.isEmpty()) return Double.POSITIVE_INFINITY
        return sqrt(
            values
                .filter { it.isFinite() }
                .map { it * it }
                .averageOrZero(),
        )
    }
}

private fun List<Double>.averageOrZero(): Double =
    if (isEmpty()) 0.0 else average()

private fun List<Double>.percentile(fraction: Double): Double {
    val sorted = filter { it.isFinite() }.sorted()
    if (sorted.isEmpty()) return Double.POSITIVE_INFINITY
    if (sorted.size == 1) return sorted.first()
    val f = fraction.coerceIn(0.0, 1.0)
    val position = f * (sorted.size - 1)
    val lower = position.toInt()
    val upper = min(lower + 1, sorted.lastIndex)
    val alpha = position - lower
    return sorted[lower] * (1.0 - alpha) +
        sorted[upper] * alpha
}

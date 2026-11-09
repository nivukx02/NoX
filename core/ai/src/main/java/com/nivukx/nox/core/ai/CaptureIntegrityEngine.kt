package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.MotionCaptureCustomization
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Production frame-integrity fusion for motion capture output.
 *
 * This engine is intentionally model-agnostic. It does not invent missing evidence and it never
 * treats temporal smoothing as proof of correctness. Instead it fuses independently measured
 * evidence into an explicit decision with:
 *  - component scores that remain inspectable in the UI/session metadata;
 *  - hard safety gates for invalid geometry, stale synchronization and severe runtime pressure;
 *  - configurable operator strictness;
 *  - exponential smoothing for frame-to-frame noise;
 *  - band hysteresis so a single bad frame cannot flap an output stream;
 *  - targeted recommendations based on the weakest evidence.
 *
 * The result is suitable for deciding whether an avatar/reconstruction frame may leave the
 * reconstruction boundary and be serialized as trusted motion.
 */
class CaptureIntegrityEngine(
    private val baseConfig: Config = Config(),
) : AutoCloseable {

    enum class Band {
        EXCELLENT,
        TRUSTED,
        DEGRADED,
        HOLD,
        BLOCKED,
    }

    enum class Bottleneck {
        NONE,
        GEOMETRY,
        RIG_COVERAGE,
        TEMPORAL,
        RECONSTRUCTION,
        TRACKING,
        OCCLUSION,
        CAMERA_MOTION,
        CALIBRATION,
        PERFORMANCE,
        FACE_EYE,
        FACE_EYE_OOD,
        GLOBAL_4D,
        HAND,
        KINEMATIC,
        SURFACE,
        PHYSICS,
        DIRECTIONAL_UNCERTAINTY,
        SYNCHRONIZATION,
        DENSE_LANDMARKS,
        UNKNOWN,
    }

    data class Config(
        val minimumOutputScore: Double = 0.58,
        val excellentScore: Double = 0.90,
        val trustedScore: Double = 0.76,
        val degradedScore: Double = 0.58,
        val hardReprojectionErrorPx: Double = 12.0,
        val goodReprojectionErrorPx: Double = 3.0,
        val hardUncertaintyMeters: Double = 0.65,
        val goodUncertaintyMeters: Double = 0.12,
        val minimumSubjectConfidence: Double = 0.30,
        val minimumGeometryConfidence: Double = 0.40,
        val minimumCalibrationConfidence: Double = 0.40,
        val minimumTemporalConfidence: Double = 0.42,
        val minimumSyncConfidence: Double = 0.45,
        val maximumCameraMotionRisk: Double = 0.92,
        val minimumPerformanceScale: Double = 0.28,
        val maximumOcclusionRisk: Double = 0.94,
        val smoothingAlpha: Double = 0.22,
        val scoreHysteresis: Double = 0.035,
        val bandConfirmationFrames: Int = 3,
        val warningLimit: Int = 8,
    ) {
        init {
            require(minimumOutputScore in 0.0..1.0)
            require(excellentScore in 0.0..1.0)
            require(trustedScore in 0.0..1.0)
            require(degradedScore in 0.0..1.0)
            require(minimumOutputScore >= 0.0)
            require(excellentScore >= trustedScore)
            require(trustedScore >= degradedScore)
            require(hardReprojectionErrorPx > goodReprojectionErrorPx)
            require(hardUncertaintyMeters > goodUncertaintyMeters)
            require(minimumSubjectConfidence in 0.0..1.0)
            require(minimumGeometryConfidence in 0.0..1.0)
            require(minimumCalibrationConfidence in 0.0..1.0)
            require(minimumTemporalConfidence in 0.0..1.0)
            require(minimumSyncConfidence in 0.0..1.0)
            require(maximumCameraMotionRisk in 0.0..1.0)
            require(minimumPerformanceScale in 0.0..1.0)
            require(maximumOcclusionRisk in 0.0..1.0)
            require(smoothingAlpha in 0.01..1.0)
            require(scoreHysteresis >= 0.0)
            require(bandConfirmationFrames >= 1)
            require(warningLimit >= 1)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val reconstructionConfidence: Double,
        val reprojectionErrorPx: Double,
        val meanUncertaintyMeters: Double,
        val geometricObservability: Double,
        val rigCoverageScore: Double = 1.0,
        val temporalConfidence: Double,
        val temporalStability: Double,
        val synchronizationConfidence: Double,
        val calibrationConfidence: Double,
        val subjectConfidence: Double,
        val motionScore: Double,
        val motionEvidenceConfidence: Double,
        val microMotionScore: Double,
        val measurementUncertaintyScore: Double = 0.0,
        val occlusionRisk: Double,
        val cameraMotionRisk: Double,
        val faceEyeConfidence: Double,
        val faceEyeFusionMode: String = "HOLD",
        val faceEyeIndependentEvidenceSupport: Double = 0.0,
        val faceGeometryConfidence: Double = 1.0,
        val eyeTemporalConfidence: Double = 1.0,
        val handConfidence: Double,
        val handUncertaintyConfidence: Double = 1.0,
        val handUncertaintyOodRisk: Double = 0.0,
        val thermalScale: Double,
        val memoryScale: Double,
        val runtimeQualityScale: Double,
        val multiCamera: Boolean,
        val customization: MotionCaptureCustomization,
        val reconstructionWarnings: List<String> = emptyList(),
        val temporalWarnings: List<String> = emptyList(),
        val geometryWarnings: List<String> = emptyList(),
        val kinematicConstraintScore: Double = 1.0,
        val surfaceConfidence: Double = 1.0,
        val surfaceNormalObservability: Double = 1.0,
        val surfaceHighUncertaintyRatio: Double = 0.0,
        val surfaceCameraInformationGain: Double = 1.0,
        val surfaceEvidenceStrictness: Double = 0.84,
        val denseLandmarkCoverage: Double = 1.0,
        val denseLandmarkVisibleCoverage: Double = 1.0,
        val denseLandmarkUncertaintyP95: Double = 0.0,
        val denseLandmarkCrossViewAgreement: Double = 1.0,
        val denseLandmarkCrossViewEvidenceAvailable: Boolean = false,
        val denseLandmarkTemporalStability: Double = 1.0,
        val denseLandmarkHardFailure: Boolean = false,
        val physicsConstraintScore: Double = 1.0,
        val physicsConstraintAccepted: Boolean = true,
        val directionalUncertaintyMajor95Meters: Double = 0.0,
        val directionalUncertaintyDepthMeters: Double = 0.0,
        val directionalUncertaintyAnisotropy: Double = 1.0,
        val directionalUncertaintyConfidence: Double = 1.0,
        val directionalUncertaintyDegradedJointCount: Int = 0,
        val faceEyeFusionAccepted: Boolean = true,
        val faceEyeHardFailure: Boolean = false,
        val faceEyeShapeConfidence: Double = 1.0,
        val faceEyeEyeDynamicConsistency: Double = 1.0,
        val faceEyeCalibrationDriftScore: Double = 0.0,
        val faceEyeCalibrationDriftPersistent: Boolean = false,
        val faceEyeReliabilityTrend: Double = 0.0,
        val faceEyeOutOfDistributionRisk: Double = 0.0,
        val faceEyeFusionUncertaintyTotal: Double = 0.0,
        val faceEyeFusionUncertaintyTrustworthy: Boolean = true,
        val global4DScore: Double = 1.0,
        val global4DAccepted: Boolean = true,
        val global4DPersistentFailure: Boolean = false,
        val global4DRootMotionConsistency: Double = 1.0,
        val global4DRootVelocityAgreement: Double = 1.0,
        val global4DSceneConsistency: Double = 0.72,
        val global4DDirectionalUncertaintyConsistency: Double = 1.0,
        val global4DFaceEyeConsistency: Double = 1.0,
        val globalRigTrust: Double = 1.0,
        val worldSpaceConsistencyScore: Double = 1.0,
        val worldSpaceRobustOutlierRatio: Double = 0.0,
        val kinematicWarnings: List<String> = emptyList(),
        val surfaceWarnings: List<String> = emptyList(),
        val motionBlurRisk: Double = 0.0,
        val bodyMetricEvidenceAuthority: Double = 1.0,
        val faceMetricEvidenceAuthority: Double? = null,
        val handMetricEvidenceAuthority: Double? = null,
        val bodyEpipolarConsistency: Double = 1.0,
        val faceEpipolarConsistency: Double? = null,
        val handEpipolarConsistency: Double? = null,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(reconstructionConfidence in 0.0..1.0)
            require(reprojectionErrorPx.isFinite() || reprojectionErrorPx.isInfinite())
            require(meanUncertaintyMeters.isFinite() || meanUncertaintyMeters.isInfinite())
            require(geometricObservability in 0.0..1.0)
            require(rigCoverageScore in 0.0..1.0)
            require(temporalConfidence in 0.0..1.0)
            require(temporalStability in 0.0..1.0)
            require(synchronizationConfidence in 0.0..1.0)
            require(calibrationConfidence in 0.0..1.0)
            require(subjectConfidence in 0.0..1.0)
            require(motionScore in 0.0..1.0)
            require(motionEvidenceConfidence in 0.0..1.0)
            require(microMotionScore in 0.0..1.0)
            require(measurementUncertaintyScore in 0.0..1.0)
            require(occlusionRisk in 0.0..1.0)
            require(cameraMotionRisk in 0.0..1.0)
            require(motionBlurRisk in 0.0..1.0)
            require(faceEyeConfidence in 0.0..1.0)
            require(faceGeometryConfidence in 0.0..1.0)
            require(eyeTemporalConfidence in 0.0..1.0)
            require(handConfidence in 0.0..1.0)
            require(handUncertaintyConfidence in 0.0..1.0)
            require(handUncertaintyOodRisk in 0.0..1.0)
            require(thermalScale in 0.0..1.0)
            require(memoryScale in 0.0..1.0)
            require(runtimeQualityScale in 0.0..1.0)
            require(kinematicConstraintScore in 0.0..1.0)
            require(surfaceConfidence in 0.0..1.0)
            require(surfaceNormalObservability in 0.0..1.0)
            require(surfaceHighUncertaintyRatio in 0.0..1.0)
            require(surfaceCameraInformationGain in 0.0..1.0)
            require(surfaceEvidenceStrictness in 0.0..1.0)
            require(denseLandmarkCoverage in 0.0..1.0)
            require(denseLandmarkVisibleCoverage in 0.0..1.0)
            require(denseLandmarkUncertaintyP95 in 0.0..1.0)
            require(denseLandmarkCrossViewAgreement in 0.0..1.0)
            require(denseLandmarkTemporalStability in 0.0..1.0)
            require(physicsConstraintScore in 0.0..1.0)
            require(directionalUncertaintyMajor95Meters >= 0.0)
            require(directionalUncertaintyDepthMeters >= 0.0)
            require(directionalUncertaintyAnisotropy >= 1.0)
            require(directionalUncertaintyConfidence in 0.0..1.0)
            require(directionalUncertaintyDegradedJointCount >= 0)
            require(globalRigTrust in 0.0..1.0)
            require(faceEyeCalibrationDriftScore in 0.0..1.0)
            require(faceEyeReliabilityTrend in -1.0..1.0)
            require(faceEyeOutOfDistributionRisk in 0.0..1.0)
            require(faceEyeFusionUncertaintyTotal in 0.0..1.0)
            require(global4DScore in 0.0..1.0)
            require(global4DRootMotionConsistency in 0.0..1.0)
            require(global4DSceneConsistency in 0.0..1.0)
            require(worldSpaceConsistencyScore in 0.0..1.0)
            require(worldSpaceRobustOutlierRatio in 0.0..1.0)
            require(faceEyeShapeConfidence in 0.0..1.0)
            require(faceEyeEyeDynamicConsistency in 0.0..1.0)
            require(faceEyeFusionAccepted || faceEyeConfidence >= 0.0)
            require(bodyMetricEvidenceAuthority in 0.0..1.0)
            listOfNotNull(
                faceMetricEvidenceAuthority,
                handMetricEvidenceAuthority,
                faceEpipolarConsistency,
                handEpipolarConsistency,
            ).forEach {
                require(it in 0.0..1.0)
            }
            require(bodyEpipolarConsistency in 0.0..1.0)
        }
    }

    data class Component(
        val reconstruction: Double,
        val geometry: Double,
        val rigCoverage: Double,
        val temporal: Double,
        val synchronization: Double,
        val calibration: Double,
        val tracking: Double,
        val occlusion: Double,
        val cameraMotion: Double,
        val performance: Double,
        val microMotion: Double,
        val faceGeometry: Double,
        val eyeTemporal: Double,
        val faceEye: Double,
        val faceEyeOod: Double,
        val global4D: Double,
        val hand: Double,
        val handUncertainty: Double,
        val kinematic: Double,
        val surface: Double,
        val denseLandmarks: Double,
        val physics: Double,
        val worldSpace: Double,
        val directionalUncertainty: Double,
        val final: Double,
    ) {
        val minimum: Double
            get() =
                minOf(
                    reconstruction,
                    geometry,
                    rigCoverage,
                    temporal,
                    synchronization,
                    calibration,
                    tracking,
                    occlusion,
                    cameraMotion,
                    performance,
                    microMotion,
                    faceGeometry,
                    eyeTemporal,
                    faceEye,
                    faceEyeOod,
                    global4D,
                    hand,
                    handUncertainty,
                    kinematic,
                    surface,
                    denseLandmarks,
                    physics,
                    worldSpace,
                    directionalUncertainty,
                )
    }

    data class Result(
        val timestampNanos: Long,
        val rawScore: Double,
        val smoothedScore: Double,
        val band: Band,
        val outputAllowed: Boolean,
        val hardFailures: List<String>,
        val warnings: List<String>,
        val recommendations: List<String>,
        val bottleneck: Bottleneck,
        val components: Component,
        val frameIndex: Long,
        val acceptedFrames: Long,
        val heldFrames: Long,
    ) {
        val trustworthy: Boolean
            get() = outputAllowed && band != Band.BLOCKED

        val captureReady: Boolean
            get() = outputAllowed && band in setOf(Band.EXCELLENT, Band.TRUSTED)
    }

    private var smoothedScore = 0.0
    private var initialized = false
    private var frameIndex = 0L
    private var acceptedFrames = 0L
    private var heldFrames = 0L
    private var currentBand = Band.BLOCKED
    private var pendingBand = Band.BLOCKED
    private var pendingCount = 0

    @Synchronized
    fun evaluate(input: Input): Result {
        frameIndex += 1L

        val effective = effectiveConfig(input.customization)
        val hardFailures = mutableListOf<String>()
        val warnings = mutableListOf<String>()

        val reconstruction =
            reconstructionScore(
                confidence = input.reconstructionConfidence,
                reprojectionErrorPx = input.reprojectionErrorPx,
                uncertaintyMeters = input.meanUncertaintyMeters,
            )
        val geometry = input.geometricObservability.coerceIn(0.0, 1.0)
        val rigCoverage =
            if (input.multiCamera) {
                minOf(
                    input.rigCoverageScore.coerceIn(0.0, 1.0),
                    input.globalRigTrust.coerceIn(0.0, 1.0),
                )
            } else {
                1.0
            }
        val temporal =
            (
                input.temporalConfidence * 0.58 +
                    input.temporalStability * 0.42
            ).coerceIn(0.0, 1.0)
        val synchronization =
            if (input.multiCamera) {
                input.synchronizationConfidence.coerceIn(0.0, 1.0)
            } else {
                1.0
            }
        val calibration =
            if (input.multiCamera) {
                input.calibrationConfidence.coerceIn(0.0, 1.0)
            } else {
                max(
                    input.calibrationConfidence,
                    0.55,
                ).coerceIn(0.0, 1.0)
            }
        val tracking =
            (
                input.subjectConfidence * 0.62 +
                    input.motionEvidenceConfidence * 0.38
            ).coerceIn(0.0, 1.0)
        val measurementQuality =
            (
                1.0 -
                    input.measurementUncertaintyScore
            ).coerceIn(0.0, 1.0)
        val occlusion =
            (
                1.0 - input.occlusionRisk
            ).coerceIn(0.0, 1.0)
        val cameraMotion =
            (
                1.0 -
                    input.cameraMotionRisk *
                    input.customization.cameraMotionSuppression
            ).coerceIn(0.0, 1.0)
        val performance =
            min(
                min(input.thermalScale, input.memoryScale),
                input.runtimeQualityScale,
            ).coerceIn(0.0, 1.0)
        val microMotion =
            (
                input.microMotionScore *
                    input.customization.microMotionPreservation +
                    input.motionScore *
                    (1.0 - input.customization.microMotionPreservation)
            ).coerceIn(0.0, 1.0)
        val faceEyeOod =
            if (!input.customization.recordFaceEyeFusion) {
                1.0
            } else {
                (
                    1.0 -
                        input.faceEyeOutOfDistributionRisk
                            .coerceIn(0.0, 1.0)
                ).coerceIn(0.0, 1.0)
            }

        val faceEye =
            if (!input.customization.recordFaceEyeFusion) {
                1.0
            } else {
                val geometryTemporal =
                    (
                        input.faceGeometryConfidence * 0.55 +
                            input.eyeTemporalConfidence * 0.45
                    ).coerceIn(0.0, 1.0)
                val uncertaintyPenalty =
                    (
                        1.0 -
                            input.faceEyeFusionUncertaintyTotal * 0.22
                        ).coerceIn(0.74, 1.0)
                val shapeDynamicEvidence =
                    (
                        input.faceEyeShapeConfidence * 0.55 +
                            input.faceEyeEyeDynamicConsistency * 0.45
                        ).coerceIn(0.0, 1.0)

                val baseFaceEye =
                    (
                        (
                            (
                                input.faceEyeConfidence *
                                    input.customization.faceEyeFusionWeight +
                                input.subjectConfidence *
                                    (1.0 -
                                        input.customization.faceEyeFusionWeight)
                            ) *
                                (0.60 + 0.40 * geometryTemporal)
                        ) *
                            (0.60 + 0.40 * shapeDynamicEvidence) *
                            (
                                1.0 -
                                    input.faceEyeCalibrationDriftScore * 0.54
                            ) *
                            uncertaintyPenalty
                        ).coerceIn(0.0, 1.0)

                (
                    baseFaceEye *
                        faceEyeReliabilityTrendScore(
                            input.faceEyeReliabilityTrend,
                        )
                    ).coerceIn(0.0, 1.0)
            }
        
        val hand =
            if (!input.customization.recordHand3D) {
                1.0
            } else {
                input.handConfidence.coerceIn(0.0, 1.0)
            }

        val kinematic =
            input.kinematicConstraintScore.coerceIn(0.0, 1.0)

        val surface =
            (
                input.surfaceConfidence.coerceIn(0.0, 1.0) * 0.52 +
                    input.surfaceNormalObservability.coerceIn(0.0, 1.0) * 0.20 +
                    (
                        1.0 -
                            input.surfaceHighUncertaintyRatio.coerceIn(0.0, 1.0)
                        ) * 0.18 +
                    input.surfaceCameraInformationGain.coerceIn(0.0, 1.0) * 0.10
            ).coerceIn(0.0, 1.0)

        val physics =
            input.physicsConstraintScore.coerceIn(0.0, 1.0)

        val denseLandmarks =
            scoreDenseLandmarks(
                coverage = input.denseLandmarkCoverage,
                visibleCoverage = input.denseLandmarkVisibleCoverage,
                uncertaintyP95 = input.denseLandmarkUncertaintyP95,
                crossViewAgreement = input.denseLandmarkCrossViewAgreement,
                crossViewEvidenceAvailable =
                    input.denseLandmarkCrossViewEvidenceAvailable,
                temporalStability = input.denseLandmarkTemporalStability,
                hardFailure = input.denseLandmarkHardFailure,
            )

        val directionalUncertainty =
            directionalUncertaintyScore(
                major95Meters = input.directionalUncertaintyMajor95Meters,
                depthMeters = input.directionalUncertaintyDepthMeters,
                anisotropy = input.directionalUncertaintyAnisotropy,
                confidence = input.directionalUncertaintyConfidence,
                degradedJointCount =
                    input.directionalUncertaintyDegradedJointCount,
            )

        val global4D =
            input.global4DScore.coerceIn(0.0, 1.0)

        val handUncertainty =
            (
                input.handUncertaintyConfidence *
                    (
                        1.0 -
                            input.handUncertaintyOodRisk * 0.65
                    )
            ).coerceIn(0.0, 1.0)

        val worldSpace =
            (
                input.worldSpaceConsistencyScore *
                    (
                        1.0 -
                            input.worldSpaceRobustOutlierRatio * 0.65
                    )
            ).coerceIn(0.0, 1.0)

        if (input.reprojectionErrorPx.isNaN() ||
            input.reprojectionErrorPx >= effective.hardReprojectionErrorPx
        ) {
            hardFailures += "reprojection-error-hard-gate"
        }
        if (input.meanUncertaintyMeters.isNaN() ||
            input.meanUncertaintyMeters >= effective.hardUncertaintyMeters
        ) {
            hardFailures += "metric-uncertainty-hard-gate"
        }
        if (tracking < effective.minimumSubjectConfidence) {
            hardFailures += "subject-tracking-below-floor"
        }
        if (measurementQuality < 0.22) {
            hardFailures += "measurement-uncertainty-hard-gate"
        }
        if (input.motionBlurRisk >= 0.92) {
            hardFailures += "optical-motion-blur-hard-gate"
        }

        val crossModalMetricAuthority =
            listOfNotNull(
                input.bodyMetricEvidenceAuthority,
                input.faceMetricEvidenceAuthority,
                input.handMetricEvidenceAuthority,
            ).minOrNull()
                ?.coerceIn(0.0, 1.0)
                ?: 0.0
        val crossModalEpipolarAuthority =
            listOfNotNull(
                input.bodyEpipolarConsistency,
                input.faceEpipolarConsistency,
                input.handEpipolarConsistency,
            ).minOrNull()
                ?.coerceIn(0.0, 1.0)
                ?: 0.0

        if (input.bodyMetricEvidenceAuthority < 0.16 ||
            input.faceMetricEvidenceAuthority?.let { it < 0.16 } == true ||
            input.handMetricEvidenceAuthority?.let { it < 0.16 } == true
        ) {
            hardFailures += "cross-modal-metric-authority-hard-gate"
        }
        if (input.bodyEpipolarConsistency < 0.18 ||
            input.faceEpipolarConsistency?.let { it < 0.18 } == true ||
            input.handEpipolarConsistency?.let { it < 0.18 } == true
        ) {
            hardFailures += "cross-modal-epipolar-authority-hard-gate"
        }
        if (geometry < effective.minimumGeometryConfidence &&
            input.multiCamera
        ) {
            hardFailures += "geometric-observability-below-floor"
        }
        if (input.multiCamera && rigCoverage < 0.20) {
            hardFailures += "rig-coverage-critical"
        }
        if (calibration < effective.minimumCalibrationConfidence &&
            input.multiCamera
        ) {
            hardFailures += "calibration-confidence-below-floor"
        }
        if (temporal < effective.minimumTemporalConfidence) {
            hardFailures += "temporal-confidence-below-floor"
        }
        if (synchronization < effective.minimumSyncConfidence &&
            input.multiCamera
        ) {
            hardFailures += "multi-camera-sync-below-floor"
        }
        if (input.cameraMotionRisk > effective.maximumCameraMotionRisk) {
            hardFailures += "camera-motion-risk-hard-gate"
        }
        if (performance < effective.minimumPerformanceScale) {
            hardFailures += "runtime-performance-hard-gate"
        }
        if (input.occlusionRisk > effective.maximumOcclusionRisk) {
            hardFailures += "occlusion-risk-hard-gate"
        }
        if (kinematic < 0.20) {
            hardFailures += "kinematic-constraint-hard-gate"
        }
        if (surface < 0.20) {
            hardFailures += "surface-confidence-hard-gate"
        }
        if (
            input.surfaceNormalObservability <
                (0.10 + input.surfaceEvidenceStrictness * 0.18) ||
            input.surfaceHighUncertaintyRatio >
                (0.98 - input.surfaceEvidenceStrictness * 0.30)
        ) {
            hardFailures += "surface-normal-hard-gate"
        }
        if (
            input.surfaceCameraInformationGain <
                (0.12 + input.surfaceEvidenceStrictness * 0.30)
        ) {
            hardFailures += "surface-view-information-hard-gate"
        }
        if (input.denseLandmarkHardFailure) {
            hardFailures += "dense-landmark-hard-gate"
        }
        if (input.denseLandmarkCoverage < 0.16) {
            hardFailures += "dense-landmark-coverage-hard-gate"
        }
        if (input.denseLandmarkUncertaintyP95 > 0.94) {
            hardFailures += "dense-landmark-uncertainty-hard-gate"
        }
        if (
            input.denseLandmarkCrossViewEvidenceAvailable &&
            input.denseLandmarkCrossViewAgreement < 0.14
        ) {
            hardFailures += "dense-landmark-cross-view-hard-gate"
        }
        if (!input.physicsConstraintAccepted || physics < 0.20) {
            hardFailures += "physics-constraint-hard-gate"
        }
        if (
            input.handUncertaintyConfidence < 0.12 ||
            input.handUncertaintyOodRisk > 0.96
        ) {
            hardFailures += "hand-uncertainty-hard-gate"
        }
        if (
            input.worldSpaceConsistencyScore < 0.12 ||
            input.worldSpaceRobustOutlierRatio > 0.88
        ) {
            hardFailures += "world-space-hard-gate"
        }
        if (
            input.directionalUncertaintyConfidence < 0.10 ||
            input.directionalUncertaintyAnisotropy >= 18.0 ||
            input.directionalUncertaintyDepthMeters >= 1.0
        ) {
            hardFailures += "directional-uncertainty-hard-gate"
        }
        val faceEyeHardThreshold =
            (
                0.08 +
                    input.customization.faceEyeFusionStrictness
                        .coerceIn(0.0, 1.0) * 0.10
            ).coerceIn(0.08, 0.18)

        if (
            !input.faceEyeFusionAccepted &&
            input.faceEyeConfidence <
                faceEyeHardThreshold
        ) {
            hardFailures += "face-eye-fusion-hard-gate"
        }
        if (
            input.faceEyeHardFailure ||
            input.faceEyeShapeConfidence < 0.16 ||
            input.faceEyeEyeDynamicConsistency < 0.12
        ) {
            hardFailures += "face-eye-hard-gate"
        }
        if (input.faceEyeCalibrationDriftScore >= 0.92) {
            hardFailures += "face-eye-calibration-drift-hard-gate"
        }
        if (
            input.faceEyeCalibrationDriftPersistent &&
            input.faceEyeCalibrationDriftScore >= 0.62
        ) {
            hardFailures += "face-eye-calibration-drift-persistent-hard-gate"
        }
        if (input.faceEyeReliabilityTrend <= -0.72) {
            hardFailures += "face-eye-reliability-collapse-hard-gate"
        }
        if (input.faceEyeOutOfDistributionRisk >= 0.92) {
            hardFailures += "face-eye-out-of-distribution-hard-gate"
        }
        if (input.faceEyeFusionUncertaintyTotal >= 0.82) {
            hardFailures += "face-eye-fusion-uncertainty-hard-gate"
        }
        if (
            input.faceEyeFusionUncertaintyTotal >= 0.72 ||
            !input.faceEyeFusionUncertaintyTrustworthy
        ) {
            warnings += "face-eye-fusion-uncertainty-elevated"
        }
        if (
            !input.global4DAccepted ||
            input.global4DScore < 0.18 ||
            input.global4DPersistentFailure
        ) {
            hardFailures += "global-4d-consistency-hard-gate"
        }
        if (input.globalRigTrust < 0.16) {
            hardFailures += "global-rig-trust-hard-gate"
        }

        warnings += input.reconstructionWarnings
        warnings += input.temporalWarnings
        warnings += input.geometryWarnings
        if (input.microMotionScore < 0.12 &&
            input.motionScore > 0.15 &&
            input.customization.microMotionPreservation > 0.80
        ) {
            warnings += "micro-motion-evidence-low"
        }
        if (input.multiCamera && rigCoverage < 0.50) {
            warnings += "rig-coverage-limited"
        }
        if (cameraMotion < 0.45) {
            warnings += "camera-motion-compensation-dominant"
        }
        if (performance < 0.55) {
            warnings += "runtime-performance-reduced"
        }
        if (input.customization.recordFaceEyeFusion &&
            input.faceGeometryConfidence < 0.45
        ) {
            warnings += "face-3d-temporal-evidence-limited"
        }
        if (input.customization.recordFaceEyeFusion &&
            input.eyeTemporalConfidence < 0.45
        ) {
            warnings += "eye-3d-temporal-evidence-limited"
        }
        if (input.customization.recordFaceEyeFusion &&
            faceEye < 0.45
        ) {
            warnings += "face-eye-evidence-limited"
        }
        if (
            input.surfaceNormalObservability <
                0.55 + input.surfaceEvidenceStrictness * 0.18
        ) {
            warnings += "surface-normal-observability-limited"
        }
        if (
            input.surfaceHighUncertaintyRatio >
                (0.45 - input.surfaceEvidenceStrictness * 0.18)
                    .coerceAtLeast(0.18)
        ) {
            warnings += "surface-high-uncertainty-ratio"
        }
        if (
            input.surfaceCameraInformationGain <
                (0.42 + input.surfaceEvidenceStrictness * 0.18)
                    .coerceAtMost(0.70)
        ) {
            warnings += "surface-view-information-limited"
        }
        if (input.faceEyeShapeConfidence < 0.55) {
            warnings += "face-eye-shape-evidence-limited"
        }
        if (input.faceEyeEyeDynamicConsistency < 0.55) {
            warnings += "face-eye-dynamic-evidence-limited"
        }
        if (input.faceEyeCalibrationDriftScore >= 0.62) {
            warnings += "face-eye-calibration-drift-elevated"
        }
        if (input.faceEyeCalibrationDriftPersistent) {
            warnings += "face-eye-calibration-drift-persistent"
        }
        if (input.faceEyeReliabilityTrend <= -0.18) {
            warnings += "face-eye-reliability-declining"
        }
        if (input.faceEyeReliabilityTrend <= -0.45) {
            warnings += "face-eye-reliability-collapse-risk"
        }
        if (input.faceEyeOutOfDistributionRisk >= 0.62) {
            warnings += "face-eye-out-of-distribution-risk-elevated"
        }
        if (
            input.global4DScore < 0.62 ||
            !input.global4DAccepted ||
            input.global4DPersistentFailure
        ) {
            warnings += "global-4d-consistency-evidence-limited"
        }
        if (input.faceEyeCalibrationDriftScore >= 0.82) {
            warnings += "face-eye-recalibration-recommended"
        }
        if (!input.faceEyeFusionAccepted) {
            warnings += "face-eye-fusion-rejected"
            if (
                input.faceEyeConfidence >=
                    faceEyeHardThreshold
            ) {
                warnings +=
                    "face-eye-recovery-only-body-output-preserved"
            }
        }
        if (
            input.directionalUncertaintyConfidence < 0.55 ||
            input.directionalUncertaintyAnisotropy > 6.0 ||
            input.directionalUncertaintyDepthMeters > 0.34
        ) {
            warnings += "directional-uncertainty-evidence-limited"
        }
        if (input.globalRigTrust < 0.55) {
            warnings += "global-rig-trust-degraded"
        }
        if (
            input.handUncertaintyConfidence < 0.55 ||
            input.handUncertaintyOodRisk > 0.62
        ) {
            warnings += "hand-uncertainty-evidence-limited"
        }
        if (
            input.worldSpaceConsistencyScore < 0.55 ||
            input.worldSpaceRobustOutlierRatio > 0.42
        ) {
            warnings += "world-space-evidence-limited"
        }
        if (input.customization.recordHand3D &&
            input.handConfidence < 0.40
        ) {
            warnings += "hand-evidence-limited"
        }

        warnings += input.kinematicWarnings
        warnings += input.surfaceWarnings
        if (kinematic < 0.55) {
            warnings += "kinematic-constraint-evidence-limited"
        }
        if (surface < 0.55) {
            warnings += "dense-surface-confidence-limited"
        }
        if (input.denseLandmarkCoverage < 0.55) {
            warnings += "dense-landmark-coverage-degraded"
        }
        if (input.denseLandmarkVisibleCoverage < 0.55) {
            warnings += "dense-landmark-visible-coverage-degraded"
        }
        if (input.denseLandmarkUncertaintyP95 > 0.35) {
            warnings += "dense-landmark-uncertainty-elevated"
        }
        if (
            input.denseLandmarkCrossViewEvidenceAvailable &&
            input.denseLandmarkCrossViewAgreement < 0.55
        ) {
            warnings += "dense-landmark-cross-view-degraded"
        }
        if (input.denseLandmarkTemporalStability < 0.55) {
            warnings += "dense-landmark-temporal-stability-degraded"
        }
        if (input.motionBlurRisk > 0.48) {
            warnings += "optical-motion-blur-risk-elevated"
        }
        if (input.motionBlurRisk > 0.72) {
            warnings += "optical-motion-blur-risk-high"
        }
        if (crossModalMetricAuthority < 0.55) {
            warnings += "cross-modal-metric-authority-reduced"
        }
        if (crossModalEpipolarAuthority < 0.55) {
            warnings += "cross-modal-epipolar-authority-reduced"
        }

        val weightedRaw =
            (
                (
                    reconstruction * 0.17 +
                        geometry * 0.07 +
                        rigCoverage * 0.05 +
                        temporal * 0.11 +
                        synchronization * 0.08 +
                        calibration * 0.06 +
                        tracking * 0.08 +
                        measurementQuality * 0.02 +
                        occlusion * 0.05 +
                        cameraMotion * 0.06 +
                        performance * 0.05 +
                        microMotion * 0.03 +
                        faceEye * 0.03 +
                        faceEyeOod * 0.04 +
                        global4D * 0.06 +
                        hand * 0.01 +
                        handUncertainty * 0.06 +
                        kinematic * 0.05 +
                        surface * 0.05 +
                        denseLandmarks * 0.06 +
                        physics * 0.06 +
                        worldSpace * 0.06 +
                        directionalUncertainty * 0.04 +
                        crossModalMetricAuthority * 0.03 +
                        crossModalEpipolarAuthority * 0.05
                    ) /
                    1.36
            ).coerceIn(0.0, 1.0)

        val bottleneck = bottleneckOf(
            reconstruction = reconstruction,
            geometry = geometry,
            rigCoverage = rigCoverage,
            temporal = temporal,
            tracking = tracking,
            occlusion = occlusion,
            cameraMotion = cameraMotion,
            calibration = calibration,
            performance = performance,
            synchronization = synchronization,
            faceEye = faceEye,
            faceEyeOod = faceEyeOod,
            global4D = global4D,
            hand = hand,
            handUncertainty = handUncertainty,
            kinematic = kinematic,
            surface = surface,
            physics = physics,
            worldSpace = worldSpace,
            directionalUncertainty = directionalUncertainty,
            denseLandmarks = denseLandmarks,
        )

        val bottleneckPenalty =
            when {
                hardFailures.isNotEmpty() -> 0.0
                bottleneck == Bottleneck.RECONSTRUCTION &&
                    reconstruction < 0.45 -> 0.90
                bottleneck == Bottleneck.GEOMETRY &&
                    geometry < 0.45 -> 0.91
                bottleneck == Bottleneck.TEMPORAL &&
                    temporal < 0.45 -> 0.92
                else -> 1.0
            }

        val rawScore =
            (
                weightedRaw *
                    bottleneckPenalty
            ).coerceIn(0.0, 1.0)

        if (!initialized) {
            smoothedScore = rawScore
            initialized = true
        } else {
            smoothedScore +=
                (rawScore - smoothedScore) *
                    effective.smoothingAlpha
        }

        val desiredBand =
            desiredBand(
                score = smoothedScore,
                hardFailures = hardFailures,
                config = effective,
            )
        val band = advanceBand(
            desired = desiredBand,
            score = smoothedScore,
            config = effective,
        )

        val outputAllowed =
            hardFailures.isEmpty() &&
                smoothedScore >= effective.minimumOutputScore &&
                band != Band.BLOCKED &&
                band != Band.HOLD

        if (outputAllowed) {
            acceptedFrames += 1L
        } else {
            heldFrames += 1L
        }

        val cleanedWarnings =
            warnings
                .asSequence()
                .filter { it.isNotBlank() }
                .distinct()
                .take(effective.warningLimit)
                .toList()
        val recommendations =
            recommendationsFor(
                bottleneck = bottleneck,
                score = smoothedScore,
                input = input,
                outputAllowed = outputAllowed,
            )

        return Result(
            timestampNanos = input.timestampNanos,
            rawScore = rawScore,
            smoothedScore = smoothedScore,
            band = band,
            outputAllowed = outputAllowed,
            hardFailures = hardFailures.distinct(),
            warnings = cleanedWarnings,
            recommendations = recommendations,
            bottleneck = bottleneck,
            components = Component(
                reconstruction = reconstruction,
                geometry = geometry,
                rigCoverage = rigCoverage,
                temporal = temporal,
                synchronization = synchronization,
                calibration = calibration,
                tracking = tracking,
                occlusion = occlusion,
                cameraMotion = cameraMotion,
                performance = performance,
                microMotion = microMotion,
                faceGeometry =
                    input.faceGeometryConfidence.coerceIn(0.0, 1.0),
                eyeTemporal =
                    input.eyeTemporalConfidence.coerceIn(0.0, 1.0),
                faceEye = faceEye,
                faceEyeOod = faceEyeOod,
                global4D = global4D,
                hand = hand,
                handUncertainty = handUncertainty,
                kinematic = kinematic,
                surface = surface,
                physics = physics,
                worldSpace = worldSpace,
                directionalUncertainty = directionalUncertainty,
                crossModalMetricAuthority = crossModalMetricAuthority,
                crossModalEpipolarAuthority = crossModalEpipolarAuthority,
                denseLandmarks = denseLandmarks,
                final = smoothedScore,
            ),
            frameIndex = frameIndex,
            acceptedFrames = acceptedFrames,
            heldFrames = heldFrames,
        )
    }

    @Synchronized
    fun reset() {
        smoothedScore = 0.0
        initialized = false
        frameIndex = 0L
        acceptedFrames = 0L
        heldFrames = 0L
        currentBand = Band.BLOCKED
        pendingBand = Band.BLOCKED
        pendingCount = 0
    }

    private fun faceEyeReliabilityTrendScore(
        trend: Double,
    ): Double {
        val bounded = trend.coerceIn(-1.0, 1.0)
        return when {
            bounded >= 0.0 ->
                (0.88 + bounded * 0.12).coerceIn(0.0, 1.0)
            bounded >= -0.08 ->
                (0.82 + (bounded + 0.08) / 0.08 * 0.06)
                    .coerceIn(0.0, 1.0)
            bounded >= -0.35 ->
                (
                    0.82 -
                        (-bounded - 0.08) / 0.27 * 0.42
                    ).coerceIn(0.34, 0.82)
            else ->
                (
                    0.40 -
                        (-bounded - 0.35) / 0.65 * 0.32
                    ).coerceIn(0.08, 0.40)
        }
    }

    private fun directionalUncertaintyScore(
        major95Meters: Double,
        depthMeters: Double,
        anisotropy: Double,
        confidence: Double,
        degradedJointCount: Int,
    ): Double {
        val major =
            kotlin.math.exp(
                -major95Meters.coerceAtLeast(0.0) / 0.42,
            ).coerceIn(0.0, 1.0)
        val depth =
            kotlin.math.exp(
                -depthMeters.coerceAtLeast(0.0) / 0.34,
            ).coerceIn(0.0, 1.0)
        val shape =
            (
                1.0 -
                    (anisotropy.coerceAtLeast(1.0) - 1.0) / 12.0
                ).coerceIn(0.0, 1.0)
        val degraded =
            kotlin.math.exp(
                -degradedJointCount.coerceAtLeast(0) / 12.0,
            ).coerceIn(0.20, 1.0)
        return (
            major * 0.30 +
                depth * 0.24 +
                shape * 0.16 +
                confidence.coerceIn(0.0, 1.0) * 0.22 +
                degraded * 0.08
            ).coerceIn(0.0, 1.0)
    }

    private fun scoreDenseLandmarks(
        coverage: Double,
        visibleCoverage: Double,
        uncertaintyP95: Double,
        crossViewAgreement: Double,
        crossViewEvidenceAvailable: Boolean,
        temporalStability: Double,
        hardFailure: Boolean,
    ): Double {
        if (hardFailure) return 0.0

        val crossView =
            if (crossViewEvidenceAvailable) {
                crossViewAgreement.coerceIn(0.0, 1.0)
            } else {
                0.72
            }

        return (
            coverage.coerceIn(0.0, 1.0) * 0.24 +
                visibleCoverage.coerceIn(0.0, 1.0) * 0.16 +
                (1.0 - uncertaintyP95.coerceIn(0.0, 1.0)) * 0.26 +
                crossView * 0.20 +
                temporalStability.coerceIn(0.0, 1.0) * 0.14
            ).coerceIn(0.0, 1.0)
    }

    private fun reconstructionScore(
        confidence: Double,
        reprojectionErrorPx: Double,
        uncertaintyMeters: Double,
    ): Double {
        val confidencePart = confidence.coerceIn(0.0, 1.0)
        val reprojectionPart =
            when {
                reprojectionErrorPx.isNaN() ||
                    reprojectionErrorPx.isInfinite() -> 0.0
                reprojectionErrorPx <= baseConfig.goodReprojectionErrorPx -> 1.0
                else ->
                    exp(
                        -(
                            reprojectionErrorPx -
                                baseConfig.goodReprojectionErrorPx
                            ) /
                            max(
                                baseConfig.goodReprojectionErrorPx,
                                0.5,
                            ),
                    ).coerceIn(0.0, 1.0)
            }
        val uncertaintyPart =
            when {
                uncertaintyMeters.isNaN() ||
                    uncertaintyMeters.isInfinite() -> 0.0
                uncertaintyMeters <= baseConfig.goodUncertaintyMeters -> 1.0
                else ->
                    exp(
                        -(
                            uncertaintyMeters -
                                baseConfig.goodUncertaintyMeters
                            ) /
                            max(
                                baseConfig.goodUncertaintyMeters,
                                0.01,
                            ),
                    ).coerceIn(0.0, 1.0)
            }

        return (
            confidencePart * 0.62 +
                reprojectionPart * 0.22 +
                uncertaintyPart * 0.16
        ).coerceIn(0.0, 1.0)
    }

    private fun effectiveConfig(
        customization: MotionCaptureCustomization,
    ): Config {
        val strictness =
            max(
                customization.temporalAlignmentStrictness,
                customization.outputIntegrityStrictness,
            )
        val strictnessScale =
            0.82 + strictness * 0.26
        return baseConfig.copy(
            minimumOutputScore =
                (
                    baseConfig.minimumOutputScore +
                        strictness * 0.08
                ).coerceIn(0.0, 0.95),
            minimumTemporalConfidence =
                (
                    baseConfig.minimumTemporalConfidence *
                        strictnessScale
                ).coerceIn(0.30, 0.92),
            minimumSyncConfidence =
                (
                    baseConfig.minimumSyncConfidence *
                        strictnessScale
                ).coerceIn(0.30, 0.92),
            minimumGeometryConfidence =
                (
                    baseConfig.minimumGeometryConfidence +
                        customization.worldConsistencyDetail * 0.04
                ).coerceIn(0.30, 0.92),
            bandConfirmationFrames =
                max(
                    1,
                    (
                        baseConfig.bandConfirmationFrames +
                            if (strictness > 0.82) 1 else 0
                        ),
                ),
        )
    }

    private fun desiredBand(
        score: Double,
        hardFailures: List<String>,
        config: Config,
    ): Band {
        if (hardFailures.isNotEmpty()) return Band.BLOCKED
        return when {
            score >= config.excellentScore -> Band.EXCELLENT
            score >= config.trustedScore -> Band.TRUSTED
            score >= config.degradedScore -> Band.DEGRADED
            score >= config.minimumOutputScore -> Band.HOLD
            else -> Band.BLOCKED
        }
    }

    private fun advanceBand(
        desired: Band,
        score: Double,
        config: Config,
    ): Band {
        if (desired == Band.BLOCKED) {
            pendingBand = Band.BLOCKED
            pendingCount = 0
            currentBand = Band.BLOCKED
            return currentBand
        }

        if (currentBand == Band.BLOCKED &&
            desired != Band.BLOCKED
        ) {
            if (pendingBand == desired) {
                pendingCount += 1
            } else {
                pendingBand = desired
                pendingCount = 1
            }
            if (pendingCount < config.bandConfirmationFrames) {
                return Band.HOLD
            }
        } else if (
            desired.ordinal > currentBand.ordinal &&
            abs(score - thresholdFor(currentBand, config)) <=
                config.scoreHysteresis
        ) {
            return currentBand
        }

        if (desired != currentBand) {
            if (pendingBand == desired) {
                pendingCount += 1
            } else {
                pendingBand = desired
                pendingCount = 1
            }
            if (pendingCount >= config.bandConfirmationFrames) {
                currentBand = desired
                pendingCount = 0
            }
        } else {
            pendingBand = currentBand
            pendingCount = 0
        }

        return currentBand
    }

    private fun thresholdFor(
        band: Band,
        config: Config,
    ): Double =
        when (band) {
            Band.EXCELLENT -> config.excellentScore
            Band.TRUSTED -> config.trustedScore
            Band.DEGRADED -> config.degradedScore
            Band.HOLD -> config.minimumOutputScore
            Band.BLOCKED -> 0.0
        }

    private fun bottleneckOf(
        reconstruction: Double,
        geometry: Double,
        rigCoverage: Double,
        temporal: Double,
        tracking: Double,
        occlusion: Double,
        cameraMotion: Double,
        calibration: Double,
        performance: Double,
        synchronization: Double,
        faceEye: Double,
        hand: Double,
        kinematic: Double,
        surface: Double,
        physics: Double,
        directionalUncertainty: Double,
        denseLandmarks: Double,
        faceEyeOod: Double,
        global4D: Double,
        handUncertainty: Double = 1.0,
        worldSpace: Double = 1.0,
    ): Bottleneck {
        val values = listOf(
            Bottleneck.RECONSTRUCTION to reconstruction,
            Bottleneck.GEOMETRY to geometry,
            Bottleneck.RIG_COVERAGE to rigCoverage,
            Bottleneck.TEMPORAL to temporal,
            Bottleneck.TRACKING to tracking,
            Bottleneck.OCCLUSION to occlusion,
            Bottleneck.CAMERA_MOTION to cameraMotion,
            Bottleneck.CALIBRATION to calibration,
            Bottleneck.PERFORMANCE to performance,
            Bottleneck.SYNCHRONIZATION to synchronization,
            Bottleneck.FACE_EYE to faceEye,
            Bottleneck.FACE_EYE_OOD to faceEyeOod,
            Bottleneck.GLOBAL_4D to global4D,
            Bottleneck.HAND to hand,
            Bottleneck.KINEMATIC to kinematic,
            Bottleneck.SURFACE to surface,
            Bottleneck.PHYSICS to physics,
            Bottleneck.DIRECTIONAL_UNCERTAINTY to directionalUncertainty,
            Bottleneck.DENSE_LANDMARKS to denseLandmarks,
            Bottleneck.HAND to handUncertainty,
        )
        return values.minByOrNull { it.second }?.first ?: Bottleneck.UNKNOWN
    }

    private fun recommendationsFor(
        bottleneck: Bottleneck,
        score: Double,
        input: Input,
        outputAllowed: Boolean,
    ): List<String> {
        val result = mutableListOf<String>()

        when (bottleneck) {
            Bottleneck.GEOMETRY ->
                result += "Increase camera baseline or move to a better-observed pose."

            Bottleneck.RIG_COVERAGE ->
                result += "Add or reposition cameras to cover the subject volume before recording."

            Bottleneck.TEMPORAL ->
                result += "Tighten temporal alignment before accepting multi-view output."

            Bottleneck.SYNCHRONIZATION ->
                result += "Re-establish synchronized camera timestamps."

            Bottleneck.RECONSTRUCTION ->
                result += "Capture a more observable pose and reduce reprojection residual."

            Bottleneck.TRACKING ->
                result += "Reacquire the subject before committing another trusted frame."

            Bottleneck.OCCLUSION ->
                result += "Expose hidden joints or increase camera coverage before extending prediction."

            Bottleneck.CAMERA_MOTION ->
                result += "Reduce camera motion or increase camera-motion compensation."

            Bottleneck.CALIBRATION ->
                result += "Refresh calibration with a wider pose and stable timestamp alignment."

            Bottleneck.PERFORMANCE ->
                result += "Reduce runtime load or allow thermal recovery before preserving detail."

            Bottleneck.FACE_EYE ->
                result += "Keep the face visible for stable eye and head fusion."

            Bottleneck.FACE_EYE_OOD ->
                result += "Recover in-distribution face and eye evidence before committing metric gaze or head-state output."

            Bottleneck.HAND ->
                result += "Keep the hands inside the capture volume for articulated hand output."

            Bottleneck.KINEMATIC ->
                result += "Reduce skeletal geometry residuals or relax correction only when the measured motion supports it."

            Bottleneck.SURFACE ->
                result += "Increase surface coverage and multi-view evidence before trusting dense mesh detail."

            Bottleneck.PHYSICS ->
                result += "Hold metric output and inspect bone-length, dynamics, symmetry, and floor-contact consistency."

            Bottleneck.DIRECTIONAL_UNCERTAINTY ->
                result += "Increase camera baseline and synchronized view diversity before trusting depth-sensitive metric output."

            Bottleneck.DENSE_LANDMARKS ->
                result += "Improve dense landmark visibility, uncertainty, and view agreement before trusting detailed reconstruction output."

            Bottleneck.GLOBAL_4D ->
                result += "Hold metric output until human trajectory, camera alignment, rig trust, and scene consistency agree."

            Bottleneck.NONE,
            Bottleneck.UNKNOWN -> Unit
        }

        if (input.customization.preserveMicroMotion &&
            input.microMotionScore < 0.10
        ) {
            result += "Micro-motion evidence is weak; avoid over-smoothing the next take."
        }

        if (score >= 0.90 && outputAllowed) {
            result += "Frame meets the trusted motion-output integrity gate."
        } else if (!outputAllowed) {
            result += "Hold this frame outside trusted output until integrity recovers."
        }

        return result.distinct().take(4)
    }

    override fun close() {
        reset()
    }
}

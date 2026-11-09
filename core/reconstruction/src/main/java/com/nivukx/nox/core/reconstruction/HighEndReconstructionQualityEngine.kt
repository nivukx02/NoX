package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ProductionCaptureProfile
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Cross-pipeline production quality and compute-policy orchestrator.
 *
 * Existing NoX engines remain the authoritative sources for perception,
 * calibration, temporal state, geometry, face/eye/hand evidence and TSDF.
 * This layer does not replace them. It converts their independent evidence
 * streams into one conservative operating policy and an explicit compute
 * budget with hysteresis.
 *
 * The central design choice is a weighted geometric blend rather than a plain
 * arithmetic mean. A very strong body track must not completely hide a broken
 * calibration or synchronization channel.
 */
class HighEndReconstructionQualityEngine {

    enum class Policy {
        ACCEPT,
        HOLD,
        DEGRADE,
        REJECT,
    }

    enum class Bottleneck {
        NONE,
        RECONSTRUCTION,
        GEOMETRY,
        TEMPORAL,
        SYNCHRONIZATION,
        CALIBRATION,
        TRACKING,
        FACE_EYE,
        HAND,
        HAND_UNCERTAINTY_CALIBRATION,
        SURFACE,
        DENSE_LANDMARKS,
        PERFORMANCE,
        THERMAL,
        MEMORY,
        UNCERTAINTY_CALIBRATION,
        PHYSICS_CONSISTENCY,
        DIRECTIONAL_UNCERTAINTY,
        METRIC_PRECISION,
        FACE_EYE_OOD,
        FACE_EYE_CAPTURE_DOMAIN,
        GLOBAL_4D_ROOT_VELOCITY,
        WORLD_SPACE_CONSISTENCY,
        GLOBAL_4D_CONSISTENCY,
        OPTICAL_CAPTURE,
    }

    data class Input(
        val timestampNanos: Long,
        val profile: ProductionCaptureProfile,
        val reconstructionConfidence: Double,
        val reprojectionErrorPx: Double,
        val meanUncertaintyMeters: Double,
        val geometricObservability: Double,
        val temporalConfidence: Double,
        val synchronizationConfidence: Double,
        val synchronizationResidualP95Milliseconds: Double = 0.0,
        val synchronizationOutlierRatio: Double = 0.0,
        val calibrationConfidence: Double,
        val trackingConfidence: Double,
        val identitySwapRisk: Double,
        val identityContinuity: Double = 1.0,
        val occlusionRisk: Double,
        val cameraMotionRisk: Double,
        val faceEyeConfidence: Double,
        val handConfidence: Double,
        val handUncertaintyConfidence: Double = 1.0,
        val handUncertaintyOodRisk: Double = 0.0,
        val surfaceConfidence: Double,
        val surfaceNormalObservability: Double = 1.0,
        val surfaceHighUncertaintyRatio: Double = 0.0,
        val denseLandmarkCoverage: Double = 1.0,
        val denseLandmarkVisibleCoverage: Double = 1.0,
        val denseLandmarkUncertaintyP95: Double = 0.0,
        val denseLandmarkCrossViewAgreement: Double = 1.0,
        val denseLandmarkCrossViewEvidenceAvailable: Boolean = false,
        val denseLandmarkTemporalStability: Double = 1.0,
        val denseLandmarkContactCoverage: Double = 0.0,
        val denseLandmarkPredictionBridgeRatio: Double = 0.0,
        val denseLandmarkHardFailure: Boolean = false,
        val metricPrecision95Meters: Double = 0.0,
        val metricPrecisionConfidence: Double = 1.0,
        val metricPrecisionReferenceGradeEligible: Boolean = false,
        val metricPrecisionHardFailure: Boolean = false,
        val cameraInformationGain: Double = 1.0,
        val surfaceEvidenceStrictness: Double = 0.84,
        val thermalScale: Double,
        val memoryScale: Double,
        val runtimeQualityScale: Double,
        val runtimeInferenceScale: Double,
        val gpuDispatchMillis: Double,
        val connectedCameraCount: Int,
        val observedSubjectCount: Int,
        val calibratedUncertaintyConfidence: Double = 1.0,
        val motionOutOfDistributionRisk: Double = 0.0,
        val motionUncertaintyCoverage: Double = 1.0,
        val motionDynamicStress: Double = 0.0,
        val motionDynamicsScore: Double = 0.0,
        val motionLowFrequencyScore: Double = 0.0,
        val motionHighFrequencyScore: Double = 0.0,
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
        val faceEyeOutOfDistributionRisk: Double = 0.0,
        val faceEyeCalibrationDriftScore: Double = 0.0,
        val faceEyeCalibrationDriftPersistent: Boolean = false,
        val faceEyeUncertaintyCalibrationConfidence: Double = 0.0,
        val faceEyeUncertaintyCalibrationReady: Boolean = false,
        val faceUncertaintyCoverage: Double = 0.0,
        val eyeUncertaintyCoverage: Double = 0.0,
        val faceUncertaintyCalibrationMultiplier: Double = 1.0,
        val eyeUncertaintyCalibrationMultiplier: Double = 1.0,
        val faceEyeFusionStrictness: Double = 0.84,
        val faceEyeReliabilityTrend: Double = 0.0,
        val faceEyeCaptureDomainConfidence: Double = 1.0,
        val faceEyeSaccadeVelocityDegreesPerSecond: Double = 0.0,
        val faceEyeDominantFailure: String = "NONE",
        val global4DScore: Double = 1.0,
        val global4DAccepted: Boolean = true,
        val global4DPersistentFailure: Boolean = false,
        val global4DRootMotionConsistency: Double = 1.0,
        val global4DRootVelocityAgreement: Double = 1.0,
        val worldSpaceConsistencyScore: Double = 1.0,
        val worldSpaceRobustOutlierRatio: Double = 0.0,
        val global4DSceneConsistency: Double = 0.72,
        val global4DDirectionalUncertaintyConsistency: Double = 1.0,
        val global4DFaceEyeConsistency: Double = 1.0,
        val globalRigTrust: Double = 1.0,
        val adaptiveDetailScore: Double = 1.0,
        val adaptiveDetailHold: Boolean = false,
        val thermalDriftConfidence: Double = 1.0,
        val thermalDriftHardFailure: Boolean = false,
        val concurrentRigScore: Double = 1.0,
        val concurrentRigAdmissible: Boolean = true,
        val faceEyeMetricEvidence: Double = 1.0,
        val hardwareReadoutTimingAvailable: Boolean = false,
        val hardwareReadoutTimingConfidence: Double = 1.0,
        val hardwareReadoutTimingHardFailure: Boolean = false,
        /**
         * Fraction of admitted multi-person 3D actors currently represented by prediction instead
         * of a fresh reconstruction. Prediction is a continuity mechanism, not new geometry.
         */
        val multiPersonPredictionRatio: Double = 0.0,
        val hardCaptureFailure: Boolean = false,
        val warnings: List<String> = emptyList(),
        val motionBlurRisk: Double = 0.0,
        val bodyMetricEvidenceAuthority: Double = 1.0,
        val faceMetricEvidenceAuthority: Double = 1.0,
        val handMetricEvidenceAuthority: Double = 1.0,
        val bodyEpipolarConsistency: Double = 1.0,
        val faceEpipolarConsistency: Double = 1.0,
        val handEpipolarConsistency: Double = 1.0,
    ) {
        init {
            require(timestampNanos > 0L)
            require(reprojectionErrorPx.isFinite() || reprojectionErrorPx.isInfinite())
            require(meanUncertaintyMeters.isFinite() || meanUncertaintyMeters.isInfinite())
            require(connectedCameraCount >= 0)
            require(observedSubjectCount >= 0)
            require(identityContinuity in 0.0..1.0)
            require(calibratedUncertaintyConfidence in 0.0..1.0)
            require(
                synchronizationResidualP95Milliseconds.isFinite() &&
                    synchronizationResidualP95Milliseconds >= 0.0
            )
            require(synchronizationOutlierRatio in 0.0..1.0)
            require(motionOutOfDistributionRisk in 0.0..1.0)
            require(motionUncertaintyCoverage in 0.0..1.0)
            require(motionDynamicStress in 0.0..1.0)
            require(motionDynamicsScore in 0.0..1.0)
            require(motionLowFrequencyScore in 0.0..1.0)
            require(motionHighFrequencyScore in 0.0..1.0)
            require(physicsConstraintScore in 0.0..1.0)
            require(directionalUncertaintyMajor95Meters >= 0.0)
            require(directionalUncertaintyDepthMeters >= 0.0)
            require(directionalUncertaintyAnisotropy >= 1.0)
            require(directionalUncertaintyConfidence in 0.0..1.0)
            require(directionalUncertaintyDegradedJointCount >= 0)
            require(faceEyeOutOfDistributionRisk in 0.0..1.0)
            require(faceEyeCalibrationDriftScore in 0.0..1.0)
            require(faceEyeUncertaintyCalibrationConfidence in 0.0..1.0)
            require(faceUncertaintyCoverage in 0.0..1.0)
            require(eyeUncertaintyCoverage in 0.0..1.0)
            require(faceUncertaintyCalibrationMultiplier > 0.0)
            require(eyeUncertaintyCalibrationMultiplier > 0.0)
            require(faceEyeShapeConfidence in 0.0..1.0)
            require(faceEyeEyeDynamicConsistency in 0.0..1.0)
            require(faceEyeFusionStrictness in 0.0..1.0)
            require(faceEyeReliabilityTrend in -1.0..1.0)
            require(faceEyeCaptureDomainConfidence in 0.0..1.0)
            require(handUncertaintyConfidence in 0.0..1.0)
            require(handUncertaintyOodRisk in 0.0..1.0)
            require(global4DRootVelocityAgreement in 0.0..1.0)
            require(worldSpaceConsistencyScore in 0.0..1.0)
            require(worldSpaceRobustOutlierRatio in 0.0..1.0)
            require(
                faceEyeSaccadeVelocityDegreesPerSecond.isFinite() &&
                    faceEyeSaccadeVelocityDegreesPerSecond >= 0.0,
            )
            require(faceEyeDominantFailure.isNotBlank())
            require(faceEyeFusionAccepted || faceEyeConfidence >= 0.0)
            require(surfaceNormalObservability in 0.0..1.0)
            require(surfaceHighUncertaintyRatio in 0.0..1.0)
            require(denseLandmarkCoverage in 0.0..1.0)
            require(denseLandmarkVisibleCoverage in 0.0..1.0)
            require(denseLandmarkUncertaintyP95 in 0.0..1.0)
            require(denseLandmarkCrossViewAgreement in 0.0..1.0)
            require(denseLandmarkTemporalStability in 0.0..1.0)
            require(denseLandmarkContactCoverage in 0.0..1.0)
            require(denseLandmarkPredictionBridgeRatio in 0.0..1.0)
            require(metricPrecision95Meters >= 0.0)
            require(metricPrecisionConfidence in 0.0..1.0)
            require(cameraInformationGain in 0.0..1.0)
            require(surfaceEvidenceStrictness in 0.0..1.0)
            require(global4DDirectionalUncertaintyConsistency in 0.0..1.0)
            require(global4DFaceEyeConsistency in 0.0..1.0)
            require(globalRigTrust in 0.0..1.0)
            require(adaptiveDetailScore in 0.0..1.0)
            require(thermalDriftConfidence in 0.0..1.0)
            require(concurrentRigScore in 0.0..1.0)
            require(faceEyeMetricEvidence in 0.0..1.0)
            require(hardwareReadoutTimingConfidence in 0.0..1.0)
            require(multiPersonPredictionRatio in 0.0..1.0)
            require(motionBlurRisk in 0.0..1.0)
            require(bodyMetricEvidenceAuthority in 0.0..1.0)
            require(faceMetricEvidenceAuthority in 0.0..1.0)
            require(handMetricEvidenceAuthority in 0.0..1.0)
            require(bodyEpipolarConsistency in 0.0..1.0)
            require(faceEpipolarConsistency in 0.0..1.0)
            require(handEpipolarConsistency in 0.0..1.0)
        }
    }

    data class ComputeBudget(
        val detectorIntervalFrames: Int,
        val targetInferenceFps: Int,
        val targetCaptureFps: Int,
        val denseMeshHz: Int,
        val faceMeshHz: Int,
        val eyeTrackingHz: Int,
        val handTrackingHz: Int,
        val bundleAdjustmentIterations: Int,
        val tsdfStride: Int,
        val meshVertexBudget: Int,
        val allowDenseSurface: Boolean,
        val allowReferenceOptimization: Boolean,
    )

    data class Decision(
        val timestampNanos: Long,
        val policy: Policy,
        val bottleneck: Bottleneck,
        val score: Double,
        val rawScore: Double,
        val hysteresisScore: Double,
        val hardFailure: Boolean,
        val outputAllowed: Boolean,
        val warnings: List<String>,
        val componentScores: Map<String, Double>,
        val budget: ComputeBudget,
    )

    private var previousScore = 0.0
    private var previousPolicy = Policy.HOLD
    private var previousTimestampNanos = Long.MIN_VALUE
    private var stableSamples = 0
    private var lastDecision: Decision? = null

    @Synchronized
    fun reset() {
        previousScore = 0.0
        previousPolicy = Policy.HOLD
        previousTimestampNanos = Long.MIN_VALUE
        stableSamples = 0
        lastDecision = null
    }

    @Synchronized
    fun evaluate(input: Input): Decision {
        lastDecision?.let { previous ->
            if (input.timestampNanos <= previous.timestampNanos) {
                return previous.copy(
                    warnings =
                        (
                            previous.warnings +
                                "stale-quality-frame-rejected"
                            ).distinct(),
                )
            }
        }

        val p = input.profile.validated()
        val components = componentScores(input)
        val rawScore = weightedGeometricMean(components)

        val dtSeconds =
            if (previousTimestampNanos == Long.MIN_VALUE) {
                1.0 / 30.0
            } else {
                ((input.timestampNanos - previousTimestampNanos).coerceAtLeast(1L) / 1e9)
                    .coerceIn(1e-4, 2.0)
            }

        val attackScale =
            (1.0 - exp(-dtSeconds * 12.0)).coerceIn(0.02, 1.0)
        val responsiveness =
            when (previousPolicy) {
                Policy.REJECT -> 0.86
                Policy.DEGRADE -> 0.72
                Policy.HOLD -> 0.62
                Policy.ACCEPT -> 0.48
            }

        val smoothingGain =
            (attackScale * responsiveness).coerceIn(0.06, 0.92)

        val hysteresisScore =
            if (previousTimestampNanos == Long.MIN_VALUE) {
                rawScore
            } else {
                previousScore + (rawScore - previousScore) * smoothingGain
            }

        val hardFailure =
            input.hardCaptureFailure ||
                input.reconstructionConfidence <= 0.0 ||
                input.synchronizationConfidence < 0.18 ||
                input.synchronizationResidualP95Milliseconds > 12.0 ||
                input.synchronizationOutlierRatio > 0.20 ||
                input.calibrationConfidence < 0.18 ||
                input.identitySwapRisk > 0.97 ||
                (
                    input.observedSubjectCount >= 4 &&
                        input.identityContinuity < 0.18
                    ) ||
                input.motionOutOfDistributionRisk > 0.995 ||
                input.faceEyeHardFailure ||
                !input.physicsConstraintAccepted ||
                input.physicsConstraintScore < 0.16 ||
                input.faceEyeCaptureDomainConfidence < 0.16 ||
                input.global4DRootVelocityAgreement < 0.14 ||
                input.handUncertaintyConfidence < 0.12 ||
                input.handUncertaintyOodRisk > 0.96 ||
                input.worldSpaceConsistencyScore < 0.14 ||
                input.worldSpaceRobustOutlierRatio > 0.82 ||
                input.directionalUncertaintyConfidence < 0.10 ||
                input.directionalUncertaintyAnisotropy >= 18.0 ||
                input.faceEyeOutOfDistributionRisk >= 0.985 ||
                (
                    input.faceEyeCalibrationDriftPersistent &&
                        input.faceEyeCalibrationDriftScore >= 0.62
                    ) ||
                !input.global4DAccepted ||
                input.global4DScore < 0.16 ||
                input.global4DPersistentFailure ||
                (
                    input.faceEyeDominantFailure != "NONE" &&
                        !input.faceEyeFusionAccepted &&
                        input.faceEyeConfidence < 0.18
                    ) ||
                input.hardwareReadoutTimingAvailable &&
                    input.hardwareReadoutTimingHardFailure ||
                input.hardwareReadoutTimingAvailable &&
                    input.hardwareReadoutTimingConfidence < 0.16 ||
                input.motionBlurRisk >= 0.92 ||
                input.bodyMetricEvidenceAuthority < 0.16 ||
                input.faceMetricEvidenceAuthority < 0.16 ||
                input.handMetricEvidenceAuthority < 0.16 ||
                input.bodyEpipolarConsistency < 0.18 ||
                input.faceEpipolarConsistency < 0.18 ||
                input.handEpipolarConsistency < 0.18 ||
                input.faceEyeReliabilityTrend <= -0.72 ||
                (
                    !input.faceEyeFusionAccepted &&
                        input.faceEyeConfidence <
                            (
                                0.55 +
                                    input.faceEyeFusionStrictness
                                        .coerceIn(0.0, 1.0) * 0.10
                                )
                    ) ||
                input.globalRigTrust < 0.16 ||
                input.surfaceNormalObservability <
                    (0.12 + input.surfaceEvidenceStrictness * 0.18) ||
                input.surfaceHighUncertaintyRatio >
                    (0.96 - input.surfaceEvidenceStrictness * 0.30) ||
                input.cameraInformationGain <
                    (0.16 + input.surfaceEvidenceStrictness * 0.34) ||
                input.denseLandmarkHardFailure ||
                input.metricPrecisionHardFailure ||
                input.denseLandmarkCoverage < 0.16 ||
                input.denseLandmarkUncertaintyP95 > 0.94 ||
                (
                    input.denseLandmarkCrossViewEvidenceAvailable &&
                        input.denseLandmarkCrossViewAgreement < 0.14
                    ) ||
                input.adaptiveDetailHold ||
                input.thermalDriftHardFailure ||
                (
                    input.observedSubjectCount >= 4 &&
                        input.multiPersonPredictionRatio > 0.92
                    ) ||
                (
                    p.multiViewRequired &&
                        (
                            !input.concurrentRigAdmissible ||
                                input.concurrentRigScore < 0.46
                        )
                    )
        val policy = classifyPolicy(
            score = hysteresisScore,
            profile = p,
            hardFailure = hardFailure,
            synchronization = input.synchronizationConfidence,
            calibration = input.calibrationConfidence,
            thermal = input.thermalScale,
            runtime = input.runtimeQualityScale,
        )

        val bottleneck = dominantBottleneck(
            components = components,
            input = input,
        )

        val identityWarnings =
            buildList {
                if (
                    input.observedSubjectCount >= 4 &&
                    input.identityContinuity < 0.55
                ) {
                    add(
                        "identity continuity degraded in multi-person scene",
                    )
                }
                if (
                    input.observedSubjectCount >= 4 &&
                    input.identityContinuity < 0.35
                ) {
                    add(
                        "identity continuity is limiting production confidence",
                    )
                }
            }

        stableSamples =
            if (policy == previousPolicy) {
                (stableSamples + 1).coerceAtMost(120)
            } else {
                0
            }

        val budget = computeBudget(
            profile = p,
            policy = policy,
            score = hysteresisScore,
            thermalScale = input.thermalScale,
            memoryScale = input.memoryScale,
            runtimeQualityScale = input.runtimeQualityScale,
            runtimeInferenceScale = input.runtimeInferenceScale,
            gpuDispatchMillis = input.gpuDispatchMillis,
            cameraCount = input.connectedCameraCount,
            dynamicStress = input.motionDynamicStress,
            motionDynamicsScore = input.motionDynamicsScore,
            motionLowFrequencyScore = input.motionLowFrequencyScore,
            motionHighFrequencyScore = input.motionHighFrequencyScore,
            uncertaintyOodRisk = input.motionOutOfDistributionRisk,
            physicsConsistencyScore = input.physicsConstraintScore,
            rootVelocityAgreement = input.global4DRootVelocityAgreement,
            directionalUncertaintyConfidence =
                input.directionalUncertaintyConfidence,
            directionalUncertaintyAnisotropy =
                input.directionalUncertaintyAnisotropy,
            directionalUncertaintyDepthMeters =
                input.directionalUncertaintyDepthMeters,
            globalRigTrust = input.globalRigTrust,
            faceEyeFusionConfidence = input.faceEyeConfidence,
            faceEyeFusionAccepted = input.faceEyeFusionAccepted,
            faceEyeOutOfDistributionRisk =
                input.faceEyeOutOfDistributionRisk,
            faceEyeCalibrationDriftScore =
                input.faceEyeCalibrationDriftScore,
            faceEyeCalibrationDriftPersistent =
                input.faceEyeCalibrationDriftPersistent,
            faceEyeFusionStrictness =
                input.faceEyeFusionStrictness,
            faceEyeSaccadeVelocityDegreesPerSecond =
                input.faceEyeSaccadeVelocityDegreesPerSecond,
            worldSpaceConsistencyScore = input.worldSpaceConsistencyScore,
            worldSpaceRobustOutlierRatio = input.worldSpaceRobustOutlierRatio,
            global4DConsistencyScore =
                input.global4DScore.coerceIn(0.0, 1.0),
            global4DAccepted =
                input.global4DAccepted,
            global4DPersistentFailure =
                input.global4DPersistentFailure,
        )

        previousTimestampNanos = input.timestampNanos
        previousScore = hysteresisScore
        previousPolicy = policy

        val warnings =
            (
                input.warnings +
                    policyWarnings(policy, bottleneck, input, p) +
                    buildEvidenceWarnings(input) +
                    identityWarnings
                )
                .distinct()
                .take(20)

        val decision = Decision(
            timestampNanos = input.timestampNanos,
            policy = policy,
            bottleneck = bottleneck,
            score = hysteresisScore,
            rawScore = rawScore,
            hysteresisScore = hysteresisScore,
            hardFailure = hardFailure,
            outputAllowed = !hardFailure && policy != Policy.REJECT,
            warnings = warnings,
            componentScores = components,
            budget = budget,
        )
        lastDecision = decision
        return decision
    }

    private fun scorePredictionReliability(
        ratio: Double,
        observedSubjects: Int,
    ): Double {
        if (observedSubjects < 4) return 1.0
        return when {
            ratio <= 0.10 -> 1.0
            ratio >= 0.92 -> 0.0
            ratio <= 0.50 ->
                1.0 -
                    (ratio - 0.10) /
                        (0.50 - 0.10) * 0.24
            else ->
                0.76 -
                    (ratio - 0.50) /
                        (0.92 - 0.50) * 0.76
        }.coerceIn(0.0, 1.0)
    }

    private fun buildEvidenceWarnings(input: Input): List<String> =
        buildList {
            if (input.adaptiveDetailHold) {
                add("adaptive-reconstruction-detail-hold")
            }
            if (input.adaptiveDetailScore < 0.55) {
                add("adaptive-reconstruction-detail-evidence-low")
            }
            if (input.thermalDriftConfidence < 0.60) {
                add("thermal-extrinsic-drift-confidence-low")
            }
            if (input.thermalDriftHardFailure) {
                add("thermal-extrinsic-drift-hard-failure")
            }
            if (input.concurrentRigScore < 0.62) {
                add("concurrent-rig-admission-score-reduced")
            }
            if (!input.concurrentRigAdmissible) {
                add("concurrent-rig-admission-rejected")
            }
            if (
                input.observedSubjectCount >= 4 &&
                input.multiPersonPredictionRatio > 0.50
            ) {
                add("multi-person-3d-prediction-pressure-elevated")
            }
            if (
                input.observedSubjectCount >= 4 &&
                input.multiPersonPredictionRatio > 0.92
            ) {
                add("multi-person-3d-prediction-pressure-hard-limit")
            }
            if (input.faceEyeMetricEvidence < 0.60) {
                add("face-eye-metric-evidence-reduced")
            }
            if (
                minOf(
                    input.bodyMetricEvidenceAuthority,
                    input.faceMetricEvidenceAuthority,
                    input.handMetricEvidenceAuthority,
                ) < 0.55
            ) {
                add("cross-modal-metric-authority-reduced")
            }
            if (
                minOf(
                    input.bodyEpipolarConsistency,
                    input.faceEpipolarConsistency,
                    input.handEpipolarConsistency,
                ) < 0.55
            ) {
                add("cross-modal-epipolar-authority-reduced")
            }
            if (
                input.hardwareReadoutTimingAvailable &&
                input.hardwareReadoutTimingConfidence < 0.60
            ) {
                add("hardware-readout-timing-evidence-reduced")
            }
            if (
                input.hardwareReadoutTimingAvailable &&
                input.hardwareReadoutTimingHardFailure
            ) {
                add("hardware-readout-timing-hard-failure")
            }
            if (input.motionBlurRisk > 0.48) {
                add("optical-motion-blur-risk-elevated")
            }
            if (input.motionBlurRisk > 0.72) {
                add("optical-motion-blur-risk-high")
            }
            if (input.motionBlurRisk >= 0.92) {
                add("optical-motion-blur-hard-limit")
            }
        }.distinct()

    private fun componentScores(input: Input): LinkedHashMap<String, Double> {
        val p = input.profile
        return linkedMapOf(
            "reconstruction" to scoreConfidence(input.reconstructionConfidence),
            "geometry" to scoreGeometry(
                reprojectionErrorPx = input.reprojectionErrorPx,
                observability = input.geometricObservability,
            ),
            "uncertainty" to scoreInverseUncertainty(
                meanUncertaintyMeters = input.meanUncertaintyMeters,
                limitMeters = p.maxMeanUncertaintyMeters,
            ),
            "temporal" to scoreConfidence(input.temporalConfidence),
            "synchronization" to scoreSynchronizationEvidence(
                confidence = input.synchronizationConfidence,
                residualP95Milliseconds =
                    input.synchronizationResidualP95Milliseconds,
                outlierRatio = input.synchronizationOutlierRatio,
            ),
            "calibration" to scoreConfidence(input.calibrationConfidence),
            "tracking" to scoreTracking(
                trackingConfidence = input.trackingConfidence,
                identitySwapRisk = input.identitySwapRisk,
                occlusionRisk = input.occlusionRisk,
                cameraMotionRisk = input.cameraMotionRisk,
            ),
            "opticalCapture" to
                (1.0 - input.motionBlurRisk)
                    .coerceIn(0.0, 1.0),
            "uncertaintyCalibration" to scoreUncertaintyCalibration(
                confidence = input.calibratedUncertaintyConfidence,
                coverage = input.motionUncertaintyCoverage,
                oodRisk = input.motionOutOfDistributionRisk,
                targetCoverage = 0.95,
            ),
            "physicsConsistency" to scorePhysicsConsistency(
                score = input.physicsConstraintScore,
                accepted = input.physicsConstraintAccepted,
            ),
            "directionalUncertainty" to scoreDirectionalUncertainty(
                major95Meters = input.directionalUncertaintyMajor95Meters,
                depthMeters = input.directionalUncertaintyDepthMeters,
                anisotropy = input.directionalUncertaintyAnisotropy,
                confidence = input.directionalUncertaintyConfidence,
                degradedJointCount =
                    input.directionalUncertaintyDegradedJointCount,
            ),
            "identityContinuity" to
                input.identityContinuity.coerceIn(0.0, 1.0),
            "globalRigTrust" to
                input.globalRigTrust.coerceIn(0.0, 1.0),
            "adaptiveDetail" to
                input.adaptiveDetailScore.coerceIn(0.0, 1.0),
            "thermalDrift" to
                input.thermalDriftConfidence.coerceIn(0.0, 1.0),
            "concurrentRig" to
                input.concurrentRigScore.coerceIn(0.0, 1.0),
            "faceEyeMetric" to
                input.faceEyeMetricEvidence.coerceIn(0.0, 1.0),
            "crossModalMetricAuthority" to
                minOf(
                    input.bodyMetricEvidenceAuthority,
                    input.faceMetricEvidenceAuthority,
                    input.handMetricEvidenceAuthority,
                ).coerceIn(0.0, 1.0),
            "crossModalEpipolarAuthority" to
                minOf(
                    input.bodyEpipolarConsistency,
                    input.faceEpipolarConsistency,
                    input.handEpipolarConsistency,
                ).coerceIn(0.0, 1.0),
            "hardwareReadoutTiming" to
                if (input.hardwareReadoutTimingAvailable) {
                    input.hardwareReadoutTimingConfidence
                        .coerceIn(0.0, 1.0)
                } else {
                    1.0
                },
            "multiPersonPrediction" to
                scorePredictionReliability(
                    ratio = input.multiPersonPredictionRatio,
                    observedSubjects = input.observedSubjectCount,
                ),
            "faceEye" to scoreFaceEyeEvidence(
                confidence = input.faceEyeConfidence,
                accepted = input.faceEyeFusionAccepted,
            ),
            "faceEyeTrend" to scoreFaceEyeReliabilityTrend(
                trend = input.faceEyeReliabilityTrend,
            ),
            "faceEyeShape" to
                input.faceEyeShapeConfidence.coerceIn(0.0, 1.0),
            "faceEyeEyeDynamics" to
                input.faceEyeEyeDynamicConsistency.coerceIn(0.0, 1.0),
            "faceEyeOod" to scoreFaceEyeOod(
                risk = input.faceEyeOutOfDistributionRisk,
            ),
            "faceEyeCaptureDomain" to
                input.faceEyeCaptureDomainConfidence
                    .coerceIn(0.0, 1.0),
            "handUncertaintyCalibration" to
                (
                    input.handUncertaintyConfidence *
                        (
                            1.0 -
                                input.handUncertaintyOodRisk * 0.72
                            ).coerceIn(0.0, 1.0)
                    ).coerceIn(0.0, 1.0),
            "worldSpaceConsistency" to
                (
                    input.worldSpaceConsistencyScore *
                        (
                            1.0 -
                                input.worldSpaceRobustOutlierRatio * 0.62
                            ).coerceIn(0.0, 1.0)
                    ).coerceIn(0.0, 1.0),
            "global4DRootVelocity" to
                input.global4DRootVelocityAgreement
                    .coerceIn(0.0, 1.0),
            "faceEyeUncertaintyCalibration" to
                scoreFaceEyeUncertaintyCalibration(
                    confidence =
                        input.faceEyeUncertaintyCalibrationConfidence,
                    ready =
                        input.faceEyeUncertaintyCalibrationReady,
                    faceCoverage =
                        input.faceUncertaintyCoverage,
                    eyeCoverage =
                        input.eyeUncertaintyCoverage,
                    faceMultiplier =
                        input.faceUncertaintyCalibrationMultiplier,
                    eyeMultiplier =
                        input.eyeUncertaintyCalibrationMultiplier,
                ),
            "global4DConsistency" to
                scoreGlobal4DConsistency(
                    score = input.global4DScore,
                    accepted = input.global4DAccepted,
                    persistentFailure =
                        input.global4DPersistentFailure,
                    rootMotionConsistency =
                        input.global4DRootMotionConsistency,
                    sceneConsistency =
                        input.global4DSceneConsistency,
                    directionalUncertaintyConsistency =
                        input.global4DDirectionalUncertaintyConsistency,
                    faceEyeConsistency =
                        input.global4DFaceEyeConsistency,
                ),
            "hand" to scoreConfidence(input.handConfidence),
            "surface" to scoreSurface(
                surfaceConfidence = input.surfaceConfidence,
                surfaceNormalObservability =
                    input.surfaceNormalObservability,
                surfaceHighUncertaintyRatio =
                    input.surfaceHighUncertaintyRatio,
                cameraInformationGain =
                    input.cameraInformationGain,
                profile = p,
            ),
            "metricPrecision" to scoreMetricPrecision(
                uncertainty95Meters = input.metricPrecision95Meters,
                confidence = input.metricPrecisionConfidence,
                referenceGradeEligible =
                    input.metricPrecisionReferenceGradeEligible,
            ),
            "denseLandmarks" to scoreDenseLandmarks(
                coverage = input.denseLandmarkCoverage,
                visibleCoverage = input.denseLandmarkVisibleCoverage,
                uncertaintyP95 = input.denseLandmarkUncertaintyP95,
                crossViewAgreement =
                    input.denseLandmarkCrossViewAgreement,
                crossViewEvidenceAvailable =
                    input.denseLandmarkCrossViewEvidenceAvailable,
                temporalStability =
                    input.denseLandmarkTemporalStability,
                contactCoverage =
                    input.denseLandmarkContactCoverage,
                predictionBridgeRatio =
                    input.denseLandmarkPredictionBridgeRatio,
                hardFailure = input.denseLandmarkHardFailure,
            ),
            "surfaceNormalObservability" to
                input.surfaceNormalObservability.coerceIn(0.0, 1.0),
            "surfaceUncertainty" to
                (
                    1.0 -
                        input.surfaceHighUncertaintyRatio
                ).coerceIn(0.0, 1.0),
            "viewInformation" to
                input.cameraInformationGain.coerceIn(0.0, 1.0),
            "runtime" to scorePerformance(
                input.runtimeQualityScale,
                input.runtimeInferenceScale,
                input.gpuDispatchMillis,
            ),
            "thermal" to scoreConfidence(input.thermalScale),
            "memory" to scoreConfidence(input.memoryScale),
            "coverage" to scoreCoverage(
                connectedCameraCount = input.connectedCameraCount,
                observedSubjectCount = input.observedSubjectCount,
                profile = p,
            ),
        )
    }

    private fun scoreMetricPrecision(
        uncertainty95Meters: Double,
        confidence: Double,
        referenceGradeEligible: Boolean,
    ): Double {
        val uncertaintyScore =
            exp(
                -uncertainty95Meters.coerceAtLeast(0.0) / 0.025,
            ).coerceIn(0.0, 1.0)
        val empiricalModifier =
            if (referenceGradeEligible) 1.0 else 0.78
        return (
            uncertaintyScore * 0.62 +
                confidence.coerceIn(0.0, 1.0) * 0.38
            ) * empiricalModifier
    }

    private fun scoreGlobal4DConsistency(
        score: Double,
        accepted: Boolean,
        persistentFailure: Boolean,
        rootMotionConsistency: Double,
        sceneConsistency: Double,
        directionalUncertaintyConsistency: Double,
        faceEyeConsistency: Double,
    ): Double {
        val bounded =
            score.coerceIn(0.0, 1.0)
        val root =
            rootMotionConsistency.coerceIn(0.0, 1.0)
        val scene =
            sceneConsistency.coerceIn(0.0, 1.0)
        val directional =
            directionalUncertaintyConsistency
                .coerceIn(0.0, 1.0)
        val faceEye =
            faceEyeConsistency
                .coerceIn(0.0, 1.0)
        val base =
            bounded * 0.46 +
                root * 0.20 +
                scene * 0.10 +
                directional * 0.14 +
                faceEye * 0.10
        return (
            if (accepted) base else base * 0.54
        ).let {
            if (persistentFailure) {
                it * 0.42
            } else {
                it
            }
        }.coerceIn(0.0, 1.0)
    }

    private fun scoreDirectionalUncertainty(
        major95Meters: Double,
        depthMeters: Double,
        anisotropy: Double,
        confidence: Double,
        degradedJointCount: Int,
    ): Double {
        val majorScore =
            exp(-major95Meters.coerceAtLeast(0.0) / 0.42)
                .coerceIn(0.0, 1.0)
        val depthScore =
            exp(-depthMeters.coerceAtLeast(0.0) / 0.34)
                .coerceIn(0.0, 1.0)
        val anisotropyScore =
            (
                1.0 -
                    (anisotropy.coerceAtLeast(1.0) - 1.0) / 12.0
                ).coerceIn(0.0, 1.0)
        val degradedJointScale =
            exp(
                -degradedJointCount.coerceAtLeast(0) / 12.0,
            ).coerceIn(0.20, 1.0)
        return (
            majorScore * 0.28 +
                depthScore * 0.24 +
                anisotropyScore * 0.16 +
                confidence.coerceIn(0.0, 1.0) * 0.24 +
                degradedJointScale * 0.08
            ).coerceIn(0.0, 1.0)
    }

    private fun scorePhysicsConsistency(
        score: Double,
        accepted: Boolean,
    ): Double {
        val bounded = score.coerceIn(0.0, 1.0)
        return if (accepted) {
            bounded
        } else {
            (bounded * 0.58).coerceIn(0.0, 1.0)
        }
    }

    private fun scoreFaceEyeUncertaintyCalibration(
        confidence: Double,
        ready: Boolean,
        faceCoverage: Double,
        eyeCoverage: Double,
        faceMultiplier: Double,
        eyeMultiplier: Double,
    ): Double {
        val coverageScore =
            (
                (
                    1.0 -
                        kotlin.math.abs(faceCoverage - 0.95)
                    ).coerceIn(0.0, 1.0) * 0.45 +
                    (
                        1.0 -
                            kotlin.math.abs(eyeCoverage - 0.95)
                        ).coerceIn(0.0, 1.0) * 0.45 +
                    confidence.coerceIn(0.0, 1.0) * 0.10
            ).coerceIn(0.0, 1.0)

        val wideningPenalty =
            (
                1.0 -
                    (
                        (
                            faceMultiplier.coerceAtLeast(1.0) - 1.0
                        ) / 3.0 +
                            (
                                eyeMultiplier.coerceAtLeast(1.0) - 1.0
                            ) / 3.0
                        ) * 0.50
                ).coerceIn(0.55, 1.0)

        return (
            coverageScore *
                wideningPenalty *
                if (ready) 1.0 else 0.82
            ).coerceIn(0.0, 1.0)
    }

    private fun scoreFaceEyeOod(
        risk: Double,
    ): Double =
        (
            1.0 -
                risk.coerceIn(0.0, 1.0) * 0.92
            ).coerceIn(0.0, 1.0)

    private fun scoreFaceEyeReliabilityTrend(
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

    private fun scoreFaceEyeEvidence(
        confidence: Double,
        accepted: Boolean,
    ): Double {
        val bounded =
            confidence.coerceIn(0.0, 1.0)
        return if (accepted) {
            bounded
        } else {
            (
                bounded * 0.52
            ).coerceIn(0.0, 1.0)
        }
    }

    private fun scoreConfidence(value: Double): Double =
        when {
            !value.isFinite() -> 0.0
            else -> value.coerceIn(0.0, 1.0)
        }

    private fun scoreGeometry(
        reprojectionErrorPx: Double,
        observability: Double,
    ): Double {
        val errorScore =
            if (!reprojectionErrorPx.isFinite()) {
                0.05
            } else {
                exp(-0.34 * reprojectionErrorPx.coerceAtLeast(0.0))
                    .coerceIn(0.0, 1.0)
            }

        return (
            errorScore * 0.66 +
                observability.coerceIn(0.0, 1.0) * 0.34
            ).coerceIn(0.0, 1.0)
    }

    private fun scoreInverseUncertainty(
        meanUncertaintyMeters: Double,
        limitMeters: Double,
    ): Double {
        if (!meanUncertaintyMeters.isFinite()) return 0.0
        if (meanUncertaintyMeters <= 0.0) return 1.0

        return exp(
            -1.10 *
                (meanUncertaintyMeters / limitMeters.coerceAtLeast(1e-4)),
        ).coerceIn(0.0, 1.0)
    }

    private fun scoreTracking(
        trackingConfidence: Double,
        identitySwapRisk: Double,
        occlusionRisk: Double,
        cameraMotionRisk: Double,
    ): Double {
        val identity = 1.0 - identitySwapRisk.coerceIn(0.0, 1.0)
        val occlusion = 1.0 - occlusionRisk.coerceIn(0.0, 1.0)
        val camera = 1.0 - cameraMotionRisk.coerceIn(0.0, 1.0)

        return (
            trackingConfidence.coerceIn(0.0, 1.0) * 0.54 +
                identity * 0.22 +
                occlusion * 0.14 +
                camera * 0.10
            ).coerceIn(0.0, 1.0)
    }

    private fun scoreUncertaintyCalibration(
        confidence: Double,
        coverage: Double,
        oodRisk: Double,
        targetCoverage: Double,
    ): Double {
        val coverageDeficit =
            (
                targetCoverage -
                    coverage.coerceIn(0.0, 1.0)
                ).coerceAtLeast(0.0)

        val coverageScore =
            (
                1.0 -
                    coverageDeficit /
                        targetCoverage.coerceAtLeast(1e-4)
                ).coerceIn(0.0, 1.0)

        return (
            confidence.coerceIn(0.0, 1.0) * 0.44 +
                coverageScore * 0.34 +
                (1.0 - oodRisk.coerceIn(0.0, 1.0)) * 0.22
            ).coerceIn(0.0, 1.0)
    }

    private fun scoreSynchronizationEvidence(
        confidence: Double,
        residualP95Milliseconds: Double,
        outlierRatio: Double,
    ): Double {
        val confidenceScore =
            confidence.coerceIn(0.0, 1.0)
        val residualScore =
            (1.0 - residualP95Milliseconds / 12.0)
                .coerceIn(0.0, 1.0)
        val outlierScore =
            (1.0 - outlierRatio / 0.20)
                .coerceIn(0.0, 1.0)
        return (
            confidenceScore * 0.58 +
                residualScore * 0.27 +
                outlierScore * 0.15
            ).coerceIn(0.0, 1.0)
    }

    private fun scoreDenseLandmarks(
        coverage: Double,
        visibleCoverage: Double,
        uncertaintyP95: Double,
        crossViewAgreement: Double,
        crossViewEvidenceAvailable: Boolean,
        temporalStability: Double,
        contactCoverage: Double,
        predictionBridgeRatio: Double,
        hardFailure: Boolean,
    ): Double {
        if (hardFailure) return 0.0

        val coverageScore =
            coverage.coerceIn(0.0, 1.0)
        val visibleScore =
            visibleCoverage.coerceIn(0.0, 1.0)
        val uncertaintyScore =
            (1.0 - uncertaintyP95.coerceIn(0.0, 1.0))
                .coerceIn(0.0, 1.0)
        val temporalScore =
            temporalStability.coerceIn(0.0, 1.0)
        val contactScore =
            contactCoverage.coerceIn(0.0, 1.0)
        val predictionScore =
            predictionBridgeRatio.coerceIn(0.0, 1.0)

        val crossViewScore =
            if (crossViewEvidenceAvailable) {
                crossViewAgreement.coerceIn(0.0, 1.0)
            } else {
                0.72
            }

        return (
            coverageScore * 0.22 +
                visibleScore * 0.16 +
                uncertaintyScore * 0.24 +
                crossViewScore * 0.18 +
                temporalScore * 0.14 +
                contactScore * 0.02 +
                predictionScore * 0.04
            ).coerceIn(0.0, 1.0)
    }

    private fun scoreSurface(
        surfaceConfidence: Double,
        surfaceNormalObservability: Double,
        surfaceHighUncertaintyRatio: Double,
        cameraInformationGain: Double,
        profile: ProductionCaptureProfile,
    ): Double {
        if (profile.surfacePolicy == com.nivukx.nox.core.model.SurfacePolicy.SKELETON_ONLY) {
            return 1.0
        }

        val normal =
            surfaceNormalObservability.coerceIn(0.0, 1.0)
        val uncertainty =
            (
                1.0 -
                    surfaceHighUncertaintyRatio.coerceIn(0.0, 1.0)
                )
                .coerceIn(0.0, 1.0)
        val viewInformation =
            cameraInformationGain.coerceIn(0.0, 1.0)

        return (
            (
                surfaceConfidence.coerceIn(0.0, 1.0) * 0.52 +
                    normal * 0.20 +
                    uncertainty * 0.18 +
                    viewInformation * 0.10
                ) *
                (
                    0.78 +
                        profile.surfaceObservationWeight * 0.22
                    )
            ).coerceIn(0.0, 1.0)
    }

    private fun scorePerformance(
        runtimeQualityScale: Double,
        runtimeInferenceScale: Double,
        gpuDispatchMillis: Double,
    ): Double {
        val runtime =
            runtimeQualityScale.coerceIn(0.0, 1.0) * 0.58 +
                runtimeInferenceScale.coerceIn(0.0, 1.0) * 0.28

        val gpu =
            if (!gpuDispatchMillis.isFinite() || gpuDispatchMillis <= 0.0) {
                1.0
            } else {
                exp(-gpuDispatchMillis / 35.0).coerceIn(0.10, 1.0)
            }

        return (runtime * 0.82 + gpu * 0.18).coerceIn(0.0, 1.0)
    }

    private fun scoreCoverage(
        connectedCameraCount: Int,
        observedSubjectCount: Int,
        profile: ProductionCaptureProfile,
    ): Double {
        val cameras =
            if (profile.multiViewRequired) {
                when {
                    connectedCameraCount >= profile.multiViewConsensusThreshold -> 1.0
                    connectedCameraCount <= 0 -> 0.0
                    else ->
                        connectedCameraCount.toDouble() /
                            profile.multiViewConsensusThreshold.toDouble()
                }
            } else {
                1.0
            }

        val subjects =
            when {
                observedSubjectCount <= 0 -> 0.35
                observedSubjectCount <= profile.maxSubjects -> 1.0
                else -> 0.65
            }

        return (cameras * 0.70 + subjects * 0.30)
            .coerceIn(0.0, 1.0)
    }

    private fun weightedGeometricMean(
        components: Map<String, Double>,
    ): Double {
        if (components.isEmpty()) return 0.0

        val weights = mapOf(
            "reconstruction" to 1.16,
            "geometry" to 1.25,
            "uncertainty" to 1.16,
            "temporal" to 1.10,
            "synchronization" to 1.24,
            "calibration" to 1.28,
            "tracking" to 1.10,
            "opticalCapture" to 0.86,
            "uncertaintyCalibration" to 1.18,
            "physicsConsistency" to 1.14,
            "directionalUncertainty" to 1.18,
            "globalRigTrust" to 1.06,
            "metricPrecision" to 1.30,
            "faceEye" to 0.92,
            "faceEyeOod" to 0.88,
            "faceEyeCalibrationDrift" to 0.94,
            "faceEyeCaptureDomain" to 0.78,
            "handUncertaintyCalibration" to 0.68,
            "global4DRootVelocity" to 0.78,
            "worldSpaceConsistency" to 0.74,
            "global4DConsistency" to 1.08,
            "hand" to 0.58,
            "surface" to 0.70,
            "surfaceNormalObservability" to 0.44,
            "surfaceUncertainty" to 0.50,
            "viewInformation" to 0.46,
            "runtime" to 0.70,
            "thermal" to 0.60,
            "memory" to 0.52,
            "coverage" to 0.72,
        )

        var weightedLog = 0.0
        var totalWeight = 0.0

        for ((key, value) in components) {
            val weight = weights[key] ?: 1.0
            val floor = 0.015
            weightedLog += weight * ln(value.coerceIn(floor, 1.0))
            totalWeight += weight
        }

        return exp(weightedLog / totalWeight.coerceAtLeast(1e-9))
            .coerceIn(0.0, 1.0)
    }

    private fun classifyPolicy(
        score: Double,
        profile: ProductionCaptureProfile,
        hardFailure: Boolean,
        synchronization: Double,
        calibration: Double,
        thermal: Double,
        runtime: Double,
    ): Policy {
        if (hardFailure) return Policy.REJECT

        val referencePenalty =
            if (profile.qualityTier == com.nivukx.nox.core.model.CaptureQualityTier.REFERENCE) {
                0.04
            } else {
                0.0
            }

        val degradeThreshold =
            (0.52 + referencePenalty).coerceIn(0.45, 0.65)
        val acceptThreshold =
            (0.76 + referencePenalty).coerceIn(0.70, 0.86)

        if (synchronization < 0.40 || calibration < 0.42) {
            return Policy.HOLD
        }

        if (thermal < 0.38 || runtime < 0.36) {
            return Policy.DEGRADE
        }

        return when {
            score >= acceptThreshold -> Policy.ACCEPT
            score >= degradeThreshold -> Policy.HOLD
            score >= 0.30 -> Policy.DEGRADE
            else -> Policy.REJECT
        }
    }

    private fun dominantBottleneck(
        components: Map<String, Double>,
        input: Input,
    ): Bottleneck {
        if (input.synchronizationConfidence < 0.40) {
            return Bottleneck.SYNCHRONIZATION
        }
        if (input.calibrationConfidence < 0.42) {
            return Bottleneck.CALIBRATION
        }
        val weakest = components.minByOrNull { it.value }?.key
            ?: return Bottleneck.NONE

        return when (weakest) {
            "reconstruction" -> Bottleneck.RECONSTRUCTION
            "geometry", "uncertainty", "coverage" -> Bottleneck.GEOMETRY
            "temporal" -> Bottleneck.TEMPORAL
            "synchronization" -> Bottleneck.SYNCHRONIZATION
            "calibration" -> Bottleneck.CALIBRATION
            "tracking" -> Bottleneck.TRACKING
            "faceEye" -> Bottleneck.FACE_EYE
            "hand" -> Bottleneck.HAND
            "surface", "surfaceNormalObservability", "surfaceUncertainty",
            "viewInformation" -> Bottleneck.SURFACE
            "denseLandmarks" -> Bottleneck.DENSE_LANDMARKS
            "runtime" -> Bottleneck.PERFORMANCE
            "thermal" -> Bottleneck.THERMAL
            "memory" -> Bottleneck.MEMORY
            "uncertaintyCalibration" -> Bottleneck.UNCERTAINTY_CALIBRATION
            "physicsConsistency" -> Bottleneck.PHYSICS_CONSISTENCY
            "directionalUncertainty" -> Bottleneck.DIRECTIONAL_UNCERTAINTY
            "faceEyeOod" -> Bottleneck.FACE_EYE_OOD
            "faceEyeCaptureDomain" -> Bottleneck.FACE_EYE_CAPTURE_DOMAIN
            "handUncertaintyCalibration" ->
                Bottleneck.HAND_UNCERTAINTY_CALIBRATION
            "global4DRootVelocity" -> Bottleneck.GLOBAL_4D_ROOT_VELOCITY
            "global4DConsistency" -> Bottleneck.GLOBAL_4D_CONSISTENCY
            else -> Bottleneck.NONE
        }
    }

    private fun policyWarnings(
        policy: Policy,
        bottleneck: Bottleneck,
        input: Input,
        profile: ProductionCaptureProfile,
    ): List<String> {
        val warnings = ArrayList<String>()

        if (policy != Policy.ACCEPT) {
            warnings += "production-policy-" + policy.name.lowercase()
        }
        if (bottleneck != Bottleneck.NONE) {
            warnings += "bottleneck-" + bottleneck.name.lowercase()
        }
        if (input.connectedCameraCount < profile.multiViewConsensusThreshold &&
            profile.multiViewRequired
        ) {
            warnings += "multi-view-coverage-below-profile"
        }
        if (input.identitySwapRisk > 0.70) {
            warnings += "identity-swap-risk"
        }
        if (input.synchronizationResidualP95Milliseconds > 8.0) {
            warnings += "synchronization-residual-p95-elevated"
        }
        if (input.synchronizationOutlierRatio > 0.10) {
            warnings += "synchronization-outlier-ratio-elevated"
        }
        if (input.meanUncertaintyMeters > profile.maxMeanUncertaintyMeters) {
            warnings += "mean-geometry-uncertainty-above-profile"
        }
        if (input.thermalScale < 0.60) {
            warnings += "thermal-budget-reduced"
        }
        if (input.memoryScale < 0.60) {
            warnings += "memory-budget-reduced"
        }
        if (input.motionOutOfDistributionRisk > 0.58) {
            warnings += "motion-out-of-distribution-risk"
        }
        if (input.motionUncertaintyCoverage < 0.85) {
            warnings += "motion-uncertainty-coverage-below-production-target"
        }
        if (input.calibratedUncertaintyConfidence < 0.60) {
            warnings += "motion-uncertainty-calibration-low-confidence"
        }
        if (input.faceEyeCaptureDomainConfidence < 0.55) {
            warnings += "face-eye-capture-domain-degraded"
        }
        if (input.global4DRootVelocityAgreement < 0.55) {
            warnings += "global-4d-root-velocity-agreement-degraded"
        }
        if (
            input.worldSpaceConsistencyScore < 0.55 ||
            input.worldSpaceRobustOutlierRatio > 0.42
        ) {
            warnings += "world-space-consistency-degraded"
        }
        if (
            input.handUncertaintyConfidence < 0.55 ||
            input.handUncertaintyOodRisk > 0.62
        ) {
            warnings += "hand-uncertainty-calibration-degraded"
        }
        if (
            !input.faceEyeUncertaintyCalibrationReady &&
            input.faceEyeConfidence > 0.45
        ) {
            warnings += "face-eye-uncertainty-calibration-warmup"
        }
        if (
            input.faceEyeUncertaintyCalibrationConfidence > 0.0 &&
            (
                input.faceUncertaintyCoverage < 0.84 ||
                    input.eyeUncertaintyCoverage < 0.84
                )
        ) {
            warnings += "face-eye-uncertainty-coverage-limited"
        }

        if (
            input.physicsConstraintScore < 0.55 ||
            !input.physicsConstraintAccepted
        ) {
            warnings += "physics-consistency-evidence-limited"
        }
        if (
            input.directionalUncertaintyConfidence < 0.55 ||
            input.directionalUncertaintyAnisotropy > 6.0 ||
            input.directionalUncertaintyDepthMeters > 0.34
        ) {
            warnings += "directional-uncertainty-evidence-limited"
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
            warnings += "dense-landmark-cross-view-agreement-degraded"
        }
        if (input.denseLandmarkTemporalStability < 0.55) {
            warnings += "dense-landmark-temporal-stability-degraded"
        }
        if (input.metricPrecisionHardFailure) {
            warnings += "metric precision budget entered hard-failure state"
        } else if (
            input.metricPrecision95Meters > 0.025
        ) {
            warnings += "metric precision uncertainty is elevated"
        }
        if (input.denseLandmarkHardFailure) {
            warnings += "dense-landmark-hard-failure"
        }
        if (
            input.faceEyeConfidence < 0.55 ||
            !input.faceEyeFusionAccepted
        ) {
            warnings += "face-eye-consistency-evidence-limited"
        }
        if (input.faceEyeReliabilityTrend <= -0.18) {
            warnings += "face-eye-reliability-declining"
        }
        if (input.faceEyeReliabilityTrend <= -0.45) {
            warnings += "face-eye-reliability-collapse-risk"
        }
        if (input.faceEyeOutOfDistributionRisk >= 0.72) {
            warnings += "face-eye-out-of-distribution-risk-elevated"
        }
        if (input.faceEyeCalibrationDriftScore >= 0.62) {
            warnings += "face-eye-calibration-drift-elevated"
        }
        if (input.faceEyeCalibrationDriftPersistent) {
            warnings += "face-eye-calibration-drift-persistent"
        }
        if (input.faceEyeDominantFailure != "NONE") {
            warnings +=
                "face-eye-dominant-failure-" +
                    input.faceEyeDominantFailure.lowercase()
        }
        if (
            input.global4DScore < 0.62 ||
            !input.global4DAccepted ||
            input.global4DPersistentFailure
        ) {
            warnings += "global-4d-consistency-evidence-limited"
        }
        if (input.faceEyeShapeConfidence < 0.55) {
            warnings += "face-eye-shape-evidence-limited"
        }
        if (input.faceEyeEyeDynamicConsistency < 0.55) {
            warnings += "face-eye-dynamic-consistency-limited"
        }
        if (input.faceEyeHardFailure) {
            warnings += "face-eye-hard-failure-gate"
        }
        if (!input.faceEyeFusionAccepted) {
            warnings += "face-eye-fusion-rejected"
        }
        if (
            !input.faceEyeFusionAccepted &&
            input.faceEyeConfidence >=
                (
                    0.08 +
                        input.faceEyeFusionStrictness
                            .coerceIn(0.0, 1.0) * 0.10
                    )
        ) {
            warnings += "face-eye-recovery-only-body-output-preserved"
        }
        val surfaceStrictness =
            input.surfaceEvidenceStrictness.coerceIn(0.0, 1.0)
        if (
            input.surfaceNormalObservability <
                0.55 + surfaceStrictness * 0.20
        ) {
            warnings += "surface-normal-observability-limited"
        }
        if (
            input.surfaceHighUncertaintyRatio >
                (0.45 - surfaceStrictness * 0.20)
                    .coerceAtLeast(0.20)
        ) {
            warnings += "surface-high-uncertainty-ratio"
        }
        if (
            input.cameraInformationGain <
                (0.42 + surfaceStrictness * 0.20)
                    .coerceAtMost(0.72)
        ) {
            warnings += "surface-view-information-limited"
        }
        if (input.globalRigTrust < 0.55) {
            warnings += "global-rig-trust-degraded"
        }

        return warnings
    }

    private fun computeBudget(
        profile: ProductionCaptureProfile,
        policy: Policy,
        score: Double,
        thermalScale: Double,
        memoryScale: Double,
        runtimeQualityScale: Double,
        runtimeInferenceScale: Double,
        gpuDispatchMillis: Double,
        cameraCount: Int,
        dynamicStress: Double,
        motionDynamicsScore: Double,
        motionLowFrequencyScore: Double,
        motionHighFrequencyScore: Double,
        uncertaintyOodRisk: Double,
        physicsConsistencyScore: Double,
        rootVelocityAgreement: Double,
        directionalUncertaintyConfidence: Double,
        directionalUncertaintyAnisotropy: Double,
        directionalUncertaintyDepthMeters: Double,
        globalRigTrust: Double,
        faceEyeFusionConfidence: Double,
        faceEyeFusionAccepted: Boolean,
        faceEyeOutOfDistributionRisk: Double,
        faceEyeCalibrationDriftScore: Double,
        faceEyeCalibrationDriftPersistent: Boolean,
        faceEyeFusionStrictness: Double,
        faceEyeSaccadeVelocityDegreesPerSecond: Double,
        worldSpaceConsistencyScore: Double,
        worldSpaceRobustOutlierRatio: Double,
        global4DConsistencyScore: Double,
        global4DAccepted: Boolean,
        global4DPersistentFailure: Boolean,
    ): ComputeBudget {
        val qualityScale =
            when (policy) {
                Policy.ACCEPT -> 1.00
                Policy.HOLD -> 0.88
                Policy.DEGRADE -> 0.66
                Policy.REJECT -> 0.45
            }

        val thermal =
            thermalScale.coerceIn(0.25, 1.0)
        val memory =
            memoryScale.coerceIn(0.25, 1.0)
        val runtime =
            runtimeQualityScale.coerceIn(0.30, 1.0)
        val inferenceRuntime =
            runtimeInferenceScale.coerceIn(0.30, 1.0)

        val combined =
            min(
                min(qualityScale, thermal),
                min(memory, runtime),
            )

        val stress =
            dynamicStress.coerceIn(0.0, 1.0)
        val motionScore =
            motionDynamicsScore.coerceIn(0.0, 1.0)
        val lowFrequencyMotion =
            motionLowFrequencyScore.coerceIn(0.0, 1.0)
        val highFrequencyMotion =
            motionHighFrequencyScore.coerceIn(0.0, 1.0)
        val oodRisk =
            uncertaintyOodRisk.coerceIn(0.0, 1.0)
        val physicsScore =
            physicsConsistencyScore.coerceIn(0.0, 1.0)
        val physicsRecoveryPressure =
            (1.0 - physicsScore).coerceIn(0.0, 1.0)
        val rootVelocityRecoveryPressure =
            (1.0 -
                rootVelocityAgreement.coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)
        val worldSpaceRecoveryPressure =
            (
                (
                    1.0 -
                        worldSpaceConsistencyScore.coerceIn(0.0, 1.0)
                    ) * 0.72 +
                    worldSpaceRobustOutlierRatio.coerceIn(0.0, 1.0) * 0.28
                ).coerceIn(0.0, 1.0)
        val directionalConfidence =
            directionalUncertaintyConfidence.coerceIn(0.0, 1.0)
        val directionalDepthPressure =
            (directionalUncertaintyDepthMeters / 0.34)
                .coerceIn(0.0, 1.0)
        val directionalUncertaintyAnisotropyPressure =
            ((directionalUncertaintyAnisotropy - 1.0) / 8.0)
                .coerceIn(0.0, 1.0)
        val directionalRecoveryPressure =
            (
                (1.0 - directionalConfidence) * 0.56 +
                    directionalDepthPressure * 0.26 +
                    directionalUncertaintyAnisotropyPressure * 0.18
                ).coerceIn(0.0, 1.0)

        val rigTrust =
            globalRigTrust.coerceIn(0.0, 1.0)
        val rigRecoveryPressure =
            (1.0 - rigTrust).coerceIn(0.0, 1.0)
        val faceEyeScore =
            faceEyeFusionConfidence.coerceIn(0.0, 1.0)
        val faceEyeOodPressure =
            faceEyeOutOfDistributionRisk.coerceIn(0.0, 1.0)
        val faceEyeCalibrationPressure =
            (
                faceEyeCalibrationDriftScore.coerceIn(0.0, 1.0) * 0.72 +
                    if (faceEyeCalibrationDriftPersistent) 0.34 else 0.0
                ).coerceIn(0.0, 1.0)
        val faceEyeStrictness =
            faceEyeFusionStrictness
                .coerceIn(0.0, 1.0)
        val global4DScore =
            global4DConsistencyScore.coerceIn(0.0, 1.0)
        val global4DRecoveryPressure =
            (
                1.0 - global4DScore
            ).coerceIn(0.0, 1.0) +
                if (global4DAccepted) 0.0 else 0.16 +
                if (global4DPersistentFailure) 0.32 else 0.0

        val faceEyeRecoveryPressure =
            (
                1.0 -
                    faceEyeScore
                ).coerceIn(0.0, 1.0) +
                faceEyeOodPressure * 0.48 +
                faceEyeCalibrationPressure * 0.58 +
                if (faceEyeFusionAccepted) 0.0 else
                    0.18 * (0.55 + faceEyeStrictness * 0.45)

        val gazeBurstPressure =
            (
                faceEyeSaccadeVelocityDegreesPerSecond
                    .coerceIn(0.0, 1_200.0) / 900.0
            ).coerceIn(0.0, 1.0)

        val motionPriority =
            (
                stress * 0.48 +
                    motionScore * 0.18 +
                    highFrequencyMotion * 0.22 +
                    lowFrequencyMotion * 0.06 +
                    oodRisk * 0.06
                ).coerceIn(0.0, 1.0)

        val referenceMultiplier =
            if (
                policy == Policy.ACCEPT &&
                    score >= 0.90 &&
                    cameraCount >= profile.multiViewConsensusThreshold &&
                    gpuDispatchMillis.isFinite()
            ) {
                1.0
            } else {
                combined
            }

        val targetInference =
            (
                profile.inferenceFps *
                    inferenceRuntime *
                    referenceMultiplier *
                    (
                        1.0 +
                            motionPriority * 0.22 +
                            gazeBurstPressure * 0.12 +
                            rootVelocityRecoveryPressure * 0.08 +
                            worldSpaceRecoveryPressure * 0.10
                        )
                )
                .toInt()
                .coerceIn(6, profile.targetCaptureFps)

        val denseSurfaceScale =
            (
                combined *
                    (1.0 - stress * 0.20) *
                    (1.0 - physicsRecoveryPressure * 0.24) *
                    (1.0 - directionalRecoveryPressure * 0.28) *
                    (1.0 - faceEyeRecoveryPressure * 0.16) *
                    (1.0 - faceEyeCalibrationPressure * 0.20) *
                    (1.0 - rigRecoveryPressure * 0.32) *
                    (1.0 - global4DRecoveryPressure * 0.20) *
                    (1.0 - rootVelocityRecoveryPressure * 0.16) *
                    (1.0 - worldSpaceRecoveryPressure * 0.20)
                ).coerceIn(0.24, 1.0)

        val denseMesh =
            (profile.denseMeshHz * denseSurfaceScale)
                .toInt()
                .coerceIn(1, profile.denseMeshHz)

        val faceEyeRecoveryFloor =
            if (
                !faceEyeFusionAccepted ||
                faceEyeCalibrationDriftPersistent
            ) {
                1.05
            } else {
                0.0
            }

        val faceMeshScale =
            maxOf(
                combined *
                    (
                        1.0 +
                            oodRisk * 0.10 +
                            faceEyeRecoveryPressure * 0.44 +
                            faceEyeCalibrationPressure * 0.24 +
                            gazeBurstPressure * 0.08
                    ),
                faceEyeRecoveryFloor,
            )

        val faceMesh =
            (
                profile.faceMeshHz *
                    faceMeshScale
                )
                .toInt()
                .coerceIn(1, profile.faceMeshHz)

        val eyeTrackingScale =
            maxOf(
                referenceMultiplier *
                    (
                        1.0 +
                            motionPriority * 0.16 +
                            highFrequencyMotion * 0.16 +
                            faceEyeRecoveryPressure * 0.52 +
                            faceEyeCalibrationPressure * 0.34 +
                            gazeBurstPressure * 0.38
                    ),
                faceEyeRecoveryFloor,
            )

        val eyes =
            (
                profile.eyeTrackingHz *
                    eyeTrackingScale
                )
                .toInt()
                .coerceIn(1, profile.eyeTrackingHz)

        val hands =
            (
                profile.handTrackingHz *
                    combined *
                    (
                        1.0 +
                            stress * 0.18 +
                            highFrequencyMotion * 0.34
                        )
                )
                .toInt()
                .coerceIn(1, profile.handTrackingHz)

        val ba =
            (
                profile.bundleAdjustmentIterations *
                    referenceMultiplier *
                    (
                        1.0 +
                            physicsRecoveryPressure * 0.28 +
                            directionalRecoveryPressure * 0.36 +
                            global4DRecoveryPressure * 0.20 +
                            faceEyeCalibrationPressure * 0.16 +
                            rootVelocityRecoveryPressure * 0.16 +
                            worldSpaceRecoveryPressure * 0.22
                        )
                )
                .toInt()
                .coerceIn(
                    2,
                    (
                        profile.bundleAdjustmentIterations * 1.28
                    ).toInt().coerceAtLeast(2),
                )

        val meshBudget =
            (profile.meshVertexBudget * combined)
                .toInt()
                .coerceIn(8_000, profile.meshVertexBudget)

        val tsdfStride =
            when {
                combined >= 0.90 -> 1
                combined >= 0.72 -> 2
                combined >= 0.52 -> 3
                else -> 4
            }

        val detectorInterval =
            when {
                motionPriority >= 0.78 ||
                    highFrequencyMotion >= 0.72 ||
                    gazeBurstPressure >= 0.72 ->
                    profile.detectorIntervalFrames.coerceAtLeast(1)
                policy == Policy.ACCEPT && score >= 0.84 ->
                    profile.detectorIntervalFrames
                uncertaintyOodRisk >= 0.58 ||
                    directionalRecoveryPressure >= 0.52 ->
                    profile.detectorIntervalFrames.coerceAtMost(3)
                else ->
                    (profile.detectorIntervalFrames + 1).coerceAtMost(8)
            }

        val allowDense =
            combined >= 0.62 &&
                physicsScore >= 0.46 &&
                directionalConfidence >= 0.32 &&
                directionalUncertaintyAnisotropy < 12.0 &&
                worldSpaceConsistencyScore >= 0.46 &&
                rigTrust >= 0.46 &&
                profile.surfacePolicy != com.nivukx.nox.core.model.SurfacePolicy.SKELETON_ONLY

        val allowReference =
            profile.qualityTier == com.nivukx.nox.core.model.CaptureQualityTier.REFERENCE &&
                policy == Policy.ACCEPT &&
                score >= 0.88 &&
                uncertaintyOodRisk < 0.42 &&
                physicsScore >= 0.78 &&
                directionalConfidence >= 0.72 &&
                directionalUncertaintyAnisotropy < 6.0 &&
                global4DScore >= 0.82 &&
                global4DAccepted &&
                !global4DPersistentFailure &&
                !faceEyeCalibrationDriftPersistent &&
                faceEyeCalibrationPressure < 0.45 &&
                rigTrust >= 0.82 &&
                cameraCount >= profile.multiViewConsensusThreshold

        return ComputeBudget(
            detectorIntervalFrames = detectorInterval,
            targetInferenceFps = targetInference,
            targetCaptureFps = profile.targetCaptureFps,
            denseMeshHz = denseMesh,
            faceMeshHz = faceMesh,
            eyeTrackingHz = eyes,
            handTrackingHz = hands,
            bundleAdjustmentIterations = ba,
            tsdfStride = tsdfStride,
            meshVertexBudget = meshBudget,
            allowDenseSurface = allowDense,
            allowReferenceOptimization = allowReference,
        )
    }
}

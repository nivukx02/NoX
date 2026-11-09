package com.nivukx.nox.core.model

/**
 * Immutable session record of the production reconstruction policy selected
 * for a frame or evaluation interval.
 *
 * This is provenance data, not a UI snapshot. It allows post-session tooling
 * to explain why a frame was accepted, held, degraded, or rejected and what
 * compute envelope was active at that point.
 */
data class ProductionPolicyRecord(
    val timestampNanos: Long,
    val profileId: String,
    val policy: String,
    val bottleneck: String,
    val score: Double,
    val rawScore: Double,
    val outputAllowed: Boolean,
    val hardFailure: Boolean,
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
    val calibratedUncertaintyConfidence: Double = 1.0,
    val motionOutOfDistributionRisk: Double = 0.0,
    val motionUncertaintyCoverage: Double = 1.0,
    val globalRigTrust: Double = 1.0,
    val motionDynamicStress: Double = 0.0,
    val motionDynamicsScore: Double = 0.0,
    val motionLowFrequencyScore: Double = 0.0,
    val motionHighFrequencyScore: Double = 0.0,
    val motionFrequencyBandSensitivity: Double = 0.82,
    val physicsConstraintScore: Double = 1.0,
    val physicsConstraintAccepted: Boolean = true,
    val surfaceNormalObservability: Double = 1.0,
    val surfaceHighUncertaintyRatio: Double = 0.0,
    val surfaceCameraInformationGain: Double = 1.0,
    val surfaceEvidenceStrictness: Double = 0.84,
    val faceEyeFusionAccepted: Boolean = true,
    val faceEyeFusionMode: String = "HOLD",
    val faceEyeIndependentEvidenceSupport: Double = 0.0,
    val faceEyeReliabilityTrend: Double = 0.0,
    val faceEyeHardFailure: Boolean = false,
    val faceEyeFusionConfidence: Double = 0.0,
    val faceEyeShapeConfidence: Double = 1.0,
    val faceEyeEyeDynamicConsistency: Double = 1.0,
    val faceEyeHardFailureReasonCount: Int = 0,
    val faceEyeOutOfDistributionRisk: Double = 0.0,
    val faceEyeFusionStrictness: Double = 0.84,
    val faceEyeSaccadeVelocityDegreesPerSecond: Double = 0.0,
    val faceEyeDominantFailure: String = "NONE",
    val faceEyeCalibrationDriftScore: Double = 0.0,
    val faceEyeCalibrationDriftPersistent: Boolean = false,
    val faceEyeIpdDriftToleranceMillimeters: Double = 12.0,
    val physicsConstraintStrictness: Double = 0.84,
    val global4DConsistencyStrictness: Double = 0.86,
    val physicsMeanBoneResidualMeters: Double = 0.0,
    val physicsP95BoneResidualMeters: Double = 0.0,
    val physicsMaxBoneResidualMeters: Double = 0.0,
    val physicsSymmetryScore: Double = 1.0,
    val physicsFloorContactScore: Double = 1.0,
    val physicsCorrectedJointCount: Int = 0,
    val physicsConstrainedEdgeCount: Int = 0,
    val directionalUncertaintyMajor95Meters: Double = 0.0,
    val directionalUncertaintyDepthMeters: Double = 0.0,
    val directionalUncertaintyAnisotropy: Double = 1.0,
    val directionalUncertaintyConfidence: Double = 1.0,
    val directionalUncertaintyDegradedJointCount: Int = 0,
    val global4DScore: Double = 1.0,
    val global4DAccepted: Boolean = true,
    val global4DPersistentFailure: Boolean = false,
    val global4DRootMotionConsistency: Double = 1.0,
    val global4DRootVelocityAgreement: Double = 1.0,
    val global4DSceneConsistency: Double = 0.72,
    val global4DDirectionalUncertaintyConsistency: Double = 1.0,
    val global4DFaceEyeConsistency: Double = 1.0,
    val handUncertaintyConfidence: Double = 1.0,
    val handUncertaintyRadiusMeters: Double = 0.0,
    val handUncertaintyCoverage: Double = 0.0,
    val handUncertaintyOodRisk: Double = 0.0,
    val handUncertaintyPolicy: String = "NORMAL",
    val worldSpaceConsistencyScore: Double = 1.0,
    val worldSpaceRobustOutlierRatio: Double = 0.0,
    val warnings: List<String>,
) {
    init {
        require(timestampNanos > 0L)
        require(profileId.isNotBlank())
        require(policy.isNotBlank())
        require(bottleneck.isNotBlank())
        require(score.isFinite())
        require(rawScore.isFinite())
        require(detectorIntervalFrames >= 1)
        require(targetInferenceFps >= 1)
        require(targetCaptureFps >= targetInferenceFps)
        require(denseMeshHz >= 1)
        require(faceMeshHz >= 1)
        require(eyeTrackingHz >= 1)
        require(handTrackingHz >= 1)
        require(bundleAdjustmentIterations >= 0)
        require(tsdfStride in 1..16)
        require(meshVertexBudget >= 1_000)
        require(calibratedUncertaintyConfidence in 0.0..1.0)
        require(motionOutOfDistributionRisk in 0.0..1.0)
        require(motionUncertaintyCoverage in 0.0..1.0)
        require(globalRigTrust in 0.0..1.0)
        require(motionDynamicStress in 0.0..1.0)
        require(motionDynamicsScore in 0.0..1.0)
        require(motionLowFrequencyScore in 0.0..1.0)
        require(motionHighFrequencyScore in 0.0..1.0)
        require(motionFrequencyBandSensitivity in 0.0..1.0)
        require(physicsConstraintScore in 0.0..1.0)
        require(surfaceNormalObservability in 0.0..1.0)
        require(surfaceHighUncertaintyRatio in 0.0..1.0)
        require(surfaceCameraInformationGain in 0.0..1.0)
        require(surfaceEvidenceStrictness in 0.0..1.0)
        require(faceEyeFusionMode.isNotBlank())
        require(faceEyeIndependentEvidenceSupport in 0.0..1.0)
        require(faceEyeReliabilityTrend in -1.0..1.0)
        require(faceEyeFusionConfidence in 0.0..1.0)
        require(faceEyeShapeConfidence in 0.0..1.0)
        require(faceEyeEyeDynamicConsistency in 0.0..1.0)
        require(faceEyeHardFailureReasonCount >= 0)
        require(faceEyeOutOfDistributionRisk in 0.0..1.0)
        require(faceEyeFusionStrictness in 0.0..1.0)
        require(
            faceEyeSaccadeVelocityDegreesPerSecond.isFinite() &&
                faceEyeSaccadeVelocityDegreesPerSecond >= 0.0,
        )
        require(faceEyeDominantFailure.isNotBlank())
        require(faceEyeCalibrationDriftScore in 0.0..1.0)
        require(faceEyeIpdDriftToleranceMillimeters in 4.0..30.0)
        require(physicsConstraintStrictness in 0.0..1.0)
        require(global4DConsistencyStrictness in 0.0..1.0)
        require(physicsMeanBoneResidualMeters.isFinite() && physicsMeanBoneResidualMeters >= 0.0)
        require(physicsP95BoneResidualMeters.isFinite() && physicsP95BoneResidualMeters >= 0.0)
        require(physicsMaxBoneResidualMeters.isFinite() && physicsMaxBoneResidualMeters >= 0.0)
        require(physicsSymmetryScore in 0.0..1.0)
        require(physicsFloorContactScore in 0.0..1.0)
        require(physicsCorrectedJointCount >= 0)
        require(physicsConstrainedEdgeCount >= 0)
        require(
            directionalUncertaintyMajor95Meters.isFinite() &&
                directionalUncertaintyMajor95Meters >= 0.0,
        )
        require(
            directionalUncertaintyDepthMeters.isFinite() &&
                directionalUncertaintyDepthMeters >= 0.0,
        )
        require(directionalUncertaintyAnisotropy >= 1.0)
        require(directionalUncertaintyConfidence in 0.0..1.0)
        require(directionalUncertaintyDegradedJointCount >= 0)
        require(global4DScore in 0.0..1.0)
        require(global4DRootMotionConsistency in 0.0..1.0)
        require(global4DRootVelocityAgreement in 0.0..1.0)
        require(global4DSceneConsistency in 0.0..1.0)
        require(global4DDirectionalUncertaintyConsistency in 0.0..1.0)
        require(global4DFaceEyeConsistency in 0.0..1.0)
        require(handUncertaintyConfidence in 0.0..1.0)
        require(handUncertaintyRadiusMeters.isFinite() && handUncertaintyRadiusMeters >= 0.0)
        require(handUncertaintyCoverage in 0.0..1.0)
        require(handUncertaintyOodRisk in 0.0..1.0)
        require(handUncertaintyPolicy.isNotBlank())
        require(worldSpaceConsistencyScore in 0.0..1.0)
        require(worldSpaceRobustOutlierRatio in 0.0..1.0)
    }
}

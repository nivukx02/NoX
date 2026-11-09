package com.nivukx.nox.ui

import com.nivukx.nox.core.model.ActorDetailPriorityMode
import com.nivukx.nox.core.model.MotionCaptureMode
import com.nivukx.nox.core.model.MotionCaptureCustomization

enum class FrontierCapturePreset {
    CROWD_50_LIVE,
    PRECISION_QUALIFICATION,
    RECONSTRUCTION_MASTER,
    PERFORMANCE_STABLE,
}

fun MotionCaptureCustomization.applyFrontierPreset(
    preset: FrontierCapturePreset,
): MotionCaptureCustomization =
    copyWith {
        when (preset) {
            FrontierCapturePreset.CROWD_50_LIVE -> {
                maxSubjects = 50
                fullDetailActorBudget = 16
                faceDetailActorBudget = 24
                eyeDetailActorBudget = 24
                handDetailActorBudget = 24
                denseSurfaceActorBudget = 20
                actorDetailPriorityMode =
                    com.nivukx.nox.core.model.ActorDetailPriorityMode.MOTION_FIRST
                actorIdentityStrictness = 0.92
                actorOcclusionRecovery = 0.92
                actorCrossCameraAssociation = 0.94
                denseMeshDetail = 0.82
                surfaceEvidenceStrictness = 0.86
                worldConsistencyDetail = 0.90
                temporalSupersamplingPriority = 0.82
                faceDetail = 0.82
                eyeDetail = 0.84
                gazePriority = 0.80
                handDetail = 0.84
                depthDetail = 0.90
                personSegmentationEnabled = true
                preferNpuForDensePerception = true
                dynamicTopology = true
            }

            FrontierCapturePreset.PRECISION_QUALIFICATION -> {
                mode = com.nivukx.nox.core.model.MotionCaptureMode.MULTI_VIEW_RGB
                targetFps = 60
                inferenceFps = 60
                maxSubjects = 12
                fullDetailActorBudget = 12
                faceDetailActorBudget = 12
                eyeDetailActorBudget = 12
                handDetailActorBudget = 12
                denseSurfaceActorBudget = 12
                actorIdentityStrictness = 0.98
                actorOcclusionRecovery = 0.96
                actorCrossCameraAssociation = 0.98
                bodyDetail = 1.0
                faceDetail = 1.0
                eyeDetail = 1.0
                handDetail = 1.0
                depthDetail = 1.0
                denseMeshDetail = 1.0
                worldConsistencyDetail = 1.0
                topologyConstraintStrength = 0.99
                multiViewFusionWeight = 0.99
                multiViewEvidenceAdaptivity = 0.99
                temporalSupersamplingPriority = 1.0
                rollingShutterCompensation = 1.0
                ultraPrecisionEnabled = true
                ultraPrecisionQualificationMode = true
                ultraPrecisionMatchedReferenceRequired = true
                correlatedEvidenceEnabled = true
                correlatedEvidenceQualificationMode = true
                correlatedEvidenceMinimumCoverage = 0.80
                compactSession = false
            }

            FrontierCapturePreset.RECONSTRUCTION_MASTER -> {
                mode = com.nivukx.nox.core.model.MotionCaptureMode.MULTI_VIEW_RGB_DEPTH
                maxSubjects = 24
                fullDetailActorBudget = 24
                faceDetailActorBudget = 24
                eyeDetailActorBudget = 24
                handDetailActorBudget = 24
                denseSurfaceActorBudget = 24
                actorIdentityStrictness = 0.96
                actorOcclusionRecovery = 1.0
                actorCrossCameraAssociation = 0.98
                bodyDetail = 1.0
                faceDetail = 0.96
                eyeDetail = 0.96
                handDetail = 0.96
                depthDetail = 1.0
                denseMeshDetail = 1.0
                surfaceEvidenceStrictness = 1.0
                worldConsistencyDetail = 1.0
                topologyConstraintStrength = 1.0
                multiViewFusionWeight = 1.0
                multiViewEvidenceAdaptivity = 1.0
                global4DConsistencyStrictness = 1.0
                temporalSupersamplingPriority = 1.0
                dynamicTopology = true
                cameraDepthSidecarEnabled = true
                cameraRawSidecarEnabled = true
                compactSession = false
            }

            FrontierCapturePreset.PERFORMANCE_STABLE -> {
                maxSubjects = 50
                fullDetailActorBudget = 8
                faceDetailActorBudget = 12
                eyeDetailActorBudget = 12
                handDetailActorBudget = 12
                denseSurfaceActorBudget = 8
                actorIdentityStrictness = 0.86
                actorOcclusionRecovery = 0.86
                actorCrossCameraAssociation = 0.88
                denseMeshDetail = 0.66
                surfaceEvidenceStrictness = 0.70
                worldConsistencyDetail = 0.78
                temporalSupersamplingPriority = 0.64
                faceDetail = 0.68
                eyeDetail = 0.66
                gazePriority = 0.62
                handDetail = 0.66
                depthDetail = 0.72
                preferNpuForDensePerception = true
                dynamicTopology = true
            }
        }

        val boundedTargetSubjects =
            multiPersonCapacityPolicy.targetSubjects.coerceAtMost(maxSubjects)
        multiPersonCapacityPolicy = multiPersonCapacityPolicy.copy(
            targetSubjects = boundedTargetSubjects,
            hardSubjectCeiling = maxOf(
                multiPersonCapacityPolicy.hardSubjectCeiling,
                boundedTargetSubjects,
            ).coerceAtMost(64),
        )
    }


/**
 * Returns true only while the profile's defining settings remain intact.
 *
 * A profile is an editable starting point, not a lock. If an operator changes one of the
 * profile-defining controls, the UI correctly labels the configuration as Custom instead of
 * pretending the preset is still active.
 */
fun MotionCaptureCustomization.matchesFrontierPreset(
    preset: FrontierCapturePreset,
): Boolean = when (preset) {
    FrontierCapturePreset.CROWD_50_LIVE ->
        maxSubjects == 50 &&
            fullDetailActorBudget == 16 &&
            faceDetailActorBudget == 24 &&
            eyeDetailActorBudget == 24 &&
            handDetailActorBudget == 24 &&
            denseSurfaceActorBudget == 20 &&
            actorDetailPriorityMode == ActorDetailPriorityMode.MOTION_FIRST &&
            actorIdentityStrictness == 0.92 &&
            actorOcclusionRecovery == 0.92 &&
            actorCrossCameraAssociation == 0.94 &&
            denseMeshDetail == 0.82 &&
            surfaceEvidenceStrictness == 0.86 &&
            worldConsistencyDetail == 0.90 &&
            temporalSupersamplingPriority == 0.82 &&
            faceDetail == 0.82 &&
            eyeDetail == 0.84 &&
            gazePriority == 0.80 &&
            handDetail == 0.84 &&
            depthDetail == 0.90 &&
            personSegmentationEnabled &&
            preferNpuForDensePerception &&
            dynamicTopology

    FrontierCapturePreset.PRECISION_QUALIFICATION ->
        mode == MotionCaptureMode.MULTI_VIEW_RGB &&
            targetFps == 60 &&
            inferenceFps == 60 &&
            maxSubjects == 12 &&
            fullDetailActorBudget == 12 &&
            faceDetailActorBudget == 12 &&
            eyeDetailActorBudget == 12 &&
            handDetailActorBudget == 12 &&
            denseSurfaceActorBudget == 12 &&
            actorIdentityStrictness == 0.98 &&
            actorOcclusionRecovery == 0.96 &&
            actorCrossCameraAssociation == 0.98 &&
            bodyDetail == 1.0 &&
            faceDetail == 1.0 &&
            eyeDetail == 1.0 &&
            handDetail == 1.0 &&
            depthDetail == 1.0 &&
            denseMeshDetail == 1.0 &&
            worldConsistencyDetail == 1.0 &&
            topologyConstraintStrength == 0.99 &&
            multiViewFusionWeight == 0.99 &&
            multiViewEvidenceAdaptivity == 0.99 &&
            temporalSupersamplingPriority == 1.0 &&
            rollingShutterCompensation == 1.0 &&
            ultraPrecisionEnabled &&
            ultraPrecisionQualificationMode &&
            ultraPrecisionMatchedReferenceRequired &&
            correlatedEvidenceEnabled &&
            correlatedEvidenceQualificationMode &&
            correlatedEvidenceMinimumCoverage == 0.80 &&
            !compactSession

    FrontierCapturePreset.RECONSTRUCTION_MASTER ->
        mode == MotionCaptureMode.MULTI_VIEW_RGB_DEPTH &&
            maxSubjects == 24 &&
            fullDetailActorBudget == 24 &&
            faceDetailActorBudget == 24 &&
            eyeDetailActorBudget == 24 &&
            handDetailActorBudget == 24 &&
            denseSurfaceActorBudget == 24 &&
            actorIdentityStrictness == 0.96 &&
            actorOcclusionRecovery == 1.0 &&
            actorCrossCameraAssociation == 0.98 &&
            bodyDetail == 1.0 &&
            faceDetail == 0.96 &&
            eyeDetail == 0.96 &&
            handDetail == 0.96 &&
            depthDetail == 1.0 &&
            denseMeshDetail == 1.0 &&
            surfaceEvidenceStrictness == 1.0 &&
            worldConsistencyDetail == 1.0 &&
            topologyConstraintStrength == 1.0 &&
            multiViewFusionWeight == 1.0 &&
            multiViewEvidenceAdaptivity == 1.0 &&
            global4DConsistencyStrictness == 1.0 &&
            temporalSupersamplingPriority == 1.0 &&
            dynamicTopology &&
            cameraDepthSidecarEnabled &&
            cameraRawSidecarEnabled &&
            !compactSession

    FrontierCapturePreset.PERFORMANCE_STABLE ->
        maxSubjects == 50 &&
            fullDetailActorBudget == 8 &&
            faceDetailActorBudget == 12 &&
            eyeDetailActorBudget == 12 &&
            handDetailActorBudget == 12 &&
            denseSurfaceActorBudget == 8 &&
            actorIdentityStrictness == 0.86 &&
            actorOcclusionRecovery == 0.86 &&
            actorCrossCameraAssociation == 0.88 &&
            denseMeshDetail == 0.66 &&
            surfaceEvidenceStrictness == 0.70 &&
            worldConsistencyDetail == 0.78 &&
            temporalSupersamplingPriority == 0.64 &&
            faceDetail == 0.68 &&
            eyeDetail == 0.66 &&
            gazePriority == 0.62 &&
            handDetail == 0.66 &&
            depthDetail == 0.72 &&
            preferNpuForDensePerception &&
            dynamicTopology
}


/**
 * Applies the actor target as one coherent policy update so model invariants hold during edits.
 */
fun MotionCaptureCustomization.withTargetActorLimit(
    requestedSubjects: Int,
): MotionCaptureCustomization {
    val target = requestedSubjects.coerceIn(1, 64)
    return copyWith {
        maxSubjects = target
        val priorPolicy = multiPersonCapacityPolicy
        multiPersonCapacityPolicy = priorPolicy.copy(
            targetSubjects = target,
            hardSubjectCeiling = maxOf(priorPolicy.hardSubjectCeiling, target).coerceAtMost(64),
        )
    }
}

/**
 * Shrinking the camera envelope clamps both dependent budgets in the same immutable update.
 */
fun MotionCaptureCustomization.withGlobalCameraEnvelope(
    requestedCameras: Int,
): MotionCaptureCustomization {
    val envelope = requestedCameras.coerceIn(2, 128)
    val authorityLimit = envelope.coerceAtMost(32)
    val shardLimit = envelope.coerceAtMost(16)
    return copyWith {
        distributedGlobalCameraEnvelope = envelope
        distributedAuthoritativeCameraBudget =
            distributedAuthoritativeCameraBudget.coerceIn(2, authorityLimit)
        distributedMaximumShards =
            distributedMaximumShards.coerceIn(1, shardLimit)
    }
}

package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.ai.PerceptionPipeline
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.LandmarkObservation
import com.nivukx.nox.core.model.LandmarkSpace
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.MotionCaptureCustomization
import kotlin.math.max

/**
 * Re-enters recorded multimodal perception results into the canonical calibrated 3D fusion engine.
 *
 * This bridge is intentionally thin. It does not re-implement triangulation, uncertainty
 * propagation, kinematic constraints, dense surface reconstruction or global 4D consistency. Those
 * remain owned by MultiViewFusionEngine and its specialized downstream stages.
 *
 * A local recording can be processed through this bridge when its camera is present in the active
 * calibration snapshot. Multiple camera recordings can share one bridge instance and push their
 * results into the same fusion buffer before reconstruction is requested for an actor.
 */
class PostCaptureReconstructionBridge(
    calibrationProvider: () -> com.nivukx.nox.core.model.CalibrationSnapshot?,
    private val fusion: MultiViewFusionEngine =
        MultiViewFusionEngine(
            calibrationProvider = calibrationProvider,
        ),
    private val distributedRigPlanProvider:
        () -> DistributedCameraRigCoordinator.Plan? = { null },
) : AutoCloseable {

    data class ReconstructedActor(
        val actorId: Long,
        val skeleton: SkeletonFrame?,
        val warnings: List<String>,
    )

    data class FrameResult(
        val timestampNanos: Long,
        val cameraId: DeviceId,
        val actorCountObserved: Int,
        val actorResults: List<ReconstructedActor>,
        val acceptedActorCount: Int,
        val denseSurfaceResult:
            DenseMultiViewSurfaceReconstructionEngine.Result? = null,
        val warnings: List<String>,
    )

    fun pushAndReconstruct(
        result: PerceptionPipeline.Result,
        cameraId: DeviceId,
        customization: MotionCaptureCustomization =
            MotionCaptureCustomization.production(),
    ): FrameResult {
        val multi = result.multiActorPose
        if (multi == null) {
            return FrameResult(
                timestampNanos = result.timestampNanos,
                cameraId = cameraId,
                actorCountObserved = 0,
                actorResults = emptyList(),
                acceptedActorCount = 0,
                warnings = listOf("POST_CAPTURE_MULTI_ACTOR_RESULT_UNAVAILABLE"),
            )
        }

        val width = multi.inputWidth.coerceAtLeast(1)
        val height = multi.inputHeight.coerceAtLeast(1)
        val distributedRigPlan = distributedRigPlanProvider()
        val cameraAdmittedByRig =
            distributedRigPlan?.let {
                cameraId.value in it.selectedCameraIds
            } ?: true

        if (!cameraAdmittedByRig) {
            return FrameResult(
                timestampNanos = result.timestampNanos,
                cameraId = cameraId,
                actorCountObserved = result.actorTracks.size,
                actorResults = emptyList(),
                acceptedActorCount = 0,
                warnings = listOf(
                    "POST_CAPTURE_CAMERA_OUTSIDE_DISTRIBUTED_RIG_AUTHORITY",
                ) +
                    (distributedRigPlan?.warnings ?: emptyList())
                        .takeLast(12),
            )
        }

        val segmentation =
            result.personSegmentation
                ?.takeIf {
                    it.state ==
                        com.nivukx.nox.core.ai.PersonSegmentationEngine.State.READY
                }
                ?.takeIf {
                    val segmentationTimestampNanos =
                        it.timestampMs.coerceAtLeast(0L) *
                            1_000_000L
                    kotlin.math.abs(
                        segmentationTimestampNanos -
                            result.timestampNanos,
                    ) <=
                        280_000_000L
                }
                ?.takeIf {
                    val segmentationTimestampNanos =
                        it.timestampMs * 1_000_000L
                    val ageNanos =
                        result.timestampNanos -
                            segmentationTimestampNanos
                    ageNanos in
                        0L..280_000_000L
                }

        result.denseLandmarkEvidence?.let { dense ->
            val denseInputs =
                dense.landmarks
                    .filter {
                        it.state !=
                            com.nivukx.nox.core.ai
                                .DenseLandmarkEvidenceFusionEngine
                                .EvidenceState.REJECTED
                    }
                    .map { landmark ->
                        MultiViewFusionEngine.DenseLandmarkInput(
                            subjectId = landmark.key.subjectId,
                            landmarkId = landmark.key.landmarkId,
                            normalizedX = landmark.normalizedX,
                            normalizedY = landmark.normalizedY,
                            confidence = landmark.confidence,
                            uncertainty = landmark.uncertainty,
                            visibility = landmark.visibility,
                            contactProbability =
                                landmark.contactProbability,
                            temporalStability = landmark.temporalStability,
                            crossViewAgreement = landmark.crossViewAgreement,
                            reprojectionResidualNormalized =
                                landmark.weightedReprojectionResidualNormalized,
                            predicted =
                                landmark.state ==
                                    com.nivukx.nox.core.ai
                                        .DenseLandmarkEvidenceFusionEngine
                                        .EvidenceState
                                        .TEMPORALLY_PREDICTED,
                            frameHeightPx = height,
                            captureTimingConfidence =
                                result.subframeTiming
                                    ?.timingConfidence
                                    ?: 1.0,
                            probabilitySigmaXNormalized =
                                landmark.probabilitySigmaXNormalized,
                            probabilitySigmaYNormalized =
                                landmark.probabilitySigmaYNormalized,
                            silhouetteConsistency =
                                segmentation?.let {
                                    (
                                        it.silhouetteSupport(
                                            landmark.normalizedX,
                                            landmark.normalizedY,
                                        ) * 0.70 +
                                            it.boundaryStability * 0.15 +
                                            it.meanForegroundConfidence * 0.15
                                        ).coerceIn(0.0, 1.0)
                                } ?: 1.0,
                        )
                    }

            if (denseInputs.isNotEmpty()) {
                fusion.pushDenseLandmarks(
                    deviceId = cameraId,
                    timestampNanos = result.timestampNanos,
                    landmarks = denseInputs,
                )
            }
        }

        val denseSurfaceResult =
            runCatching {
                fusion.reconstructBufferedDenseSurface(
                    timestampNanos = result.timestampNanos,
                    maximumSkewNanos = 25_000_000L,
                )
            }.getOrNull()

        val trackResults = result.actorTracks
            .take(
                customization.maxSubjects
                    .coerceIn(1, 64),
            )
            .map { actor ->
                val observations =
                    actor.joints.map { joint ->
                        LandmarkObservation(
                            landmarkId = joint.landmarkId,
                            x = joint.x * width,
                            y = joint.y * height,
                            confidence = joint.confidence.coerceIn(0.0, 1.0),
                            timestampNanos = result.timestampNanos,
                            space = LandmarkSpace.PIXELS,
                            trackingId =
                                actor.actorId
                                    .coerceAtMost(Int.MAX_VALUE.toLong())
                                    .toInt(),
                        )
                    }

                if (observations.size >= 4) {
                    fusion.push(
                        deviceId = cameraId,
                        timestampNanos = result.timestampNanos,
                        landmarks = observations,
                    )
                }

                actor to observations
            }

        val actorResults =
            trackResults.map { (actor, _) ->
                val skeleton =
                    runCatching {
                        fusion.reconstruct(
                            timestampNanos = result.timestampNanos,
                            trackId = actor.actorId,
                            subjectType = SubjectType.HUMAN,
                        )
                    }.getOrNull()

                ReconstructedActor(
                    actorId = actor.actorId,
                    skeleton = skeleton,
                    warnings =
                        if (skeleton == null) {
                            listOf(
                                "POST_CAPTURE_ACTOR_RECONSTRUCTION_UNAVAILABLE",
                            )
                        } else {
                            emptyList()
                        },
                )
            }

        val accepted =
            actorResults.count { it.skeleton != null }

        return FrameResult(
            timestampNanos = result.timestampNanos,
            cameraId = cameraId,
            actorCountObserved = result.actorTracks.size,
            actorResults = actorResults,
            acceptedActorCount = accepted,
            denseSurfaceResult = denseSurfaceResult,
            warnings =
                (
                    multi.warnings +
                        actorResults.flatMap { it.warnings } +
                        (
                            distributedRigPlan
                                ?.takeIf { it.authoritative }
                                ?.warnings
                                .orEmpty()
                        )
                    )
                    .distinct()
                    .takeLast(24),
        )
    }

    fun reset() {
        fusion.reset()
    }

    override fun close() {
        fusion.reset()
    }
}

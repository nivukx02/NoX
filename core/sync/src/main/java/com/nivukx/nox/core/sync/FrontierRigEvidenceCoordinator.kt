package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.HighEndMotionCaptureSettings
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.calibration.UltraMultiCameraCalibrationAuthorityEngine
import kotlin.math.PI

/**
 * Combines synchronization quality with bounded distributed camera authority selection.
 * Remote camera participation remains distributed evidence; sensor ownership is never centralized here.
 */
class FrontierRigEvidenceCoordinator(
    private val synchronization: MultiCameraSynchronizationQualityEngine = MultiCameraSynchronizationQualityEngine(),
    private val authority: FrontierCameraAuthorityFederationEngine = FrontierCameraAuthorityFederationEngine(),
    private val calibrationAuthority: UltraMultiCameraCalibrationAuthorityEngine =
        UltraMultiCameraCalibrationAuthorityEngine(),
    private val actorCameraLoadBalancer: UltraMultiActorCameraLoadBalancer =
        UltraMultiActorCameraLoadBalancer(),
) : AutoCloseable {

    private var closed = false
    @Synchronized
    fun configure(settings: HighEndMotionCaptureSettings) {
        check(!closed) {
            "FrontierRigEvidenceCoordinator is closed"
        }
        actorCameraLoadBalancer.configure(
            UltraMultiActorCameraLoadBalancer.Config(
                maximumActors =
                    settings.maxSubjects.coerceIn(1, 64),
                authorityCamerasPerActor =
                    12,
                minimumAuthorityCamerasPerActor =
                    settings.frontierMinimumObservedViews
                        .coerceIn(2, 12),
                maximumAuthorityActorLoad =
                    24,
                maximumWitnessCamerasPerActor =
                    24,
                adaptiveAuthorityDepthEnabled =
                    settings.adaptiveAuthorityDepthEnabled,
                maximumAdaptiveAuthorityBonus =
                    settings.maximumAdaptiveAuthorityBonus,
                adaptiveDifficultyThreshold =
                    settings.adaptiveAuthorityDifficultyThreshold,
                adaptivePriorityThreshold =
                    settings.adaptiveAuthorityPriorityThreshold,
                maximumAdaptiveAuthorityCamerasPerActor =
                    settings.maximumAdaptiveAuthorityCamerasPerActor
                        .coerceIn(12, 32),
                metricGeometryWeight =
                    (
                        0.08 +
                            settings.cameraDiversityPriority *
                                0.10
                    ).coerceIn(0.08, 0.20),
                metricNoveltyWeight =
                    (
                        0.08 +
                            settings.cameraDiversityPriority *
                                0.10
                    ).coerceIn(0.08, 0.20),
            ),
        )
    }

    data class CameraInput(
        val evidence: FrontierCameraAuthorityFederationEngine.CameraEvidence,
        val peerSample: MultiCameraSynchronizationQualityEngine.PeerSample?,
        val calibrationUncertainty: CalibrationUncertaintyProfile? = null,
        val positionMeters: Vec3? = null,
        val forward: Vec3? = null,
        val reprojectionErrorPixels: Double = 0.0,
        val coverage: Double = 1.0,
        val timingResidualMillis: Double = 0.0,
    )

    data class Result(
        val synchronization: MultiCameraSynchronizationQualityEngine.Result,
        val cameraAuthority: FrontierCameraAuthorityFederationEngine.Result,
        val authoritativeReady: Boolean,
        val warnings: List<String>,
        val cameraAuthorityByActor: Map<Int, FrontierCameraAuthorityFederationEngine.Result> = emptyMap(),
        val calibrationAuthority: UltraMultiCameraCalibrationAuthorityEngine.Result? = null,
        val actorCameraAllocation: UltraMultiActorCameraLoadBalancer.Result? = null,
    )

    @Synchronized
    fun evaluate(
        timestampNanos: Long,
        cameras: List<CameraInput>,
        actorIds: Set<Int> = emptySet(),
        actorPositionsMeters: Map<Int, Vec3> = emptyMap(),
        actorPriorities: Map<Int, Double> = emptyMap(),
        actorOcclusionRisk: Map<Int, Double> = emptyMap(),
    ): Result {
        check(!closed) {
            "FrontierRigEvidenceCoordinator is closed"
        }
        val syncSamples = cameras.mapNotNull { it.peerSample }
        val syncResult = synchronization.evaluate(timestampNanos, syncSamples)
        val adjusted = cameras.map { input ->
            val sample = input.peerSample
            val synchronizedPeer = sample?.let { peer -> syncSamples.firstOrNull { it.deviceId == peer.deviceId } }
            val uncertainty = input.calibrationUncertainty
            input.evidence.copy(
                positionMeters = input.positionMeters,
                forward = input.forward,
                synchronization =
                    if (synchronizedPeer != null) {
                        input.evidence.synchronization * syncResult.confidence
                    } else {
                        input.evidence.synchronization
                    },
                calibration =
                    (
                        input.evidence.calibration *
                            (uncertainty?.evidenceConfidence ?: 1.0)
                    ).coerceIn(0.0, 1.0),
                poseUncertaintyMeters =
                    uncertainty?.translationSigmaMeters ?: input.evidence.poseUncertaintyMeters,
                poseAngularUncertaintyDegrees =
                    uncertainty?.rotationSigmaRadians
                        ?.times(180.0 / PI)
                        ?: input.evidence.poseAngularUncertaintyDegrees,
            )
        }
        val adjustedEvidence = adjusted.map { it.evidence }

        val calibrationInputs =
            cameras.mapNotNull { input ->
                val position = input.positionMeters
                    ?: return@mapNotNull null
                val forward = input.forward
                    ?: return@mapNotNull null
                val uncertainty =
                    input.calibrationUncertainty
                        ?: CalibrationUncertaintyProfile()

                UltraMultiCameraCalibrationAuthorityEngine.Camera(
                    id = input.evidence.cameraId,
                    sector = input.evidence.sector,
                    positionMeters = position,
                    forward = forward,
                    reprojectionErrorPixels =
                        input.reprojectionErrorPixels.coerceAtLeast(0.0),
                    synchronizationConfidence =
                        input.evidence.synchronization,
                    calibrationConfidence =
                        input.evidence.calibration,
                    coverage =
                        input.coverage.coerceIn(0.0, 1.0),
                    uncertainty = uncertainty,
                    timingResidualMillis =
                        input.timingResidualMillis.coerceAtLeast(0.0),
                    available =
                        input.evidence.health > 0.0,
                )
            }

        val calibrationResult =
            calibrationInputs
                .takeIf { it.size >= 2 }
                ?.let(calibrationAuthority::evaluate)

        val cameraResult = authority.select(adjustedEvidence)

        val loadBalancerActors =
            actorIds
                .filter { it >= 0 }
                .sorted()
                .map { actorId ->
                    UltraMultiActorCameraLoadBalancer.Actor(
                        actorId = actorId,
                        priority =
                            actorPriorities[actorId]
                                ?.coerceIn(0.0, 1.0)
                                ?: 0.50,
                        visibilityByCamera =
                            adjustedEvidence.associate { evidence ->
                                evidence.cameraId to
                                    (
                                        evidence.actorVisibilityById[actorId]
                                            ?: evidence.actorVisibility
                                    )
                            },
                        worldPositionMeters =
                            actorPositionsMeters[actorId],
                        occlusionRisk =
                            actorOcclusionRisk[actorId]
                                ?.coerceIn(0.0, 1.0)
                                ?: 0.0,
                    )
                }

        val actorCameraAllocation =
            actorCameraLoadBalancer.allocate(
                actors = loadBalancerActors,
                cameras = adjustedEvidence,
            )

        val cameraAuthorityByActor = actorIds
            .filter { it >= 0 }
            .sorted()
            .associateWith { actorId ->
                authority.selectForActor(actorId, adjustedEvidence)
            }
        val calibrationReady =
            calibrationResult?.metricReady != false
        val ready = cameraResult.authoritative.size >= 2 &&
            syncResult.confidence >= 0.70 &&
            cameraResult.authorityScore >= 0.60 &&
            calibrationReady
        return Result(
            synchronization = syncResult,
            cameraAuthority = cameraResult,
            authoritativeReady = ready,
            cameraAuthorityByActor = cameraAuthorityByActor,
            warnings = buildList {
                if (!ready) add("frontier-rig-not-authoritative")
                calibrationResult?.warnings?.forEach { add("calibration-$it") }
                if (cameraResult.authoritative.size < 4) add("frontier-rig-low-authoritative-camera-diversity")
                actorCameraAllocation.warnings.forEach { add("actor-camera-$it") }
                cameraAuthorityByActor.forEach { (actorId, result) ->
                    if (result.authoritative.size < 2) add("actor-$actorId-low-authoritative-camera-support")
                }
            },
            calibrationAuthority = calibrationResult,
            actorCameraAllocation = actorCameraAllocation.takeIf {
                actorIds.isNotEmpty()
            },
        )
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        synchronization.reset()
        authority.reset()
        actorCameraLoadBalancer.reset()
    }
}
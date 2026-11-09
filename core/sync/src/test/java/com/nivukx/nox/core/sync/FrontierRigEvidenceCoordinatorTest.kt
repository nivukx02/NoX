package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DeviceId
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierRigEvidenceCoordinatorTest {
    @Test
    fun closeIsIdempotentAndRejectsFurtherConfiguration() {
        val coordinator = FrontierRigEvidenceCoordinator()
        coordinator.close()
        coordinator.close()

        kotlin.test.assertFailsWith<IllegalStateException> {
            coordinator.configure(
                com.nivukx.nox.core.model.HighEndMotionCaptureSettings(),
            )
        }
    }

    @Test
    fun authoritySelectionRemainsBounded() {
        val coordinator=FrontierRigEvidenceCoordinator()
        val result=coordinator.evaluate(100L,(0 until 128).map { id ->
            FrontierRigEvidenceCoordinator.CameraInput(
                FrontierCameraAuthorityFederationEngine.CameraEvidence("c$id",id%16,0.95,0.95,0.95,0.95,0.9,0.9,0.9,0.95),
                MultiCameraSynchronizationQualityEngine.PeerSample(DeviceId("d$id"),0L,10_000L,0.0,10.0),
            )
        })
        assertTrue(result.cameraAuthority.authoritative.size <= 32)
        assertTrue(result.cameraAuthority.witnesses.size <= 96)
    }
    @Test
    fun actorSpecificAuthorityUsesActorVisibilityEvidence() {
        val coordinator = FrontierRigEvidenceCoordinator()
        val cameras = (0 until 8).map { id ->
            FrontierRigEvidenceCoordinator.CameraInput(
                evidence = FrontierCameraAuthorityFederationEngine.CameraEvidence(
                    cameraId = "cam-$id",
                    sector = id,
                    health = 0.95,
                    synchronization = 0.95,
                    calibration = 0.95,
                    coverage = 0.90,
                    actorVisibility = 0.20,
                    actorVisibilityById = mapOf(
                        1 to if (id < 4) 0.98 else 0.15,
                        2 to if (id >= 4) 0.98 else 0.15,
                    ),
                    thermalHeadroom = 0.95,
                    networkHeadroom = 0.95,
                    freshness = 1.0,
                ),
                peerSample = null,
            )
        }
        val result = coordinator.evaluate(
            timestampNanos = 1_000L,
            cameras = cameras,
            actorIds = setOf(1,2),
        )
        val a = result.cameraAuthorityByActor[1]!!.authoritative.map { it.cameraId }.toSet()
        val b = result.cameraAuthorityByActor[2]!!.authoritative.map { it.cameraId }.toSet()
        assertTrue(a != b)
        assertTrue(a.any { it in setOf("cam-0","cam-1","cam-2","cam-3") })
        assertTrue(b.any { it in setOf("cam-4","cam-5","cam-6","cam-7") })
    }
    @Test
    fun runtime_settings_control_adaptive_authority_depth() {
        val coordinator = FrontierRigEvidenceCoordinator()
        coordinator.configure(
            com.nivukx.nox.core.model.HighEndMotionCaptureSettings(
                maxSubjects = 50,
                adaptiveAuthorityDepthEnabled = true,
                maximumAdaptiveAuthorityBonus = 4,
                adaptiveAuthorityDifficultyThreshold = 0.40,
                adaptiveAuthorityPriorityThreshold = 0.80,
                maximumAdaptiveAuthorityCamerasPerActor = 16,
            ),
        )

        val cameras =
            (0 until 16).map { id ->
                FrontierRigEvidenceCoordinator.CameraInput(
                    evidence =
                        FrontierCameraAuthorityFederationEngine.CameraEvidence(
                            cameraId = "runtime-$id",
                            sector = id,
                            health = 0.96,
                            synchronization = 0.96,
                            calibration = 0.96,
                            coverage = 0.90,
                            actorVisibility = 0.45,
                            actorVisibilityById =
                                mapOf(7 to 0.45),
                            thermalHeadroom = 0.96,
                            networkHeadroom = 0.96,
                            freshness = 1.0,
                        ),
                    peerSample = null,
                )
            }

        val result =
            coordinator.evaluate(
                timestampNanos = 42L,
                cameras = cameras,
                actorIds = setOf(7),
                actorPriorities = mapOf(7 to 0.95),
                actorOcclusionRisk = mapOf(7 to 0.90),
                actorPositionsMeters =
                    mapOf(
                        7 to
                            com.nivukx.nox.core.model.Vec3(
                                0.0,
                                0.0,
                                0.0,
                            ),
                    ),
            )

        val allocation =
            result.actorCameraAllocation

        assertTrue(allocation != null)
        assertTrue(
            (allocation!!.requestedAuthorityDepthByActor[7] ?: 0) > 12,
        )
    }

    @Test
    fun calibrationUncertaintyPropagatesIntoAuthorityScore() {
        val coordinator = FrontierRigEvidenceCoordinator()
        fun camera(id: String) =
            FrontierRigEvidenceCoordinator.CameraInput(
                evidence =
                    FrontierCameraAuthorityFederationEngine.CameraEvidence(
                        cameraId = id,
                        sector = 0,
                        health = 0.98,
                        synchronization = 0.98,
                        calibration = 0.98,
                        coverage = 0.95,
                        actorVisibility = 0.95,
                        thermalHeadroom = 0.95,
                        networkHeadroom = 0.95,
                        freshness = 1.0,
                    ),
                peerSample = null,
            )
        val clean = coordinator.evaluate(
            timestampNanos = 10L,
            cameras = listOf(
                camera("clean").copy(
                    calibrationUncertainty =
                        com.nivukx.nox.core.model.CalibrationUncertaintyProfile(
                            translationSigmaMeters = 0.002,
                            rotationSigmaRadians = 0.001,
                            evidenceConfidence = 0.98,
                        ),
                ),
                camera("clean-2"),
            ),
        )
        val uncertain = coordinator.evaluate(
            timestampNanos = 11L,
            cameras = listOf(
                camera("uncertain").copy(
                    calibrationUncertainty =
                        com.nivukx.nox.core.model.CalibrationUncertaintyProfile(
                            translationSigmaMeters = 0.20,
                            rotationSigmaRadians = 0.10,
                            evidenceConfidence = 0.20,
                        ),
                ),
                camera("uncertain-2"),
            ),
        )
        assertTrue(
            clean.cameraAuthority.authorityScore >
                uncertain.cameraAuthority.authorityScore,
        )
    }
}
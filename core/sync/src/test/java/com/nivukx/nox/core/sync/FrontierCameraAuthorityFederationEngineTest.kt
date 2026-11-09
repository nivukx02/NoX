package com.nivukx.nox.core.sync

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierCameraAuthorityFederationEngineTest {
    private fun camera(id: Int, sector: Int = id % 8, bad: Int = 0) =
        FrontierCameraAuthorityFederationEngine.CameraEvidence("cam-$id", sector, 0.95, 0.95, 0.97, 0.9, 0.9, 0.8, 0.8, 0.95, bad)

    @Test
    fun selectsThirtyTwoAuthoritiesFromOneHundredTwentyEightParticipants() {
        val result = FrontierCameraAuthorityFederationEngine().select((0 until 128).map(::camera))
        assertEquals(32, result.authoritative.size)
        assertEquals(96, result.witnesses.size)
        assertEquals(8, result.sectorCoverage.size)
    }

    @Test
    fun quarantinesPersistentBadCamera() {
        val result = FrontierCameraAuthorityFederationEngine().select(listOf(camera(1), camera(2, bad=3)))
        assertTrue(result.quarantined.any { it.cameraId == "cam-2" })
        assertTrue(result.authoritative.none { it.cameraId == "cam-2" })
    }

    @Test
    fun unhealthyCameraCannotBecomeAuthority() {
        val bad = camera(1).copy(health = 0.1)
        val result = FrontierCameraAuthorityFederationEngine().select(listOf(bad, camera(2)))
        assertTrue(result.authoritative.none { it.cameraId == "cam-1" })
    }
    @Test
    fun authoritySelectionUsesPreviousAuthorityAsStableTieBreak() {
        val engine = FrontierCameraAuthorityFederationEngine()
        val first = engine.select(
            listOf(
                camera(1, sector = 0),
                camera(2, sector = 1),
                camera(3, sector = 2),
            )
        )
        val second = engine.select(
            listOf(
                camera(1, sector = 0).copy(health = 0.86),
                camera(2, sector = 1).copy(health = 0.86),
                camera(3, sector = 2).copy(health = 0.861),
            )
        )
        assertTrue(first.authoritative.map { it.cameraId }.toSet().containsAll(second.authoritative.map { it.cameraId }.toSet()))
    }
    @Test
    fun strongNewCameraCanReplaceWeakPreviousAuthority() {
        val engine = FrontierCameraAuthorityFederationEngine(
            FrontierCameraAuthorityFederationEngine.Config(
                authoritativeLimit = 2,
                maximumWitnessCount = 2,
                authorityHysteresisScoreDrop = 0.04,
            )
        )
        engine.select(listOf(camera(1,0),camera(2,1),camera(3,2)))
        val result = engine.select(
            listOf(
                camera(1,0).copy(health=0.40),
                camera(2,1).copy(health=0.40),
                camera(3,2).copy(health=1.0),
            )
        )
        assertTrue(result.authoritative.any { it.cameraId == "cam-3" })
    }
    @Test
    fun actorSpecificSelectionUsesPerActorVisibilityEvidence() {
        val engine = FrontierCameraAuthorityFederationEngine(
            FrontierCameraAuthorityFederationEngine.Config(
                authoritativeLimit = 2,
                maximumWitnessCount = 2,
            ),
        )
        val result = engine.selectForActor(
            42,
            listOf(
                camera(1, 0).copy(actorVisibility = 0.4, actorVisibilityById = mapOf(42 to 0.95)),
                camera(2, 1).copy(actorVisibility = 0.9, actorVisibilityById = mapOf(42 to 0.30)),
                camera(3, 2).copy(actorVisibility = 0.9, actorVisibilityById = mapOf(42 to 0.20)),
            ),
        )
        assertTrue(result.authoritative.any { it.cameraId == "cam-1" })
    }
    @Test
    fun severeOpticalBlurCannotEnterAuthoritySet() {
        val engine =
            FrontierCameraAuthorityFederationEngine(
                FrontierCameraAuthorityFederationEngine.Config(
                    authoritativeLimit = 2,
                    maximumWitnessCount = 2,
                    maximumMotionBlurRiskForAuthority = 0.72,
                ),
            )

        val result =
            engine.select(
                listOf(
                    camera(1, 0).copy(motionBlurRisk = 0.93),
                    camera(2, 1).copy(motionBlurRisk = 0.04),
                    camera(3, 2).copy(motionBlurRisk = 0.08),
                ),
            )

        assertTrue(
            result.authoritative.none { it.cameraId == "cam-1" },
        )
        assertTrue(
            result.authoritative.any { it.cameraId == "cam-2" },
        )
        assertTrue(
            result.authoritative.any { it.cameraId == "cam-3" },
        )
    }

    @Test
    fun lowOpticalBlurPreservesLegacyAuthoritySelectionQuality() {
        val clean =
            camera(9).copy(motionBlurRisk = 0.05)
        val unavailable =
            camera(9).copy(motionBlurRisk = null)

        assertTrue(
            clean.score() > unavailable.score() - 0.10,
        )
        assertTrue(
            clean.eligible(
                FrontierCameraAuthorityFederationEngine.Config(),
            ),
        )
        assertTrue(
            unavailable.eligible(
                FrontierCameraAuthorityFederationEngine.Config(),
            ),
        )
    }

    @Test
    fun extrinsicUncertaintyMonotonicallyReducesCameraAuthority() {
        fun camera(poseUncertainty: Double, angular: Double, drift: Double) =
            FrontierCameraAuthorityFederationEngine.CameraEvidence(
                cameraId = "c",
                sector = 0,
                health = 0.98,
                synchronization = 0.98,
                calibration = 0.98,
                coverage = 0.95,
                actorVisibility = 0.95,
                thermalHeadroom = 0.95,
                networkHeadroom = 0.95,
                freshness = 1.0,
                calibrationDriftRisk = drift,
                poseUncertaintyMeters = poseUncertainty,
                poseAngularUncertaintyDegrees = angular,
            )

        val engine = FrontierCameraAuthorityFederationEngine()
        val clean = engine.select(
            listOf(camera(0.002, 0.05, 0.01)),
        )
        val uncertain = engine.select(
            listOf(camera(0.20, 8.0, 0.90)),
        )
        assertTrue(clean.authorityScore > uncertain.authorityScore)
    )
}
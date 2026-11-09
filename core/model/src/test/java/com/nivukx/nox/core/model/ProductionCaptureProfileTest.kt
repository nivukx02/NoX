package com.nivukx.nox.core.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ProductionCaptureProfileTest {

    @Test
    fun allProfilesValidateAndHaveDistinctIds() {
        val profiles = ProductionCaptureProfile.all()

        assertTrue(profiles.size >= 6)
        assertEquals(
            profiles.size,
            profiles.map { it.id.uppercase() }.distinct().size,
        )

        profiles.forEach { profile ->
            assertEquals(profile, profile.validated())
            assertTrue(profile.targetCaptureFps >= profile.inferenceFps)
            assertTrue(profile.meshVertexBudget >= 4_000)
            assertTrue(profile.tsdfTruncationMeters > profile.tsdfVoxelSizeMeters)
            assertTrue(profile.operatorSummary().isNotEmpty())
        }
    }

    @Test
    fun lookupIsCaseInsensitiveAndUnknownFallsBackToDefault() {
        assertEquals(
            ProductionCaptureProfile.CINEMATIC_ULTRA,
            ProductionCaptureProfile.byId("cinematic_ultra"),
        )

        assertEquals(
            ProductionCaptureProfile.default,
            ProductionCaptureProfile.byId("not-a-real-profile"),
        )
    }

    @Test
    fun referenceBakeUsesSmallestVoxelPolicy() {
        val profile = ProductionCaptureProfile.REFERENCE_BAKE

        assertTrue(profile.tsdfVoxelSizeMeters < 0.005)
        assertEquals("REFERENCE", profile.tsdfResolutionClass)
        assertTrue(profile.bundleAdjustmentIterations >= 64)
        assertTrue(profile.multiViewRequired)
    }

    @Test
    fun sustainedMobilePrefersMonocularFirst() {
        val profile = ProductionCaptureProfile.SUSTAINED_MOBILE

        assertEquals(
            ReconstructionMode.MONOCULAR_FIRST,
            profile.reconstructionMode,
        )
        assertEquals(
            ThermalPolicy.SUSTAINED,
            profile.thermalPolicy,
        )
        assertTrue(profile.targetCaptureFps < ProductionCaptureProfile.LIVE_ACTION.targetCaptureFps)
    }
    @Test
    fun nanoMetrologyRequiresPhysicalQualificationEvidence() {
        val profile =
            ProductionCaptureProfile.NANO_METROLOGY_REFERENCE

        assertTrue(profile.requireRigStabilityEvidence)
        assertTrue(profile.requireDynamicMetrologyEvidence)
        assertTrue(profile.minimumRigStabilityConfidence >= 0.90)
        assertEquals(
            1.0,
            profile.dynamicMetrologyP95Millimeters,
            0.0,
        )
        assertTrue(
            profile.operatorSummary()
                .any { it.label == "Rig stability" },
        )
    }

}

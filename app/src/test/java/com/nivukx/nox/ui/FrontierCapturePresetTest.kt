package com.nivukx.nox.ui

import com.nivukx.nox.core.model.MotionCaptureCustomization
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierCapturePresetTest {
    @Test
    fun everyPresetIsRecognizedAfterItIsApplied() {
        FrontierCapturePreset.entries.forEach { preset ->
            val configured = MotionCaptureCustomization().applyFrontierPreset(preset)

            assertTrue(
                "${preset.name} should be recognized after application",
                configured.matchesFrontierPreset(preset),
            )
            assertEquals(
                "Only one profile should be active for ${preset.name}",
                1,
                FrontierCapturePreset.entries.count(configured::matchesFrontierPreset),
            )
        }
    }

    @Test
    fun changingADefiningBudgetTurnsThePresetIntoCustomConfiguration() {
        val configured = MotionCaptureCustomization()
            .applyFrontierPreset(FrontierCapturePreset.CROWD_50_LIVE)
            .copyWith {
                fullDetailActorBudget = 18
            }

        assertFalse(
            configured.matchesFrontierPreset(FrontierCapturePreset.CROWD_50_LIVE),
        )
    }

    @Test
    fun presetsClampMultiPersonTargetToTheirActorLimit() {
        val precision = MotionCaptureCustomization()
            .applyFrontierPreset(FrontierCapturePreset.PRECISION_QUALIFICATION)
        val reconstruction = MotionCaptureCustomization()
            .applyFrontierPreset(FrontierCapturePreset.RECONSTRUCTION_MASTER)

        assertEquals(12, precision.maxSubjects)
        assertEquals(12, precision.multiPersonCapacityPolicy.targetSubjects)
        assertEquals(24, reconstruction.maxSubjects)
        assertEquals(24, reconstruction.multiPersonCapacityPolicy.targetSubjects)
    }

    @Test
    fun actorLimitMutationUpdatesCapacityPolicyAtomically() {
        val configured = MotionCaptureCustomization().withTargetActorLimit(12)

        assertEquals(12, configured.maxSubjects)
        assertEquals(12, configured.multiPersonCapacityPolicy.targetSubjects)
        assertTrue(configured.multiPersonCapacityPolicy.hardSubjectCeiling >= 12)
    }

    @Test
    fun cameraEnvelopeMutationClampsAuthorityAndShardBudgets() {
        val configured = MotionCaptureCustomization().withGlobalCameraEnvelope(2)

        assertEquals(2, configured.distributedGlobalCameraEnvelope)
        assertEquals(2, configured.distributedAuthoritativeCameraBudget)
        assertEquals(2, configured.distributedMaximumShards)
    }

    @Test
    fun precisionAndCrowdProfilesAreNotInterchangeable() {
        val precision = MotionCaptureCustomization()
            .applyFrontierPreset(FrontierCapturePreset.PRECISION_QUALIFICATION)

        assertTrue(
            precision.matchesFrontierPreset(FrontierCapturePreset.PRECISION_QUALIFICATION),
        )
        assertFalse(
            precision.matchesFrontierPreset(FrontierCapturePreset.CROWD_50_LIVE),
        )
    }
}

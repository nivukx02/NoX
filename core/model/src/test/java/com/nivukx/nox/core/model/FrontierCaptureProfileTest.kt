package com.nivukx.nox.core.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierCaptureProfileTest {

    @Test
    fun copyBuilderKeepsManualFocusAndColorTemperaturePropertiesIndependent() {
        val configured = MotionCaptureCustomization().copyWith {
            cameraManualFocusDistanceDiopters = 2.5
            cameraPreciseColorTemperatureEnabled = true
        }

        assertEquals(2.5, configured.cameraManualFocusDistanceDiopters!!, 0.0)
        assertTrue(configured.cameraPreciseColorTemperatureEnabled)
    }


    @Test fun fingerprintIsDeterministic() {
        val profile=FrontierCaptureProfile.frontierPrecision()
        assertEquals(profile.fingerprint(),profile.fingerprint())
        assertTrue(profile.fingerprint().length == 64)
    }

    @Test fun canonicalFormIsStableAndVersioned() {
        val profile = FrontierCaptureProfile.frontierPrecision()
        assertTrue(profile.canonicalForm().contains("frontier-precision"))
        assertTrue(profile.canonicalForm().contains("120"))
    }

    @Test
    fun frontierPrecisionPresetCarriesStrictAuthorityPolicy() {
        val settings =
            FrontierCaptureProfile.frontierPrecision().applyTo()

        assertTrue(settings.maximumMotionBlurRiskForAuthority <= 0.30)
        assertTrue(settings.minimumEpipolarConsistencyForAuthority >= 0.80)
        assertTrue(settings.minimumMetricEvidenceAuthorityForAuthority >= 0.78)
        assertTrue(settings.minimumCaptureAuthorityScore >= 0.88)
        assertTrue(settings.precisionAuthorityMinimumScore >= 0.93)
        assertTrue(settings.precisionAuthorityMinimumSamples >= 1_200)
    }

    @Test fun profileCanAdaptExistingSettingsWithoutDroppingSafetyGates() {
        val settings = FrontierCaptureProfile.frontierPrecision().applyTo()
        assertEquals(120, settings.targetFps)
        assertEquals(60, settings.inferenceFps)
        assertEquals(128, settings.distributedGlobalCameraEnvelope)
        assertEquals(32, settings.distributedAuthoritativeCameraBudget)
        assertTrue(settings.frontierPrecisionQualificationRequiresReference)
        assertTrue(settings.ultraPrecisionMatchedReferenceRequired)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsInferenceRateAboveCaptureRate() {
        FrontierCaptureProfile(targetFps = 30, inferenceFps = 60)
    }

    @Test fun changedCapturePolicyChangesFingerprint() {
        val a=FrontierCaptureProfile.frontierPrecision()
        val b=a.copy(targetFps=60)
        assertNotEquals(a.fingerprint(),b.fingerprint())
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsActorCeilingBelowTarget() {
        FrontierCaptureProfile(targetActorCount=50,hardActorCeiling=20)
    }
    @Test
    fun preciseCctPolicy_changesFingerprint_and_reachesSettings() {
        val a =
            FrontierCaptureProfile.frontierPrecision()

        val b =
            a.copy(
                preciseColorTemperatureEnabled = true,
                colorTemperatureKelvin = 3200,
                colorTint = 14,
            )

        assertNotEquals(
            a.fingerprint(),
            b.fingerprint(),
        )

        val settings = b.applyTo()

        assertTrue(
            settings.cameraPreciseColorTemperatureEnabled,
        )
        assertEquals(
            3200,
            settings.cameraColorTemperatureKelvin,
        )
        assertEquals(
            14,
            settings.cameraColorTint,
        )
    }

}
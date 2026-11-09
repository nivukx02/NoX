package com.nivukx.nox.core.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MotionCaptureCustomizationProfileTest {
    @Test
    fun multiviewPrecisionTightensSpatialAndTemporalGates() {
        val profile = MotionCaptureCustomization.multiViewPrecision()
        assertEquals(60, profile.targetFps)
        assertEquals(30, profile.inferenceFps)
        assertTrue(profile.multiViewFusionWeight >= 0.95)
        assertTrue(profile.multiViewEvidenceAdaptivity >= 0.99)
        assertTrue(profile.maximumTemporalSkewMillis <= 8.0)
        assertTrue(profile.maximumReprojectionErrorPx <= 3.5)
        assertTrue(profile.maximumReconstructionUncertaintyMeters <= 0.22)
    }

    @Test
    fun faceEyePrecisionPrioritizesFacialEvidence() {
        val profile = MotionCaptureCustomization.faceEyePrecision()
        assertEquals(60, profile.targetFps)
        assertEquals(60, profile.inferenceFps)
        assertEquals(1.0, profile.faceDetail, 1e-9)
        assertEquals(1.0, profile.eyeDetail, 1e-9)
        assertTrue(profile.faceEyeFusionWeight >= 0.90)
        assertTrue(profile.gazePriority >= 0.99)
        assertTrue(profile.eyeTemporalStrictness >= 0.99)
        assertTrue(profile.maximumReprojectionErrorPx <= 2.5)
    }
}

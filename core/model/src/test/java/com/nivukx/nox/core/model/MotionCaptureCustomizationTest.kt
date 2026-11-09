package com.nivukx.nox.core.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MotionCaptureCustomizationTest {

    @Test
    fun productionCustomizationUsesConservativeHighEndDefaults() {
        val customization = MotionCaptureCustomization.production()

        assertTrue(customization.microMotionPreservation >= 0.80)
        assertTrue(customization.cameraMotionSuppression >= 0.60)
        assertTrue(customization.faceEyeFusionWeight in 0.0..1.0)
        assertTrue(customization.temporalAlignmentStrictness >= 0.70)
        assertTrue(customization.subframeAlignmentStrictness >= 0.70)
        assertTrue(customization.rollingShutterCompensation >= 0.80)
        assertTrue(customization.outputIntegrityStrictness in 0.0..1.0)
        assertTrue(customization.minimumParallaxDegrees > 0.0)
        assertTrue(customization.preferredTemporalSkewMillis < customization.maximumTemporalSkewMillis)
        assertTrue(customization.maximumReprojectionErrorPx > 0.0)
        assertTrue(customization.maximumReconstructionUncertaintyMeters > 0.0)
    }

    @Test
    fun qualityFirstProfileRaisesStrictnessWithoutChangingContract() {
        val customization = MotionCaptureCustomization.qualityFirst()

        assertEquals(1.0, customization.microMotionPreservation, 0.0)
        assertTrue(
            customization.cameraMotionSuppression >
                MotionCaptureCustomization.production()
                    .cameraMotionSuppression,
        )
        assertTrue(
            customization.temporalAlignmentStrictness >
                MotionCaptureCustomization.production()
                    .temporalAlignmentStrictness,
        )
        assertTrue(
            customization.outputIntegrityStrictness >
                MotionCaptureCustomization.production()
                    .outputIntegrityStrictness,
        )
        assertTrue(
            customization.subframeAlignmentStrictness >
                MotionCaptureCustomization.production()
                    .subframeAlignmentStrictness,
        )
        assertTrue(
            customization.rollingShutterCompensation >
                MotionCaptureCustomization.production()
                    .rollingShutterCompensation,
        )
        assertTrue(customization.targetFps >= customization.inferenceFps)
        assertTrue(
            customization.minimumParallaxDegrees >
                MotionCaptureCustomization.production()
                    .minimumParallaxDegrees,
        )
        assertTrue(
            customization.maximumTemporalSkewMillis <
                MotionCaptureCustomization.production()
                    .maximumTemporalSkewMillis,
        )
        assertTrue(
            customization.maximumReprojectionErrorPx <
                MotionCaptureCustomization.production()
                    .maximumReprojectionErrorPx,
        )
        assertTrue(
            customization.maximumReconstructionUncertaintyMeters <
                MotionCaptureCustomization.production()
                    .maximumReconstructionUncertaintyMeters,
        )
    }
    @Test
    fun faceEyeFusionStrictnessIsIndependentAndPresetAware() {
        val production =
            MotionCaptureCustomization.production()
        val quality =
            MotionCaptureCustomization.qualityFirst()
        val lowLatency =
            MotionCaptureCustomization.lowLatency()

        assertTrue(
            production.faceEyeFusionStrictness in 0.0..1.0,
        )
        assertTrue(
            quality.faceEyeFusionStrictness >=
                production.faceEyeFusionStrictness,
        )
        assertTrue(
            lowLatency.faceEyeFusionStrictness <
                quality.faceEyeFusionStrictness,
        )
    }

    @Test
    fun ipdDriftToleranceIsBoundedAndQualityPresetIsStricter() {
        val production = MotionCaptureCustomization.production()
        val quality = MotionCaptureCustomization.qualityFirst()
        val lowLatency = MotionCaptureCustomization.lowLatency()

        assertTrue(
            production.faceEyeIpdDriftToleranceMillimeters in 4.0..30.0,
        )
        assertTrue(
            quality.faceEyeIpdDriftToleranceMillimeters <=
                production.faceEyeIpdDriftToleranceMillimeters,
        )
        assertTrue(
            lowLatency.faceEyeIpdDriftToleranceMillimeters >=
                quality.faceEyeIpdDriftToleranceMillimeters,
        )
    }

    @Test
    fun lowLatencyProfileRelaxesTemporalQualityControlsWithinContract() {
        val production =
            MotionCaptureCustomization.production()
        val lowLatency =
            MotionCaptureCustomization.lowLatency()

        assertTrue(
            lowLatency.subframeAlignmentStrictness <
                production.subframeAlignmentStrictness,
        )
        assertTrue(
            lowLatency.rollingShutterCompensation <
                production.rollingShutterCompensation,
        )
        assertTrue(
            lowLatency.subframeAlignmentStrictness in 0.0..1.0,
        )
        assertTrue(
            lowLatency.rollingShutterCompensation in 0.0..1.0,
        )
    }

    @Test
    fun preciseCctFields_survive_named_copy_builder_updates() {
        val base =
            MotionCaptureCustomization.production()

        val updated =
            base.copyWith {
                cameraPreciseColorTemperatureEnabled = true
                cameraColorTemperatureKelvin = 6500
                cameraColorTint = 22
            }

        assertTrue(
            updated.cameraPreciseColorTemperatureEnabled,
        )
        assertEquals(
            6500,
            updated.cameraColorTemperatureKelvin,
        )
        assertEquals(
            22,
            updated.cameraColorTint,
        )
        assertTrue(
            updated != base,
        )
    }

}

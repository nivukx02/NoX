package com.nivukx.nox.core.model

import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import org.junit.Test

class MotionCaptureCustomizationEvidenceTuningTest {

    @Test
    fun copyWithPreservesPhotometricExposurePolicy() {
        val original =
            MotionCaptureCustomization.production().copyWith {
                photometricExposureShiftSoftLimit = 0.18
                photometricExposureShiftHardLimit = 0.52
                photometricExposureReliabilityFloor = 0.42
            }

        val mutated =
            original.copyWith {
                bodyDetail = 1.0
                surfaceEvidenceStrictness = 0.99
            }

        assertEquals(
            0.18,
            mutated.photometricExposureShiftSoftLimit,
            1e-9,
        )
        assertEquals(
            0.52,
            mutated.photometricExposureShiftHardLimit,
            1e-9,
        )
        assertEquals(
            0.42,
            mutated.photometricExposureReliabilityFloor,
            1e-9,
        )
    }

    @Test
    fun photometric_exposure_policy_defaults_are_bounded() {
        val customization = MotionCaptureCustomization.production()

        assertEquals(
            0.12,
            customization.photometricExposureShiftSoftLimit,
            1e-9,
        )
        assertEquals(
            0.40,
            customization.photometricExposureShiftHardLimit,
            1e-9,
        )
        assertEquals(
            0.35,
            customization.photometricExposureReliabilityFloor,
            1e-9,
        )
    }

    @Test
    fun photometric_exposure_policy_rejects_invalid_ordering() {
        assertFailsWith<IllegalArgumentException> {
            MotionCaptureCustomization(
                photometricExposureShiftSoftLimit = 0.40,
                photometricExposureShiftHardLimit = 0.20,
            )
        }
    }

    @Test
    fun productionProfileKeepsStrictEvidenceDefaults() {
        val customization = MotionCaptureCustomization.production()

        assertEquals(0.92, customization.reconstructionEvidenceFusionStrength, 1e-9)
        assertEquals(
            0.12,
            customization.reconstructionEvidenceDisagreementToleranceMeters,
            1e-9,
        )
        assertEquals(
            0.20,
            customization.reconstructionEvidenceMinimumSourceDiversity,
            1e-9,
        )
    }

    @Test
    fun evidenceToleranceRejectsUnboundedValues() {
        assertFailsWith<IllegalArgumentException> {
            MotionCaptureCustomization(
                reconstructionEvidenceDisagreementToleranceMeters = 0.01,
            )
        }

        assertFailsWith<IllegalArgumentException> {
            MotionCaptureCustomization(
                reconstructionEvidenceDisagreementToleranceMeters = 1.01,
            )
        }
    }
}

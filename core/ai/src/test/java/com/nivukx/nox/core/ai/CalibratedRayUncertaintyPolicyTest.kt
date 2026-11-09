package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CalibratedRayUncertaintyPolicyTest {
    @Test
    fun missingUncertaintyIsNotTreatedAsPerfectCalibration() {
        val resolved =
            resolveCalibratedRayUncertainty(
                CalibrationUncertaintyProfile(),
            )

        assertEquals(
            UNKNOWN_CALIBRATION_TRANSLATION_SIGMA_METERS,
            resolved.translationSigmaMeters,
        )
        assertEquals(2.0, resolved.rotationPenalty)
        assertEquals(2.0, resolved.intrinsicPenalty)
        assertEquals(
            UNKNOWN_CALIBRATION_EVIDENCE_CONFIDENCE,
            resolved.evidenceConfidence,
        )
        val resultingConfidence =
            resolved.evidenceConfidence *
                (1.0 - 0.20 * resolved.rotationPenalty) *
                (1.0 - 0.15 * resolved.intrinsicPenalty)
        assertTrue(resultingConfidence < 0.20)
    }

    @Test
    fun measuredPositiveSigmasArePreservedWhileMissingFieldsRemainConservative() {
        val resolved =
            resolveCalibratedRayUncertainty(
                CalibrationUncertaintyProfile(
                    translationSigmaMeters = 0.004,
                    rotationSigmaRadians = 0.005,
                    intrinsicRelativeSigma = 0.0,
                    evidenceConfidence = 0.90,
                ),
            )

        assertEquals(0.004, resolved.translationSigmaMeters)
        assertEquals(0.5, resolved.rotationPenalty)
        assertEquals(2.0, resolved.intrinsicPenalty)
        assertEquals(0.90, resolved.evidenceConfidence)
    }

    @Test
    fun confidenceAndPenaltiesRemainBoundedForExtremeProfiles() {
        val resolved =
            resolveCalibratedRayUncertainty(
                CalibrationUncertaintyProfile(
                    translationSigmaMeters = 0.25,
                    rotationSigmaRadians = 1.0,
                    intrinsicRelativeSigma = 2.0,
                    evidenceConfidence = 1.0,
                ),
            )

        assertEquals(0.25, resolved.translationSigmaMeters)
        assertEquals(2.0, resolved.rotationPenalty)
        assertEquals(2.0, resolved.intrinsicPenalty)
        assertTrue(resolved.evidenceConfidence in 0.0..1.0)
    }
}

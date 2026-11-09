package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ActiveOpticalReferenceParityEngineTest {

    private val activeOptical =
        CapturePrecisionBenchmarkEngine.ReferenceType
            .ACTIVE_OPTICAL_MARKER_REFERENCE

    private fun strongInput(
        noxP95: Double = 0.62,
        noxP99: Double = 1.30,
        noxUncertainty: Double = 0.14,
        referenceP95: Double = 0.92,
        referenceP99: Double = 1.80,
        referenceUncertainty: Double = 0.12,
        matchedConditions: Boolean = true,
        sampleCount: Int = 720,
        independentSessions: Int = 4,
        referenceType:
            CapturePrecisionBenchmarkEngine.ReferenceType? =
            activeOptical,
    ) =
        ActiveOpticalReferenceParityEngine.Input(
            referenceType = referenceType,
            matchedConditions = matchedConditions,
            noxP95Millimeters = noxP95,
            noxP99Millimeters = noxP99,
            noxMeasurementUncertaintyMillimeters = noxUncertainty,
            referenceP95Millimeters = referenceP95,
            referenceP99Millimeters = referenceP99,
            referenceUncertaintyMillimeters =
                referenceUncertainty,
            sampleCount = sampleCount,
            independentSessions = independentSessions,
            commonTargetDistanceMeters = 3.0,
            commonTargetAngularSpeedDegreesPerSecond = 180.0,
        )

    @Test
    fun missing_reference_family_is_not_ready() {
        val result =
            ActiveOpticalReferenceParityEngine().evaluate(
                strongInput(referenceType = null),
            )

        assertEquals(
            ActiveOpticalReferenceParityEngine.State.INCONCLUSIVE,
            result.state,
        )
        assertFalse(result.eligible)
        assertFalse(result.referenceFamilyMatched)
        assertTrue(
            result.warnings.contains(
                "ACTIVE_OPTICAL_REFERENCE_FAMILY_NOT_CONFIRMED",
            ),
        )
    }

    @Test
    fun matched_active_optical_reference_can_prove_superiority_evidence() {
        val result =
            ActiveOpticalReferenceParityEngine().evaluate(
                strongInput(),
            )

        assertEquals(
            ActiveOpticalReferenceParityEngine.State.NOX_SUPERIOR_EVIDENCE,
            result.state,
        )
        assertTrue(result.eligible)
        assertTrue(result.referenceUncertaintyAdequate)
        assertTrue(result.noxMeasurementUncertaintyAdequate)
        assertTrue(
            result.conservativeNoxP95Millimeters!! <
                result.referenceP95Millimeters!!,
        )
    }

    @Test
    fun reference_remains_superior_when_nox_conservative_error_is_higher() {
        val result =
            ActiveOpticalReferenceParityEngine().evaluate(
                strongInput(
                    noxP95 = 1.30,
                    noxP99 = 3.10,
                    referenceP95 = 0.90,
                    referenceP99 = 1.90,
                ),
            )

        assertEquals(
            ActiveOpticalReferenceParityEngine.State.REFERENCE_SUPERIOR,
            result.state,
        )
        assertTrue(result.eligible)
        assertTrue(
            result.warnings.contains(
                "REFERENCE_P95_LOWER_THAN_NOX",
            ),
        )
    }

    @Test
    fun mismatched_conditions_block_superiority_even_with_good_numbers() {
        val result =
            ActiveOpticalReferenceParityEngine().evaluate(
                strongInput(matchedConditions = false),
            )

        assertEquals(
            ActiveOpticalReferenceParityEngine.State.INCONCLUSIVE,
            result.state,
        )
        assertFalse(result.eligible)
        assertTrue(
            result.warnings.contains(
                "ACTIVE_OPTICAL_REFERENCE_CONDITIONS_NOT_MATCHED",
            ),
        )
    }

    @Test
    fun insufficient_reference_uncertainty_blocks_physical_superiority() {
        val result =
            ActiveOpticalReferenceParityEngine().evaluate(
                strongInput(referenceUncertainty = 0.80),
            )

        assertEquals(
            ActiveOpticalReferenceParityEngine.State.INCONCLUSIVE,
            result.state,
        )
        assertFalse(result.referenceUncertaintyAdequate)
        assertTrue(
            result.warnings.contains(
                "ACTIVE_OPTICAL_REFERENCE_UNCERTAINTY_TOO_LARGE",
            ),
        )
    }
}

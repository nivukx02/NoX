package com.nivukx.nox.ui.studio

import kotlin.test.Test
import kotlin.test.assertTrue

class StudioHealthModelReferenceComparisonTest {

    private fun base(
        outcome: String?,
        qualified: Boolean?,
    ) =
        StudioTelemetryState(
            camerasConnected = 8,
            camerasExpected = 8,
            synchronizationConfidence = 0.98,
            calibrationReady = true,
            trackingConfidence = 0.98,
            reconstructionConfidence = 0.98,
            droppedObservations = 0L,
            thermalConstrained = false,
            exportReady = true,
            infraredReferenceOutcome = outcome,
            infraredReferenceQualified = qualified,
        )

    @Test
    fun explicit_nox_lower_error_is_visible_to_operator() {
        val summary =
            StudioHealthModel.summarize(
                base(
                    outcome = "NOX_LOWER_ERROR",
                    qualified = true,
                ),
            )

        assertTrue(
            summary.warnings.any {
                it.contains("active-optical reference shows lower")
            },
        )
    }

    @Test
    fun reference_worse_blocks_authoritative_health_state() {
        val summary =
            StudioHealthModel.summarize(
                base(
                    outcome = "REFERENCE_WORSE",
                    qualified = true,
                ),
            )

        assertTrue(
            summary.band == StudioHealthBand.BLOCKED,
        )
    }

    @Test
    fun inconclusive_comparison_never_claims_superiority() {
        val summary =
            StudioHealthModel.summarize(
                base(
                    outcome = "INCONCLUSIVE",
                    qualified = false,
                ),
            )

        assertTrue(
            summary.warnings.none {
                it.contains(
                    "lower measured error for NoX",
                )
            },
        )
    }

    @Test
    fun engineNativeReferenceLowerErrorBlocksAuthoritativeHealthState() {
        val summary =
            StudioHealthModel.summarize(
                base(
                    outcome = "REFERENCE_LOWER_ERROR",
                    qualified = true,
                ),
            )

        assertTrue(summary.band == StudioHealthBand.BLOCKED)
        assertTrue(
            summary.warnings.any {
                it.contains("lower measured error than NoX")
            },
        )
    }

}

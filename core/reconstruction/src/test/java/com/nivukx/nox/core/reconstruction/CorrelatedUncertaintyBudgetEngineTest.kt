package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CorrelatedUncertaintyBudgetEngineTest {
    @Test
    fun correlatedBudgetIsNotLinearSum() {
        val result = CorrelatedUncertaintyBudgetEngine().combine(
            CorrelatedUncertaintyBudgetEngine.Input(
                reconstructionMm = 0.20,
                directionalMm = 0.20,
                geometricMm = 0.20,
                surfaceMm = 0.20,
                motionMm = 0.10,
                temporalMm = 0.10,
            ),
        )
        assertTrue(result.oneSigmaMillimeters < 1.0)
        assertTrue(result.finalP95Millimeters > result.oneSigmaMillimeters)
    }

    @Test
    fun poorEvidenceInflatesTheBudget() {
        val engine = CorrelatedUncertaintyBudgetEngine()
        val good = engine.combine(
            CorrelatedUncertaintyBudgetEngine.Input(
                reconstructionMm = 0.20,
                directionalMm = 0.10,
                geometricMm = 0.10,
                surfaceMm = 0.10,
                motionMm = 0.05,
                temporalMm = 0.05,
            ),
        )
        val degraded = engine.combine(
            CorrelatedUncertaintyBudgetEngine.Input(
                reconstructionMm = 0.20,
                directionalMm = 0.10,
                geometricMm = 0.10,
                surfaceMm = 0.10,
                motionMm = 0.05,
                temporalMm = 0.05,
                observability = 0.45,
                synchronization = 0.70,
                occlusionRisk = 0.60,
            ),
        )
        assertTrue(degraded.finalP95Millimeters > good.finalP95Millimeters)
        assertTrue(degraded.warnings.isNotEmpty())
    }

    @Test
    fun duplicate_correlation_pairs_are_counted_once_and_reported() {
        val input =
            CorrelatedUncertaintyBudgetEngine.Input(
                reconstructionMm = 0.50,
                directionalMm = 0.50,
                geometricMm = 0.0,
                surfaceMm = 0.0,
                motionMm = 0.0,
                temporalMm = 0.0,
                correlations =
                    listOf(
                        CorrelatedUncertaintyBudgetEngine.Correlation(
                            CorrelatedUncertaintyBudgetEngine.Component.RECONSTRUCTION,
                            CorrelatedUncertaintyBudgetEngine.Component.DIRECTIONAL,
                            0.50,
                        ),
                        CorrelatedUncertaintyBudgetEngine.Correlation(
                            CorrelatedUncertaintyBudgetEngine.Component.DIRECTIONAL,
                            CorrelatedUncertaintyBudgetEngine.Component.RECONSTRUCTION,
                            0.50,
                        ),
                    ),
            )

        val result =
            CorrelatedUncertaintyBudgetEngine()
                .combine(input)

        assertEquals(1, result.correlationsApplied)
        assertTrue(
            result.warnings.any {
                it.startsWith("DUPLICATE_CORRELATION_PAIR_IGNORED:")
            },
        )
    }

    @Test
    fun correlationIsBounded() {
        val result = CorrelatedUncertaintyBudgetEngine().combine(
            CorrelatedUncertaintyBudgetEngine.Input(
                reconstructionMm = 1.0,
                directionalMm = 1.0,
                geometricMm = 0.0,
                surfaceMm = 0.0,
                motionMm = 0.0,
                temporalMm = 0.0,
                correlations = listOf(
                    CorrelatedUncertaintyBudgetEngine.Correlation(
                        CorrelatedUncertaintyBudgetEngine.Component.RECONSTRUCTION,
                        CorrelatedUncertaintyBudgetEngine.Component.DIRECTIONAL,
                        5.0,
                    ),
                ),
            ),
        )
        assertTrue(result.oneSigmaMillimeters.isFinite())
        assertEquals(1, result.correlationsApplied)
    }
}

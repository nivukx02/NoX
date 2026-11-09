package com.nivukx.nox.ui.studio

import org.junit.Assert.assertTrue
import org.junit.Test

class StudioVisualBudgetEngineTest {

    @Test
    fun healthySessionPreservesFullVisualQuality() {
        val engine = StudioVisualBudgetEngine()
        val budget = engine.evaluate(
            StudioVisualBudgetEngine.Input(
                gpuLatencyMs = 7.0,
                sessionHealth = 0.95,
                captureIntegrity = 0.96,
            ),
        )

        assertTrue(budget.tier == StudioVisualBudgetEngine.Tier.FULL)
        assertTrue(
            budget.effectScale >= 0.90f,
        )
        assertTrue(
            budget.telemetryDensity ==
                StudioVisualBudgetEngine.TelemetryDensity.DENSE,
        )
    }

    @Test
    fun gpuPressureReducesEffectsWithoutRemovingViewportDetail() {
        val engine = StudioVisualBudgetEngine()
        val budget = engine.evaluate(
            StudioVisualBudgetEngine.Input(
                gpuLatencyMs = 22.0,
                sessionHealth = 0.70,
                captureIntegrity = 0.72,
            ),
        )

        assertTrue(
            budget.tier ==
                StudioVisualBudgetEngine.Tier.REDUCED ||
                budget.tier ==
                StudioVisualBudgetEngine.Tier.ADAPTIVE,
        )
        assertTrue(budget.effectScale < 1.0f)
        assertTrue(budget.preserveViewportDetail)
        assertTrue(budget.diagnosticRefreshHz < 30.0)
    }

    @Test
    fun criticalCapturePressureUsesMinimalRendering() {
        val engine = StudioVisualBudgetEngine()
        val budget = engine.evaluate(
            StudioVisualBudgetEngine.Input(
                gpuLatencyMs = 42.0,
                sessionHealth = 0.20,
                captureIntegrity = 0.18,
            ),
        )

        assertTrue(
            budget.tier ==
                StudioVisualBudgetEngine.Tier.MINIMAL,
        )
        assertTrue(
            budget.telemetryDensity ==
                StudioVisualBudgetEngine.TelemetryDensity.CRITICAL,
        )
        assertTrue(
            budget.warnings.any {
                it.contains("critical")
            },
        )
    }
}

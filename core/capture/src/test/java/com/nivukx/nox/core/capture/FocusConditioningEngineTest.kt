package com.nivukx.nox.core.capture

import kotlin.test.Test
import kotlin.test.assertTrue

class FocusConditioningEngineTest {

    @Test
    fun stableFocusProducesHighConditioning() {
        val result =
            FocusConditioningEngine().evaluate(
                FocusConditioningEngine.Input(
                    calibrationFocusDistanceDiopters = 2.0,
                    currentFocusDistanceDiopters = 2.01,
                    focusConfidence = 0.98,
                    manualFocus = true,
                ),
            )

        assertTrue(result.conditioningScore > 0.95)
        assertTrue(result.referenceGradeCompatible)
    }

    @Test
    fun focusShiftProducesRecalibrationWarning() {
        val result =
            FocusConditioningEngine().evaluate(
                FocusConditioningEngine.Input(
                    calibrationFocusDistanceDiopters = 1.0,
                    currentFocusDistanceDiopters = 2.0,
                    focusConfidence = 0.98,
                    manualFocus = false,
                ),
            )

        assertTrue(result.conditioningScore < 0.70)
        assertTrue(
            result.warnings.contains(
                "FOCUS_CONDITIONING_DRIFT_DETECTED",
            ),
        )
        assertTrue(!result.referenceGradeCompatible)
    }

    @Test
    fun unavailableFocusStateFailsConservatively() {
        val result =
            FocusConditioningEngine().evaluate(
                FocusConditioningEngine.Input(
                    calibrationFocusDistanceDiopters = null,
                    currentFocusDistanceDiopters = null,
                    focusConfidence = 0.0,
                    manualFocus = false,
                ),
            )

        assertTrue(result.conditioningScore < 0.70)
        assertTrue(!result.referenceGradeCompatible)
    }
}

package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertTrue

class MetricPrecisionTimingAuthorityTest {

    @Test
    fun strongMeasuredTimingReducesModeledTimingContribution() {
        val engine = MetricPrecisionBudgetEngine()

        val strong =
            engine.evaluate(
                MetricPrecisionBudgetEngine.Input(
                    temporalSkewMilliseconds = 0.3,
                    timingAuthorityConfidence = 0.99,
                    timingAuthorityResidualP95Milliseconds = 0.4,
                    referenceGradeEnabled = true,
                ),
            )

        val weak =
            engine.evaluate(
                MetricPrecisionBudgetEngine.Input(
                    temporalSkewMilliseconds = 0.3,
                    timingAuthorityConfidence = 0.60,
                    timingAuthorityResidualP95Milliseconds = 4.0,
                    referenceGradeEnabled = true,
                ),
            )

        assertTrue(
            strong.contribution.timingMillimeters <
                weak.contribution.timingMillimeters,
        )
    }

    @Test
    fun lowTimingAuthorityRaisesAQualificationWarning() {
        val result =
            MetricPrecisionBudgetEngine().evaluate(
                MetricPrecisionBudgetEngine.Input(
                    referenceGradeEnabled = true,
                    timingAuthorityConfidence = 0.45,
                    timingAuthorityResidualP95Milliseconds = 6.0,
                ),
            )

        assertTrue(
            result.warnings.contains(
                "timing-authority-quality-low",
            ),
        )
    }
}

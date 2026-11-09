package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ProductionMotionCaptureQualityOrchestratorTimingTest {

    @Test
    fun referenceContractRejectsWeakTimingAuthority() {
        val engine =
            ProductionMotionCaptureQualityOrchestrator()

        val decision =
            engine.evaluate(
                ProductionMotionCaptureQualityOrchestrator.Evidence(
                    timestampNanos = 1_000_000L,
                    timingAuthorityConfidence = 0.62,
                    timingAuthorityResidualP95Milliseconds = 4.0,
                    timingAuthorityReferenceGrade = false,
                ),
                ProductionMotionCaptureQualityOrchestrator.QualityContract(
                    minimumTimingAuthorityConfidence = 0.90,
                    maximumTimingAuthorityResidualMilliseconds = 1.5,
                    requireReferenceGradeTimingAuthority = true,
                ),
            )

        assertFalse(decision.contractSatisfied)
        assertTrue(
            decision.warningReasons.any {
                it.contains("timing-authority")
            },
        )
    }

    @Test
    fun strongTimingAuthoritySatisfiesReferenceTimingContract() {
        val decision =
            ProductionMotionCaptureQualityOrchestrator()
                .evaluate(
                    ProductionMotionCaptureQualityOrchestrator.Evidence(
                        timestampNanos = 2_000_000L,
                        timingAuthorityConfidence = 0.98,
                        timingAuthorityResidualP95Milliseconds = 0.7,
                        timingAuthorityReferenceGrade = true,
                    ),
                    ProductionMotionCaptureQualityOrchestrator.QualityContract(
                        minimumTimingAuthorityConfidence = 0.90,
                        maximumTimingAuthorityResidualMilliseconds = 1.5,
                        requireReferenceGradeTimingAuthority = true,
                    ),
                )

        assertTrue(decision.contractSatisfied)
        assertTrue(decision.isReferenceGrade)
    }
}

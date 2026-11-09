package com.nivukx.nox.core.reconstruction

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class PrecisionQualificationProtocolEngineTest {

    @Test
    fun emptyBenchmarkCannotBeQualified() {
        val protocol = PrecisionQualificationProtocolEngine()
        val result = protocol.evaluate(
            ReferenceMetrologyBenchmarkEngine.Result.empty(),
            liveQualified = false,
        )

        assertEquals(
            PrecisionQualificationProtocolEngine.Status.NOT_READY,
            result.status,
        )
        assertTrue(!result.qualificationReady)
    }

    @Test
    fun protocolRequiresDynamicAndStaticCohorts() {
        val benchmark = ReferenceMetrologyBenchmarkEngine.Result.empty()
        val result =
            PrecisionQualificationProtocolEngine().evaluate(
                benchmark = benchmark,
                liveQualified = true,
            )

        assertTrue(
            result.warnings.any {
                it.contains("DYNAMIC_REFERENCE_COHORT")
            },
        )
        assertTrue(
            result.warnings.any {
                it.contains("STATIC_REFERENCE_COHORT")
            },
        )
    }
}

package com.nivukx.nox.core.calibration

import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.Test

class MetricScaleTemporalStabilityEngineTest {

    @Test
    fun stableScaleBecomesAuthoritativeAfterWarmup() {
        val engine = MetricScaleTemporalStabilityEngine()

        var result =
            engine.observe(
                timestampNanos = 1_000_000_000L,
                scaleMetersPerUnit = 1.80,
                confidence = 0.98,
            )

        repeat(20) { index ->
            result =
                engine.observe(
                    timestampNanos =
                        1_000_000_000L +
                            (index + 1L) * 5_000_000_000L,
                    scaleMetersPerUnit =
                        1.80 + if (index % 2 == 0) 0.00005 else -0.00005,
                    confidence = 0.98,
                )
        }

        assertTrue(result.sampleCount >= 12)
        assertTrue(result.authoritative)
        assertTrue(result.stable)
        assertTrue(result.driftPercentPerMinute < 0.25)
        assertTrue(result.scaleJitterPercent < 0.20)
    }

    @Test
    fun largeScaleJumpInvalidatesQualification() {
        val engine = MetricScaleTemporalStabilityEngine()

        repeat(12) { index ->
            engine.observe(
                timestampNanos =
                    1_000_000_000L +
                        index * 1_000_000_000L,
                scaleMetersPerUnit = 1.80,
                confidence = 0.98,
            )
        }

        val result =
            engine.observe(
                timestampNanos = 14_000_000_000L,
                scaleMetersPerUnit = 1.90,
                confidence = 0.98,
            )

        assertFalse(result.authoritative)
        assertTrue(
            result.status ==
                MetricScaleTemporalStabilityEngine.Status.DRIFTING ||
                result.status ==
                MetricScaleTemporalStabilityEngine.Status.UNSTABLE,
        )
        assertTrue(result.warnings.isNotEmpty())
    }
}

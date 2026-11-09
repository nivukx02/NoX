package com.nivukx.nox.core.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CaptureTimestampIntegrityEngineTest {

    @Test
    fun stableThirtyFpsSequenceBecomesHealthy() {
        val engine = CaptureTimestampIntegrityEngine(
            CaptureTimestampIntegrityEngine.Config(
                warmupSamples = 4,
            ),
        )

        val base = 10_000_000_000L
        repeat(12) { index ->
            val t = base + index * 33_333_333L
            val h = base + index * 33_333_334L
            engine.observe(
                CaptureTimestampIntegrityEngine.Sample(
                    sensorTimestampNanos = t,
                    monotonicTimestampNanos = h,
                    frameDurationNanos = 33_333_333L,
                    exposureTimeNanos = 8_000_000L,
                    rollingShutterSkewNanos = 1_000_000L,
                ),
            )
        }

        val quality = engine.quality()

        assertTrue(quality.sampleCount == 12)
        assertTrue(quality.validSampleCount >= 11)
        assertTrue(quality.confidence > 0.75)
        assertTrue(quality.healthy)
        assertEquals(30.0, quality.cadenceFps, 0.02)
    }

    @Test
    fun qualitySeparatesIntegerAndFloatingPointTimingStatistics() {
        val engine = CaptureTimestampIntegrityEngine(
            CaptureTimestampIntegrityEngine.Config(
                warmupSamples = 2,
            ),
        )

        val base = 2_000_000_000L
        repeat(6) { index ->
            val t = base + index * 33_333_333L
            engine.observe(
                CaptureTimestampIntegrityEngine.Sample(
                    sensorTimestampNanos = t,
                    monotonicTimestampNanos = t,
                    frameDurationNanos = 33_333_333L,
                    exposureTimeNanos = 8_333_333L,
                    rollingShutterSkewNanos = 1_666_666L,
                ),
            )
        }

        val quality = engine.quality()

        assertTrue(
            quality.meanIntervalNanos in 33_333_332.0..33_333_334.0,
        )
        assertTrue(
            quality.rollingShutterFraction > 0.04 &&
                quality.rollingShutterFraction < 0.06,
        )
        assertTrue(
            quality.exposureDutyCycle > 0.24 &&
                quality.exposureDutyCycle < 0.26,
        )
    }

    @Test
    fun duplicateAndOutOfOrderTimestampsAreRejected() {
        val engine = CaptureTimestampIntegrityEngine()

        engine.observe(
            CaptureTimestampIntegrityEngine.Sample(
                sensorTimestampNanos = 1_000_000_000L,
                monotonicTimestampNanos = 1_000_000_000L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 5_000_000L,
                rollingShutterSkewNanos = 500_000L,
            ),
        )

        val duplicate = engine.observe(
            CaptureTimestampIntegrityEngine.Sample(
                sensorTimestampNanos = 1_000_000_000L,
                monotonicTimestampNanos = 1_033_000_000L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 5_000_000L,
                rollingShutterSkewNanos = 500_000L,
            ),
        )

        val backwards = engine.observe(
            CaptureTimestampIntegrityEngine.Sample(
                sensorTimestampNanos = 900_000_000L,
                monotonicTimestampNanos = 1_066_000_000L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 5_000_000L,
                rollingShutterSkewNanos = 500_000L,
            ),
        )

        assertTrue(
            duplicate.verdict ==
                CaptureTimestampIntegrityEngine.Verdict.DUPLICATE,
        )
        assertTrue(
            backwards.verdict ==
                CaptureTimestampIntegrityEngine.Verdict.OUT_OF_ORDER,
        )
        assertTrue(engine.quality().duplicateCount == 1)
        assertTrue(engine.quality().outOfOrderCount == 1)
    }

    @Test
    fun hostClockReversalDoesNotCorruptTrustedSensorAnchor() {
        val engine = CaptureTimestampIntegrityEngine()

        engine.observe(
            CaptureTimestampIntegrityEngine.Sample(
                sensorTimestampNanos = 1_000_000_000L,
                monotonicTimestampNanos = 1_000_000_000L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 5_000_000L,
                rollingShutterSkewNanos = 500_000L,
            ),
        )

        val invalidHost = engine.observe(
            CaptureTimestampIntegrityEngine.Sample(
                sensorTimestampNanos = 1_033_333_333L,
                monotonicTimestampNanos = 900_000_000L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 5_000_000L,
                rollingShutterSkewNanos = 500_000L,
            ),
        )

        val next = engine.observe(
            CaptureTimestampIntegrityEngine.Sample(
                sensorTimestampNanos = 1_066_666_666L,
                monotonicTimestampNanos = 1_066_666_666L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 5_000_000L,
                rollingShutterSkewNanos = 500_000L,
            ),
        )

        assertTrue(invalidHost.verdict ==
            CaptureTimestampIntegrityEngine.Verdict.INVALID)
        assertTrue(engine.quality().hostOutOfOrderCount == 1)
        assertTrue(next.acceptedForCadence)
        assertTrue(next.deltaNanos != null)
        assertTrue(next.deltaNanos!! in 66_000_000L..67_000_000L)
    }

    @Test
    fun repeatedTimingFailuresExposeRecoveryBudget() {
        val engine = CaptureTimestampIntegrityEngine(
            CaptureTimestampIntegrityEngine.Config(
                maximumConsecutiveInvalidSamples = 3,
            ),
        )

        engine.observe(
            CaptureTimestampIntegrityEngine.Sample(
                sensorTimestampNanos = 1_000_000_000L,
                monotonicTimestampNanos = 1_000_000_000L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 5_000_000L,
                rollingShutterSkewNanos = 500_000L,
            ),
        )

        repeat(3) { index ->
            val bad = engine.observe(
                CaptureTimestampIntegrityEngine.Sample(
                    sensorTimestampNanos = 1_000_000_000L,
                    monotonicTimestampNanos =
                        1_000_000_000L + (index + 1) * 1_000_000L,
                    frameDurationNanos = 33_333_333L,
                    exposureTimeNanos = 5_000_000L,
                    rollingShutterSkewNanos = 500_000L,
                ),
            )
            if (index == 2) {
                assertTrue(
                    bad.warnings.any {
                        it.contains("recovery budget")
                    },
                )
            }
        }
    }

    @Test
    fun severeGapAndRollingShutterPressureAreVisible() {
        val engine = CaptureTimestampIntegrityEngine()

        engine.observe(
            CaptureTimestampIntegrityEngine.Sample(
                sensorTimestampNanos = 0L,
                monotonicTimestampNanos = 0L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 8_000_000L,
                rollingShutterSkewNanos = 1_000_000L,
            ),
        )

        val assessment = engine.observe(
            CaptureTimestampIntegrityEngine.Sample(
                sensorTimestampNanos = 200_000_000L,
                monotonicTimestampNanos = 200_000_000L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 20_000_000L,
                rollingShutterSkewNanos = 32_000_000L,
            ),
        )

        assertTrue(
            assessment.verdict ==
                CaptureTimestampIntegrityEngine.Verdict.GAP,
        )
        assertTrue(assessment.rollingShutterFraction > 0.9)
        assertTrue(assessment.exposureDutyCycle > 0.5)
    }
}

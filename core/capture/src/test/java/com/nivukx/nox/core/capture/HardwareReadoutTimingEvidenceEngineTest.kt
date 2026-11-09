package com.nivukx.nox.core.capture

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class HardwareReadoutTimingEvidenceEngineTest {

    @Test
    fun stableReadoutTimingReachesStableState() {
        val engine = HardwareReadoutTimingEvidenceEngine()
        repeat(12) { index ->
            engine.observe(
                HardwareReadoutTimingEvidenceEngine.Observation(
                    sequence = index.toLong(),
                    sensorTimestampNanos = index * 16_666_667L,
                    readoutTimestampNanos =
                        index * 16_666_667L +
                            4_000_000L,
                ),
            )
        }

        val result = engine.evaluate()
        assertEquals(
            HardwareReadoutTimingEvidenceEngine.State.STABLE,
            result.state,
        )
        assertTrue(result.confidence > 0.80)
        assertTrue(result.p95JitterMillis < 0.01)
    }

    @Test
    fun invalidReadoutSamplesAreRejected() {
        val engine = HardwareReadoutTimingEvidenceEngine()
        repeat(8) { index ->
            val sensor = index * 16_666_667L
            engine.observe(
                HardwareReadoutTimingEvidenceEngine.Observation(
                    sequence = index.toLong(),
                    sensorTimestampNanos = sensor,
                    readoutTimestampNanos =
                        if (index == 0) sensor - 1_000_000L
                        else sensor + 4_000_000L,
                ),
            )
        }

        val result = engine.evaluate()
        assertEquals(7, result.validSampleCount)
        assertEquals(1, result.invalidSampleCount)
        assertTrue(
            result.warnings.contains(
                "HARDWARE_READOUT_INVALID_SAMPLES_REJECTED",
            ),
        )
    }

    @Test
    fun unstableReadoutMovesOutOfStableState() {
        val engine = HardwareReadoutTimingEvidenceEngine()
        repeat(16) { index ->
            val delayMillis =
                if (index % 2 == 0) 0.0 else 8.0
            val sensor = index * 16_666_667L
            engine.observe(
                HardwareReadoutTimingEvidenceEngine.Observation(
                    sequence = index.toLong(),
                    sensorTimestampNanos = sensor,
                    readoutTimestampNanos =
                        sensor +
                            (delayMillis * 1_000_000.0).toLong(),
                ),
            )
        }

        val result = engine.evaluate()
        assertTrue(
            result.state ==
                HardwareReadoutTimingEvidenceEngine.State.DEGRADED ||
                result.state ==
                HardwareReadoutTimingEvidenceEngine.State.WATCH,
        )
        assertTrue(result.p95JitterMillis > 2.0)
    }
}

package com.nivukx.nox.core.sync

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

class RecordedCameraClockFitEngineTest {

    @Test
    fun fitsOffsetAndDriftFromNoisySamples() {
        val engine =
            RecordedCameraClockFitEngine()

        val offset = 4_500_000.0
        val driftPpm = 18.0
        val slope = 1.0 + driftPpm * 1e-6

        val samples =
            (0 until 64).map { index ->
                val device = 1_000_000_000L + index * 50_000_000L
                val noise =
                    when (index % 5) {
                        0 -> 80_000.0
                        1 -> -55_000.0
                        else -> 25_000.0
                    }
                val master =
                    offset +
                        device.toDouble() * slope +
                        noise

                RecordedCameraClockFitEngine.Sample(
                    deviceTimestampNanos = device,
                    masterTimestampNanos = master.toLong(),
                )
            }

        val fit =
            engine.fit(
                samples = samples,
                minimumSamples = 12,
            )

        assertTrue(fit.accepted)
        assertTrue(abs(fit.clockDriftPpm - driftPpm) < 2.5)
        assertTrue(fit.residualRmsNanos < 500_000.0)
    }
}

package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ContinuousTimeObservationRefinerTest {
    private val engine = ContinuousTimeObservationRefiner()

    private fun state(
        time: Long,
        x: Double,
        vx: Double,
    ) = ContinuousTimeObservationRefiner.State(
        timestampNanos = time,
        xMeters = x,
        yMeters = 0.0,
        zMeters = 2.0,
        vxMetersPerSecond = vx,
        vyMetersPerSecond = 0.0,
        vzMetersPerSecond = 0.0,
    )

    @Test
    fun midpointInterpolationIsStable() {
        val result = engine.refine(
            state(0L, 0.0, 1.0),
            state(100_000_000L, 0.1, 1.0),
            ContinuousTimeObservationRefiner.RowTime(
                frameTimestampNanos = 50_000_000L,
                normalizedRow = 0.5,
                readoutTimeNanos = 0L,
            ),
        )
        assertEquals(0.05, result.xMeters, 1e-6)
        assertTrue(result.temporalConfidence > 0.70)
        assertEquals(0L, result.temporalResidualNanos)
    }

    @Test
    fun rowTimeShiftsTimestamp() {
        val row = ContinuousTimeObservationRefiner.RowTime(
            frameTimestampNanos = 1_000_000_000L,
            normalizedRow = 1.0,
            readoutTimeNanos = 10_000_000L,
        )
        assertEquals(1_005_000_000L, row.timestampNanos())
    }

    @Test
    fun measurementConflictGetsRobustlyDownweighted() {
        val result = engine.refine(
            state(0L, 0.0, 1.0),
            state(100_000_000L, 0.1, 1.0),
            ContinuousTimeObservationRefiner.RowTime(
                frameTimestampNanos = 50_000_000L,
                normalizedRow = 0.5,
                readoutTimeNanos = 0L,
            ),
            measured = Triple(1.0, 1.0, 3.0),
        )
        assertTrue(result.robustWeight < 0.30)
        assertTrue(result.warnings.contains("TEMPORAL_MODEL_AND_MEASUREMENT_DISAGREE"))
    }

    @Test
    fun excessiveSupportGapIsReported() {
        val result = engine.refine(
            state(0L, 0.0, 0.0),
            state(500_000_000L, 0.1, 0.0),
            ContinuousTimeObservationRefiner.RowTime(
                frameTimestampNanos = 250_000_000L,
                normalizedRow = 0.5,
                readoutTimeNanos = 0L,
            ),
        )
        assertTrue(result.warnings.contains("SUPPORT_INTERVAL_TOO_LONG"))
    }
}

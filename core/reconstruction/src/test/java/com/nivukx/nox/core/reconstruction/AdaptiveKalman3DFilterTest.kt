package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AdaptiveKalman3DFilterTest {

    @Test
    fun firstMeasurementCreatesBoundedState() {
        val filter = AdaptiveKalman3DFilter(maximumTrackedStates = 32)
        val state = filter.update(
            id = 1,
            measurement = Vec3(1.0, 2.0, 3.0),
            measurementConfidence = 0.95,
            reprojectionErrorPx = 0.8,
            timestampNanos = 1_000_000L,
        )

        assertEquals(1, filter.activeStateCount())
        assertTrue(state.uncertaintyMeters > 0.0)
        assertTrue(state.confidence > 0.8)
    }

    @Test
    fun reversedTimestampResetsInsteadOfGeneratingHugeVelocity() {
        val filter = AdaptiveKalman3DFilter()
        filter.update(
            id = 7,
            measurement = Vec3(0.0, 0.0, 0.0),
            measurementConfidence = 1.0,
            reprojectionErrorPx = 0.5,
            timestampNanos = 2_000_000_000L,
        )
        val reset = filter.update(
            id = 7,
            measurement = Vec3(2.0, 0.0, 0.0),
            measurementConfidence = 1.0,
            reprojectionErrorPx = 0.5,
            timestampNanos = 1_000_000_000L,
        )

        assertTrue(reset.timestampReset)
        assertTrue(reset.velocity.norm() < 1e-9)
        assertEquals(1, filter.activeStateCount())
    }

    @Test
    fun stateCapacityIsBoundedAndEvictionIsDeterministic() {
        val filter = AdaptiveKalman3DFilter(maximumTrackedStates = 32)
        repeat(64) { id ->
            filter.update(
                id = id,
                measurement = Vec3(id.toDouble(), 0.0, 0.0),
                measurementConfidence = if (id == 0) 0.10 else 0.95,
                reprojectionErrorPx = 0.8,
                timestampNanos = (id + 1L) * 1_000_000L,
            )
        }

        assertEquals(32, filter.activeStateCount())
        assertTrue(filter.snapshot().keys.maxOrNull()!! <= 63)
    }

    @Test
    fun staleStatesAreReclaimedWithoutExplicitRemove() {
        val filter = AdaptiveKalman3DFilter(
            maximumTrackedStates = 32,
            staleStateTimeoutNanos = 100_000_000L,
        )
        filter.update(
            id = 4,
            measurement = Vec3(0.0, 0.0, 0.0),
            measurementConfidence = 0.9,
            reprojectionErrorPx = 1.0,
            timestampNanos = 0L,
        )

        filter.prune(200_000_000L)
        assertEquals(0, filter.activeStateCount())
    }

    @Test
    fun forecastProjectsStateForwardWithoutMutatingFilter() {
        val filter = AdaptiveKalman3DFilter()
        filter.update(
            id = 12,
            measurement = Vec3(0.0, 0.0, 0.0),
            measurementConfidence = 1.0,
            reprojectionErrorPx = 0.5,
            timestampNanos = 1_000_000_000L,
        )
        filter.update(
            id = 12,
            measurement = Vec3(0.10, 0.0, 0.0),
            measurementConfidence = 1.0,
            reprojectionErrorPx = 0.5,
            timestampNanos = 1_100_000_000L,
        )

        val before = filter.snapshot().getValue(12)
        val forecast =
            filter.forecast(
                id = 12,
                timestampNanos = 1_150_000_000L,
            )

        assertTrue(forecast != null)
        assertTrue(
            forecast!!.position.x >= before.position.x,
        )
        assertTrue(
            forecast.position.x.isFinite(),
        )
        assertTrue(
            forecast.uncertaintyMeters >= before.uncertaintyMeters,
        )

        val after = filter.snapshot().getValue(12)
        assertEquals(
            before.position,
            after.position,
        )
    }

    @Test
    fun largeInnovationIsDownWeighted() {
        val filter = AdaptiveKalman3DFilter()
        filter.update(
            id = 2,
            measurement = Vec3(0.0, 0.0, 0.0),
            measurementConfidence = 1.0,
            reprojectionErrorPx = 0.4,
            timestampNanos = 1_000_000_000L,
        )
        val state = filter.update(
            id = 2,
            measurement = Vec3(5.0, 0.0, 0.0),
            measurementConfidence = 0.25,
            reprojectionErrorPx = 6.0,
            timestampNanos = 1_016_000_000L,
        )

        assertTrue(state.measurementWeight < 1.0)
        assertTrue(state.confidence < 1.0)
        assertTrue(state.innovationMeters > 0.0)
    }
}

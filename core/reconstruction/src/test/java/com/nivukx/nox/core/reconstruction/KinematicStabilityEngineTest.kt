package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class KinematicStabilityEngineTest {

    @Test
    fun robustCubicModelTracksSmoothMotion() {
        val engine = KinematicStabilityEngine(
            KinematicStabilityEngine.Config(
                minimumRegressionSamples = 4,
            ),
        )

        var result = engine.latest()

        repeat(8) { index ->
            val t = index * 0.02
            val position = Vec3(
                0.10 * t + 0.5 * t * t,
                -0.04 * t + 0.25 * t * t,
                1.0 + 0.02 * t,
            )
            result = engine.observe(
                KinematicStabilityEngine.Observation(
                    timestampNanos = (t * 1_000_000_000.0).toLong(),
                    positionMeters = position,
                    confidence = 0.99,
                ),
            )
        }

        assertEquals(
            KinematicStabilityEngine.State.OBSERVED,
            result.state,
        )
        assertTrue(result.velocityMetersPerSecond.isFinite())
        assertTrue(result.accelerationMetersPerSecond2.isFinite())
        assertTrue(result.jerkMetersPerSecond3.isFinite())
        assertTrue(result.modelConfidence > 0.50)
        assertTrue(result.motionStability > 0.40)

        engine.close()
    }

    @Test
    fun isolatedOutlierDoesNotCreateNonFiniteState() {
        val engine = KinematicStabilityEngine()

        var timestamp = 0L
        repeat(5) {
            val position = Vec3(
                it * 0.003,
                0.0,
                1.0,
            )
            engine.observe(
                KinematicStabilityEngine.Observation(
                    timestampNanos = timestamp,
                    positionMeters = position,
                    confidence = 0.98,
                ),
            )
            timestamp += 20_000_000L
        }

        val result = engine.observe(
            KinematicStabilityEngine.Observation(
                timestampNanos = timestamp,
                positionMeters = Vec3(0.8, -0.6, 2.5),
                confidence = 0.20,
            ),
        )

        assertTrue(result.finite)
        assertTrue(result.modelConfidence <= 1.0)
        assertTrue(result.motionStability >= 0.0)

        engine.close()
    }

    @Test
    fun predictionIsBoundedAndEventuallyStales() {
        val engine = KinematicStabilityEngine(
            KinematicStabilityEngine.Config(
                maximumPredictionHorizonSeconds = 0.05,
            ),
        )

        repeat(6) { index ->
            val t = index * 0.016
            engine.observe(
                KinematicStabilityEngine.Observation(
                    timestampNanos = (t * 1_000_000_000.0).toLong(),
                    positionMeters = Vec3(
                        t,
                        0.0,
                        1.0,
                    ),
                    confidence = 1.0,
                ),
            )
        }

        val predicted = engine.predict(6 * 16_000_000L + 20_000_000L)
        assertEquals(
            KinematicStabilityEngine.State.PREDICTED,
            predicted.state,
        )
        assertTrue(predicted.positionMeters.isFinite())
        assertTrue(predicted.uncertaintyMeters >= 0.0)

        val stale = engine.predict(6 * 16_000_000L + 200_000_000L)
        assertEquals(
            KinematicStabilityEngine.State.STALE,
            stale.state,
        )
        assertTrue(stale.modelConfidence < predicted.modelConfidence)

        engine.close()
    }

    @Test
    fun timestampReversalFailsClosed() {
        val engine = KinematicStabilityEngine()

        engine.observe(
            KinematicStabilityEngine.Observation(
                timestampNanos = 1_000_000L,
                positionMeters = Vec3(0.0, 0.0, 1.0),
                confidence = 1.0,
            ),
        )
        val result = engine.observe(
            KinematicStabilityEngine.Observation(
                timestampNanos = 900_000L,
                positionMeters = Vec3(0.0, 0.0, 1.0),
                confidence = 1.0,
            ),
        )

        assertEquals(
            KinematicStabilityEngine.State.REJECTED,
            result.state,
        )
        assertTrue(
            result.warnings.any {
                it.contains("timestamp-reversal")
            },
        )

        engine.close()
    }
}

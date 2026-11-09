package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MotionAwareRecoveryEngineTest {

    @Test
    fun firstObservationCreatesAStableTrack() {
        val engine = MotionAwareRecoveryEngine()

        val result =
            engine.update(
                MotionAwareRecoveryEngine.Observation(
                    trackId = 7,
                    positionMeters = Vec3(0.0, 0.0, 2.0),
                    confidence = 0.95,
                    reprojectionErrorPx = 0.8,
                    timestampNanos = 1_000_000L,
                ),
            )

        assertEquals(
            MotionAwareRecoveryEngine.State.FUSED,
            result.state,
        )
        assertTrue(result.confidence > 0.70)
        assertEquals(1, engine.activeTrackCount())
    }

    @Test
    fun shortOcclusionUsesPredictionWithBoundedConfidence() {
        val engine =
            MotionAwareRecoveryEngine(
                MotionAwareRecoveryEngine.Config.precision(),
            )

        engine.update(
            MotionAwareRecoveryEngine.Observation(
                trackId = 3,
                positionMeters = Vec3(0.0, 0.0, 2.0),
                confidence = 0.94,
                reprojectionErrorPx = 0.9,
                timestampNanos = 1_000_000_000L,
            ),
        )

        engine.update(
            MotionAwareRecoveryEngine.Observation(
                trackId = 3,
                positionMeters = Vec3(0.04, 0.0, 2.0),
                confidence = 0.93,
                reprojectionErrorPx = 0.9,
                timestampNanos = 1_033_333_333L,
            ),
        )

        val predicted =
            engine.predict(
                trackId = 3,
                timestampNanos = 1_066_666_666L,
            )

        assertTrue(predicted != null)
        assertEquals(
            MotionAwareRecoveryEngine.State.PREDICTED,
            predicted!!.state,
        )
        assertTrue(
            predicted.confidence in 0.18..1.0,
        )
        assertTrue(
            predicted.positionMeters.isFinite(),
        )
    }

    @Test
    fun lowConfidenceMeasurementDoesNotOverwriteStableTrack() {
        val engine =
            MotionAwareRecoveryEngine(
                MotionAwareRecoveryEngine.Config(
                    minimumMeasurementConfidence = 0.30,
                ),
            )

        engine.update(
            MotionAwareRecoveryEngine.Observation(
                trackId = 21,
                positionMeters = Vec3(0.0, 0.0, 2.0),
                confidence = 0.95,
                reprojectionErrorPx = 0.7,
                timestampNanos = 1_000_000_000L,
            ),
        )
        engine.update(
            MotionAwareRecoveryEngine.Observation(
                trackId = 21,
                positionMeters = Vec3(0.10, 0.0, 2.0),
                confidence = 0.95,
                reprojectionErrorPx = 0.7,
                timestampNanos = 1_033_333_333L,
            ),
        )

        val held =
            engine.update(
                MotionAwareRecoveryEngine.Observation(
                    trackId = 21,
                    positionMeters = Vec3(8.0, 0.0, 2.0),
                    confidence = 0.05,
                    reprojectionErrorPx = 9.0,
                    timestampNanos = 1_066_666_666L,
                ),
            )

        assertEquals(
            MotionAwareRecoveryEngine.State.PREDICTED,
            held.state,
        )
        assertTrue(
            held.warnings.contains(
                "low-confidence-measurement-held-out",
            ),
        )
        assertTrue(
            held.positionMeters.x < 1.0,
        )
    }

    @Test
    fun staleTracksExpireDeterministically() {
        val engine =
            MotionAwareRecoveryEngine(
                MotionAwareRecoveryEngine.Config(
                    maximumTracks = 32,
                    maximumGapMillis = 20.0,
                    staleTimeoutMillis = 40.0,
                ),
            )

        engine.update(
            MotionAwareRecoveryEngine.Observation(
                trackId = 11,
                positionMeters = Vec3(0.0, 0.0, 1.0),
                confidence = 0.90,
                reprojectionErrorPx = 1.0,
                timestampNanos = 1_000_000L,
            ),
        )

        val prediction =
            engine.predict(
                trackId = 11,
                timestampNanos = 60_000_000L,
            )

        assertTrue(prediction == null)
        assertEquals(0, engine.activeTrackCount())
    }

    @Test
    fun trackCountRemainsBoundedUnderChurn() {
        val engine =
            MotionAwareRecoveryEngine(
                MotionAwareRecoveryEngine.Config(
                    maximumTracks = 32,
                    maximumGapMillis = 80.0,
                    staleTimeoutMillis = 120.0,
                ),
            )

        repeat(200) { index ->
            engine.update(
                MotionAwareRecoveryEngine.Observation(
                    trackId = index,
                    positionMeters =
                        Vec3(
                            index * 0.001,
                            0.0,
                            2.0,
                        ),
                    confidence = 0.82,
                    reprojectionErrorPx = 1.1,
                    timestampNanos =
                        1_000_000L +
                            index * 1_000_000L,
                ),
            )
        }

        assertTrue(
            engine.activeTrackCount() <= 32,
        )
    }

    @Test
    fun timestampReversalReinitializesInsteadOfCreatingVelocitySpike() {
        val engine = MotionAwareRecoveryEngine()

        engine.update(
            MotionAwareRecoveryEngine.Observation(
                trackId = 5,
                positionMeters = Vec3(0.0, 0.0, 2.0),
                confidence = 0.90,
                reprojectionErrorPx = 1.0,
                timestampNanos = 20_000_000L,
            ),
        )

        val reset =
            engine.update(
                MotionAwareRecoveryEngine.Observation(
                    trackId = 5,
                    positionMeters = Vec3(0.5, 0.0, 2.0),
                    confidence = 0.88,
                    reprojectionErrorPx = 1.0,
                    timestampNanos = 10_000_000L,
                ),
            )

        assertTrue(
            reset.warnings.any {
                it.contains("timestamp")
            },
        )
        assertTrue(
            reset.velocityMetersPerSecond.norm() < 10.0,
        )
    }
}

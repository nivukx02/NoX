package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EyeSide
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class Eye3DTemporalConsistencyEngineTest {

    @Test
    fun coherentBilateralEyesAreStable() {
        val engine = Eye3DTemporalConsistencyEngine()
        val result = engine.evaluate(
            input(
                timestamp = 1_000_000_000L,
                left = eye(EyeSide.LEFT, Vec3(-0.032, 0.0, 1.60), Vec3(0.0, 0.0, 1.0)),
                right = eye(EyeSide.RIGHT, Vec3(0.032, 0.0, 1.60), Vec3(0.0, 0.0, 1.0)),
            ),
        )
        assertEquals(Eye3DTemporalConsistencyEngine.State.STABLE, result.state)
        assertTrue(result.confidence > 0.65)
    }

    @Test
    fun sustainedCrossEyeDisagreementDegrades() {
        val engine = Eye3DTemporalConsistencyEngine(
            Eye3DTemporalConsistencyEngine.Config(
                persistentRejectFrames = 3,
            ),
        )
        repeat(4) { index ->
            val result = engine.evaluate(
                input(
                    timestamp = 2_000_000_000L + index * 16_000_000L,
                    left = eye(EyeSide.LEFT, Vec3(-0.032, 0.0, 1.60), Vec3(0.0, 0.0, 1.0)),
                    right = eye(EyeSide.RIGHT, Vec3(0.032, 0.0, 1.60), Vec3(0.9, 0.0, 0.435)),
                ),
            )
            if (index == 3) {
                assertEquals(
                    Eye3DTemporalConsistencyEngine.State.DEGRADED,
                    result.state,
                )
                assertTrue(result.warnings.any { it.contains("left-right") })
            }
        }
    }

    @Test
    fun triangulationParallaxContributesToEyeConfidence() {
        val lowParallaxEngine = Eye3DTemporalConsistencyEngine()
        val highParallaxEngine = Eye3DTemporalConsistencyEngine()

        val low =
            lowParallaxEngine.evaluate(
                input(
                    timestamp = 4_000_000_000L,
                    left = eye(
                        EyeSide.LEFT,
                        Vec3(-0.032, 0.0, 1.60),
                        Vec3(0.0, 0.0, 1.0),
                        parallax = 0.25,
                    ),
                    right = eye(
                        EyeSide.RIGHT,
                        Vec3(0.032, 0.0, 1.60),
                        Vec3(0.0, 0.0, 1.0),
                        parallax = 0.25,
                    ),
                ),
            )

        val high =
            highParallaxEngine.evaluate(
                input(
                    timestamp = 4_000_000_000L,
                    left = eye(
                        EyeSide.LEFT,
                        Vec3(-0.032, 0.0, 1.60),
                        Vec3(0.0, 0.0, 1.0),
                        parallax = 4.0,
                    ),
                    right = eye(
                        EyeSide.RIGHT,
                        Vec3(0.032, 0.0, 1.60),
                        Vec3(0.0, 0.0, 1.0),
                        parallax = 4.0,
                    ),
                ),
            )

        assertTrue(
            "better observed eye geometry should improve confidence",
            high.confidence > low.confidence,
        )
    }

    @Test
    fun realFastGazeCanBeReportedAsRapidInsteadOfRejected() {
        val engine = Eye3DTemporalConsistencyEngine()
        val first = eye(
            EyeSide.LEFT,
            Vec3(-0.032, 0.0, 1.60),
            Vec3(1.0, 0.0, 0.0),
            velocity = 850.0,
        )
        val firstRight = eye(
            EyeSide.RIGHT,
            Vec3(0.032, 0.0, 1.60),
            Vec3(1.0, 0.0, 0.0),
            velocity = 850.0,
        )

        val result = engine.evaluate(
            input(
                timestamp = 3_000_000_000L,
                left = first,
                right = firstRight,
            ),
        )

        assertTrue(
            result.state ==
                Eye3DTemporalConsistencyEngine.State.RAPID_GAZE ||
                result.state ==
                Eye3DTemporalConsistencyEngine.State.STABLE,
        )
        assertTrue(result.confidence > 0.50)
    }
    @Test
    fun stricterConfigurationChangesGatePolicy() {
        val engine = Eye3DTemporalConsistencyEngine()
        engine.configureStrictness(1.0)

        val result = engine.evaluate(
            input(
                timestamp = 5_000_000_000L,
                left = eye(
                    EyeSide.LEFT,
                    Vec3(-0.032, 0.0, 1.60),
                    Vec3(0.0, 0.35, 0.94),
                ),
                right = eye(
                    EyeSide.RIGHT,
                    Vec3(0.032, 0.0, 1.60),
                    Vec3(0.0, -0.35, 0.94),
                ),
            ),
        )

        assertTrue(result.confidence < 0.80)
        assertTrue(
            result.state !=
                Eye3DTemporalConsistencyEngine.State.STABLE,
        )
    }



    @Test
    fun stale_eye_frame_does_not_rewrite_bilateral_temporal_state() {
        val engine = Eye3DTemporalConsistencyEngine()

        val first = engine.evaluate(
            input(
                timestamp = 6_000_000_000L,
                left = eye(
                    EyeSide.LEFT,
                    Vec3(-0.032, 0.0, 1.60),
                    Vec3(0.0, 0.0, 1.0),
                ),
                right = eye(
                    EyeSide.RIGHT,
                    Vec3(0.032, 0.0, 1.60),
                    Vec3(0.0, 0.0, 1.0),
                ),
            ),
        )

        val stale = engine.evaluate(
            input(
                timestamp = 5_900_000_000L,
                left = eye(
                    EyeSide.LEFT,
                    Vec3(-0.032, 0.0, 1.20),
                    Vec3(1.0, 0.0, 0.0),
                    velocity = 1200.0,
                ),
                right = eye(
                    EyeSide.RIGHT,
                    Vec3(0.032, 0.0, 1.20),
                    Vec3(1.0, 0.0, 0.0),
                    velocity = 1200.0,
                ),
            ),
        )

        assertEquals(first.timestampNanos, stale.timestampNanos)
        assertTrue(
            stale.warnings.contains("stale-eye-frame-rejected"),
        )
    }

    private fun input(
        timestamp: Long,
        left: Eye3DReconstructionEngine.Result?,
        right: Eye3DReconstructionEngine.Result?,
    ) = Eye3DTemporalConsistencyEngine.Input(
        timestampNanos = timestamp,
        left = left,
        right = right,
    )

    private fun eye(
        side: EyeSide,
        center: Vec3,
        direction: Vec3,
        velocity: Double = 0.0,
        parallax: Double = 0.0,
    ) = Eye3DReconstructionEngine.Result(
        eye = side,
        centerWorld = center,
        gazeDirectionWorld = direction.normalized(),
        confidence = 0.92,
        angularSpreadDegrees = 2.0,
        observations = 4,
        centerUncertaintyMeters = 0.004,
        gazeStability = 0.92,
        angularVelocityDegreesPerSecond = velocity,
        parallaxDegrees = parallax,
    )
}

package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class KinematicSkeletonPredictionEngineTest {

    @Test
    fun predictionUsesVelocityAndInflatesUncertainty() {
        val engine = KinematicSkeletonPredictionEngine()

        val initial = skeleton(
            timestampNanos = 1_000_000_000L,
            position = Vec3(0.0, 0.0, 0.0),
            velocity = Vec3(1.0, 0.0, 0.0),
        )

        engine.observe(initial)

        val result = engine.predict(
            trackId = initial.trackId,
            targetTimestampNanos = 1_100_000_000L,
        )

        requireNotNull(result)
        assertTrue(result.predicted)
        assertEquals(0.10, result.skeleton.joints.single().position.x, 1e-6)
        assertTrue(
            result.skeleton.joints.single().positionUncertaintyMeters >=
                initial.joints.single().positionUncertaintyMeters,
        )
        assertTrue(result.confidenceScale < 1.0)
    }

    @Test
    fun horizonExceededNeverClaimsFreshPrediction() {
        val engine = KinematicSkeletonPredictionEngine(
            KinematicSkeletonPredictionEngine.Config(
                maximumPredictionHorizonMillis = 50L,
            ),
        )

        val initial = skeleton(
            timestampNanos = 1_000_000_000L,
            position = Vec3(1.0, 2.0, 3.0),
            velocity = Vec3(0.0, 0.0, 0.0),
        )

        engine.observe(initial)

        val result = engine.predict(
            trackId = initial.trackId,
            targetTimestampNanos = 1_100_000_000L,
        )

        requireNotNull(result)
        assertTrue(result.predicted)
        assertTrue(
            result.warnings.contains(
                "ACTOR_3D_PREDICTION_HORIZON_EXCEEDED",
            ),
        )
        assertEquals(initial, result.skeleton)
    }

    @Test
    fun resetRemovesActorState() {
        val engine = KinematicSkeletonPredictionEngine()
        val initial = skeleton(
            timestampNanos = 1_000_000_000L,
            position = Vec3(0.0, 0.0, 0.0),
            velocity = Vec3(0.0, 0.0, 0.0),
        )

        engine.observe(initial)
        assertTrue(engine.trackedActorIds().contains(initial.trackId))

        engine.reset()
        assertTrue(engine.trackedActorIds().isEmpty())
    }

    private fun skeleton(
        timestampNanos: Long,
        position: Vec3,
        velocity: Vec3,
    ) =
        SkeletonFrame(
            timestampNanos = timestampNanos,
            subjectType = SubjectType.HUMAN,
            trackId = 7L,
            joints = listOf(
                ReconstructionJoint(
                    landmarkId = 0,
                    position = position,
                    velocity = velocity,
                    confidence = 0.95,
                    reprojectionErrorPx = 0.5,
                    observationCount = 4,
                    positionUncertaintyMeters = 0.004,
                ),
            ),
            globalConfidence = 0.92,
        )
}

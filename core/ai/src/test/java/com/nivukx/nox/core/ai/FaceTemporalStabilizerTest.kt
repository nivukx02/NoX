package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.FaceMeshPoint
import com.nivukx.nox.core.model.Vec2
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceTemporalStabilizerTest {
    @Test
    fun reduces_single_frame_landmark_jump() {
        val stabilizer = FaceTemporalStabilizer()

        stabilizer.refine(face(1_000_000_000L, 100.0))
        stabilizer.refine(face(1_033_333_333L, 104.0))
        val refined = stabilizer.refine(
            face(1_066_666_666L, 420.0),
        )

        val x = refined.points.first().point.x
        assertTrue(x < 420.0)
        assertTrue(x > 100.0)
    }


    @Test
    fun accelerated_face_motion_uses_bounded_second_order_prediction() {
        val stabilizer = FaceTemporalStabilizer()

        stabilizer.refine(face(3_000_000_000L, 100.0))
        stabilizer.refine(face(3_033_333_333L, 104.0))
        val accelerated =
            stabilizer.refine(
                face(3_066_666_666L, 112.0),
            )

        val point = accelerated.points.first().point
        assertTrue(point.x.isFinite())
        assertTrue(point.y.isFinite())
        assertTrue(point.x > 108.0)
        assertTrue(point.x < 116.0)
    }


    @Test
    fun third_order_face_prediction_remains_finite_under_acceleration() {
        val stabilizer = FaceTemporalStabilizer()

        stabilizer.refine(face(4_000_000_000L, 100.0))
        stabilizer.refine(face(4_033_333_333L, 104.0))
        stabilizer.refine(face(4_066_666_666L, 112.0))
        val predicted =
            stabilizer.refine(
                face(
                    4_099_999_999L,
                    121.0,
                ),
            )

        val point =
            predicted.points.first().point
        assertTrue(point.x.isFinite())
        assertTrue(point.y.isFinite())
        assertTrue(point.x > 118.0)
        assertTrue(point.x < 124.0)
    }

    @Test
    fun stale_face_frame_does_not_rewrite_temporal_landmark_state() {
        val stabilizer = FaceTemporalStabilizer()

        stabilizer.refine(face(2_000_000_000L, 100.0))
        val current = stabilizer.refine(
            face(2_033_333_333L, 106.0),
        )
        val stale = stabilizer.refine(
            face(1_900_000_000L, 420.0),
        )

        assertTrue(
            stale.points.first().point.x ==
                current.points.first().point.x,
        )
    }

    private fun face(
        timestamp: Long,
        x: Double,
    ) = FaceFrame(
        timestampNanos = timestamp,
        trackingId = 7,
        boundsLeft = 70f,
        boundsTop = 60f,
        boundsRight = 220f,
        boundsBottom = 240f,
        yawDegrees = 0f,
        rollDegrees = 0f,
        points = listOf(
            FaceMeshPoint(
                index = 0,
                point = Vec2(x, 120.0),
                confidence = 0.95,
            ),
        ),
        leftEyeCenter = Vec2(x - 25.0, 110.0),
        rightEyeCenter = Vec2(x + 25.0, 110.0),
        leftEyeOpenProbability = 1.0,
        rightEyeOpenProbability = 1.0,
    )
}

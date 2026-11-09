package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.EyeObservation
import com.nivukx.nox.core.model.EyeSide
import com.nivukx.nox.core.model.EyeTrackingFrame
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EyeWireCodecTest {
    @Test
    fun roundTripsBothEyes() {
        val frame = EyeTrackingFrame(
            timestampNanos = 123456789L,
            trackingId = 9,
            left = EyeObservation(
                eye = EyeSide.LEFT,
                center = Vec2(100.0, 120.0),
                irisCenter = Vec2(101.0, 121.0),
                normalizedOffset = Vec2(0.12, -0.08),
                openness = 0.91,
                blink = false,
                confidence = 0.94,
                gazeDirection = Vec3(0.12, -0.05, 0.99).normalized(),
            ),
            right = EyeObservation(
                eye = EyeSide.RIGHT,
                center = Vec2(200.0, 121.0),
                irisCenter = Vec2(201.0, 122.0),
                normalizedOffset = Vec2(-0.10, -0.06),
                openness = 0.87,
                blink = false,
                confidence = 0.92,
            ),
            combinedGaze = Vec3(0.02, -0.01, 0.999),
            fixationConfidence = 0.93,
        )

        val decoded = EyeWireCodec.decode(EyeWireCodec.encode(frame))

        assertEquals(frame.timestampNanos, decoded.timestampNanos)
        assertEquals(frame.trackingId, decoded.trackingId)
        assertTrue(decoded.left != null)
        assertTrue(decoded.right != null)
        assertEquals(frame.left!!.irisCenter!!.x, decoded.left!!.irisCenter!!.x, 1e-4)
        assertEquals(frame.right!!.confidence, decoded.right!!.confidence, 1e-4)
        assertEquals(frame.left!!.gazeDirection.x, decoded.left!!.gazeDirection.x, 1e-4)
        assertEquals(
            frame.combinedGaze.z,
            decoded.combinedGaze.z,
            1e-4,
        )
    }
}

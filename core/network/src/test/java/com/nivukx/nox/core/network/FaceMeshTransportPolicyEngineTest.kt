package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.FaceMeshPoint3D
import com.nivukx.nox.core.model.FaceMeshTriangle
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceMeshTransportPolicyEngineTest {

    private fun frame(
        timestampNanos: Long,
        centerX: Float = 100f,
        centerY: Float = 100f,
        scale: Float = 1f,
        trackingId: Int? = 1,
    ): FaceMeshFrame {
        val points = listOf(
            FaceMeshPoint3D(
                0,
                Vec3(
                    centerX.toDouble(),
                    centerY.toDouble(),
                    0.0,
                ),
            ),
            FaceMeshPoint3D(
                1,
                Vec3(
                    (centerX + 10f * scale).toDouble(),
                    centerY.toDouble(),
                    0.2,
                ),
            ),
            FaceMeshPoint3D(
                2,
                Vec3(
                    centerX.toDouble(),
                    (centerY + 10f * scale).toDouble(),
                    0.4,
                ),
            ),
        )

        return FaceMeshFrame(
            timestampNanos = timestampNanos,
            imageWidth = 1920,
            imageHeight = 1080,
            rotationDegrees = 0,
            trackingId = trackingId,
            boundsLeft = centerX - 50f * scale,
            boundsTop = centerY - 50f * scale,
            boundsRight = centerX + 50f * scale,
            boundsBottom = centerY + 50f * scale,
            confidence = 0.95,
            pointCoverage = 0.9,
            triangleCoverage = 0.9,
            points = points,
            triangles = listOf(FaceMeshTriangle(0, 1, 2)),
        )
    }

    @Test
    fun firstFrameIsAlwaysKeyframe() {
        val engine = FaceMeshTransportPolicyEngine()
        val decision = engine.evaluate(frame(1_000_000_000L), 2)

        assertTrue(decision.send)
        assertTrue(decision.keyframe)
        assertEquals("initial-keyframe", decision.reason)
    }

    @Test
    fun stableSurfaceIsSuppressedBeforeInterval() {
        val engine = FaceMeshTransportPolicyEngine()
        engine.evaluate(frame(1_000_000_000L), 4)

        val decision =
            engine.evaluate(
                frame(1_010_000_000L),
                4,
            )

        assertFalse(decision.send)
        assertEquals("interval-gate", decision.reason)
    }

    @Test
    fun surfaceChangeForcesUpdateAfterInterval() {
        val engine = FaceMeshTransportPolicyEngine(
            deformationThresholdPixels = 5.0,
            centerThresholdPixels = 2.0,
        )
        engine.evaluate(frame(1_000_000_000L), 1)

        val decision =
            engine.evaluate(
                frame(
                    timestampNanos = 1_100_000_000L,
                    centerX = 103f,
                ),
                1,
            )

        assertTrue(decision.send)
        assertTrue(!decision.keyframe)
        assertEquals("surface-change", decision.reason)
    }

    @Test
    fun cumulativeSurfaceDriftTriggersAgainstLastSentFrame() {
        val engine =
            FaceMeshTransportPolicyEngine(
                minimumConfidence = 0.10,
                deformationThresholdPixels = 3.0,
                centerThresholdPixels = 3.0,
            )

        engine.evaluate(
            frame(
                timestampNanos = 1_000_000_000L,
            ),
            1,
        )

        engine.evaluate(
            frame(
                timestampNanos = 1_100_000_000L,
                centerX = 102f,
                scale = 1f,
            ),
            1,
        )

        val decision =
            engine.evaluate(
                frame(
                    timestampNanos = 1_200_000_000L,
                    centerX = 108f,
                    scale = 1f,
                ),
                1,
            )

        assertTrue(decision.send)
        assertTrue(
            decision.deformationPixels >= 3.0 ||
                decision.centerDeltaPixels >= 3.0,
        )
    }

    @Test
    fun trackChangeForcesKeyframe() {
        val engine = FaceMeshTransportPolicyEngine()
        engine.evaluate(
            frame(
                timestampNanos = 1_000_000_000L,
                trackingId = 3,
            ),
            2,
        )

        val decision =
            engine.evaluate(
                frame(
                    timestampNanos = 1_100_000_000L,
                    trackingId = 4,
                ),
                2,
            )

        assertTrue(decision.send)
        assertTrue(decision.keyframe)
        assertEquals("track-change", decision.reason)
    }

    @Test
    fun lowConfidenceIsDroppedWithoutMutatingSendState() {
        val engine = FaceMeshTransportPolicyEngine()
        val first = engine.evaluate(
            frame(1_000_000_000L),
            2,
        )
        assertTrue(first.send)

        val low =
            frame(1_020_000_000L).copy(
                confidence = 0.05,
            )

        val decision =
            engine.evaluate(
                low,
                2,
            )

        assertFalse(decision.send)

        val next =
            engine.evaluate(
                frame(1_030_000_000L),
                2,
            )

        assertFalse(next.send)
    }

    @Test
    fun peerTopologyChangeForcesKeyframeForNewCamera() {
        val engine = FaceMeshTransportPolicyEngine(
            minimumConfidence = 0.20,
        )

        engine.evaluate(
            frame(1_000_000_000L),
            peerCount = 1,
            peerTopologyKey = "peer-a",
        )

        val decision =
            engine.evaluate(
                frame(1_100_000_000L),
                peerCount = 2,
                peerTopologyKey = "peer-a|peer-b",
            )

        assertTrue(decision.send)
        assertTrue(decision.keyframe)
        assertEquals("peer-topology-change", decision.reason)
    }

    @Test
    fun resetRestoresInitialKeyframeBehavior() {
        val engine = FaceMeshTransportPolicyEngine()
        engine.evaluate(frame(1_000_000_000L), 2)
        engine.reset()

        val decision =
            engine.evaluate(
                frame(2_000_000_000L),
                2,
            )

        assertTrue(decision.send)
        assertTrue(decision.keyframe)
    }
}

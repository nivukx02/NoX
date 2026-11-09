package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MonocularDepthFusionEngineTest {

    private val intrinsics = CameraIntrinsics(
        fx = 900.0,
        fy = 900.0,
        cx = 960.0,
        cy = 540.0,
        imageWidth = 1920,
        imageHeight = 1080,
    )

    @Test
    fun reconstructionProducesFiniteMetricGeometryWithCalibratedScale() {
        val topology = DynamicTopologyEngine().build(
            DynamicTopologyEngine.BuildInput(
                timestampNanos = 1_000_000_000L,
                subjectType = SubjectType.HUMAN,
                trackId = 1L,
                nodes = listOf(
                    DynamicTopologyEngine.SemanticNode(11, "left_shoulder", Vec2(0.42, 0.40), 0.95),
                    DynamicTopologyEngine.SemanticNode(12, "right_shoulder", Vec2(0.58, 0.40), 0.95),
                    DynamicTopologyEngine.SemanticNode(23, "left_hip", Vec2(0.45, 0.62), 0.95),
                    DynamicTopologyEngine.SemanticNode(24, "right_hip", Vec2(0.55, 0.62), 0.95),
                ),
            ),
        )

        val result = MonocularDepthFusionEngine().reconstruct(
            timestampNanos = 1_000_000_000L,
            subjectType = SubjectType.HUMAN,
            trackId = 1L,
            imageWidth = 1920,
            imageHeight = 1080,
            intrinsics = intrinsics,
            observations = listOf(
                MonocularDepthFusionEngine.Observation(11, Vec2(0.42, 0.40), 0.95),
                MonocularDepthFusionEngine.Observation(12, Vec2(0.58, 0.40), 0.95),
                MonocularDepthFusionEngine.Observation(23, Vec2(0.45, 0.62), 0.95),
                MonocularDepthFusionEngine.Observation(24, Vec2(0.55, 0.62), 0.95),
            ),
            topology = topology,
            calibratedScaleMeters = 1.72,
        )

        assertEquals(4, result.frame.joints.size)
        assertTrue(result.frame.rootPositionMeters.x.isFinite())
        assertTrue(result.frame.rootPositionMeters.y.isFinite())
        assertTrue(result.frame.rootPositionMeters.z.isFinite())
        assertTrue(result.frame.meanUncertaintyMeters.isFinite())
        assertEquals(1.72, result.frame.globalScaleMeters, 1e-9)
    }

    @Test
    fun secondFrameUsesTemporalContinuity() {
        val engine = MonocularDepthFusionEngine()
        val topology = DynamicTopologyEngine().build(
            DynamicTopologyEngine.BuildInput(
                timestampNanos = 1L,
                subjectType = SubjectType.HUMAN,
                trackId = 2L,
                nodes = listOf(
                    DynamicTopologyEngine.SemanticNode(23, "left_hip", Vec2(0.45, 0.62), 0.9),
                    DynamicTopologyEngine.SemanticNode(24, "right_hip", Vec2(0.55, 0.62), 0.9),
                ),
            ),
        )

        val first = engine.reconstruct(
            1_000_000_000L, SubjectType.HUMAN, 2L, 1920, 1080, intrinsics,
            listOf(
                MonocularDepthFusionEngine.Observation(23, Vec2(0.45, 0.62), 0.9),
                MonocularDepthFusionEngine.Observation(24, Vec2(0.55, 0.62), 0.9),
            ),
            topology,
            calibratedScaleMeters = 1.72,
        )

        val second = engine.reconstruct(
            1_016_666_666L, SubjectType.HUMAN, 2L, 1920, 1080, intrinsics,
            listOf(
                MonocularDepthFusionEngine.Observation(23, Vec2(0.451, 0.621), 0.9),
                MonocularDepthFusionEngine.Observation(24, Vec2(0.551, 0.621), 0.9),
            ),
            topology,
            previous = first.frame,
            calibratedScaleMeters = 1.72,
        )

        assertTrue(second.frame.temporalStability > 0.0)
    }
}

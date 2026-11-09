package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DynamicTopology
import com.nivukx.nox.core.model.MotionEdge
import com.nivukx.nox.core.model.MotionNode
import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.ScaleSource
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TopologyKind
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class WorldSpaceMotionConsistencyEngineTest {

    @Test
    fun stableTrajectoryPreservesMotionAndTopology() {
        val engine = WorldSpaceMotionConsistencyEngine()

        val topology = topology()
        val first = frame(
            timestampNanos = 1_000_000_000L,
            root = Vec3(0.0, 0.0, 2.0),
            child = Vec3(0.0, 1.0, 2.0),
        )

        val second = frame(
            timestampNanos = 1_016_666_667L,
            root = Vec3(0.001, 0.0, 2.002),
            child = Vec3(0.001, 1.0, 2.002),
        )

        engine.optimize(
            WorldSpaceMotionConsistencyEngine.Input(
                frame = first,
                topology = topology,
            ),
        )

        val result = engine.optimize(
            WorldSpaceMotionConsistencyEngine.Input(
                frame = second,
                topology = topology,
            ),
        )

        assertEquals(SubjectType.HUMAN, result.skeleton.subjectType)
        assertEquals(7L, result.skeleton.trackId)
        assertTrue(result.score > 0.50)
        assertTrue(result.temporalStability > 0.35)
        assertTrue(result.topologyStability > 0.50)
        assertTrue(result.motionContinuity > 0.35)
    }

    @Test
    fun largeLowConfidenceInnovationIsRejected() {
        val engine = WorldSpaceMotionConsistencyEngine()

        val topology = topology()
        val first = frame(
            timestampNanos = 2_000_000_000L,
            root = Vec3(0.0, 0.0, 2.0),
            child = Vec3(0.0, 1.0, 2.0),
        )

        val outlier = frame(
            timestampNanos = 2_016_666_667L,
            root = Vec3(1.5, 0.0, 2.0),
            child = Vec3(1.5, 1.0, 2.0),
            confidence = 0.18,
        )

        engine.optimize(
            WorldSpaceMotionConsistencyEngine.Input(
                frame = first,
                topology = topology,
            ),
        )

        val result = engine.optimize(
            WorldSpaceMotionConsistencyEngine.Input(
                frame = outlier,
                topology = topology,
            ),
        )

        assertTrue(result.rejectedJointCount >= 1)
        assertTrue(result.diagnostics.any { it.rejected })
        assertTrue(
            result.warnings.any {
                it.contains("outliers")
            },
        )
    }

    @Test
    fun poorTimingAndCameraQualityReduceReliability() {
        val engine = WorldSpaceMotionConsistencyEngine()

        val result = engine.optimize(
            WorldSpaceMotionConsistencyEngine.Input(
                frame = frame(
                    timestampNanos = 3_000_000_000L,
                    root = Vec3(0.0, 0.0, 2.0),
                    child = Vec3(0.0, 1.0, 2.0),
                ),
                topology = topology(),
                captureTimingConfidence = 0.25,
                lightingConfidence = 0.25,
                cameraMotionRisk = 0.85,
            ),
        )

        assertTrue(result.score < 0.90)
        assertTrue(
            result.warnings.any {
                it.contains("runtime")
            } || result.temporalStability < 0.90,
        )
    }


    @Test
    fun stale_world_space_frame_is_rejected_without_state_rewind() {
        val engine = WorldSpaceMotionConsistencyEngine()
        val topology = topology()

        val first = frame(
            timestampNanos = 4_000_000_000L,
            root = Vec3(0.0, 0.0, 2.0),
            child = Vec3(0.0, 1.0, 2.0),
        )

        val current = engine.optimize(
            WorldSpaceMotionConsistencyEngine.Input(
                frame = first,
                topology = topology,
            ),
        )

        val stale = engine.optimize(
            WorldSpaceMotionConsistencyEngine.Input(
                frame = frame(
                    timestampNanos = 3_900_000_000L,
                    root = Vec3(1.0, 0.0, 2.0),
                    child = Vec3(1.0, 1.0, 2.0),
                    confidence = 0.20,
                ),
                topology = topology,
            ),
        )

        assertTrue(stale.staleFrameRejected)
        assertEquals(current.skeleton, stale.skeleton)
        assertTrue(
            stale.warnings.contains("stale-frame-rejected"),
        )
        assertTrue(
            stale.warnings.contains("timestamp-reversal"),
        )
    }

    private fun frame(
        timestampNanos: Long,
        root: Vec3,
        child: Vec3,
        confidence: Double = 0.94,
    ): SkeletonFrame =
        SkeletonFrame(
            timestampNanos = timestampNanos,
            subjectType = SubjectType.HUMAN,
            trackId = 7L,
            joints = listOf(
                ReconstructionJoint(
                    landmarkId = 0,
                    position = root,
                    velocity = Vec3(0.0, 0.0, 0.12),
                    confidence = confidence,
                    reprojectionErrorPx = 1.5,
                    observationCount = 2,
                    positionUncertaintyMeters = 0.02,
                ),
                ReconstructionJoint(
                    landmarkId = 1,
                    position = child,
                    velocity = Vec3(0.0, 0.0, 0.12),
                    confidence = confidence,
                    reprojectionErrorPx = 1.5,
                    observationCount = 2,
                    positionUncertaintyMeters = 0.02,
                ),
            ),
            globalConfidence = confidence,
        )

    private fun topology(): DynamicTopology =
        DynamicTopology(
            topologyId = "test-human",
            kind = TopologyKind.HUMAN,
            nodes = listOf(
                MotionNode(
                    id = 0,
                    semanticName = "root",
                    position2D = Vec2(0.5, 0.5),
                    position3D = Vec3(0.0, 0.0, 2.0),
                    confidence = 0.95,
                    uncertaintyMeters = 0.02,
                    predicted = false,
                    visible = true,
                ),
                MotionNode(
                    id = 1,
                    semanticName = "head",
                    position2D = Vec2(0.5, 0.25),
                    position3D = Vec3(0.0, 1.0, 2.0),
                    confidence = 0.95,
                    uncertaintyMeters = 0.02,
                    predicted = false,
                    visible = true,
                ),
            ),
            edges = listOf(
                MotionEdge(
                    a = 0,
                    b = 1,
                    preferredLengthMeters = 1.0,
                    minimumLengthMeters = 0.80,
                    maximumLengthMeters = 1.20,
                    stiffness = 0.90,
                    semantic = "spine",
                ),
            ),
            rootNodeId = 0,
            scaleSource = ScaleSource.CALIBRATED_METRIC,
            globalConfidence = 0.96,
            temporalStability = 0.92,
            topologyStability = 0.95,
            timestampNanos = 1_000_000_000L,
        )
}

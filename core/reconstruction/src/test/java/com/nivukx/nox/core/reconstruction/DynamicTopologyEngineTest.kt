package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.LandmarkObservation
import com.nivukx.nox.core.model.LandmarkSpace
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TopologyKind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DynamicTopologyEngineTest {

    @Test
    fun humanPoseBuildsSemanticTopology() {
        val observations = listOf(
            LandmarkObservation(11, 0.40, 0.40, 0.95, 1L, LandmarkSpace.NORMALIZED),
            LandmarkObservation(12, 0.60, 0.40, 0.95, 1L, LandmarkSpace.NORMALIZED),
            LandmarkObservation(13, 0.34, 0.52, 0.92, 1L, LandmarkSpace.NORMALIZED),
            LandmarkObservation(14, 0.66, 0.52, 0.92, 1L, LandmarkSpace.NORMALIZED),
            LandmarkObservation(15, 0.29, 0.64, 0.90, 1L, LandmarkSpace.NORMALIZED),
            LandmarkObservation(16, 0.71, 0.64, 0.90, 1L, LandmarkSpace.NORMALIZED),
            LandmarkObservation(23, 0.44, 0.65, 0.95, 1L, LandmarkSpace.NORMALIZED),
            LandmarkObservation(24, 0.56, 0.65, 0.95, 1L, LandmarkSpace.NORMALIZED),
        )

        val topology = DynamicTopologyEngine().buildFromPose(
            timestampNanos = 1L,
            subjectType = SubjectType.HUMAN,
            trackId = 7L,
            observations = observations,
        )

        assertEquals(TopologyKind.HUMAN, topology.kind)
        assertEquals(23, topology.rootNodeId)
        assertTrue(topology.edges.any { it.semantic == "left_upper_arm" })
        assertTrue(topology.globalConfidence > 0.5)
    }

    @Test
    fun topologyPersistsEdgeStructureAcrossPartialOcclusion() {
        val engine = DynamicTopologyEngine()
        val first = engine.buildFromPose(
            timestampNanos = 1L,
            subjectType = SubjectType.HUMAN,
            trackId = 9L,
            observations = listOf(
                LandmarkObservation(11, 0.40, 0.40, 1.0, 1L, LandmarkSpace.NORMALIZED),
                LandmarkObservation(13, 0.34, 0.50, 1.0, 1L, LandmarkSpace.NORMALIZED),
                LandmarkObservation(15, 0.29, 0.60, 1.0, 1L, LandmarkSpace.NORMALIZED),
            ),
        )

        val second = engine.buildFromPose(
            timestampNanos = 2L,
            subjectType = SubjectType.HUMAN,
            trackId = 9L,
            observations = listOf(
                LandmarkObservation(11, 0.41, 0.41, 0.92, 2L, LandmarkSpace.NORMALIZED),
                LandmarkObservation(15, 0.31, 0.61, 0.88, 2L, LandmarkSpace.NORMALIZED),
            ),
            previous = first,
        )

        assertTrue(second.topologyStability > 0.3)
        assertTrue(second.timestampNanos > first.timestampNanos)
    }
}

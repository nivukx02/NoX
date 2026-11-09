package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DynamicTopology
import com.nivukx.nox.core.model.MonocularReconstructionFrame
import com.nivukx.nox.core.model.MonocularReconstructionJoint
import com.nivukx.nox.core.model.ScaleSource
import com.nivukx.nox.core.model.ReconstructionConfidenceBand
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TopologyKind
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class OcclusionRecoveryEngineTest {

    @Test
    fun missingJointIsPredictedFromPreviousVelocity() {
        val previous = frame(
            timestamp = 1_000_000_000L,
            joints = listOf(
                joint(
                    id = 11,
                    position = Vec3(0.0, 0.3, 2.0),
                    velocity = Vec3(0.4, 0.0, 0.0),
                ),
                joint(
                    id = 13,
                    position = Vec3(-0.3, 0.1, 2.0),
                    velocity = Vec3(0.4, 0.0, 0.0),
                ),
            ),
        )

        val current = frame(
            timestamp = 1_033_333_333L,
            joints = listOf(
                joint(
                    id = 11,
                    position = Vec3(0.01, 0.3, 2.0),
                    velocity = Vec3(0.4, 0.0, 0.0),
                ),
            ),
        )

        val topology = DynamicTopology(
            topologyId = "test",
            kind = TopologyKind.HUMAN,
            nodes = current.joints.map {
                com.nivukx.nox.core.model.MotionNode(
                    id = it.landmarkId,
                    semanticName = "node",
                    position2D = Vec2(0.0, 0.0),
                    position3D = it.positionCameraMeters,
                    velocity3D = it.velocityCameraMetersPerSecond,
                    confidence = it.confidence,
                    uncertaintyMeters = it.uncertaintyMeters,
                    predicted = false,
                    visible = true,
                )
            },
            edges = listOf(
                com.nivukx.nox.core.model.MotionEdge(
                    a = 11,
                    b = 13,
                    preferredLengthMeters = 0.32,
                    minimumLengthMeters = 0.15,
                    maximumLengthMeters = 0.55,
                    stiffness = 0.8,
                ),
            ),
            rootNodeId = 11,
            scaleSource = ScaleSource.CALIBRATED_METRIC,
            globalConfidence = 0.9,
            temporalStability = 0.8,
            topologyStability = 0.9,
            timestampNanos = 1_033_333_333L,
        )

        val result = OcclusionRecoveryEngine().recover(
            current = current,
            topology = topology,
            previous = previous,
        )

        assertTrue(result.recoveredJointCount >= 1)
        assertTrue(
            result.frame.joints.any {
                it.landmarkId == 13 && it.predicted
            },
        )
    }

    private fun frame(
        timestamp: Long,
        joints: List<MonocularReconstructionJoint>,
    ) =
        MonocularReconstructionFrame(
            timestampNanos = timestamp,
            subjectType = SubjectType.HUMAN,
            trackId = 1L,
            joints = joints,
            rootPositionMeters = Vec3(0.0, 0.0, 2.0),
            globalScaleMeters = 1.72,
            scaleSource = ScaleSource.CALIBRATED_METRIC,
            scaleConfidence = 0.95,
            globalConfidence = 0.9,
            reprojectionErrorPx = 1.0,
            meanUncertaintyMeters = 0.06,
            temporalStability = 0.9,
            confidenceBand = ReconstructionConfidenceBand.GOOD,
        )

    private fun joint(
        id: Int,
        position: Vec3,
        velocity: Vec3,
    ) =
        MonocularReconstructionJoint(
            landmarkId = id,
            semanticName = "joint-$id",
            positionCameraMeters = position,
            velocityCameraMetersPerSecond = velocity,
            accelerationCameraMetersPerSecond2 = Vec3(0.0, 0.0, 0.0),
            confidence = 0.95,
            uncertaintyMeters = 0.05,
            depthMeters = position.z,
            depthConfidence = 0.9,
            predicted = false,
        )
}

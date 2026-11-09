package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiViewGeometricObservabilityEngineTest {

    private val intrinsics = CameraIntrinsics(
        fx = 1000.0,
        fy = 1000.0,
        cx = 500.0,
        cy = 500.0,
        imageWidth = 1000,
        imageHeight = 1000,
    )

    @Test
    fun healthyWideBaselineProducesAcceptedGeometry() {
        val engine = MultiViewGeometricObservabilityEngine()
        val result = engine.evaluate(
            MultiViewGeometricObservabilityEngine.Input(
                timestampNanos = 1_000_000_000L,
                skeleton = skeleton(),
                observationsByJoint = mapOf(
                    0 to listOf(
                        observation("A", Vec2(500.0, 500.0), 0.0),
                        observation("B", Vec2(325.0, 500.0), 0.50),
                    ),
                ),
                synchronizedCameraCount = 2,
            ),
        )

        assertTrue(result.score > 0.55)
        assertTrue(result.accepted)
        assertTrue(
            result.medianTriangulationAngleDegrees > 5.0,
        )
        assertTrue(
            result.medianInformationConditionScore > 0.10,
        )
        assertTrue(
            result.p95InformationSigmaMajor95Meters.isFinite(),
        )
        assertTrue(
            result.p95InformationSigmaDepth95Meters.isFinite(),
        )
    }

    @Test
    fun tinyBaselineProducesWeakObservability() {
        val engine = MultiViewGeometricObservabilityEngine()
        val result = engine.evaluate(
            MultiViewGeometricObservabilityEngine.Input(
                timestampNanos = 2_000_000_000L,
                skeleton = skeleton(),
                observationsByJoint = mapOf(
                    0 to listOf(
                        observation("A", Vec2(500.0, 500.0), 0.0),
                        observation("B", Vec2(495.0, 500.0), 0.01),
                    ),
                ),
                synchronizedCameraCount = 2,
            ),
        )

        assertTrue(result.score < 0.75)
        assertTrue(result.warnings.isNotEmpty())
        assertTrue(
            result.medianInformationConditionScore <
                0.95,
        )
    }

    @Test
    fun singleViewFailsClosed() {
        val engine = MultiViewGeometricObservabilityEngine()
        val result = engine.evaluate(
            MultiViewGeometricObservabilityEngine.Input(
                timestampNanos = 3_000_000_000L,
                skeleton = skeleton(),
                observationsByJoint = mapOf(
                    0 to listOf(
                        observation("A", Vec2(500.0, 500.0), 0.0),
                    ),
                ),
                synchronizedCameraCount = 1,
            ),
        )

        assertTrue(!result.accepted)
    }

    private fun skeleton() =
        SkeletonFrame(
            timestampNanos = 1_000_000_000L,
            subjectType = SubjectType.HUMAN,
            trackId = 1L,
            joints = listOf(
                ReconstructionJoint(
                    landmarkId = 0,
                    position = Vec3(0.0, 0.0, 2.0),
                    velocity = Vec3(0.0, 0.0, 0.0),
                    confidence = 0.98,
                    reprojectionErrorPx = 1.0,
                    observationCount = 2,
                    positionUncertaintyMeters = 0.01,
                ),
            ),
            globalConfidence = 0.96,
        )

    private fun observation(
        device: String,
        point: Vec2,
        x: Double,
    ) =
        CameraObservation(
            deviceId = DeviceId(device),
            point = point,
            confidence = 0.95,
            intrinsics = intrinsics,
            pose = Pose(
                position = Vec3(x, 0.0, 0.0),
                rotation = Quat.IDENTITY,
            ),
        )
}

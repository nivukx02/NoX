package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraExtrinsics
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiViewReprojectionConsistencyEngineTest {

    private val intrinsics = CameraIntrinsics(
        fx = 1000.0,
        fy = 1000.0,
        cx = 500.0,
        cy = 500.0,
        imageWidth = 1000,
        imageHeight = 1000,
    )

    @Test
    fun consistentViewsProduceStrongConsensus() {
        val engine = MultiViewReprojectionConsistencyEngine()

        val skeleton = skeleton(
            position = Vec3(0.0, 0.0, 2.0),
            velocity = Vec3(0.0, 0.0, 0.0),
        )

        val frames = listOf(
            frame("A", 1_000_000_000L, Vec2(500.0, 500.0)),
            frame("B", 1_000_000_000L, Vec2(250.0, 500.0)),
        )

        val result = engine.evaluate(
            MultiViewReprojectionConsistencyEngine.Input(
                skeleton = skeleton,
                cameraFrames = frames,
                targetTimestampNanos = 1_000_000_000L,
            ),
        )

        assertTrue(result.score > 0.75)
        assertTrue(result.diagnostics.single().accepted)
        assertTrue(result.diagnostics.single().acceptedViews == 2)
        assertTrue(result.medianResidualPx < 0.1)
    }

    @Test
    fun singleViewOutlierIsRejectedWithoutDiscardingGoodViews() {
        val engine = MultiViewReprojectionConsistencyEngine()

        val skeleton = skeleton(
            position = Vec3(0.0, 0.0, 2.0),
            velocity = Vec3(0.0, 0.0, 0.0),
        )

        val frames = listOf(
            frame("A", 2_000_000_000L, Vec2(500.0, 500.0)),
            frame("B", 2_000_000_000L, Vec2(250.0, 500.0)),
            frame("C", 2_000_000_000L, Vec2(800.0, 800.0)),
        )

        val result = engine.evaluate(
            MultiViewReprojectionConsistencyEngine.Input(
                skeleton = skeleton,
                cameraFrames = frames,
                targetTimestampNanos = 2_000_000_000L,
            ),
        )

        val joint = result.diagnostics.single()
        assertTrue(joint.acceptedViews >= 2)
        assertTrue(joint.rejectedViews >= 1)
        assertTrue(joint.medianResidualPx < 0.1)
    }

    @Test
    fun velocityCompensationImprovesTimestampSkewResidual() {
        val engine = MultiViewReprojectionConsistencyEngine()

        val skeleton = skeleton(
            position = Vec3(0.0, 0.0, 2.0),
            velocity = Vec3(1.0, 0.0, 0.0),
        )

        val frames = listOf(
            frame("A", 3_000_000_000L, Vec2(500.0, 500.0)),
            frame("D", 3_016_000_000L, Vec2(508.0, 500.0)),
        )

        val result = engine.evaluate(
            MultiViewReprojectionConsistencyEngine.Input(
                skeleton = skeleton,
                cameraFrames = frames,
                targetTimestampNanos = 3_000_000_000L,
            ),
        )

        val joint = result.diagnostics.single()
        assertTrue(joint.medianResidualPx < 1.0)
        assertTrue(joint.temporalConsistencyScore > 0.5)
    }

    @Test
    fun temporalAlignmentEvidenceChangesViewWeight() {
        val strongEngine =
            MultiViewReprojectionConsistencyEngine()
        val weakEngine =
            MultiViewReprojectionConsistencyEngine()

        val skeleton =
            skeleton(
                position = Vec3(0.0, 0.0, 2.0),
                velocity = Vec3(0.0, 0.0, 0.0),
            )

        val strong =
            strongEngine.evaluate(
                MultiViewReprojectionConsistencyEngine.Input(
                    skeleton = skeleton,
                    cameraFrames = listOf(
                        frame(
                            "A",
                            4_500_000_000L,
                            Vec2(500.0, 500.0),
                            temporalAlignmentConfidence = 0.98,
                            temporalUncertaintyNanos = 1_000_000L,
                        ),
                        frame(
                            "B",
                            4_500_000_000L,
                            Vec2(250.0, 500.0),
                            temporalAlignmentConfidence = 0.98,
                            temporalUncertaintyNanos = 1_000_000L,
                        ),
                    ),
                    targetTimestampNanos = 4_500_000_000L,
                ),
            )

        val weak =
            weakEngine.evaluate(
                MultiViewReprojectionConsistencyEngine.Input(
                    skeleton = skeleton,
                    cameraFrames = listOf(
                        frame(
                            "A",
                            4_500_000_000L,
                            Vec2(500.0, 500.0),
                            temporalAlignmentConfidence = 0.22,
                            temporalUncertaintyNanos = 36_000_000L,
                        ),
                        frame(
                            "B",
                            4_500_000_000L,
                            Vec2(250.0, 500.0),
                            temporalAlignmentConfidence = 0.22,
                            temporalUncertaintyNanos = 36_000_000L,
                        ),
                    ),
                    targetTimestampNanos = 4_500_000_000L,
                ),
            )

        val strongWeight =
            strong.diagnostics
                .single()
                .views
                .first()
                .robustWeight
        val weakWeight =
            weak.diagnostics
                .single()
                .views
                .first()
                .robustWeight

        assertTrue(strongWeight > weakWeight)
        assertTrue(
            weak.diagnostics
                .single()
                .acceptedViews <
                strong.diagnostics
                    .single()
                    .acceptedViews,
        )
    }

    @Test
    fun noCameraDataFailsClosed() {
        val engine = MultiViewReprojectionConsistencyEngine()
        val result = engine.evaluate(
            MultiViewReprojectionConsistencyEngine.Input(
                skeleton = skeleton(
                    position = Vec3(0.0, 0.0, 2.0),
                    velocity = Vec3(0.0, 0.0, 0.0),
                ),
                cameraFrames = emptyList(),
                targetTimestampNanos = 4_000_000_000L,
            ),
        )

        assertTrue(result.warnings.any { it.startsWith("fatal:") })
        assertTrue(result.score < 0.30)
    }

    @Test
    fun historyTracksPersistentResiduals() {
        val engine = MultiViewReprojectionConsistencyEngine()

        repeat(6) { index ->
            val timestamp = 5_000_000_000L + index * 16_000_000L
            val result = engine.evaluate(
                MultiViewReprojectionConsistencyEngine.Input(
                    skeleton = skeleton(
                        position = Vec3(0.0, 0.0, 2.0),
                        velocity = Vec3(0.0, 0.0, 0.0),
                    ),
                    cameraFrames = listOf(
                        frame("A", timestamp, Vec2(520.0, 500.0)),
                        frame("B", timestamp, Vec2(270.0, 500.0)),
                    ),
                    targetTimestampNanos = timestamp,
                ),
            )

            if (index == 5) {
                val persistent =
                    result.diagnostics.single().persistentResidualFrames
                assertTrue(persistent >= 2)
            }
        }
    }

    private fun skeleton(
        position: Vec3,
        velocity: Vec3,
    ): SkeletonFrame =
        SkeletonFrame(
            timestampNanos = 1_000_000_000L,
            subjectType = SubjectType.HUMAN,
            trackId = 42L,
            joints = listOf(
                ReconstructionJoint(
                    landmarkId = 0,
                    position = position,
                    velocity = velocity,
                    confidence = 0.98,
                    reprojectionErrorPx = 0.0,
                    observationCount = 2,
                    positionUncertaintyMeters = 0.004,
                ),
            ),
            globalConfidence = 0.96,
        )

    private fun frame(
        id: String,
        timestamp: Long,
        pixel: Vec2,
        temporalAlignmentConfidence: Double = 1.0,
        temporalUncertaintyNanos: Long = 0L,
    ): MultiViewReprojectionConsistencyEngine.CameraFrame {
        val cameraPosition =
            if (id == "B" || id == "C") Vec3(0.5, 0.0, 0.0)
            else Vec3(0.0, 0.0, 0.0)

        return MultiViewReprojectionConsistencyEngine.CameraFrame(
            deviceId = DeviceId(id),
            timestampNanos = timestamp,
            observations = listOf(
                MultiViewReprojectionConsistencyEngine.Observation(
                    landmarkId = 0,
                    camera = CameraObservation(
                        deviceId = DeviceId(id),
                        point = pixel,
                        confidence = 0.98,
                        intrinsics = intrinsics,
                        temporalAlignmentConfidence =
                            temporalAlignmentConfidence,
                        temporalUncertaintyNanos =
                            temporalUncertaintyNanos,
                        pose = Pose(
                            position = cameraPosition,
                            rotation = Quat.IDENTITY,
                        ),
                    ),
                ),
            ),
        )
    }
}

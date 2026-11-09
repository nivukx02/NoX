package com.nivukx.nox.core.reconstruction

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
import kotlin.test.Test
import kotlin.test.assertTrue

class DistributedSupportEvidenceEngineTest {

    private val intrinsics =
        CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
            imageWidth = 1000,
            imageHeight = 1000,
        )

    @Test
    fun consistentNonSelectedCamerasBecomeSupportingEvidence() {
        val result =
            DistributedSupportEvidenceEngine().evaluate(
                input(
                    selected = setOf("A", "B"),
                    cameras =
                        listOf(
                            frame("A", Vec2(500.0, 500.0)),
                            frame("B", Vec2(250.0, 500.0)),
                            frame("C", Vec2(375.0, 500.0)),
                            frame("D", Vec2(300.0, 500.0)),
                        ),
                ),
            )

        assertTrue(result.supportingCameraCount == 2)
        assertTrue(result.score > 0.70)
        assertTrue(result.contradictionRisk < 0.35)
    }

    @Test
    fun oneBadSupportingCameraDoesNotPoisonTheRig() {
        val result =
            DistributedSupportEvidenceEngine().evaluate(
                input(
                    selected = setOf("A", "B"),
                    cameras =
                        listOf(
                            frame("A", Vec2(500.0, 500.0)),
                            frame("B", Vec2(250.0, 500.0)),
                            frame("C", Vec2(375.0, 500.0)),
                            frame("D", Vec2(50.0, 50.0)),
                        ),
                ),
            )

        assertTrue(result.supportingCameraCount == 2)
        assertTrue(result.rejectedSupportingCameraCount >= 1)
        assertTrue(result.contradictionRisk < 0.60)
    }

    @Test
    fun widespreadSupportContradictionIsReported() {
        val result =
            DistributedSupportEvidenceEngine().evaluate(
                input(
                    selected = setOf("A", "B"),
                    cameras =
                        listOf(
                            frame("A", Vec2(500.0, 500.0)),
                            frame("B", Vec2(250.0, 500.0)),
                            frame("C", Vec2(40.0, 40.0)),
                            frame("D", Vec2(60.0, 70.0)),
                            frame("E", Vec2(80.0, 90.0)),
                            frame("F", Vec2(100.0, 110.0)),
                        ),
                ),
            )

        assertTrue(result.contradictionRisk > 0.45)
        assertTrue(
            result.warnings.contains(
                "SUPPORT_EVIDENCE_CONTRADICTION_HIGH",
            ),
        )
    }

    private fun input(
        selected: Set<String>,
        cameras:
            List<MultiViewReprojectionConsistencyEngine.CameraFrame>,
    ) =
        DistributedSupportEvidenceEngine.Input(
            skeleton =
                SkeletonFrame(
                    timestampNanos = 1_000_000_000L,
                    subjectType = SubjectType.HUMAN,
                    trackId = 42L,
                    joints =
                        listOf(
                            ReconstructionJoint(
                                landmarkId = 0,
                                position =
                                    Vec3(0.0, 0.0, 2.0),
                                velocity =
                                    Vec3(0.0, 0.0, 0.0),
                                confidence = 0.98,
                                reprojectionErrorPx = 0.0,
                                observationCount = 2,
                                positionUncertaintyMeters = 0.004,
                            ),
                        ),
                    globalConfidence = 0.96,
                ),
            selectedCameraIds = selected,
            cameraFrames = cameras,
            targetTimestampNanos = 1_000_000_000L,
        )

    private fun frame(
        id: String,
        pixel: Vec2,
    ): MultiViewReprojectionConsistencyEngine.CameraFrame {
        val position =
            when (id) {
                "B", "F" -> Vec3(0.5, 0.0, 0.0)
                "C" -> Vec3(0.25, 0.0, 0.0)
                "D" -> Vec3(0.4, 0.0, 0.0)
                "E" -> Vec3(0.5, 0.0, 0.0)
                else -> Vec3(0.0, 0.0, 0.0)
            }

        return MultiViewReprojectionConsistencyEngine.CameraFrame(
            deviceId = DeviceId(id),
            timestampNanos = 1_000_000_000L,
            observations =
                listOf(
                    MultiViewReprojectionConsistencyEngine.Observation(
                        landmarkId = 0,
                        camera =
                            CameraObservation(
                                deviceId = DeviceId(id),
                                point = pixel,
                                confidence = 0.98,
                                intrinsics = intrinsics,
                                pose =
                                    Pose(
                                        position = position,
                                        rotation = Quat.IDENTITY,
                                    ),
                            ),
                    ),
                ),
        )
    }
}

package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiViewCameraContributionPlannerTest {

    private fun frame(
        id: String,
        x: Double,
        confidence: Double,
    ) = SkeletonReconstructionEngine.CameraFrame(
        deviceId = DeviceId(id),
        timestampNanos = 1_000_000_000L,
        observations = listOf(
            CameraObservation(
                deviceId = DeviceId(id),
                point = Vec2(960.0, 540.0),
                confidence = confidence,
                intrinsics = CameraIntrinsics(
                    fx = 1200.0,
                    fy = 1200.0,
                    cx = 960.0,
                    cy = 540.0,
                    imageWidth = 1920,
                    imageHeight = 1080,
                ),
                pose = Pose(
                    position = Vec3(x, 0.0, 0.0),
                    rotation = Quat.IDENTITY,
                ),
                landmarkId = 0,
            ),
        ),
    )

    private fun status(
        id: String,
        trust: Double,
        state: AdaptiveMultiViewEvidenceWeightEngine.State =
            AdaptiveMultiViewEvidenceWeightEngine.State.HEALTHY,
    ) = AdaptiveMultiViewEvidenceWeightEngine.CameraStatus(
        deviceId = DeviceId(id),
        trust = trust,
        observationCount = 20,
        degradedJointCount = 0,
        hardFailureJointCount = 0,
        meanResidualPx = 1.5,
        state = state,
    )

    @Test
    fun plannerKeepsMinimumDiverseViews() {
        val planner = MultiViewCameraContributionPlanner(
            MultiViewCameraContributionPlanner.Config(
                maximumCameras = 2,
                minimumCameras = 2,
            ),
        )

        val plan = planner.plan(
            frames = listOf(
                frame("a", 0.0, 0.95),
                frame("b", 1.2, 0.91),
                frame("c", 2.4, 0.90),
            ),
            cameraStatuses = listOf(
                status("a", 0.94),
                status("b", 0.93),
                status("c", 0.92),
            ),
        )

        assertEquals(2, plan.selectedDeviceIds.size)
        assertTrue(
            plan.selectedDeviceIds.contains(DeviceId("a")) &&
                plan.selectedDeviceIds.contains(DeviceId("b")),
        )
        assertTrue(plan.diversitySatisfied)
    }

    @Test
    fun lowTrustCameraIsExcludedWhenHealthyAlternativesExist() {
        val planner = MultiViewCameraContributionPlanner(
            MultiViewCameraContributionPlanner.Config(
                maximumCameras = 2,
                minimumCameras = 2,
                holdStateBonus = -0.40,
            ),
        )

        val plan = planner.plan(
            frames = listOf(
                frame("a", 0.0, 0.97),
                frame("b", 1.0, 0.94),
                frame("c", 2.0, 0.30),
            ),
            cameraStatuses = listOf(
                status("a", 0.95),
                status("b", 0.92),
                status(
                    "c",
                    0.15,
                    AdaptiveMultiViewEvidenceWeightEngine.State.HOLD,
                ),
            ),
        )

        assertTrue(plan.selectedDeviceIds.contains(DeviceId("a")))
        assertTrue(plan.selectedDeviceIds.contains(DeviceId("b")))
        assertTrue(!plan.selectedDeviceIds.contains(DeviceId("c")))
        assertTrue(plan.excludedDeviceIds.contains(DeviceId("c")))
    }

    @Test
    fun measuredResidualAndTemporalAlignmentAffectInformationGain() {
        val planner =
            MultiViewCameraContributionPlanner(
                MultiViewCameraContributionPlanner.Config(
                    minimumCameras = 2,
                    maximumCameras = 2,
                    informationGainWeight = 0.72,
                    reliabilityWeight = 0.14,
                    diversityWeight = 0.10,
                    coverageWeight = 0.04,
                ),
            )

        fun measuredFrame(
            id: String,
            x: Double,
            confidence: Double,
            temporal: Double,
        ) =
            SkeletonReconstructionEngine.CameraFrame(
                deviceId = DeviceId(id),
                timestampNanos = 1_000_000_000L,
                observations = listOf(
                    CameraObservation(
                        deviceId = DeviceId(id),
                        point = Vec2(960.0, 540.0),
                        confidence = confidence,
                        intrinsics = CameraIntrinsics(
                            fx = 1200.0,
                            fy = 1200.0,
                            cx = 960.0,
                            cy = 540.0,
                            imageWidth = 1920,
                            imageHeight = 1080,
                        ),
                        pose = Pose(
                            position = Vec3(x, 0.0, 0.0),
                            rotation = Quat.IDENTITY,
                        ),
                        landmarkId = 0,
                        temporalAlignmentConfidence = temporal,
                    ),
                ),
            )

        val plan =
            planner.plan(
                frames = listOf(
                    measuredFrame("a", 0.0, 0.94, 0.98),
                    measuredFrame("b", 1.20, 0.93, 0.96),
                    measuredFrame("c", 2.40, 0.91, 0.52),
                ),
                cameraStatuses = listOf(
                    status("a", 0.94).copy(meanResidualPx = 1.2),
                    status("b", 0.93).copy(meanResidualPx = 1.5),
                    status("c", 0.92).copy(meanResidualPx = 12.0),
                ),
            )

        assertTrue(
            plan.selectedDeviceIds.contains(DeviceId("a")),
        )
        assertTrue(
            plan.selectedDeviceIds.contains(DeviceId("b")),
        )
        assertTrue(
            plan.globalInformationGainScore > 0.55,
        )

        val cameraC =
            plan.rankedCameras.single {
                it.deviceId == DeviceId("c")
            }

        assertTrue(
            cameraC.informationGainScore <
                plan.rankedCameras
                    .single {
                        it.deviceId == DeviceId("b")
                    }
                    .informationGainScore,
        )
    }

    @Test
    fun targetViewBiasReportsAlignmentWithoutRemovingReliabilityGuards() {
        val planner =
            MultiViewCameraContributionPlanner(
                MultiViewCameraContributionPlanner.Config(
                    minimumCameras = 2,
                    maximumCameras = 2,
                    targetViewDirection = Vec3(0.0, 0.0, 1.0),
                    targetViewWeight = 0.25,
                ),
            )

        val plan =
            planner.plan(
                frames = listOf(
                    frame("front-a", 0.0, 0.95),
                    frame("front-b", 1.0, 0.94),
                    frame("front-c", 2.0, 0.93),
                ),
                cameraStatuses = listOf(
                    status("front-a", 0.95),
                    status("front-b", 0.94),
                    status("front-c", 0.93),
                ),
            )

        assertEquals(2, plan.selectedDeviceIds.size)
        assertTrue(
            plan.rankedCameras.all {
                it.targetViewAlignmentScore > 0.99
            },
        )
        assertTrue(plan.diversitySatisfied)
    }

    @Test
    fun oneHundredTwentyEightCameraEnvelopePreservesAuthoritativeAndWitnessSets() {
        val planner =
            MultiViewCameraContributionPlanner(
                MultiViewCameraContributionPlanner.Config(
                    minimumCameras = 2,
                    maximumCameras = 32,
                    maximumWitnessCameras = 96,
                    minimumSelectionScore = 0.05,
                ),
            )

        val frames =
            (0 until 128).map { index ->
                frame(
                    id = "camera-$index",
                    x = index * 0.15,
                    confidence = 0.92,
                )
            }
        val statuses =
            (0 until 128).map { index ->
                status(
                    id = "camera-$index",
                    trust = 0.90 - (index % 5) * 0.01,
                )
            }

        val plan =
            planner.plan(
                frames = frames,
                cameraStatuses = statuses,
            )

        assertEquals(32, plan.selectedDeviceIds.size)
        assertEquals(96, plan.witnessDeviceIds.size)
        assertTrue(
            plan.selectedDeviceIds.intersect(
                plan.witnessDeviceIds,
            ).isEmpty(),
        )
        assertEquals(
            128,
            plan.selectedDeviceIds.size +
                plan.witnessDeviceIds.size,
        )
    }

    @Test
    fun singleAvailableViewRemainsSelectedForExplicitMonocularFallback() {
        val planner = MultiViewCameraContributionPlanner()

        val plan = planner.plan(
            frames = listOf(
                frame("solo", 0.0, 0.96),
            ),
            cameraStatuses = listOf(status("solo", 0.95)),
        )

        assertEquals(
            setOf(DeviceId("solo")),
            plan.selectedDeviceIds,
        )
        assertTrue(!plan.diversitySatisfied)
    }
}


class MultiViewCameraContributionPlannerAnisotropicConditioningTest {
    @Test
    fun rankedCamerasExposeWeakAxisConditioningEvidence() {
        val planner =
            MultiViewCameraContributionPlanner(
                MultiViewCameraContributionPlanner.Config(
                    minimumCameras = 2,
                    maximumCameras = 3,
                ),
            )

        val plan =
            planner.plan(
                frames = listOf(
                    CameraFrameFactory.frame("a", 0.0),
                    CameraFrameFactory.frame("b", 0.8),
                    CameraFrameFactory.frame("c", 1.6),
                ),
                cameraStatuses = listOf(
                    CameraStatusFactory.status("a"),
                    CameraStatusFactory.status("b"),
                    CameraStatusFactory.status("c"),
                ),
            )

        assertTrue(
            plan.rankedCameras.all {
                it.anisotropicConditioningScore in 0.0..1.0
            },
        )
        assertTrue(
            plan.rankedCameras.any {
                it.anisotropicConditioningScore > 0.0
            },
        )
    }
}

private object CameraFrameFactory {
    fun frame(id: String, x: Double) =
        SkeletonReconstructionEngine.CameraFrame(
            deviceId = DeviceId(id),
            timestampNanos = 1_000_000_000L,
            observations = listOf(
                CameraObservation(
                    deviceId = DeviceId(id),
                    point = Vec2(960.0, 540.0),
                    confidence = 0.96,
                    intrinsics = CameraIntrinsics(
                        fx = 1200.0,
                        fy = 1200.0,
                        cx = 960.0,
                        cy = 540.0,
                        imageWidth = 1920,
                        imageHeight = 1080,
                    ),
                    pose = Pose(
                        position = Vec3(x, 0.0, 0.0),
                        rotation = Quat.IDENTITY,
                    ),
                    landmarkId = 0,
                ),
            ),
        )
}

private object CameraStatusFactory {
    fun status(id: String) =
        AdaptiveMultiViewEvidenceWeightEngine.CameraStatus(
            deviceId = DeviceId(id),
            trust = 0.95,
            observationCount = 32,
            degradedJointCount = 0,
            hardFailureJointCount = 0,
            meanResidualPx = 0.8,
            state = AdaptiveMultiViewEvidenceWeightEngine.State.HEALTHY,
        )
}

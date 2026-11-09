package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiViewCameraContributionPlannerHysteresisTest {

    @Test
    fun persistentViewSelectionResistsSmallRankingOscillation() {
        val planner =
            MultiViewCameraContributionPlanner(
                MultiViewCameraContributionPlanner.Config(
                    minimumCameras = 2,
                    maximumCameras = 2,
                    selectionHysteresisBonus = 0.12,
                    maximumSelectionHysteresis = 0.16,
                ),
            )

        val first = planner.plan(
            frames = frames(
                aConfidence = 0.96,
                bConfidence = 0.95,
                cConfidence = 0.93,
            ),
            cameraStatuses = statuses(
                aTrust = 0.96,
                bTrust = 0.95,
                cTrust = 0.93,
            ),
        )

        val second = planner.plan(
            frames = frames(
                aConfidence = 0.95,
                bConfidence = 0.93,
                cConfidence = 0.97,
            ),
            cameraStatuses = statuses(
                aTrust = 0.94,
                bTrust = 0.92,
                cTrust = 0.99,
            ),
        )

        assertTrue(
            first.selectedDeviceIds.contains(DeviceId("a")),
        )
        assertTrue(
            second.selectedDeviceIds.contains(DeviceId("a")) ||
                second.selectedDeviceIds.contains(DeviceId("b")),
        )
        assertTrue(
            second.selectedDeviceIds.size == 2,
        )
    }

    @Test
    fun holdCameraCanStillExitSelectionDespiteHysteresis() {
        val planner =
            MultiViewCameraContributionPlanner(
                MultiViewCameraContributionPlanner.Config(
                    minimumCameras = 2,
                    maximumCameras = 2,
                    selectionHysteresisBonus = 0.16,
                    maximumSelectionHysteresis = 0.20,
                ),
            )

        planner.plan(
            frames = frames(0.97, 0.96, 0.90),
            cameraStatuses =
                statuses(
                    aTrust = 0.98,
                    bTrust = 0.96,
                    cTrust = 0.90,
                ),
        )

        val plan = planner.plan(
            frames = frames(0.20, 0.98, 0.96),
            cameraStatuses =
                statuses(
                    aTrust = 0.10,
                    bTrust = 0.98,
                    cTrust = 0.96,
                    aState =
                        AdaptiveMultiViewEvidenceWeightEngine.State.HOLD,
                ),
        )

        assertTrue(
            !plan.selectedDeviceIds.contains(DeviceId("a")),
        )
    }

    private fun frames(
        aConfidence: Double,
        bConfidence: Double,
        cConfidence: Double,
    ) =
        listOf(
            frame("a", 0.0, aConfidence),
            frame("b", 1.2, bConfidence),
            frame("c", 2.4, cConfidence),
        )

    private fun statuses(
        aTrust: Double,
        bTrust: Double,
        cTrust: Double,
        aState: AdaptiveMultiViewEvidenceWeightEngine.State =
            AdaptiveMultiViewEvidenceWeightEngine.State.HEALTHY,
    ) =
        listOf(
            status("a", aTrust, aState),
            status("b", bTrust),
            status("c", cTrust),
        )

    private fun frame(
        id: String,
        x: Double,
        confidence: Double,
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
                ),
            ),
        )

    private fun status(
        id: String,
        trust: Double,
        state: AdaptiveMultiViewEvidenceWeightEngine.State =
            AdaptiveMultiViewEvidenceWeightEngine.State.HEALTHY,
    ) =
        AdaptiveMultiViewEvidenceWeightEngine.CameraStatus(
            deviceId = DeviceId(id),
            trust = trust,
            observationCount = 20,
            degradedJointCount = 0,
            hardFailureJointCount = 0,
            meanResidualPx = 1.5,
            state = state,
        )
}

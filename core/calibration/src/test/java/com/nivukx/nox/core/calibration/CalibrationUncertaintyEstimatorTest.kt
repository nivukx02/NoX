package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec3
import kotlin.test.assertTrue
import org.junit.Test

class CalibrationUncertaintyEstimatorTest {

    @Test
    fun strongerGeometryAndMoreLandmarksProduceTighterEnvelope() {
        val estimator = CalibrationUncertaintyEstimator()

        val weak = estimator.estimate(
            CalibrationUncertaintyEstimator.CameraEvidence(
                cameraId = "0",
                intrinsics = CameraIntrinsics(
                    fx = 1000.0,
                    fy = 1000.0,
                    cx = 500.0,
                    cy = 400.0,
                ),
                pose = Pose(
                    position = Vec3(0.0, 0.0, 0.0),
                ),
                landmarkCount = 12,
                finalReprojectionRmsPx = 2.5,
                medianSceneDepthMeters = 4.0,
                maximumBaselineMeters = 0.25,
                cameraParticipationRatio = 0.55,
            ),
        )

        val strong = estimator.estimate(
            CalibrationUncertaintyEstimator.CameraEvidence(
                cameraId = "0",
                intrinsics = CameraIntrinsics(
                    fx = 1000.0,
                    fy = 1000.0,
                    cx = 500.0,
                    cy = 400.0,
                ),
                pose = Pose(
                    position = Vec3(0.0, 0.0, 0.0),
                ),
                landmarkCount = 96,
                finalReprojectionRmsPx = 0.4,
                medianSceneDepthMeters = 2.0,
                maximumBaselineMeters = 1.5,
                cameraParticipationRatio = 1.0,
            ),
        )

        assertTrue(
            strong.translationSigmaMeters <
                weak.translationSigmaMeters,
        )
        assertTrue(
            strong.rotationSigmaRadians <
                weak.rotationSigmaRadians,
        )
        assertTrue(
            strong.evidenceConfidence >
                weak.evidenceConfidence,
        )
    }

    @Test
    fun uncalibratedEvidenceReturnsNoEvidenceProfile() {
        val estimator = CalibrationUncertaintyEstimator()
        val result = estimator.estimate(
            CalibrationUncertaintyEstimator.CameraEvidence(
                cameraId = "0",
                intrinsics = CameraIntrinsics(
                    fx = 1000.0,
                    fy = 1000.0,
                    cx = 500.0,
                    cy = 400.0,
                ),
                pose = Pose(),
                landmarkCount = 24,
                finalReprojectionRmsPx = 0.8,
                medianSceneDepthMeters = 2.0,
                maximumBaselineMeters = 1.0,
                cameraParticipationRatio = 1.0,
                calibrated = false,
            ),
        )

        assertTrue(!result.hasEvidence)
    }
}

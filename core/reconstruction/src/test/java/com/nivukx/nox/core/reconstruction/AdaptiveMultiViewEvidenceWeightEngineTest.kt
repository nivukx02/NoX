package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptiveMultiViewEvidenceWeightEngineTest {

    @Test
    fun unseenCameraStartsNeutral() {
        val engine = AdaptiveMultiViewEvidenceWeightEngine()

        engine.beginFrame()

        val weight =
            engine.weight(
                deviceId = DeviceId("camera-a"),
                jointId = 11,
                observationConfidence = 0.9,
            )

        assertTrue(weight > 0.85)
    }

    @Test
    fun repeatedBadResidualsReduceOnlyAffectedViewAndJoint() {
        val engine = AdaptiveMultiViewEvidenceWeightEngine()
        val badCamera = DeviceId("camera-b")
        val goodCamera = DeviceId("camera-a")

        repeat(12) {
            engine.beginFrame()
            engine.update(
                reprojection(
                    badCamera = badCamera,
                    badJoint = 11,
                    badResidualPx = 14.0,
                    goodCamera = goodCamera,
                    goodResidualPx = 1.1,
                ),
            )
        }

        val badWeight =
            engine.weight(
                deviceId = badCamera,
                jointId = 11,
                observationConfidence = 1.0,
            )
        val otherJointWeight =
            engine.weight(
                deviceId = badCamera,
                jointId = 23,
                observationConfidence = 1.0,
            )
        val goodWeight =
            engine.weight(
                deviceId = goodCamera,
                jointId = 11,
                observationConfidence = 1.0,
            )

        assertTrue(badWeight < otherJointWeight)
        assertTrue(badWeight < goodWeight)
        assertTrue(goodWeight > 0.65)
    }

    @Test
    fun recoveryReturnsTrustTowardNeutral() {
        val engine = AdaptiveMultiViewEvidenceWeightEngine()
        val badCamera = DeviceId("camera-b")

        repeat(15) {
            engine.beginFrame()
            engine.update(
                reprojection(
                    badCamera = badCamera,
                    badJoint = 11,
                    badResidualPx = 16.0,
                    goodCamera = DeviceId("camera-a"),
                    goodResidualPx = 1.0,
                ),
            )
        }

        val degraded =
            engine.weight(
                deviceId = badCamera,
                jointId = 11,
                observationConfidence = 1.0,
            )

        repeat(30) {
            engine.beginFrame()
        }

        val recovered =
            engine.weight(
                deviceId = badCamera,
                jointId = 11,
                observationConfidence = 1.0,
            )

        assertTrue(recovered > degraded)
    }

    @Test
    fun globalRigTrustDownweightsAllViewsWithoutDestroyingCameraHistory() {
        val engine = AdaptiveMultiViewEvidenceWeightEngine()
        val camera = DeviceId("camera-global")

        engine.beginFrame()
        engine.update(
            reprojection(
                badCamera = camera,
                badJoint = 11,
                badResidualPx = 1.2,
                goodCamera = DeviceId("camera-other"),
                goodResidualPx = 1.0,
            ),
            calibrationConfidence = 0.96,
            synchronizationConfidence = 0.95,
        )

        val healthyWeight =
            engine.weight(
                deviceId = camera,
                jointId = 11,
                observationConfidence = 1.0,
            )

        engine.beginFrame()
        engine.update(
            reprojection(
                badCamera = camera,
                badJoint = 11,
                badResidualPx = 1.2,
                goodCamera = DeviceId("camera-other"),
                goodResidualPx = 1.0,
            ),
            calibrationConfidence = 0.28,
            synchronizationConfidence = 0.34,
        )

        val degradedRigWeight =
            engine.weight(
                deviceId = camera,
                jointId = 11,
                observationConfidence = 1.0,
            )

        assertTrue(
            degradedRigWeight < healthyWeight,
        )
        assertTrue(
            degradedRigWeight >
                engine.configuration().minimumWeight * 0.20,
        )
        assertTrue(
            engine.snapshot().warnings.any {
                it.contains("global")
            },
        )
    }

    @Test
    fun cameraAndJointStateRemainBoundedUnderChurn() {
        val engine =
            AdaptiveMultiViewEvidenceWeightEngine(
                AdaptiveMultiViewEvidenceWeightEngine.Config(
                    maximumCameras = 3,
                    maximumJointStatesPerCamera = 64,
                    jointStatePruneAfterFrames = 16,
                ),
            )

        repeat(24) { frame ->
            engine.beginFrame()
            val camera = DeviceId("camera-$frame")
            engine.weight(
                deviceId = camera,
                jointId = frame,
                observationConfidence = 1.0,
            )
        }

        assertTrue(engine.snapshot().cameraWeights.size <= 3)
    }

    @Test
    fun snapshotReportsHoldCameraAfterPersistentFailures() {
        val engine =
            AdaptiveMultiViewEvidenceWeightEngine(
                AdaptiveMultiViewEvidenceWeightEngine.Config(
                    minimumEvidenceSamples = 1,
                    trustEmaAlpha = 0.35,
                    residualEmaAlpha = 0.40,
                    cameraHoldThreshold = 0.40,
                    cameraFailureThreshold = 0.60,
                ),
            )
        val badCamera = DeviceId("camera-hold")

        repeat(24) {
            engine.beginFrame()
            engine.update(
                reprojection(
                    badCamera = badCamera,
                    badJoint = 11,
                    badResidualPx = 42.0,
                    goodCamera = DeviceId("camera-good"),
                    goodResidualPx = 1.0,
                ),
            )
        }

        val snapshot = engine.snapshot()

        assertTrue(snapshot.holdCameraCount >= 1)
        assertTrue(
            snapshot.warnings.any {
                it.contains("multi-view-camera-evidence-hold")
            },
        )
    }

    private fun reprojection(
        badCamera: DeviceId,
        badJoint: Int,
        badResidualPx: Double,
        goodCamera: DeviceId,
        goodResidualPx: Double,
    ) =
        MultiViewReprojectionConsistencyEngine.Result(
            skeleton =
                SkeletonFrame(
                    timestampNanos = 1L,
                    subjectType = SubjectType.HUMAN,
                    trackId = 1L,
                    joints = emptyList(),
                    globalConfidence = 0.9,
                ),
            score = 0.86,
            meanResidualPx = 2.0,
            medianResidualPx = 1.0,
            p95ResidualPx = 3.0,
            maximumResidualPx = badResidualPx,
            inlierRatio = 0.75,
            viewAgreementScore = 0.84,
            temporalConsistencyScore = 0.88,
            acceptedJointCount = 1,
            rejectedJointCount = 0,
            diagnostics =
                listOf(
                    MultiViewReprojectionConsistencyEngine.JointDiagnostics(
                        jointId = badJoint,
                        observedViews = 2,
                        acceptedViews = 1,
                        rejectedViews = 1,
                        inlierRatio = 0.5,
                        medianResidualPx = minOf(
                            badResidualPx,
                            goodResidualPx,
                        ),
                        rmsResidualPx = badResidualPx,
                        p95ResidualPx = badResidualPx,
                        maximumResidualPx = badResidualPx,
                        residualSpreadPx = badResidualPx,
                        measurementScore = 0.65,
                        viewAgreementScore = 0.60,
                        temporalConsistencyScore = 0.80,
                        finalScore = 0.55,
                        persistentResidualFrames = 0,
                        accepted = true,
                        views =
                            listOf(
                                view(
                                    jointId = badJoint,
                                    deviceId = badCamera,
                                    residual = badResidualPx,
                                    rejected = true,
                                ),
                                view(
                                    jointId = badJoint,
                                    deviceId = goodCamera,
                                    residual = goodResidualPx,
                                    rejected = false,
                                ),
                            ),
                    ),
                ),
            warnings = emptyList(),
            recommendations = emptyList(),
        )

    private fun view(
        jointId: Int,
        deviceId: DeviceId,
        residual: Double,
        rejected: Boolean,
    ) =
        MultiViewReprojectionConsistencyEngine.ViewDiagnostics(
            jointId = jointId,
            deviceId = deviceId,
            reprojectionErrorPx = residual,
            robustWeight = if (rejected) 0.05 else 0.92,
            observationConfidence = if (rejected) 0.90 else 0.96,
            timeOffsetMillis = 0.5,
            projectedDepthMeters = 2.0,
            inImage = true,
            rejected = rejected,
            rejectionReason = if (rejected) "residual" else null,
        )
}

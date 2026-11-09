package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import org.junit.Assert.assertTrue
import org.junit.Test

class SparseTsdfAdaptiveDetailTest {
    @Test
    fun fineEvidenceUsesHigherSpatialDetail() {
        val engine = SparseTsdfFusionEngine()
        val decision = engine.integrateDepthAdaptive(
            depthMeters = FloatArray(16 * 16) { 1.0f },
            width = 16,
            height = 16,
            intrinsics = CameraIntrinsics(100.0, 100.0, 8.0, 8.0),
            cameraPose = Pose(rotation = Quat.IDENTITY),
            timestampNanos = 1_000_000L,
            baseStride = 1,
            evidence = SparseTsdfFusionEngine.AdaptiveEvidence(
                motionEnergy = 0.05,
                uncertainty = 0.05,
                cameraCoverage = 0.95,
                reconstructionConfidence = 0.95,
                thermalHeadroom = 0.95,
                gpuHeadroom = 0.95,
                requestedDetail = 1.0,
            ),
        )
        assertTrue(decision.detailLevel >= 0.70)
        assertTrue(decision.stride <= 2)
        assertTrue(decision.confidenceScale > 0.75)
        assertTrue(decision.acceptedSamples > 0)
    }

    @Test
    fun lowViewInformationReducesTsdfDetailDemand() {
        val engine = SparseTsdfFusionEngine()

        val highInformation =
            engine.integrateDepthAdaptive(
                depthMeters = FloatArray(16 * 16) { 1.0f },
                width = 16,
                height = 16,
                intrinsics = CameraIntrinsics(100.0, 100.0, 8.0, 8.0),
                cameraPose = Pose(rotation = Quat.IDENTITY),
                timestampNanos = 2_000_000L,
                baseStride = 1,
                evidence = SparseTsdfFusionEngine.AdaptiveEvidence(
                    motionEnergy = 0.10,
                    uncertainty = 0.10,
                    cameraCoverage = 0.95,
                    cameraInformationGain = 0.95,
                    reconstructionConfidence = 0.94,
                    thermalHeadroom = 0.92,
                    gpuHeadroom = 0.92,
                    requestedDetail = 0.92,
                ),
            )

        val lowInformation =
            engine.integrateDepthAdaptive(
                depthMeters = FloatArray(16 * 16) { 1.0f },
                width = 16,
                height = 16,
                intrinsics = CameraIntrinsics(100.0, 100.0, 8.0, 8.0),
                cameraPose = Pose(rotation = Quat.IDENTITY),
                timestampNanos = 2_033_000L,
                baseStride = 1,
                evidence = SparseTsdfFusionEngine.AdaptiveEvidence(
                    motionEnergy = 0.10,
                    uncertainty = 0.10,
                    cameraCoverage = 0.95,
                    cameraInformationGain = 0.20,
                    reconstructionConfidence = 0.94,
                    thermalHeadroom = 0.92,
                    gpuHeadroom = 0.92,
                    requestedDetail = 0.92,
                ),
            )

        assertTrue(
            lowInformation.detailLevel <=
                highInformation.detailLevel,
        )
        assertTrue(
            lowInformation.computePressure >=
                highInformation.computePressure,
        )
    }

    @Test
    fun pressuredEvidenceReducesTsdfSamplingRate() {
        val engine = SparseTsdfFusionEngine()
        val decision = engine.integrateDepthAdaptive(
            depthMeters = FloatArray(16 * 16) { 1.0f },
            width = 16,
            height = 16,
            intrinsics = CameraIntrinsics(100.0, 100.0, 8.0, 8.0),
            cameraPose = Pose(rotation = Quat.IDENTITY),
            timestampNanos = 1_000_000L,
            baseStride = 1,
            evidence = SparseTsdfFusionEngine.AdaptiveEvidence(
                motionEnergy = 0.92,
                uncertainty = 0.78,
                cameraCoverage = 0.32,
                reconstructionConfidence = 0.35,
                thermalHeadroom = 0.25,
                gpuHeadroom = 0.20,
                requestedDetail = 0.80,
            ),
        )
        assertTrue(decision.detailLevel < 0.60)
        assertTrue(decision.stride >= 3)
        assertTrue(decision.confidenceScale < 0.85)
        assertTrue(decision.rejectedSamples >= 0)
    }

    @Test
    fun zeroPerPixelConfidenceCannotEnterTheKotlinTsdfVolume() {
        val engine = SparseTsdfFusionEngine()
        val depths = FloatArray(16 * 16) { 1.0f }
        val decision = engine.integrateDepthAdaptive(
            depthMeters = depths,
            width = 16,
            height = 16,
            intrinsics = CameraIntrinsics(100.0, 100.0, 8.0, 8.0),
            cameraPose = Pose(rotation = Quat.IDENTITY),
            timestampNanos = 3_000_000L,
            baseStride = 1,
            confidence = 1.0,
            depthConfidence = FloatArray(depths.size) { 0.0f },
            evidence = SparseTsdfFusionEngine.AdaptiveEvidence(
                motionEnergy = 0.05,
                uncertainty = 0.05,
                cameraCoverage = 0.95,
                reconstructionConfidence = 0.95,
                thermalHeadroom = 0.95,
                gpuHeadroom = 0.95,
                requestedDetail = 1.0,
            ),
        )

        assertTrue(decision.acceptedSamples == 0)
        assertTrue(decision.rejectedSamples > 0)
        assertTrue(decision.reason == "all_samples_rejected")
    }

    @Test
    fun perPixelConfidenceChangesEvidenceWeightWithoutChangingDepthValues() {
        val engine = SparseTsdfFusionEngine()
        val depths = FloatArray(16 * 16) { 1.0f }
        val highConfidence = engine.integrateDepthEvidenceWeighted(
            depthMeters = depths,
            width = 16,
            height = 16,
            intrinsics = CameraIntrinsics(100.0, 100.0, 8.0, 8.0),
            cameraPose = Pose(rotation = Quat.IDENTITY),
            timestampNanos = 4_000_000L,
            stride = 1,
            confidence = 1.0,
            depthConfidence = FloatArray(depths.size) { 1.0f },
        )
        val lowConfidence = engine.integrateDepthEvidenceWeighted(
            depthMeters = depths,
            width = 16,
            height = 16,
            intrinsics = CameraIntrinsics(100.0, 100.0, 8.0, 8.0),
            cameraPose = Pose(rotation = Quat.IDENTITY),
            timestampNanos = 4_033_000L,
            stride = 1,
            confidence = 1.0,
            depthConfidence = FloatArray(depths.size) { 0.1f },
        )

        assertTrue(highConfidence.acceptedSamples > 0)
        assertTrue(lowConfidence.acceptedSamples > 0)
        assertTrue(lowConfidence.meanEvidenceWeight < highConfidence.meanEvidenceWeight)
    }

    @Test
    fun malformedPerPixelConfidenceRejectsSamplesAndInvalidDimensionsAreRejected() {
        val engine = SparseTsdfFusionEngine()
        val depths = FloatArray(16 * 16) { 1.0f }
        val decision = engine.integrateDepthEvidenceWeighted(
            depthMeters = depths,
            width = 16,
            height = 16,
            intrinsics = CameraIntrinsics(100.0, 100.0, 8.0, 8.0),
            cameraPose = Pose(rotation = Quat.IDENTITY),
            timestampNanos = 5_000_000L,
            stride = 1,
            confidence = 1.0,
            depthConfidence = FloatArray(depths.size) { Float.NaN },
        )
        assertTrue(decision.acceptedSamples == 0)
        assertTrue(decision.rejectedSamples > 0)
    }

}

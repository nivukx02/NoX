package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import org.junit.Assert.assertTrue
import org.junit.Test

class DenseLandmarkEvidenceFusionEngineTest {
    private val engine = DenseLandmarkEvidenceFusionEngine()

    private fun obs(
        id: String,
        x: Double,
        y: Double,
        confidence: Double = 0.98,
        visibility: Double = 0.96,
        occlusion: Double = 0.05,
        mask: Int? = 7,
    ) = DenseLandmarkEvidenceFusionEngine.Observation(
        landmarkId = 12,
        deviceId = DeviceId(id),
        xPx = x,
        yPx = y,
        timestampNanos = 1_000_000_000L,
        pixelUncertainty = 1.2,
        detectorConfidence = confidence,
        visibilityConfidence = visibility,
        contactConfidence = 0.10,
        occlusionRisk = occlusion,
        identityConfidence = 0.98,
        semanticMaskId = mask,
        frameWidth = 1920,
        frameHeight = 1080,
    )

    @Test
    fun independentViewsProduceHighConsensus() {
        val result = engine.fuse(
            listOf(
                obs("A", 960.0, 540.0),
                obs("B", 963.0, 539.0),
                obs("C", 958.0, 542.0),
                obs("D", 961.0, 541.0),
            ),
        ).landmarks.single()

        assertTrue(result.distinctCameras == 4)
        assertTrue(result.effectiveViewCount >= 2.0)
        assertTrue(result.consensusConfidence > 0.65)
        assertTrue(result.admissibleForGeometry)
    }

    @Test
    fun severeOcclusionBlocksGeometry() {
        val result = engine.fuse(
            listOf(
                obs("A", 960.0, 540.0, visibility = 0.22, occlusion = 0.95),
                obs("B", 962.0, 541.0, visibility = 0.24, occlusion = 0.93),
            ),
        ).landmarks.single()

        assertTrue(!result.admissibleForGeometry)
        assertTrue(result.blockers.isNotEmpty())
    }

    @Test
    fun semanticDisagreementIsVisible() {
        val result = engine.fuse(
            listOf(
                obs("A", 960.0, 540.0, mask = 1),
                obs("B", 960.0, 540.0, mask = 2),
                obs("C", 960.0, 540.0, mask = 3),
            ),
        ).landmarks.single()

        assertTrue(result.semanticMaskAgreement < 0.70)
        assertTrue(result.warnings.contains("SEMANTIC_MASK_AGREEMENT_LOW"))
    }

    @Test
    fun observationsOutsideTemporalWindowAreDropped() {
        val result = engine.fuse(
            listOf(
                obs("A", 960.0, 540.0),
                obs("B", 962.0, 541.0).copy(
                    timestampNanos = 1_020_000_000L,
                ),
                obs("C", 959.0, 539.0),
            ),
            targetTimestampNanos = 1_000_000_000L,
        ).landmarks.single()

        assertTrue(result.distinctCameras == 2)
    }
}


class DenseLandmarkEvidenceFusionEngineRayGeometryTest {
    @Test
    fun convergentCalibratedRaysRemainGeometryAdmissible() {
        val engine = DenseLandmarkEvidenceFusionEngine()
        val result = engine.fuse(
            listOf(
                DenseLandmarkEvidenceFusionEngine.Observation(
                    landmarkId = 12,
                    deviceId = DeviceId("cam-a"),
                    xPx = 50.0,
                    yPx = 50.0,
                    timestampNanos = 1_000_000L,
                    pixelUncertainty = 1.0,
                    detectorConfidence = 0.98,
                    visibilityConfidence = 0.98,
                    frameWidth = 100,
                    frameHeight = 100,
                    worldRayOriginMeters = Vec3(-0.20, 0.0, 0.0),
                    worldRayDirection = Vec3(0.20, 0.0, 1.0),
                ),
                DenseLandmarkEvidenceFusionEngine.Observation(
                    landmarkId = 12,
                    deviceId = DeviceId("cam-b"),
                    xPx = 50.0,
                    yPx = 50.0,
                    timestampNanos = 1_000_000L,
                    pixelUncertainty = 1.0,
                    detectorConfidence = 0.98,
                    visibilityConfidence = 0.98,
                    frameWidth = 100,
                    frameHeight = 100,
                    worldRayOriginMeters = Vec3(0.20, 0.0, 0.0),
                    worldRayDirection = Vec3(-0.20, 0.0, 1.0),
                ),
            ),
        ).landmarks.single()

        assertTrue(result.crossViewRayConsistency > 0.90)
        assertTrue(result.admissibleForGeometry)
    }

    @Test
    fun contradictoryCalibratedRaysAreRejectedByGeometryGate() {
        val engine = DenseLandmarkEvidenceFusionEngine(
            DenseLandmarkEvidenceFusionEngine.Config(
                maximumSymmetricRayDistanceMeters = 0.01,
                minimumCrossViewRayConsistency = 0.70,
            ),
        )
        val result = engine.fuse(
            listOf(
                DenseLandmarkEvidenceFusionEngine.Observation(
                    landmarkId = 13,
                    deviceId = DeviceId("cam-a"),
                    xPx = 50.0,
                    yPx = 50.0,
                    timestampNanos = 2_000_000L,
                    pixelUncertainty = 1.0,
                    detectorConfidence = 0.98,
                    visibilityConfidence = 0.98,
                    frameWidth = 100,
                    frameHeight = 100,
                    worldRayOriginMeters = Vec3(0.0, 0.0, 0.0),
                    worldRayDirection = Vec3(1.0, 0.0, 1.0),
                ),
                DenseLandmarkEvidenceFusionEngine.Observation(
                    landmarkId = 13,
                    deviceId = DeviceId("cam-b"),
                    xPx = 50.0,
                    yPx = 50.0,
                    timestampNanos = 2_000_000L,
                    pixelUncertainty = 1.0,
                    detectorConfidence = 0.98,
                    visibilityConfidence = 0.98,
                    frameWidth = 100,
                    frameHeight = 100,
                    worldRayOriginMeters = Vec3(0.0, 1.0, 0.0),
                    worldRayDirection = Vec3(1.0, 0.0, 0.0),
                ),
            ),
        ).landmarks.single()

        assertTrue(result.blockers.contains("CROSS_VIEW_RAY_CONSISTENCY_LOW"))
        assertTrue(!result.admissibleForGeometry)
    }
}

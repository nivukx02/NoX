package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertTrue

class CameraPlacementInformationEngineTest {
    @Test
    fun frontFacingCameraOutranksRearFacingCamera() {
        val engine = CameraPlacementInformationEngine()
        val actor =
            CameraPlacementInformationEngine.ActorAnchor(
                actorId = 7,
                positionMeters = Vec3(0.0, 0.0, 0.0),
            )
        val front =
            CameraPlacementInformationEngine.CameraAnchor(
                cameraId = "front",
                positionMeters = Vec3(0.0, 0.0, 5.0),
                forward = Vec3(0.0, 0.0, -1.0),
                authorityScore = 0.95,
                sector = 1,
                calibrationConfidence = 0.98,
                synchronizationConfidence = 0.99,
                networkHeadroom = 0.95,
                thermalHeadroom = 0.95,
            )
        val rear =
            front.copy(
                cameraId = "rear",
                forward = Vec3(0.0, 0.0, 1.0),
                sector = 3,
            )
        val frontScore = engine.evaluate(actor, front)
        val rearScore = engine.evaluate(actor, rear)
        assertTrue(frontScore.geometryAvailable)
        assertTrue(frontScore.facingScore > rearScore.facingScore)
        assertTrue(frontScore.finalScore > rearScore.finalScore)
    }

    @Test
    fun orthogonalCameraAddsMoreNovelInformation() {
        val engine = CameraPlacementInformationEngine()
        val actor =
            CameraPlacementInformationEngine.ActorAnchor(
                actorId = 1,
                positionMeters = Vec3(0.0, 0.0, 0.0),
            )
        val selected =
            CameraPlacementInformationEngine.CameraAnchor(
                cameraId = "a",
                positionMeters = Vec3(0.0, 0.0, 5.0),
                forward = Vec3(0.0, 0.0, -1.0),
                authorityScore = 0.90,
                sector = 0,
                calibrationConfidence = 0.95,
                synchronizationConfidence = 0.95,
                networkHeadroom = 1.0,
                thermalHeadroom = 1.0,
            )
        val orthogonal =
            selected.copy(
                cameraId = "b",
                positionMeters = Vec3(5.0, 0.0, 0.0),
                forward = Vec3(-1.0, 0.0, 0.0),
                sector = 1,
            )
        val nearDuplicate =
            selected.copy(
                cameraId = "c",
                positionMeters = Vec3(0.5, 0.0, 5.0),
                forward = Vec3(-0.1, 0.0, -1.0),
                sector = 0,
            )
        val orthogonalScore =
            engine.evaluate(
                actor = actor,
                camera = orthogonal,
                selectedCameras = listOf(selected),
            )
        val nearDuplicateScore =
            engine.evaluate(
                actor = actor,
                camera = nearDuplicate,
                selectedCameras = listOf(selected),
            )
        assertTrue(
            orthogonalScore.noveltyDegrees >
                nearDuplicateScore.noveltyDegrees,
        )
        assertTrue(
            orthogonalScore.noveltyScore >
                nearDuplicateScore.noveltyScore,
        )
    }

    @Test
    fun measured_motion_blur_reduces_metric_placement_authority() {
        val engine = CameraPlacementInformationEngine()
        val actor =
            CameraPlacementInformationEngine.ActorAnchor(
                actorId = 5,
                positionMeters = Vec3(0.0, 0.0, 0.0),
            )

        val sharp =
            CameraPlacementInformationEngine.CameraAnchor(
                cameraId = "sharp",
                positionMeters = Vec3(0.0, 0.0, 5.0),
                forward = Vec3(0.0, 0.0, -1.0),
                authorityScore = 0.95,
                sector = 0,
                calibrationConfidence = 0.98,
                synchronizationConfidence = 0.99,
                networkHeadroom = 0.95,
                thermalHeadroom = 0.95,
                motionBlurRisk = 0.02,
            )

        val blurred =
            sharp.copy(
                cameraId = "blurred",
                sector = 1,
                motionBlurRisk = 0.82,
            )

        val sharpScore = engine.evaluate(actor, sharp)
        val blurredScore = engine.evaluate(actor, blurred)

        assertTrue(
            sharpScore.opticalIntegrityScore >
                blurredScore.opticalIntegrityScore,
        )
        assertTrue(
            sharpScore.finalScore >
                blurredScore.finalScore,
        )
        assertTrue(
            "MOTION_BLUR_RISK_HIGH" in
                blurredScore.diagnostics,
        )
    }

    @Test
    fun missingGeometryIsExplicitlyDegraded() {
        val engine = CameraPlacementInformationEngine()
        val actor =
            CameraPlacementInformationEngine.ActorAnchor(
                actorId = 1,
                positionMeters = Vec3(0.0, 0.0, 0.0),
            )
        val camera =
            CameraPlacementInformationEngine.CameraAnchor(
                cameraId = "missing-forward",
                positionMeters = Vec3(0.0, 0.0, 5.0),
                forward = Vec3(0.0, 0.0, 0.0),
                authorityScore = 0.80,
                sector = 0,
                calibrationConfidence = 0.80,
                synchronizationConfidence = 0.80,
                networkHeadroom = 0.80,
                thermalHeadroom = 0.80,
            )
        val score = engine.evaluate(actor, camera)
        assertTrue(!score.geometryAvailable)
        assertTrue(
            "CAMERA_FORWARD_OR_TARGET_DEGENERATE" in
                score.diagnostics,
        )
    }
}

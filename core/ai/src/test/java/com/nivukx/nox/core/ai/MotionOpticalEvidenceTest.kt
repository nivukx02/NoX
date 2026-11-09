package com.nivukx.nox.core.ai

import kotlin.test.assertTrue
import org.junit.Test

class MotionOpticalEvidenceTest {

    @Test
    fun motionBlurContextReducesMeasurementTrust() {
        val clear = MotionCaptureEngine.MotionContext(
            cameraMotionRisk = 0.05,
            sensorConfidence = 0.98,
            rollingShutterRisk = 0.05,
            globalMotionConfidence = 0.95,
            motionBlurRisk = 0.05,
        )
        val blurred = clear.copy(motionBlurRisk = 0.85)

        assertTrue(clear.measurementTrustScale > blurred.measurementTrustScale)
        assertTrue(blurred.measurementTrustScale >= 0.28)
    }

    @Test
    fun qualityExposesOpticalBlurRiskAndKeepsReadinessBounded() {
        val engine = MotionCaptureEngine()
        val input = PoseDetectionEngine.Result(
            timestampNanos = 1_000_000_000L,
            imageWidth = 1000,
            imageHeight = 1000,
            landmarks = List(3) { id ->
                PoseDetectionEngine.Landmark(
                    id = id,
                    name = "landmark-$id",
                    x = 300f + id * 50f,
                    y = 400f + id * 20f,
                    confidence = 0.96f,
                )
            },
            confidence = 0.96f,
        )

        val quality = engine.refine(
            input = input,
            context = MotionCaptureEngine.MotionContext(
                sensorConfidence = 0.98,
                globalMotionConfidence = 0.95,
                motionBlurRisk = 0.98,
            ),
        ).quality

        assertTrue(quality.motionBlurRisk in 0.0..1.0)
        assertTrue(quality.captureReadiness in 0.0..1.0)
        assertTrue(quality.captureSafe.not())
        engine.close()
    }
}

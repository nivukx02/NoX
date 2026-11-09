package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.CameraFocusMode
import com.nivukx.nox.core.model.FrameTelemetry
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraCaptureQualityEngineBackpressureTest {
    @Test
    fun longerExposureProducesHigherExplicitOpticalBlurRisk() {
        val base =
            FrameTelemetry(
                sensorTimestampNanos = 2_000_000_000L,
                monotonicTimestampNanos = 2_000_000_100L,
                exposureTimeNanos = 4_000_000L,
                sensorSensitivityIso = 100,
                frameDurationNanos = 16_666_667L,
                rollingShutterSkewNanos = 1_000_000L,
                aeState = 1,
                afState = 1,
                awbState = 1,
                lensFocusDistanceDiopters = 0.8f,
                focusDistanceMeters = 1.25f,
            )
        val longer =
            base.copy(
                exposureTimeNanos = 15_000_000L,
            )
        val engine = CameraCaptureQualityEngine()
        val shortExposure =
            engine.evaluate(
                telemetry = base,
                targetFps = 60,
                focusMode = CameraFocusMode.CONTINUOUS,
            )
        val longExposure =
            engine.evaluate(
                telemetry = longer,
                targetFps = 60,
                focusMode = CameraFocusMode.CONTINUOUS,
            )

        assertTrue(
            longExposure.opticalBlurRisk >=
                shortExposure.opticalBlurRisk,
        )
        assertTrue(longExposure.opticalBlurRisk > 0.0)
    }

    @Test
    fun analysisDropsReduceQualityAndHardFailAtSevereLoss() {
        val telemetry = FrameTelemetry(
            sensorTimestampNanos = 1_000_000_000L,
            monotonicTimestampNanos = 1_000_000_100L,
            exposureTimeNanos = 5_000_000L,
            sensorSensitivityIso = 100,
            frameDurationNanos = 16_666_667L,
            rollingShutterSkewNanos = 2_000_000L,
            aeState = 1,
            afState = 1,
            awbState = 1,
            lensFocusDistanceDiopters = 0.8f,
            focusDistanceMeters = 1.25f,
        )
        val engine = CameraCaptureQualityEngine()

        val healthy = engine.evaluate(
            telemetry = telemetry,
            targetFps = 60,
            focusMode = CameraFocusMode.CONTINUOUS,
            analysisDropRatio = 0.0,
        )
        val degraded = engine.evaluate(
            telemetry = telemetry,
            targetFps = 60,
            focusMode = CameraFocusMode.CONTINUOUS,
            analysisDropRatio = 0.12,
        )
        val failed = engine.evaluate(
            telemetry = telemetry,
            targetFps = 60,
            focusMode = CameraFocusMode.CONTINUOUS,
            analysisDropRatio = 0.30,
        )

        assertTrue(degraded.score < healthy.score)
        assertTrue(
            degraded.warnings.contains("analysis-backpressure-high"),
        )
        assertTrue(failed.hardFailure)
        assertTrue(
            failed.warnings.contains("analysis-backpressure-high"),
        )
    }
}

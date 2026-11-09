package com.nivukx.nox.core.ai

import org.junit.Assert.assertTrue
import org.junit.Test

class MotionPerformanceGovernorTest {
    @Test
    fun healthyTelemetryRetainsHighScale() {
        val result = MotionPerformanceGovernor().update(
            MotionPerformanceGovernor.Telemetry(
                frameTimeMillis = 11.0,
                detectorMillis = 2.0,
                trackerMillis = 1.2,
                depthMillis = 2.2,
                faceMillis = 1.0,
                eyeMillis = 0.6,
                handMillis = 0.8,
                reconstructionMillis = 2.0,
                meshMillis = 0.5,
                thermalHeadroom = 0.90,
                memoryHeadroom = 0.88,
                gpuHeadroom = 0.86,
                npuHeadroom = 0.92,
            ),
            preferredQuality = 0.95,
        )
        assertTrue(result.detectorScale > 0.50)
        assertTrue(result.reconstructionScale > 0.50)
    }

    @Test
    fun overloadedDeviceTriggersDegradationWarnings() {
        val result = MotionPerformanceGovernor().update(
            MotionPerformanceGovernor.Telemetry(
                frameTimeMillis = 36.0,
                detectorMillis = 12.0,
                trackerMillis = 5.0,
                depthMillis = 10.0,
                faceMillis = 4.0,
                eyeMillis = 3.0,
                handMillis = 3.0,
                reconstructionMillis = 8.0,
                meshMillis = 4.0,
                thermalHeadroom = 0.15,
                memoryHeadroom = 0.16,
                gpuHeadroom = 0.20,
                npuHeadroom = 0.18,
                droppedFrames = 5,
            ),
            preferredQuality = 0.95,
        )
        assertTrue(result.degraded)
        assertTrue(result.warnings.isNotEmpty())
    }

    @Test
    fun cameraHostCopiesAreVisibleInPerformanceDiagnostics() {
        val result = MotionPerformanceGovernor().update(
            MotionPerformanceGovernor.Telemetry(
                frameTimeMillis = 14.0,
                detectorMillis = 2.0,
                trackerMillis = 1.0,
                depthMillis = 2.0,
                faceMillis = 1.0,
                eyeMillis = 0.5,
                handMillis = 0.5,
                reconstructionMillis = 3.0,
                meshMillis = 1.0,
                thermalHeadroom = 0.92,
                memoryHeadroom = 0.90,
                gpuHeadroom = 0.90,
                npuHeadroom = 0.90,
                cameraInputHostCopies = 2,
                zeroCopyCameraInput = false,
            ),
        )

        assertTrue(
            result.warnings.any {
                it.contains("camera host copies")
            },
        )
        assertTrue(
            result.diagnostics.any {
                it.startsWith("cameraInputHostCopies=")
            },
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun invalidTelemetryCannotEnterGovernor() {
        MotionPerformanceGovernor().update(
            MotionPerformanceGovernor.Telemetry(
                frameTimeMillis = -1.0,
                detectorMillis = 0.0,
                trackerMillis = 0.0,
                depthMillis = 0.0,
                faceMillis = 0.0,
                eyeMillis = 0.0,
                handMillis = 0.0,
                reconstructionMillis = 0.0,
                meshMillis = 0.0,
                thermalHeadroom = 0.90,
                memoryHeadroom = 0.90,
                gpuHeadroom = 0.90,
                npuHeadroom = 0.90,
            ),
        )
    }

}

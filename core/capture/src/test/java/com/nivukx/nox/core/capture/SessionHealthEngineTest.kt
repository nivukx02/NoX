package com.nivukx.nox.core.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SessionHealthEngineTest {

    @Test
    fun productionSignalsProduceProductionState() {
        val engine = SessionHealthEngine()

        val result = engine.evaluate(healthyInput())

        assertEquals(
            SessionHealthEngine.State.PRODUCTION,
            result.state,
        )
        assertTrue(result.productionReady)
        assertTrue(result.score >= 0.78)
        assertTrue(result.hardFailures.isEmpty())
    }

    @Test
    fun brokenCaptureClockBecomesCriticalWhenConfidenceCollapses() {
        val engine = SessionHealthEngine()

        val result = engine.evaluate(
            healthyInput().copy(
                captureTimingConfidence = 0.20,
                captureTimingFps = 18.0,
                captureTimingJitterMillis = 16.0,
                captureTimingHealthy = false,
            ),
        )

        assertTrue(result.unsafe)
        assertTrue(
            result.hardFailures.any {
                it.contains("capture timing")
            },
        )
        assertTrue(result.recommendations.isNotEmpty())
    }

    @Test
    fun reconstructionAndThermalProblemsSurfaceAsBottlenecks() {
        val engine = SessionHealthEngine()

        val result = engine.evaluate(
            healthyInput().copy(
                reconstructionConfidence = 0.28,
                reconstructionReprojectionErrorPx = 12.0,
                temporalStability = 0.29,
                thermalScale = 0.34,
                memoryScale = 0.32,
            ),
        )

        assertTrue(
            result.state == SessionHealthEngine.State.CRITICAL ||
                result.state == SessionHealthEngine.State.DEGRADED,
        )
        assertTrue(result.bottleneck != null)
        assertTrue(result.warnings.isNotEmpty())
    }

    @Test
    fun synchronizationDegradationBecomesVisibleInRigHealth() {
        val engine = SessionHealthEngine()

        val result = engine.evaluate(
            healthyInput().copy(
                multiCameraSyncConfidence = 0.28,
            ),
        )

        assertTrue(
            result.state == SessionHealthEngine.State.CRITICAL ||
                result.state == SessionHealthEngine.State.DEGRADED,
        )
        assertTrue(
            result.signals.any {
                it.type == SessionHealthEngine.SignalClass.MULTI_CAMERA_SYNC
            },
        )
    }

    @Test
    fun faceEyeAndHandsCanDegradeWithoutDestroyingBodyCapture() {
        val engine = SessionHealthEngine()

        val result = engine.evaluate(
            healthyInput().copy(
                faceConfidence = 0.16,
                eyeConfidence = 0.13,
                handConfidence = 0.16,
            ),
        )

        assertTrue(result.score > 0.55)
        assertTrue(result.hardFailures.isEmpty())
    }

    private fun healthyInput() =
        SessionHealthEngine.Input(
            timestampNanos = 1_000_000_000L,
            targetCaptureFps = 60.0,
            captureTimingConfidence = 0.96,
            captureTimingFps = 60.0,
            captureTimingJitterMillis = 1.1,
            captureTimingHealthy = true,
            inferenceFps = 57.0,
            inferenceLatencyMillis = 18.0,
            droppedInferenceFrames = 1L,
            capturedFrames = 240L,
            subjectLockConfidence = 0.92,
            perceptionConfidence = 0.91,
            reconstructionConfidence = 0.90,
            reconstructionReprojectionErrorPx = 1.9,
            temporalStability = 0.93,
            calibrationConfidence = 0.96,
            cameraMotionRisk = 0.12,
            occlusionRisk = 0.13,
            meshConfidence = 0.89,
            faceConfidence = 0.88,
            eyeConfidence = 0.85,
            handConfidence = 0.84,
            thermalScale = 0.82,
            memoryScale = 0.84,
        )
    @Test
    fun lowMotionEvidenceDegradesHealthEvenWhenPerceptionIsAvailable() {
        val engine = SessionHealthEngine()

        val result = engine.evaluate(
            healthyInput().copy(
                motionEvidenceConfidence = 0.12,
                motionEvidenceDegraded = true,
            ),
        )

        assertTrue(
            result.signals.any {
                it.type == SessionHealthEngine.SignalClass.MOTION_EVIDENCE &&
                    it.hardFailure
            },
        )
    }


    @Test
    fun weakGeometricObservabilityDegradesHealth() {
        val engine = SessionHealthEngine()

        val result = engine.evaluate(
            healthyInput().copy(
                geometricObservabilityConfidence = 0.28,
                geometricObservabilityDegraded = true,
            ),
        )

        assertTrue(
            result.signals.any {
                it.type ==
                    SessionHealthEngine.SignalClass.GEOMETRIC_OBSERVABILITY &&
                    it.hardFailure
            },
        )
    }


    @Test
    fun reconstructionUncertaintyCanDegradeHealth() {
        val engine = SessionHealthEngine()
        val result = engine.evaluate(
            healthyInput().copy(
                reconstructionUncertaintyConfidence = 0.18,
                reconstructionUncertaintyDegraded = true,
                reconstructionUncertaintyP95Meters = 0.24,
            ),
        )
        assertTrue(
            result.signals.any {
                it.type ==
                    SessionHealthEngine.SignalClass.RECONSTRUCTION_UNCERTAINTY &&
                    it.hardFailure
            },
        )
    }

    @Test
    fun eyeTemporalConsistencyCanDegradeHealth() {
        val engine = SessionHealthEngine()
        val result = engine.evaluate(
            healthyInput().copy(
                eyeTemporalConfidence = 0.14,
                eyeTemporalDegraded = true,
            ),
        )

        assertTrue(
            result.signals.any {
                it.type ==
                    SessionHealthEngine.SignalClass.EYE_TEMPORAL_CONSISTENCY &&
                    it.hardFailure
            },
        )
    }


}

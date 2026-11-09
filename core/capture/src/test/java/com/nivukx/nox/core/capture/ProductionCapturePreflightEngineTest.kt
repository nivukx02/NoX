package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.MotionCaptureMode
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ProductionCapturePreflightEngineTest {

    @Test
    fun healthyMonocularSessionIsReadyWhenSubjectIsVisible() {
        val decision = ProductionCapturePreflightEngine().evaluate(
            healthyInput(),
        )

        assertTrue(decision.readiness.ready)
        assertTrue(decision.hardFailures.isEmpty())
    }

    @Test
    fun multiviewWithoutCalibrationIsRejected() {
        val decision = ProductionCapturePreflightEngine().evaluate(
            healthyInput().copy(
                mode = MotionCaptureMode.MULTI_VIEW_RGB,
                calibrationReady = false,
                calibrationConfidence = 0.0,
            ),
        )

        assertFalse(decision.readiness.ready)
        assertTrue(decision.hardFailures.any {
            it.contains("calibration")
        })
    }

    @Test
    fun unhealthyCaptureClockProducesDegradedReadiness() {
        val decision = ProductionCapturePreflightEngine().evaluate(
            healthyInput().copy(
                captureTimingConfidence = 0.42,
                captureTimingFps = 24.0,
                captureTimingJitterMillis = 8.5,
                captureTimingHealthy = false,
            ),
        )

        assertTrue(decision.degraded)
        assertTrue(decision.warnings.any {
            it.contains("capture")
        })
    }

    @Test
    fun poorRigHealthBlocksMetricCapture() {
        val decision = ProductionCapturePreflightEngine().evaluate(
            healthyInput().copy(
                mode = MotionCaptureMode.MULTI_VIEW_RGB,
                calibrationReady = true,
                calibrationConfidence = 0.92,
                rigHealthConfidence = 0.36,
                rigHealthHealthy = false,
            ),
        )

        assertFalse(decision.readiness.ready)
        assertTrue(
            decision.hardFailures.any {
                it.contains("rig camera health")
            },
        )
    }

    @Test
    fun healthySixtyFpsCaptureClockRemainsProductionReady() {
        val decision = ProductionCapturePreflightEngine().evaluate(
            healthyInput().copy(
                captureTimingConfidence = 0.96,
                captureTimingFps = 60.0,
                captureTimingJitterMillis = 1.2,
                captureTimingHealthy = true,
            ),
        )

        assertTrue(decision.readiness.ready)
        assertTrue(decision.hardFailures.isEmpty())
    }

    @Test
    fun multiviewWithPoorSynchronizationIsRejected() {
        val decision = ProductionCapturePreflightEngine().evaluate(
            healthyInput().copy(
                mode = MotionCaptureMode.MULTI_VIEW_RGB,
                calibrationReady = true,
                calibrationConfidence = 0.90,
                multiCameraSyncConfidence = 0.38,
            ),
        )

        assertFalse(decision.readiness.ready)
        assertTrue(decision.hardFailures.any {
            it.contains("synchronization")
        })
    }

    @Test
    fun forbiddenSidecarIsHardFailure() {
        val decision = ProductionCapturePreflightEngine().evaluate(
            healthyInput().copy(
                rawSidecarEnabled = true,
            ),
        )

        assertFalse(decision.readiness.ready)
        assertTrue(decision.hardFailures.isNotEmpty())
    }

    private fun healthyInput() =
        ProductionCapturePreflightEngine.Input(
            mode = MotionCaptureMode.MONOCULAR_RGB,
            cameraAvailable = true,
            modelPipelineReady = true,
            calibrationReady = false,
            calibrationConfidence = 0.0,
            thermalScale = 0.90,
            memoryScale = 0.88,
            storageAvailable = true,
            sessionPolicyReady = true,
            subjectVisible = true,
            motionQuality = 0.82,
            subjectLockConfidence = 0.86,
            cameraMotionRisk = 0.08,
            multiCameraSyncConfidence = 0.94,
            captureTimingConfidence = 0.94,
            captureTimingFps = 30.0,
            captureTimingJitterMillis = 1.1,
            captureTimingHealthy = true,
        )

    @Test
    fun nonAdmittedConcurrentRigBlocksMultiview() {
        val decision = ProductionCapturePreflightEngine().evaluate(
            healthyInput().copy(
                mode = MotionCaptureMode.MULTI_VIEW_RGB,
                calibrationReady = true,
                calibrationConfidence = 0.92,
                concurrentRigAdmissible = false,
                concurrentRigScore = 0.35,
            ),
        )
        assertFalse(decision.readiness.ready)
        assertTrue(
            decision.hardFailures.any {
                it.contains("concurrent camera rig")
            },
        )
    }

    @Test
    fun degradedThermalDriftProducesReferenceWarning() {
        val decision = ProductionCapturePreflightEngine().evaluate(
            healthyInput().copy(
                thermalDriftHealthy = false,
                thermalDriftConfidence = 0.42,
            ),
        )
        assertTrue(decision.degraded)
        assertTrue(
            decision.warnings.any {
                it.contains("thermal/extrinsic")
            },
        )
    }

    @Test
    fun weakFaceEyeEvidenceIsReportedForMultiview() {
        val decision = ProductionCapturePreflightEngine().evaluate(
            healthyInput().copy(
                mode = MotionCaptureMode.MULTI_VIEW_RGB,
                calibrationReady = true,
                calibrationConfidence = 0.92,
                faceEyeMetricReady = false,
                faceEyeMetricConfidence = 0.30,
            ),
        )
        assertTrue(
            decision.warnings.any {
                it.contains("face/eye metric")
            },
        )
    }

    @Test
    fun high3dUncertaintyBlocksMetricCapture() {
        val decision = ProductionCapturePreflightEngine().evaluate(
            healthyInput().copy(
                mode = MotionCaptureMode.MULTI_VIEW_RGB,
                calibrationReady = true,
                calibrationConfidence = 0.92,
                reconstructionUncertaintyConfidence = 0.30,
                reconstructionUncertaintyHealthy = false,
            ),
        )
        assertFalse(decision.readiness.ready)
        assertTrue(
            decision.hardFailures.any { it.contains("uncertainty") },
        )
    }

    @Test
    fun weakMultiViewGeometryBlocksMetricCapture() {
        val decision = ProductionCapturePreflightEngine().evaluate(
            healthyInput().copy(
                mode = MotionCaptureMode.MULTI_VIEW_RGB,
                geometricObservabilityConfidence = 0.31,
                geometricObservabilityHealthy = false,
            ),
        )

        assertTrue(
            decision.hardFailures.any {
                it.contains("geometric observability")
            },
        )
    }


}

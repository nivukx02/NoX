
package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.CaptureReadiness
import com.nivukx.nox.core.model.MotionCaptureMode
import kotlin.math.max

/**
 * Production preflight gate for NoX motion-only capture.
 *
 * The engine is deliberately pure: Android/UI code supplies real-time health signals and this
 * class turns them into a deterministic readiness decision. It never silently promotes degraded
 * evidence into "ready".
 */
class ProductionCapturePreflightEngine {

    data class Input(
        val mode: MotionCaptureMode,
        val cameraAvailable: Boolean,
        val modelPipelineReady: Boolean,
        val calibrationReady: Boolean,
        val calibrationConfidence: Double,
        val multiCameraSyncConfidence: Double = 1.0,
        val rigHealthConfidence: Double = 1.0,
        val rigHealthHealthy: Boolean = true,
        val geometricObservabilityConfidence: Double = 1.0,
        val geometricObservabilityHealthy: Boolean = true,
        val reconstructionUncertaintyConfidence: Double = 1.0,
        val reconstructionUncertaintyHealthy: Boolean = true,
        val thermalScale: Double,
        val memoryScale: Double,
        val storageAvailable: Boolean,
        val sessionPolicyReady: Boolean,
        val subjectVisible: Boolean,
        val motionQuality: Double,
        val subjectLockConfidence: Double,
        val cameraMotionRisk: Double,
        val captureTimingConfidence: Double = 1.0,
        val captureTimingFps: Double = 0.0,
        val captureTimingJitterMillis: Double = 0.0,
        val captureTimingHealthy: Boolean = true,
        val pipelineError: String? = null,
        val rawSidecarEnabled: Boolean = false,
        val depthSidecarEnabled: Boolean = false,
        val concurrentRigAdmissible: Boolean = true,
        val concurrentRigScore: Double = 1.0,
        val thermalDriftHealthy: Boolean = true,
        val thermalDriftConfidence: Double = 1.0,
        val faceEyeMetricReady: Boolean = true,
        val faceEyeMetricConfidence: Double = 1.0,
        val hardwareReadoutTimingAvailable: Boolean = false,
        val hardwareReadoutTimingConfidence: Double = 1.0,
        val hardwareReadoutTimingHealthy: Boolean = true,
    )

    data class Policy(
        val minimumMotionQuality: Double = 0.50,
        val minimumSubjectLockConfidence: Double = 0.30,
        val minimumThermalScale: Double = 0.28,
        val minimumMemoryScale: Double = 0.24,
        val maximumCameraMotionRisk: Double = 0.92,
        val minimumCaptureTimingConfidence: Double = 0.60,
        val minimumCaptureTimingFps: Double = 15.0,
        val maximumCaptureTimingP95JitterMillis: Double = 6.0,
        val minimumCalibrationForMultiView: Double = 0.55,
        val minimumCalibrationForMetricMode: Double = 0.78,
        val minimumMultiCameraSyncConfidence: Double = 0.55,
        val minimumRigHealthConfidence: Double = 0.55,
        val minimumGeometricObservabilityConfidence: Double = 0.52,
        val minimumReconstructionUncertaintyConfidence: Double = 0.55,
        val minimumConcurrentRigScore: Double = 0.62,
        val minimumThermalDriftConfidence: Double = 0.55,
        val minimumFaceEyeMetricConfidence: Double = 0.55,
        val minimumHardwareReadoutTimingConfidence: Double = 0.55,
    )

    data class Decision(
        val readiness: CaptureReadiness,
        val mode: MotionCaptureMode,
        val degraded: Boolean,
        val hardFailures: List<String>,
        val warnings: List<String>,
        val recommendations: List<String>,
        val diagnostics: List<String>,
    )

    fun evaluate(
        input: Input,
        policy: Policy = Policy(),
    ): Decision {
        val failures = mutableListOf<String>()
        val warnings = mutableListOf<String>()
        val recommendations = mutableListOf<String>()

        if (!input.cameraAvailable) {
            failures += "capture camera unavailable"
            recommendations += "Open a supported Camera2 stream and verify capture permissions."
        }

        if (!input.modelPipelineReady) {
            failures += "perception models are not ready"
            recommendations += "Wait for all required perception models to initialize."
        }

        if (!input.storageAvailable) {
            failures += "session storage unavailable"
            recommendations += "Free storage space before starting a motion session."
        }

        if (!input.sessionPolicyReady) {
            failures += "motion-only session policy is not ready"
            recommendations += "Disable all RGB/video/raw/depth persistent sidecars."
        }

        if (input.pipelineError != null) {
            failures += "pipeline error: " + input.pipelineError
        }

        if (input.rawSidecarEnabled) {
            failures += "RAW persistence is forbidden for NoX motion-only capture"
        }

        if (input.depthSidecarEnabled) {
            failures += "depth-frame persistence is forbidden for NoX motion-only capture"
        }

        if (
            (
                input.mode == MotionCaptureMode.MULTI_VIEW_RGB ||
                    input.mode == MotionCaptureMode.MULTI_VIEW_RGB_DEPTH
                ) &&
            (
                !input.concurrentRigAdmissible ||
                    input.concurrentRigScore <
                    policy.minimumConcurrentRigScore
                )
        ) {
            failures += "concurrent camera rig is not admitted for the requested capture topology"
            recommendations +=
                "Reduce concurrent streams or select a platform-declared camera combination with adequate headroom."
        }

        if (
            !input.thermalDriftHealthy ||
                input.thermalDriftConfidence <
                policy.minimumThermalDriftConfidence
        ) {
            warnings += "camera thermal/extrinsic drift evidence is degraded"
            recommendations +=
                "Recheck extrinsics after thermal stabilization before reference-grade capture."
        }

        if (
            (
                input.mode == MotionCaptureMode.MULTI_VIEW_RGB ||
                    input.mode == MotionCaptureMode.MULTI_VIEW_RGB_DEPTH
                ) &&
            (
                !input.faceEyeMetricReady ||
                    input.faceEyeMetricConfidence <
                    policy.minimumFaceEyeMetricConfidence
                )
        ) {
            warnings += "face/eye metric evidence is below the reference target"
        }

        if (
            input.hardwareReadoutTimingAvailable &&
            (
                !input.hardwareReadoutTimingHealthy ||
                    input.hardwareReadoutTimingConfidence <
                    policy.minimumHardwareReadoutTimingConfidence
            )
        ) {
            warnings += "hardware readout timing evidence is below the production target"
            recommendations +=
                "Preserve exposure-start timing as canonical and increase temporal uncertainty until readout timing stabilizes."
        }

        if (input.thermalScale < policy.minimumThermalScale) {
            failures += "thermal headroom below safe capture floor"
            recommendations += "Reduce inference load or allow device cooling before recording."
        } else if (input.thermalScale < 0.55) {
            warnings += "thermal headroom reduced"
        }

        if (input.memoryScale < policy.minimumMemoryScale) {
            failures += "memory headroom below safe capture floor"
            recommendations += "Reduce active models or close competing applications."
        } else if (input.memoryScale < 0.50) {
            warnings += "memory headroom reduced"
        }

        if (!input.subjectVisible) {
            failures += "no capture subject detected"
            recommendations += "Place the subject inside the active capture region."
        }

        if (input.motionQuality < policy.minimumMotionQuality) {
            failures += "motion evidence below capture threshold"
        } else if (input.motionQuality < 0.68) {
            warnings += "motion evidence is usable but below production target"
        }

        if (input.subjectLockConfidence < policy.minimumSubjectLockConfidence) {
            warnings += "subject lock confidence is weak"
        }

        if (!input.captureTimingHealthy) {
            warnings += "capture clock health is degraded"
            recommendations += "Stabilize the capture pipeline before the next take or reduce processing load."
        }

        if (
            input.captureTimingConfidence <
                policy.minimumCaptureTimingConfidence
        ) {
            warnings += "capture timing confidence is below production target"
        }

        if (
            input.captureTimingFps > 0.0 &&
            input.captureTimingFps < policy.minimumCaptureTimingFps
        ) {
            warnings += "capture cadence is below the production floor"
        }

        if (
            input.captureTimingJitterMillis >
                policy.maximumCaptureTimingP95JitterMillis
        ) {
            warnings += "capture cadence jitter is above the production target"
        }

        if (
            (
                input.mode == MotionCaptureMode.MULTI_VIEW_RGB ||
                    input.mode == MotionCaptureMode.MULTI_VIEW_RGB_DEPTH
            ) &&
            (
                !input.geometricObservabilityHealthy ||
                    input.geometricObservabilityConfidence <
                    policy.minimumGeometricObservabilityConfidence
            )
        ) {
            failures += "multi-view geometric observability is below the production floor"
            recommendations +=
                "Improve camera baseline and triangulation angle before starting metric multi-view capture."
        }

        if (input.cameraMotionRisk > policy.maximumCameraMotionRisk) {
            failures += "camera motion is too uncertain for reliable geometry"
            recommendations += "Stabilize the device or enable stronger IMU/visual compensation."
        }

        if (
            input.mode == MotionCaptureMode.MULTI_VIEW_RGB ||
            input.mode == MotionCaptureMode.MULTI_VIEW_RGB_DEPTH
        ) {
            if (
                !input.rigHealthHealthy ||
                input.rigHealthConfidence <
                    policy.minimumRigHealthConfidence
            ) {
                failures += "rig camera health is below the production floor"
                recommendations +=
                    "Recover low-trust cameras before accepting metric multi-view output."
            } else if (input.rigHealthConfidence < 0.78) {
                warnings += "rig camera health is reduced"
            }

            if (
                input.multiCameraSyncConfidence <
                    policy.minimumMultiCameraSyncConfidence
            ) {
                failures += "multi-camera clock synchronization is below the production floor"
                recommendations +=
                    "Refresh peer clock synchronization before relying on multi-view reconstruction."
            } else if (
                input.multiCameraSyncConfidence < 0.78
            ) {
                warnings += "multi-camera synchronization confidence is reduced"
            }
            if (!input.calibrationReady) {
                failures += "multi-view mode requested without calibration"
                recommendations += "Run multi-camera calibration or switch to monocular RGB mode."
            }

            if (
                !input.reconstructionUncertaintyHealthy ||
                input.reconstructionUncertaintyConfidence <
                    policy.minimumReconstructionUncertaintyConfidence
            ) {
                failures += "3D reconstruction uncertainty is below the production floor"
                recommendations +=
                    "Hold metric output and reacquire a lower-uncertainty multi-view segment."
            }

            if (input.calibrationConfidence < policy.minimumCalibrationForMultiView) {
                failures += "multi-view calibration confidence is below the production floor"
            } else if (
                input.calibrationConfidence <
                    policy.minimumCalibrationForMetricMode
            ) {
                warnings += "multi-view calibration is usable but metric confidence is reduced"
            }
        }

        if (
            input.mode == MotionCaptureMode.RGB_DEPTH_HYBRID ||
            input.mode == MotionCaptureMode.MULTI_VIEW_RGB_DEPTH
        ) {
            warnings += "depth-enhanced mode is accepted only when the depth provider is genuinely available"
        }

        val degraded =
            warnings.isNotEmpty() ||
                input.thermalScale < 0.55 ||
                input.memoryScale < 0.50 ||
                input.motionQuality < 0.68 ||
                input.subjectLockConfidence < 0.50 ||
                input.captureTimingConfidence < policy.minimumCaptureTimingConfidence ||
                !input.captureTimingHealthy ||
                (input.captureTimingFps > 0.0 && input.captureTimingFps < policy.minimumCaptureTimingFps) ||
                input.captureTimingJitterMillis > policy.maximumCaptureTimingP95JitterMillis ||
                (
                    input.mode == MotionCaptureMode.MULTI_VIEW_RGB ||
                        input.mode == MotionCaptureMode.MULTI_VIEW_RGB_DEPTH
                    ) &&
                    (
                        !input.rigHealthHealthy ||
                            input.rigHealthConfidence <
                            policy.minimumRigHealthConfidence
                    )

        val diagnostics = listOf(
            "cameraAvailable=" + input.cameraAvailable,
            "modelsReady=" + input.modelPipelineReady,
            "calibrationReady=" + input.calibrationReady,
            "calibrationConfidence=" + format(input.calibrationConfidence),
            "multiCameraSyncConfidence=" + format(input.multiCameraSyncConfidence),
            "rigHealthConfidence=" + format(input.rigHealthConfidence),
            "rigHealthHealthy=" + input.rigHealthHealthy,
            "geometricObservabilityConfidence=" +
                format(input.geometricObservabilityConfidence),
            "geometricObservabilityHealthy=" +
                input.geometricObservabilityHealthy,
            "reconstructionUncertaintyConfidence=" +
                format(input.reconstructionUncertaintyConfidence),
            "reconstructionUncertaintyHealthy=" +
                input.reconstructionUncertaintyHealthy,
            "thermalScale=" + format(input.thermalScale),
            "memoryScale=" + format(input.memoryScale),
            "concurrentRigAdmissible=" + input.concurrentRigAdmissible,
            "concurrentRigScore=" + format(input.concurrentRigScore),
            "thermalDriftHealthy=" + input.thermalDriftHealthy,
            "thermalDriftConfidence=" + format(input.thermalDriftConfidence),
            "faceEyeMetricReady=" + input.faceEyeMetricReady,
            "faceEyeMetricConfidence=" + format(input.faceEyeMetricConfidence),
            "hardwareReadoutTimingAvailable=" + input.hardwareReadoutTimingAvailable,
            "hardwareReadoutTimingConfidence=" + format(input.hardwareReadoutTimingConfidence),
            "hardwareReadoutTimingHealthy=" + input.hardwareReadoutTimingHealthy,
            "motionQuality=" + format(input.motionQuality),
            "subjectLockConfidence=" + format(input.subjectLockConfidence),
            "cameraMotionRisk=" + format(input.cameraMotionRisk),
            "captureTimingConfidence=" + format(input.captureTimingConfidence),
            "captureTimingFps=" + format(input.captureTimingFps),
            "captureTimingJitterMillis=" + format(input.captureTimingJitterMillis),
            "captureTimingHealthy=" + input.captureTimingHealthy,
            "mode=" + input.mode.name,
            "hardFailureCount=" + failures.size,
            "warningCount=" + warnings.size,
        )

        val readiness = CaptureReadiness(
            cameraReady = input.cameraAvailable,
            modelReady = input.modelPipelineReady,
            calibrationReady = input.calibrationReady,
            thermalReady = input.thermalScale >= policy.minimumThermalScale,
            storageReady = input.storageAvailable,
            sessionPolicyReady =
                input.sessionPolicyReady &&
                    !input.rawSidecarEnabled &&
                    !input.depthSidecarEnabled,
            subjectVisible = input.subjectVisible,
            motionQuality = input.motionQuality,
            reasons = failures.distinct(),
            captureTimingConfidence = input.captureTimingConfidence,
            captureTimingFps = input.captureTimingFps,
            captureTimingJitterMillis = input.captureTimingJitterMillis,
            captureTimingHealthy = input.captureTimingHealthy,
            multiCameraSyncConfidence = input.multiCameraSyncConfidence,
            multiCameraSyncHealthy =
                input.mode != MotionCaptureMode.MULTI_VIEW_RGB &&
                    input.mode != MotionCaptureMode.MULTI_VIEW_RGB_DEPTH ||
                    input.multiCameraSyncConfidence >=
                        policy.minimumMultiCameraSyncConfidence,
        )

        return Decision(
            readiness = readiness,
            mode = input.mode,
            degraded = degraded,
            hardFailures = failures.distinct(),
            warnings = warnings.distinct(),
            recommendations = recommendations.distinct(),
            diagnostics = diagnostics,
        )
    }

    private fun format(value: Double): String =
        if (!value.isFinite()) {
            "nan"
        } else {
            "%.4f".format(java.util.Locale.US, value)
        }
}

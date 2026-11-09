package com.nivukx.nox.core.capture

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

/**
 * Cross-engine production health evaluator for a live NoX motion-capture session.
 *
 * Availability is not the same as trust. A detector may be returning joints while capture timing,
 * calibration, thermal headroom, or geometric residuals are simultaneously degrading. This pure
 * evaluator turns the evidence already computed by NoX into one deterministic operator state while
 * preserving every underlying signal for diagnostics and export.
 */
class SessionHealthEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumCaptureTimingConfidence: Double = 0.60,
        val targetCaptureTimingConfidence: Double = 0.90,
        val maximumTimingJitterMillis: Double = 6.0,
        val targetTimingJitterMillis: Double = 1.5,
        val minimumInferenceFpsFraction: Double = 0.55,
        val targetInferenceFpsFraction: Double = 0.90,
        val maximumInferenceLatencyMillis: Double = 80.0,
        val targetInferenceLatencyMillis: Double = 24.0,
        val maximumDroppedInferenceFraction: Double = 0.10,
        val minimumSubjectLockConfidence: Double = 0.30,
        val targetSubjectLockConfidence: Double = 0.82,
        val minimumPerceptionConfidence: Double = 0.35,
        val targetPerceptionConfidence: Double = 0.84,
        val minimumReconstructionConfidence: Double = 0.35,
        val targetReconstructionConfidence: Double = 0.86,
        val maximumReprojectionErrorPx: Double = 8.0,
        val targetReprojectionErrorPx: Double = 2.5,
        val minimumTemporalStability: Double = 0.35,
        val targetTemporalStability: Double = 0.88,
        val minimumCalibrationConfidence: Double = 0.45,
        val targetCalibrationConfidence: Double = 0.90,
        val minimumMultiCameraSyncConfidence: Double = 0.45,
        val targetMultiCameraSyncConfidence: Double = 0.90,
        val maximumCameraMotionRisk: Double = 0.90,
        val targetCameraMotionRisk: Double = 0.18,
        val minimumMotionEvidenceConfidence: Double = 0.30,
        val targetMotionEvidenceConfidence: Double = 0.84,
        val minimumGeometricObservabilityConfidence: Double = 0.52,
        val targetGeometricObservabilityConfidence: Double = 0.84,
        val minimumReconstructionUncertaintyConfidence: Double = 0.55,
        val targetReconstructionUncertaintyConfidence: Double = 0.86,
        val maximumOcclusionRisk: Double = 0.90,
        val targetOcclusionRisk: Double = 0.20,
        val minimumMeshConfidence: Double = 0.25,
        val targetMeshConfidence: Double = 0.82,
        val minimumFaceConfidence: Double = 0.15,
        val targetFaceConfidence: Double = 0.80,
        val minimumEyeConfidence: Double = 0.12,
        val targetEyeConfidence: Double = 0.78,
        val minimumEyeTemporalConfidence: Double = 0.30,
        val targetEyeTemporalConfidence: Double = 0.82,
        val minimumHandConfidence: Double = 0.15,
        val targetHandConfidence: Double = 0.78,
        val minimumThermalScale: Double = 0.28,
        val targetThermalScale: Double = 0.75,
        val minimumMemoryScale: Double = 0.24,
        val targetMemoryScale: Double = 0.75,
        val degradedScore: Double = 0.58,
        val productionScore: Double = 0.78,
        val criticalScore: Double = 0.35,
    ) {
        init {
            require(minimumCaptureTimingConfidence in 0.0..1.0)
            require(targetCaptureTimingConfidence in 0.0..1.0)
            require(targetCaptureTimingConfidence >= minimumCaptureTimingConfidence)
            require(maximumTimingJitterMillis > 0.0)
            require(targetTimingJitterMillis >= 0.0)
            require(maximumInferenceLatencyMillis > 0.0)
            require(targetInferenceLatencyMillis >= 0.0)
            require(maximumDroppedInferenceFraction in 0.0..1.0)
            require(minimumSubjectLockConfidence in 0.0..1.0)
            require(targetSubjectLockConfidence in 0.0..1.0)
            require(minimumPerceptionConfidence in 0.0..1.0)
            require(targetPerceptionConfidence in 0.0..1.0)
            require(minimumReconstructionConfidence in 0.0..1.0)
            require(targetReconstructionConfidence in 0.0..1.0)
            require(maximumReprojectionErrorPx > 0.0)
            require(targetReprojectionErrorPx >= 0.0)
            require(minimumTemporalStability in 0.0..1.0)
            require(targetTemporalStability in 0.0..1.0)
            require(minimumCalibrationConfidence in 0.0..1.0)
            require(targetCalibrationConfidence in 0.0..1.0)
            require(minimumMultiCameraSyncConfidence in 0.0..1.0)
            require(targetMultiCameraSyncConfidence in 0.0..1.0)
            require(maximumCameraMotionRisk in 0.0..1.0)
            require(targetCameraMotionRisk in 0.0..1.0)
            require(minimumMotionEvidenceConfidence in 0.0..1.0)
            require(targetMotionEvidenceConfidence in 0.0..1.0)
            require(
                targetMotionEvidenceConfidence >=
                    minimumMotionEvidenceConfidence,
            )
            require(
                minimumGeometricObservabilityConfidence in 0.0..1.0,
            )
            require(
                targetGeometricObservabilityConfidence in 0.0..1.0,
            )
            require(
                targetGeometricObservabilityConfidence >=
                    minimumGeometricObservabilityConfidence,
            )
            require(
                minimumReconstructionUncertaintyConfidence in 0.0..1.0,
            )
            require(
                targetReconstructionUncertaintyConfidence in 0.0..1.0,
            )
            require(
                targetReconstructionUncertaintyConfidence >=
                    minimumReconstructionUncertaintyConfidence,
            )
            require(maximumOcclusionRisk in 0.0..1.0)
            require(targetOcclusionRisk in 0.0..1.0)
            require(minimumMeshConfidence in 0.0..1.0)
            require(targetMeshConfidence in 0.0..1.0)
            require(minimumFaceConfidence in 0.0..1.0)
            require(targetFaceConfidence in 0.0..1.0)
            require(minimumEyeConfidence in 0.0..1.0)
            require(targetEyeConfidence in 0.0..1.0)
            require(minimumEyeTemporalConfidence in 0.0..1.0)
            require(targetEyeTemporalConfidence in 0.0..1.0)
            require(
                targetEyeTemporalConfidence >=
                    minimumEyeTemporalConfidence,
            )
            require(minimumHandConfidence in 0.0..1.0)
            require(targetHandConfidence in 0.0..1.0)
            require(minimumThermalScale in 0.0..1.0)
            require(targetThermalScale in 0.0..1.0)
            require(minimumMemoryScale in 0.0..1.0)
            require(targetMemoryScale in 0.0..1.0)
            require(degradedScore in 0.0..1.0)
            require(productionScore in 0.0..1.0)
            require(criticalScore in 0.0..1.0)
            require(criticalScore < degradedScore)
            require(degradedScore < productionScore)
        }
    }

    enum class State {
        NO_DATA,
        PRODUCTION,
        DEGRADED,
        CRITICAL,
        UNSAFE,
    }

    enum class SignalClass {
        CAPTURE_TIMING,
        INFERENCE_RUNTIME,
        SUBJECT_LOCK,
        PERCEPTION,
        RECONSTRUCTION,
        TEMPORAL,
        CALIBRATION,
        MULTI_CAMERA_SYNC,
        CAMERA_MOTION,
        MOTION_EVIDENCE,
        GEOMETRIC_OBSERVABILITY,
        RECONSTRUCTION_UNCERTAINTY,
        OCCLUSION,
        DENSE_MESH,
        FACE,
        EYES,
        EYE_TEMPORAL_CONSISTENCY,
        HANDS,
        THERMAL,
        MEMORY,
    }

    data class Signal(
        val type: SignalClass,
        val score: Double,
        val weight: Double,
        val hardFailure: Boolean,
        val label: String,
        val detail: String,
    ) {
        init {
            require(score in 0.0..1.0)
            require(weight > 0.0)
            require(label.isNotBlank())
        }

        val weightedContribution: Double
            get() = score * weight
    }

    data class Input(
        val timestampNanos: Long,
        val targetCaptureFps: Double,
        val captureTimingConfidence: Double,
        val captureTimingFps: Double,
        val captureTimingJitterMillis: Double,
        val captureTimingHealthy: Boolean,
        val inferenceFps: Double,
        val inferenceLatencyMillis: Double,
        val droppedInferenceFrames: Long,
        val capturedFrames: Long,
        val subjectLockConfidence: Double,
        val perceptionConfidence: Double,
        val reconstructionConfidence: Double,
        val reconstructionReprojectionErrorPx: Double,
        val temporalStability: Double,
        val calibrationConfidence: Double,
        val multiCameraSyncConfidence: Double = 1.0,
        val cameraMotionRisk: Double,
        val motionEvidenceConfidence: Double = 1.0,
        val motionEvidenceDegraded: Boolean = false,
        val geometricObservabilityConfidence: Double = 1.0,
        val geometricObservabilityDegraded: Boolean = false,
        val reconstructionUncertaintyConfidence: Double = 1.0,
        val reconstructionUncertaintyDegraded: Boolean = false,
        val reconstructionUncertaintyP95Meters: Double = 0.0,
        val occlusionRisk: Double,
        val meshConfidence: Double,
        val faceConfidence: Double,
        val eyeConfidence: Double,
        val eyeTemporalConfidence: Double = 1.0,
        val eyeTemporalDegraded: Boolean = false,
        val handConfidence: Double,
        val thermalScale: Double,
        val memoryScale: Double,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(targetCaptureFps >= 0.0)
            require(captureTimingConfidence in 0.0..1.0)
            require(captureTimingFps >= 0.0)
            require(captureTimingJitterMillis >= 0.0)
            require(inferenceFps >= 0.0)
            require(inferenceLatencyMillis >= 0.0)
            require(droppedInferenceFrames >= 0L)
            require(capturedFrames >= 0L)
            require(subjectLockConfidence in 0.0..1.0)
            require(perceptionConfidence in 0.0..1.0)
            require(reconstructionConfidence in 0.0..1.0)
            require(reconstructionReprojectionErrorPx >= 0.0)
            require(temporalStability in 0.0..1.0)
            require(calibrationConfidence in 0.0..1.0)
            require(cameraMotionRisk in 0.0..1.0)
            require(motionEvidenceConfidence in 0.0..1.0)
            require(geometricObservabilityConfidence in 0.0..1.0)
            require(
                reconstructionUncertaintyConfidence in 0.0..1.0,
            )
            require(reconstructionUncertaintyP95Meters >= 0.0)
            require(occlusionRisk in 0.0..1.0)
            require(meshConfidence in 0.0..1.0)
            require(faceConfidence in 0.0..1.0)
            require(eyeConfidence in 0.0..1.0)
            require(eyeTemporalConfidence in 0.0..1.0)
            require(handConfidence in 0.0..1.0)
            require(thermalScale in 0.0..1.0)
            require(memoryScale in 0.0..1.0)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val state: State,
        val score: Double,
        val captureStability: Double,
        val perceptionStability: Double,
        val reconstructionStability: Double,
        val runtimeStability: Double,
        val operatorConfidence: Double,
        val hardFailures: List<String>,
        val warnings: List<String>,
        val recommendations: List<String>,
        val signals: List<Signal>,
        val bottleneck: Signal?,
    ) {
        val productionReady: Boolean
            get() = state == State.PRODUCTION

        val degraded: Boolean
            get() = state == State.DEGRADED

        val unsafe: Boolean
            get() = state == State.CRITICAL || state == State.UNSAFE
    }

    fun evaluate(input: Input): Result {
        val droppedFraction = if (input.capturedFrames <= 0L) {
            0.0
        } else {
            (
                input.droppedInferenceFrames.toDouble() /
                    input.capturedFrames.toDouble()
                ).coerceIn(0.0, 1.0)
        }

        val signals = listOf(
            timingSignal(input),
            inferenceSignal(input, droppedFraction),
            boundedSignal(
                type = SignalClass.SUBJECT_LOCK,
                value = input.subjectLockConfidence,
                minimum = config.minimumSubjectLockConfidence,
                target = config.targetSubjectLockConfidence,
                hardFailure =
                    input.subjectLockConfidence <
                        config.minimumSubjectLockConfidence,
                label = "subject lock",
                detail = percent(input.subjectLockConfidence) + " confidence",
                weight = 1.12,
            ),
            boundedSignal(
                type = SignalClass.PERCEPTION,
                value = input.perceptionConfidence,
                minimum = config.minimumPerceptionConfidence,
                target = config.targetPerceptionConfidence,
                hardFailure =
                    input.perceptionConfidence <
                        config.minimumPerceptionConfidence,
                label = "perception",
                detail = percent(input.perceptionConfidence) + " fused confidence",
                weight = 1.12,
            ),
            boundedSignal(
                type = SignalClass.RECONSTRUCTION,
                value = input.reconstructionConfidence,
                minimum = config.minimumReconstructionConfidence,
                target = config.targetReconstructionConfidence,
                hardFailure =
                    input.reconstructionConfidence <
                        config.minimumReconstructionConfidence,
                label = "reconstruction",
                detail =
                    percent(input.reconstructionConfidence) +
                        " geometry confidence",
                weight = 1.30,
            ),
            inverseSignal(
                type = SignalClass.RECONSTRUCTION,
                value = input.reconstructionReprojectionErrorPx,
                preferredMaximum = config.targetReprojectionErrorPx,
                hardMaximum = config.maximumReprojectionErrorPx,
                label = "reprojection residual",
                detail =
                    format(input.reconstructionReprojectionErrorPx) + " px",
                weight = 0.92,
            ),
            boundedSignal(
                type = SignalClass.TEMPORAL,
                value = input.temporalStability,
                minimum = config.minimumTemporalStability,
                target = config.targetTemporalStability,
                hardFailure =
                    input.temporalStability <
                        config.minimumTemporalStability,
                label = "temporal continuity",
                detail = percent(input.temporalStability) + " stable",
                weight = 1.08,
            ),
            boundedSignal(
                type = SignalClass.CALIBRATION,
                value = input.calibrationConfidence,
                minimum = config.minimumCalibrationConfidence,
                target = config.targetCalibrationConfidence,
                hardFailure =
                    input.calibrationConfidence <
                        config.minimumCalibrationConfidence,
                label = "calibration",
                detail =
                    percent(input.calibrationConfidence) +
                        " rig confidence",
                weight = 0.92,
            ),
            boundedSignal(
                type = SignalClass.MULTI_CAMERA_SYNC,
                value = input.multiCameraSyncConfidence,
                minimum = config.minimumMultiCameraSyncConfidence,
                target = config.targetMultiCameraSyncConfidence,
                hardFailure =
                    input.multiCameraSyncConfidence <
                        config.minimumMultiCameraSyncConfidence &&
                        input.calibrationConfidence > 0.0,
                label = "multi-camera synchronization",
                detail =
                    percent(input.multiCameraSyncConfidence) +
                        " rig clock confidence",
                weight = 0.92,
            ),
            inverseSignal(
                type = SignalClass.CAMERA_MOTION,
                value = input.cameraMotionRisk,
                preferredMaximum = config.targetCameraMotionRisk,
                hardMaximum = config.maximumCameraMotionRisk,
                label = "camera motion",
                detail =
                    percent(input.cameraMotionRisk) + " motion risk",
                weight = 0.90,
            ),
            boundedSignal(
                type = SignalClass.MOTION_EVIDENCE,
                value = input.motionEvidenceConfidence,
                minimum = config.minimumMotionEvidenceConfidence,
                target = config.targetMotionEvidenceConfidence,
                hardFailure =
                    input.motionEvidenceConfidence <
                        config.minimumMotionEvidenceConfidence ||
                        input.motionEvidenceDegraded,
                label = "motion evidence",
                detail =
                    if (input.motionEvidenceDegraded) {
                        "degraded evidence"
                    } else {
                        percent(input.motionEvidenceConfidence) +
                            " confidence"
                    },
                weight = 0.92,
            ),
            boundedSignal(
                type = SignalClass.GEOMETRIC_OBSERVABILITY,
                value = input.geometricObservabilityConfidence,
                minimum = config.minimumGeometricObservabilityConfidence,
                target = config.targetGeometricObservabilityConfidence,
                hardFailure =
                    input.geometricObservabilityConfidence <
                        config.minimumGeometricObservabilityConfidence ||
                        input.geometricObservabilityDegraded,
                label = "3D geometric observability",
                detail =
                    if (input.geometricObservabilityDegraded) {
                        "weak triangulation geometry"
                    } else {
                        percent(input.geometricObservabilityConfidence) +
                            " confidence"
                    },
                weight = 0.96,
            ),
            boundedSignal(
                type = SignalClass.RECONSTRUCTION_UNCERTAINTY,
                value = input.reconstructionUncertaintyConfidence,
                minimum =
                    config.minimumReconstructionUncertaintyConfidence,
                target =
                    config.targetReconstructionUncertaintyConfidence,
                hardFailure =
                    input.reconstructionUncertaintyConfidence <
                        config.minimumReconstructionUncertaintyConfidence ||
                        input.reconstructionUncertaintyDegraded,
                label = "3D metric uncertainty",
                detail =
                    percent(input.reconstructionUncertaintyConfidence) +
                        " confidence · P95 " +
                        format(input.reconstructionUncertaintyP95Meters) +
                        " m",
                weight = 1.08,
            ),
            inverseSignal(
                type = SignalClass.OCCLUSION,
                value = input.occlusionRisk,
                preferredMaximum = config.targetOcclusionRisk,
                hardMaximum = config.maximumOcclusionRisk,
                label = "occlusion",
                detail =
                    percent(input.occlusionRisk) + " risk",
                weight = 0.88,
            ),
            boundedSignal(
                type = SignalClass.DENSE_MESH,
                value = input.meshConfidence,
                minimum = config.minimumMeshConfidence,
                target = config.targetMeshConfidence,
                hardFailure = false,
                label = "dense mesh",
                detail = percent(input.meshConfidence) + " mesh confidence",
                weight = 0.72,
            ),
            boundedSignal(
                type = SignalClass.FACE,
                value = input.faceConfidence,
                minimum = config.minimumFaceConfidence,
                target = config.targetFaceConfidence,
                hardFailure = false,
                label = "face tracking",
                detail = percent(input.faceConfidence) + " confidence",
                weight = 0.44,
            ),
            boundedSignal(
                type = SignalClass.EYES,
                value = input.eyeConfidence,
                minimum = config.minimumEyeConfidence,
                target = config.targetEyeConfidence,
                hardFailure = false,
                label = "eye tracking",
                detail = percent(input.eyeConfidence) + " confidence",
                weight = 0.44,
            ),
            boundedSignal(
                type = SignalClass.EYE_TEMPORAL_CONSISTENCY,
                value = input.eyeTemporalConfidence,
                minimum = config.minimumEyeTemporalConfidence,
                target = config.targetEyeTemporalConfidence,
                hardFailure =
                    input.eyeTemporalConfidence <
                        config.minimumEyeTemporalConfidence ||
                        input.eyeTemporalDegraded,
                label = "3D eye temporal consistency",
                detail =
                    if (input.eyeTemporalDegraded) {
                        "bilateral eye geometry degraded"
                    } else {
                        percent(input.eyeTemporalConfidence) +
                            " confidence"
                    },
                weight = 0.56,
            ),
            boundedSignal(
                type = SignalClass.HANDS,
                value = input.handConfidence,
                minimum = config.minimumHandConfidence,
                target = config.targetHandConfidence,
                hardFailure = false,
                label = "hand tracking",
                detail = percent(input.handConfidence) + " confidence",
                weight = 0.44,
            ),
            boundedSignal(
                type = SignalClass.THERMAL,
                value = input.thermalScale,
                minimum = config.minimumThermalScale,
                target = config.targetThermalScale,
                hardFailure =
                    input.thermalScale <
                        config.minimumThermalScale,
                label = "thermal headroom",
                detail = percent(input.thermalScale) + " available",
                weight = 0.82,
            ),
            boundedSignal(
                type = SignalClass.MEMORY,
                value = input.memoryScale,
                minimum = config.minimumMemoryScale,
                target = config.targetMemoryScale,
                hardFailure =
                    input.memoryScale <
                        config.minimumMemoryScale,
                label = "memory headroom",
                detail = percent(input.memoryScale) + " available",
                weight = 0.82,
            ),
        )

        val weightedScore = weightedMean(signals)
        val geometricScore = geometricMean(signals)
        val score = (
            weightedScore * 0.68 +
                geometricScore * 0.32
            ).coerceIn(0.0, 1.0)

        val hardFailures = signals
            .filter { it.hardFailure }
            .sortedByDescending { it.weight }
            .map {
                it.label + ": " + it.detail
            }

        val warnings = signals
            .filter { !it.hardFailure && it.score < 0.58 }
            .sortedByDescending {
                it.weight * (1.0 - it.score)
            }
            .take(6)
            .map {
                "degraded " + it.label + ": " + it.detail
            }

        val recommendations = buildRecommendations(
            signals = signals,
            droppedFraction = droppedFraction,
        )

        val captureSignals = signals.filter {
            it.type == SignalClass.CAPTURE_TIMING ||
                it.type == SignalClass.CALIBRATION ||
                it.type == SignalClass.MULTI_CAMERA_SYNC ||
                it.type == SignalClass.CAMERA_MOTION ||
                it.type == SignalClass.OCCLUSION
        }

        val perceptionSignals = signals.filter {
            it.type == SignalClass.SUBJECT_LOCK ||
                it.type == SignalClass.PERCEPTION ||
                it.type == SignalClass.FACE ||
                it.type == SignalClass.EYES ||
                it.type == SignalClass.HANDS
        }

        val reconstructionSignals = signals.filter {
            it.type == SignalClass.RECONSTRUCTION ||
                it.type == SignalClass.TEMPORAL ||
                it.type == SignalClass.DENSE_MESH
        }

        val runtimeSignals = signals.filter {
            it.type == SignalClass.INFERENCE_RUNTIME ||
                it.type == SignalClass.THERMAL ||
                it.type == SignalClass.MEMORY
        }

        val captureStability = weightedMean(captureSignals)
        val perceptionStability = weightedMean(perceptionSignals)
        val reconstructionStability = weightedMean(reconstructionSignals)
        val runtimeStability = weightedMean(runtimeSignals)

        val state = when {
            score < config.criticalScore &&
                hardFailures.isNotEmpty() ->
                State.UNSAFE

            score < config.criticalScore ->
                State.CRITICAL

            hardFailures.isNotEmpty() ->
                State.CRITICAL

            score < config.productionScore ->
                State.DEGRADED

            else ->
                State.PRODUCTION
        }

        val bottleneck = signals
            .filter { it.weight >= 0.55 }
            .minByOrNull { it.score }

        val operatorConfidence = (
            score * 0.48 +
                captureStability * 0.18 +
                perceptionStability * 0.14 +
                reconstructionStability * 0.14 +
                runtimeStability * 0.06
            ).coerceIn(0.0, 1.0)

        return Result(
            timestampNanos = input.timestampNanos,
            state = state,
            score = score,
            captureStability = captureStability,
            perceptionStability = perceptionStability,
            reconstructionStability = reconstructionStability,
            runtimeStability = runtimeStability,
            operatorConfidence = operatorConfidence,
            hardFailures = hardFailures.distinct(),
            warnings = warnings.distinct(),
            recommendations = recommendations.distinct(),
            signals = signals,
            bottleneck = bottleneck,
        )
    }

    private fun timingSignal(
        input: Input,
    ): Signal {
        val cadenceScore = if (
            input.targetCaptureFps <= 0.0 ||
            input.captureTimingFps <= 0.0
        ) {
            0.68
        } else {
            val error = abs(
                input.captureTimingFps -
                    input.targetCaptureFps,
            ) /
                input.targetCaptureFps.coerceAtLeast(1.0)

            (
                1.0 - error * 2.0
            ).coerceIn(0.0, 1.0)
        }

        val jitterScore = (
            1.0 -
                (
                    input.captureTimingJitterMillis -
                        config.targetTimingJitterMillis
                ) /
                    (
                        config.maximumTimingJitterMillis -
                            config.targetTimingJitterMillis
                    ).coerceAtLeast(0.1)
            ).coerceIn(0.0, 1.0)

        val healthScore = if (
            input.captureTimingHealthy
        ) {
            1.0
        } else {
            0.25
        }

        val score = (
            input.captureTimingConfidence * 0.54 +
                cadenceScore * 0.18 +
                jitterScore * 0.18 +
                healthScore * 0.10
            ).coerceIn(0.0, 1.0)

        val hardFailure =
            !input.captureTimingHealthy &&
                input.captureTimingConfidence <
                    config.minimumCaptureTimingConfidence

        return Signal(
            type = SignalClass.CAPTURE_TIMING,
            score = score,
            weight = 1.34,
            hardFailure = hardFailure,
            label = "capture timing",
            detail =
                format(input.captureTimingFps) + " FPS, " +
                    format(input.captureTimingJitterMillis) + " ms p95 jitter, " +
                    percent(input.captureTimingConfidence) + " confidence",
        )
    }

    private fun inferenceSignal(
        input: Input,
        droppedFraction: Double,
    ): Signal {
        val fpsScore = if (
            input.targetCaptureFps <= 0.0 ||
            input.inferenceFps <= 0.0
        ) {
            0.55
        } else {
            val fraction = input.inferenceFps /
                input.targetCaptureFps

            (
                (fraction - config.minimumInferenceFpsFraction) /
                    (
                        config.targetInferenceFpsFraction -
                            config.minimumInferenceFpsFraction
                    ).coerceAtLeast(0.05)
            ).coerceIn(0.0, 1.0)
        }

        val latencyScore = (
            1.0 -
                (
                    input.inferenceLatencyMillis -
                        config.targetInferenceLatencyMillis
                ) /
                    (
                        config.maximumInferenceLatencyMillis -
                            config.targetInferenceLatencyMillis
                    ).coerceAtLeast(0.1)
            ).coerceIn(0.0, 1.0)

        val dropScore = (
            1.0 -
                droppedFraction /
                    config.maximumDroppedInferenceFraction.coerceAtLeast(0.001)
        ).coerceIn(0.0, 1.0)

        val score = (
            fpsScore * 0.42 +
                latencyScore * 0.32 +
                dropScore * 0.26
            ).coerceIn(0.0, 1.0)

        val hardFailure =
            fpsScore <= 0.0 ||
                input.inferenceLatencyMillis >
                    config.maximumInferenceLatencyMillis

        return Signal(
            type = SignalClass.INFERENCE_RUNTIME,
            score = score,
            weight = 1.02,
            hardFailure = hardFailure,
            label = "inference runtime",
            detail =
                format(input.inferenceFps) + " inference FPS, " +
                    format(input.inferenceLatencyMillis) + " ms latency, " +
                    percent(droppedFraction) + " dropped",
        )
    }

    private fun boundedSignal(
        type: SignalClass,
        value: Double,
        minimum: Double,
        target: Double,
        hardFailure: Boolean,
        label: String,
        detail: String,
        weight: Double,
    ): Signal {
        val span = (
            target - minimum
        ).coerceAtLeast(1e-6)

        val score = (
            (value - minimum) / span
        ).coerceIn(0.0, 1.0)

        return Signal(
            type = type,
            score = score,
            weight = weight,
            hardFailure = hardFailure,
            label = label,
            detail = detail,
        )
    }

    private fun inverseSignal(
        type: SignalClass,
        value: Double,
        preferredMaximum: Double,
        hardMaximum: Double,
        label: String,
        detail: String,
        weight: Double,
    ): Signal {
        val preferred = preferredMaximum.coerceAtLeast(0.0)
        val hard = max(
            hardMaximum,
            preferred + 1e-6,
        )

        val score = when {
            value <= preferred -> 1.0
            value >= hard -> 0.0
            else ->
                1.0 -
                    (value - preferred) /
                        (hard - preferred)
        }.coerceIn(0.0, 1.0)

        return Signal(
            type = type,
            score = score,
            weight = weight,
            hardFailure = value >= hard,
            label = label,
            detail = detail,
        )
    }

    private fun buildRecommendations(
        signals: List<Signal>,
        droppedFraction: Double,
    ): List<String> {
        val recommendations = mutableListOf<String>()

        signals
            .sortedByDescending {
                it.weight * (1.0 - it.score)
            }
            .take(5)
            .forEach { signal ->
                when (signal.type) {
                    SignalClass.CAPTURE_TIMING ->
                        recommendations +=
                            "Stabilize capture cadence or reduce processing contention before the next take."

                    SignalClass.INFERENCE_RUNTIME ->
                        recommendations +=
                            "Reduce active inference load or switch to a lower-latency motion profile."

                    SignalClass.SUBJECT_LOCK ->
                        recommendations +=
                            "Keep the dominant subject inside the tracking volume until identity confidence recovers."

                    SignalClass.PERCEPTION ->
                        recommendations +=
                            "Improve framing and illumination or reduce the number of concurrent subjects."

                    SignalClass.RECONSTRUCTION ->
                        recommendations +=
                            "Improve calibrated coverage and reject unstable geometry before exporting the take."

                    SignalClass.TEMPORAL ->
                        recommendations +=
                            "Reduce abrupt camera movement and preserve continuous observations around fast actions."

                    SignalClass.CALIBRATION ->
                        recommendations +=
                            "Re-run multi-camera calibration before relying on metric reconstruction."

                    SignalClass.MULTI_CAMERA_SYNC ->
                        recommendations +=
                            "Re-establish low-latency camera clock synchronization before trusting multi-view output."

                    SignalClass.MOTION_EVIDENCE ->
                        recommendations +=
                            "Improve subject motion evidence and avoid committing frames dominated by camera interference."

                    SignalClass.GEOMETRIC_OBSERVABILITY ->
                        recommendations +=
                            "Increase camera baseline, coverage, or subject visibility so 3D geometry is well-conditioned."

                    SignalClass.RECONSTRUCTION_UNCERTAINTY ->
                        recommendations +=
                            "Reduce metric uncertainty before exporting trusted 3D motion."

                    SignalClass.EYE_TEMPORAL_CONSISTENCY ->
                        recommendations +=
                            "Maintain stable face visibility and temporal eye tracking before committing gaze-driven output."

                    SignalClass.CAMERA_MOTION ->
                        recommendations +=
                            "Stabilize the camera or increase IMU/visual motion compensation."

                    SignalClass.OCCLUSION ->
                        recommendations +=
                            "Increase camera coverage or preserve a clearer view of the subject's limbs."

                    SignalClass.DENSE_MESH ->
                        recommendations +=
                            "Increase spatial coverage or reduce mesh extraction pressure during unstable frames."

                    SignalClass.FACE,
                    SignalClass.EYES ->
                        recommendations +=
                            "Keep the face within the dedicated tracking camera field of view."

                    SignalClass.HANDS ->
                        recommendations +=
                            "Keep hands separated from severe self-occlusion when hand articulation matters."

                    SignalClass.THERMAL ->
                        recommendations +=
                            "Allow thermal headroom or reduce model concurrency before long takes."

                    SignalClass.MEMORY ->
                        recommendations +=
                            "Reduce active buffers and competing workloads before recording a long session."
                }
            }

        if (
            droppedFraction >
                config.maximumDroppedInferenceFraction
        ) {
            recommendations +=
                "Investigate inference queue pressure before trusting the full-resolution export."
        }

        return recommendations
    }

    private fun weightedMean(
        signals: List<Signal>,
    ): Double {
        if (signals.isEmpty()) return 0.0

        val denominator = signals
            .sumOf { it.weight }
            .coerceAtLeast(1e-6)

        return (
            signals.sumOf { it.weightedContribution } /
                denominator
            ).coerceIn(0.0, 1.0)
    }

    private fun geometricMean(
        signals: List<Signal>,
    ): Double {
        if (signals.isEmpty()) return 0.0

        val denominator = signals
            .sumOf { it.weight }
            .coerceAtLeast(1e-6)

        var weightedLog = 0.0

        signals.forEach { signal ->
            val safeScore = signal.score.coerceIn(0.005, 1.0)
            weightedLog +=
                kotlin.math.ln(safeScore) *
                    signal.weight
        }

        return exp(
            weightedLog / denominator,
        ).coerceIn(0.0, 1.0)
    }

    private fun percent(value: Double): String =
        (value.coerceIn(0.0, 1.0) * 100.0)
            .toInt()
            .toString() + "%"

    private fun format(value: Double): String =
        if (!value.isFinite()) {
            "nan"
        } else {
            "%.2f".format(java.util.Locale.US, value)
        }
}

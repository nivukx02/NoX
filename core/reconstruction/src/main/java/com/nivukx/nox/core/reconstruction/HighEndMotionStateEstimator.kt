package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * High-end temporal motion state estimator.
 *
 * This stage is deliberately model/provider neutral. It operates after perception and geometric
 * reconstruction and before the motion/session contract is finalized. Its job is not to invent
 * missing anatomy. It predicts a bounded state from recent evidence, rejects implausible
 * measurements with a robust innovation gate, preserves measured micro-motion, and exposes
 * diagnostics that upstream quality gates can use.
 *
 * Design goals:
 *  - second-order state prediction using position, velocity and acceleration;
 *  - confidence-aware measurement blending;
 *  - Huber-style robust weighting for outlier rejection;
 *  - bounded occlusion carry-forward;
 *  - velocity, acceleration and jerk limits;
 *  - explicit camera-motion and occlusion penalties;
 *  - no hidden smoothing of high-confidence high-frequency motion;
 *  - deterministic behavior for identical inputs;
 *  - timestamp regression recovery;
 *  - per-joint uncertainty accounting;
 *  - explicit quality diagnostics for production gating.
 *
 * The estimator is intentionally conservative. A low-confidence measurement can be held against a
 * prediction for a bounded number of frames, but the stage refuses to extrapolate indefinitely.
 */
class HighEndMotionStateEstimator(
    private var config: Config = Config(),
) {
    data class Config(
        val minimumConfidence: Double = 0.10,
        val positionBlendMin: Double = 0.12,
        val positionBlendMax: Double = 0.96,
        val velocityGain: Double = 0.68,
        val accelerationGain: Double = 0.34,
        val jerkGain: Double = 0.18,
        val predictionHorizonSeconds: Double = 0.12,
        val maximumOcclusionFrames: Int = 7,
        val softInnovationGateMeters: Double = 0.075,
        val hardInnovationGateMeters: Double = 0.22,
        val uncertaintyFloorMeters: Double = 0.0025,
        val uncertaintyCeilingMeters: Double = 0.30,
        val measurementNoiseMeters: Double = 0.018,
        val velocityLimitMetersPerSecond: Double = 12.0,
        val accelerationLimitMetersPerSecond2: Double = 55.0,
        val jerkLimitMetersPerSecond3: Double = 420.0,
        val cameraMotionSuppression: Double = 0.65,
        val occlusionSuppression: Double = 0.60,
        val microMotionPreservation: Double = 0.88,
        val temporalContinuityWeight: Double = 0.76,
        val highConfidenceResidualTrust: Double = 0.90,
        val lowConfidencePredictionTrust: Double = 0.86,
        val uncertaintyGrowthPerMissingFrame: Double = 0.16,
        val uncertaintyGrowthPerSecond: Double = 0.18,
        val qualityResidualScaleMeters: Double = 0.045,
        val qualityVelocityScaleMetersPerSecond: Double = 3.0,
        val qualityAccelerationScaleMetersPerSecond2: Double = 14.0,
        val temporalMemoryAlpha: Double = 0.24,
        val temporalMemoryMinimumTrust: Double = 0.35,
    ) {
        init {
            require(minimumConfidence in 0.0..1.0)
            require(positionBlendMin in 0.0..1.0)
            require(positionBlendMax in positionBlendMin..1.0)
            require(velocityGain in 0.0..1.0)
            require(accelerationGain in 0.0..1.0)
            require(jerkGain in 0.0..1.0)
            require(predictionHorizonSeconds in 0.01..0.50)
            require(maximumOcclusionFrames in 0..60)
            require(softInnovationGateMeters > 0.0)
            require(hardInnovationGateMeters >= softInnovationGateMeters)
            require(uncertaintyFloorMeters > 0.0)
            require(uncertaintyCeilingMeters >= uncertaintyFloorMeters)
            require(measurementNoiseMeters > 0.0)
            require(velocityLimitMetersPerSecond > 0.0)
            require(accelerationLimitMetersPerSecond2 > 0.0)
            require(jerkLimitMetersPerSecond3 > 0.0)
            require(cameraMotionSuppression in 0.0..1.0)
            require(occlusionSuppression in 0.0..1.0)
            require(microMotionPreservation in 0.0..1.0)
            require(temporalContinuityWeight in 0.0..1.0)
            require(highConfidenceResidualTrust in 0.0..1.0)
            require(lowConfidencePredictionTrust in 0.0..1.0)
            require(uncertaintyGrowthPerMissingFrame >= 0.0)
            require(uncertaintyGrowthPerSecond >= 0.0)
            require(temporalMemoryAlpha in 0.01..1.0)
            require(temporalMemoryMinimumTrust in 0.0..1.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val skeleton: SkeletonFrame,
        val cameraMotionRisk: Double = 0.0,
        val occlusionRisk: Double = 0.0,
        val synchronizationConfidence: Double = 1.0,
        val preserveMicroMotion: Boolean = true,
    ) {
        init {
            require(timestampNanos > 0L)
        }
    }

    data class JointDiagnostics(
        val landmarkId: Int,
        val confidence: Double,
        val innovationMeters: Double,
        val normalizedInnovation: Double,
        val measurementWeight: Double,
        val predictionWeight: Double,
        val velocityMetersPerSecond: Double,
        val accelerationMetersPerSecond2: Double,
        val uncertaintyMeters: Double,
        val occlusionFrames: Int,
        val measurementAccepted: Boolean,
        val predictionOnly: Boolean,
        val microMotionMeters: Double,
        val temporalMemoryScore: Double = 1.0,
    )

    data class Diagnostics(
        val timestampNanos: Long,
        val inputJointCount: Int,
        val outputJointCount: Int,
        val acceptedMeasurements: Int,
        val rejectedMeasurements: Int,
        val predictionOnlyJoints: Int,
        val meanInnovationMeters: Double,
        val p95InnovationMeters: Double,
        val meanUncertaintyMeters: Double,
        val meanVelocityMetersPerSecond: Double,
        val meanAccelerationMetersPerSecond2: Double,
        val microMotionRetention: Double,
        val continuityScore: Double,
        val temporalScore: Double,
        val confidenceScore: Double,
        val meanTemporalMemoryScore: Double = 1.0,
        val finalScore: Double,
        val warnings: List<String>,
        val jointDiagnostics: List<JointDiagnostics>,
    )

    data class Result(
        val skeleton: SkeletonFrame,
        val diagnostics: Diagnostics,
        val accepted: Boolean,
    )

    private data class JointState(
        var position: Vec3,
        var velocity: Vec3,
        var acceleration: Vec3,
        var uncertaintyMeters: Double,
        var lastTimestampNanos: Long,
        var confidence: Double,
        var occlusionFrames: Int,
        var residualMeters: Double,
        var initialized: Boolean,
        var temporalMemoryScore: Double = 1.0,
    )

    private val states = LinkedHashMap<Int, JointState>()
    private var lastTimestampNanos = Long.MIN_VALUE

    fun updateConfig(newConfig: Config) {
        config = newConfig
    }

    fun config(): Config = config

    fun reset() {
        states.clear()
        lastTimestampNanos = Long.MIN_VALUE
    }

    fun stateCount(): Int = states.size

    fun estimate(input: Input): Result {
        if (lastTimestampNanos != Long.MIN_VALUE &&
            input.timestampNanos <= lastTimestampNanos
        ) {
            reset()
        }

        val dtSeconds = computeDeltaSeconds(
            previousTimestampNanos = lastTimestampNanos,
            timestampNanos = input.timestampNanos,
        )

        val cameraRisk = input.cameraMotionRisk.coerceIn(0.0, 1.0)
        val occlusionRisk = input.occlusionRisk.coerceIn(0.0, 1.0)
        val synchronization = input.synchronizationConfidence.coerceIn(0.0, 1.0)

        val byId = input.skeleton.joints.associateBy { it.landmarkId }
        val output = ArrayList<ReconstructionJoint>(input.skeleton.joints.size)
        val diagnostics = ArrayList<JointDiagnostics>(input.skeleton.joints.size)

        for (joint in input.skeleton.joints) {
            val sanitized = sanitizeJoint(joint)
            val state = states.getOrPut(joint.landmarkId) {
                newState(
                    joint = sanitized,
                    timestampNanos = input.timestampNanos,
                )
            }

            val diagnostic = updateJoint(
                incoming = sanitized,
                state = state,
                dtSeconds = dtSeconds,
                cameraMotionRisk = cameraRisk,
                occlusionRisk = occlusionRisk,
                synchronizationConfidence = synchronization,
                preserveMicroMotion = input.preserveMicroMotion,
            )

            output += sanitized.copy(
                position = state.position,
                velocity = state.velocity,
                acceleration = state.acceleration,
                confidence = diagnostic.confidence,
                positionUncertaintyMeters = state.uncertaintyMeters,
            )
            diagnostics += diagnostic
        }

        carryOccludedJoints(
            currentIds = byId.keys,
            timestampNanos = input.timestampNanos,
            dtSeconds = dtSeconds,
            cameraMotionRisk = cameraRisk,
            occlusionRisk = occlusionRisk,
            synchronizationConfidence = synchronization,
            preserveMicroMotion = input.preserveMicroMotion,
            output = output,
            diagnostics = diagnostics,
        )

        val aggregate = aggregateDiagnostics(
            timestampNanos = input.timestampNanos,
            input = input.skeleton,
            diagnostics = diagnostics,
            synchronizationConfidence = synchronization,
        )

        lastTimestampNanos = input.timestampNanos

        pruneExpiredStates()

        val globalConfidence = (
            input.skeleton.globalConfidence.coerceIn(0.0, 1.0) * 0.44 +
                aggregate.confidenceScore * 0.28 +
                aggregate.continuityScore * 0.18 +
                synchronization * 0.10
            ).coerceIn(0.0, 1.0)

        val resultSkeleton = input.skeleton.copy(
            joints = output.sortedBy { it.landmarkId },
            globalConfidence = globalConfidence,
        )

        val accepted = aggregate.finalScore >= 0.34 &&
            aggregate.outputJointCount > 0 &&
            diagnostics.none { it.predictionOnly && it.occlusionFrames > config.maximumOcclusionFrames }

        return Result(
            skeleton = resultSkeleton,
            diagnostics = aggregate,
            accepted = accepted,
        )
    }

    private fun updateJoint(
        incoming: ReconstructionJoint,
        state: JointState,
        dtSeconds: Double,
        cameraMotionRisk: Double,
        occlusionRisk: Double,
        synchronizationConfidence: Double,
        preserveMicroMotion: Boolean,
    ): JointDiagnostics {
        val measurementConfidence = incoming.confidence
            .coerceIn(0.0, 1.0)

        if (!state.initialized) {
            state.position = incoming.position
            state.velocity = finiteOrZero(incoming.velocity)
            state.acceleration = Vec3(0.0, 0.0, 0.0)
            state.uncertaintyMeters = uncertaintyFromIncoming(incoming)
            state.confidence = measurementConfidence
            state.occlusionFrames = 0
            state.residualMeters = 0.0
            state.lastTimestampNanos = incomingTimestampForState(state)
            state.initialized = true

            return JointDiagnostics(
                landmarkId = incoming.landmarkId,
                confidence = measurementConfidence,
                innovationMeters = 0.0,
                normalizedInnovation = 0.0,
                measurementWeight = 1.0,
                predictionWeight = 0.0,
                velocityMetersPerSecond = state.velocity.norm(),
                accelerationMetersPerSecond2 = state.acceleration.norm(),
                uncertaintyMeters = state.uncertaintyMeters,
                occlusionFrames = 0,
                measurementAccepted = true,
                predictionOnly = false,
                microMotionMeters = 0.0,
            )
        }

        val prediction = predict(
            state = state,
            dtSeconds = dtSeconds,
            cameraMotionRisk = cameraMotionRisk,
        )

        val residual = incoming.position - prediction.position
        val innovationMeters = finiteNorm(residual)
        val measurementUncertainty = uncertaintyFromIncoming(incoming)
        val combinedUncertainty = (
            measurementUncertainty +
                state.uncertaintyMeters * 0.65 +
                config.measurementNoiseMeters
            ).coerceIn(
                config.uncertaintyFloorMeters,
                config.uncertaintyCeilingMeters,
            )

        val normalizedInnovation = innovationMeters /
            combinedUncertainty.coerceAtLeast(1e-5)

        val observedDelta =
            incoming.position -
                state.position

        val observedMotionMagnitude =
            finiteNorm(observedDelta)

        val priorVelocityMagnitude =
            finiteNorm(state.velocity)

        val directionalConsistency =
            if (
                observedMotionMagnitude < 1e-6 ||
                    priorVelocityMagnitude < 1e-6
            ) {
                1.0
            } else {
                (
                    observedDelta.normalized()
                        .dot(
                            state.velocity.normalized(),
                        ) + 1.0
                    ) * 0.5
            }.coerceIn(0.0, 1.0)

        val normalizedMemoryResidual =
            (
                innovationMeters /
                    combinedUncertainty.coerceAtLeast(1e-6)
            ).takeIf(Double::isFinite)
                ?.let {
                    (
                        it /
                            config.hardInnovationGateMeters
                                .coerceAtLeast(1e-6)
                    ).coerceIn(0.0, 1.0)
                } ?: 1.0

        val memoryObservation =
            (
                directionalConsistency * 0.62 +
                    (1.0 - normalizedMemoryResidual) * 0.38
                ).coerceIn(0.0, 1.0)

        state.temporalMemoryScore =
            (
                state.temporalMemoryScore *
                    (1.0 - config.temporalMemoryAlpha) +
                    memoryObservation *
                    config.temporalMemoryAlpha
                ).coerceIn(0.0, 1.0)

        val softGate = config.softInnovationGateMeters *
            (1.0 + cameraMotionRisk * 0.85) *
            (1.0 + occlusionRisk * 0.45)

        val hardGate = config.hardInnovationGateMeters *
            (1.0 + cameraMotionRisk * 0.35)

        val robustWeight = robustResidualWeight(
            residualMeters = innovationMeters,
            softGateMeters = softGate,
            hardGateMeters = hardGate,
        )

        val confidenceWeight = confidenceMeasurementWeight(
            confidence = measurementConfidence,
            stateConfidence = state.confidence,
            synchronizationConfidence = synchronizationConfidence,
        )

        val cameraPenalty = 1.0 -
            config.cameraMotionSuppression * cameraMotionRisk * 0.45
        val occlusionPenalty = 1.0 -
            config.occlusionSuppression * occlusionRisk * 0.40

        val hardRejected = innovationMeters > hardGate &&
            measurementConfidence < 0.72

        val accepted = !hardRejected &&
            measurementConfidence >= config.minimumConfidence

        val predictionTrust = if (accepted) {
            (
                1.0 -
                    (
                        robustWeight *
                            confidenceWeight *
                            cameraPenalty *
                            occlusionPenalty
                        )
                ).coerceIn(
                    config.positionBlendMin,
                    config.positionBlendMax,
                )
        } else {
            config.lowConfidencePredictionTrust
                .coerceIn(
                    config.positionBlendMin,
                    config.positionBlendMax,
                )
        }

        val temporalMemoryTrust =
            (
                config.temporalMemoryMinimumTrust +
                    (
                        1.0 -
                            config.temporalMemoryMinimumTrust
                        ) *
                        state.temporalMemoryScore
                ).coerceIn(
                    config.temporalMemoryMinimumTrust,
                    1.0,
                )

        val adjustedPredictionTrust =
            (
                predictionTrust *
                    temporalMemoryTrust
            ).coerceIn(
                config.positionBlendMin,
                config.positionBlendMax,
            )

        val measurementTrust = 1.0 - adjustedPredictionTrust

        val measuredMicroMotion = residual -
            state.velocity * dtSeconds

        val microMotionWeight = if (preserveMicroMotion) {
            config.microMotionPreservation *
                measurementConfidence *
                (1.0 - cameraMotionRisk * 0.50) *
                (1.0 - occlusionRisk * 0.35)
        } else {
            0.0
        }

        val blendedPosition = (
            prediction.position * adjustedPredictionTrust +
                incoming.position * measurementTrust +
                measuredMicroMotion * microMotionWeight * 0.22
            )

        val boundedPosition = limitPositionJump(
            previous = state.position,
            candidate = blendedPosition,
            dtSeconds = dtSeconds,
            confidence = measurementConfidence,
            cameraMotionRisk = cameraMotionRisk,
        )

        val observedVelocity = (
            boundedPosition - state.position
            ) / dtSeconds.coerceAtLeast(1e-4)

        val predictedVelocity = prediction.velocity

        val velocityBlend = (
            config.velocityGain *
                confidenceWeight *
                (1.0 - cameraMotionRisk * 0.35) *
                (1.0 - occlusionRisk * 0.30)
            ).coerceIn(0.05, 1.0)

        var velocity = lerp(
            predictedVelocity,
            observedVelocity,
            velocityBlend,
        )

        val observedAcceleration = (
            velocity - state.velocity
            ) / dtSeconds.coerceAtLeast(1e-4)

        val accelerationBlend = (
            config.accelerationGain *
                confidenceWeight
            ).coerceIn(0.05, 1.0)

        var acceleration = lerp(
            state.acceleration,
            observedAcceleration,
            accelerationBlend,
        )

        val jerk = (
            acceleration - state.acceleration
            ) / dtSeconds.coerceAtLeast(1e-4)

        val jerkLimited = clampMagnitude(
            jerk,
            config.jerkLimitMetersPerSecond3,
        )

        acceleration = state.acceleration +
            jerkLimited *
            dtSeconds.coerceIn(1e-4, 0.10) *
            config.jerkGain

        acceleration = clampMagnitude(
            acceleration,
            config.accelerationLimitMetersPerSecond2 *
                (
                    1.0 +
                        (1.0 - measurementConfidence) * 0.65
                    ),
        )

        velocity = clampMagnitude(
            velocity,
            config.velocityLimitMetersPerSecond *
                (
                    1.0 +
                        (1.0 - measurementConfidence) * 0.25
                    ),
        )

        val uncertaintyGrowth =
            if (accepted) {
                dtSeconds * config.uncertaintyGrowthPerSecond
            } else {
                dtSeconds * config.uncertaintyGrowthPerSecond * 1.75
            }

        val residualReduction = if (accepted) {
            (0.18 + confidenceWeight * 0.42)
        } else {
            0.03
        }

        state.uncertaintyMeters = (
            combinedUncertainty *
                (1.0 - residualReduction) +
                uncertaintyGrowth +
                if (!accepted) 0.004 else 0.0
            ).coerceIn(
                config.uncertaintyFloorMeters,
                config.uncertaintyCeilingMeters,
            )

        state.position = boundedPosition
        state.velocity = velocity
        state.acceleration = acceleration
        state.confidence = (
            measurementConfidence * 0.58 +
                state.confidence * 0.22 +
                robustWeight * 0.20
            ).coerceIn(0.0, 1.0)
        state.occlusionFrames = if (accepted) 0 else {
            state.occlusionFrames + 1
        }
        state.residualMeters = innovationMeters
        state.lastTimestampNanos += (
            dtSeconds * 1e9
        ).toLong().coerceAtLeast(1L)

        val effectiveMicroMotion = finiteNorm(
            measuredMicroMotion *
                microMotionWeight,
        )

        return JointDiagnostics(
            landmarkId = incoming.landmarkId,
            confidence = state.confidence,
            innovationMeters = innovationMeters,
            normalizedInnovation = normalizedInnovation,
            measurementWeight = measurementTrust,
            predictionWeight = predictionTrust,
            velocityMetersPerSecond = velocity.norm(),
            accelerationMetersPerSecond2 = acceleration.norm(),
            uncertaintyMeters = state.uncertaintyMeters,
            occlusionFrames = state.occlusionFrames,
            measurementAccepted = accepted,
            predictionOnly = !accepted,
            microMotionMeters = effectiveMicroMotion,
            temporalMemoryScore =
                state.temporalMemoryScore,
        )
    }

    private data class Prediction(
        val position: Vec3,
        val velocity: Vec3,
    )

    private fun predict(
        state: JointState,
        dtSeconds: Double,
        cameraMotionRisk: Double,
    ): Prediction {
        val boundedDt = dtSeconds
            .coerceIn(
                1.0 / 480.0,
                config.predictionHorizonSeconds.coerceAtLeast(0.05),
            )

        val accelerationScale = (
            1.0 -
                cameraMotionRisk *
                config.cameraMotionSuppression *
                0.40
            ).coerceIn(0.35, 1.0)

        val velocity = state.velocity +
            state.acceleration *
            boundedDt *
            accelerationScale

        val position = state.position +
            velocity * boundedDt +
            state.acceleration *
            (0.5 * boundedDt * boundedDt) *
            accelerationScale

        return Prediction(
            position = finiteOrZero(position),
            velocity = finiteOrZero(velocity),
        )
    }

    private fun carryOccludedJoints(
        currentIds: Set<Int>,
        timestampNanos: Long,
        dtSeconds: Double,
        cameraMotionRisk: Double,
        occlusionRisk: Double,
        synchronizationConfidence: Double,
        preserveMicroMotion: Boolean,
        output: MutableList<ReconstructionJoint>,
        diagnostics: MutableList<JointDiagnostics>,
    ) {
        val missing = states.keys
            .filter { it !in currentIds }
            .toList()

        for (landmarkId in missing) {
            val state = states[landmarkId]
                ?: continue

            if (state.occlusionFrames >= config.maximumOcclusionFrames) {
                continue
            }

            val prediction = predict(
                state = state,
                dtSeconds = dtSeconds,
                cameraMotionRisk = cameraMotionRisk,
            )

            val uncertaintyMultiplier = (
                1.0 +
                    config.uncertaintyGrowthPerMissingFrame *
                    (state.occlusionFrames + 1)
                ).coerceAtMost(3.0)

            state.position = prediction.position
            state.velocity = clampMagnitude(
                prediction.velocity,
                config.velocityLimitMetersPerSecond,
            )
            state.acceleration = clampMagnitude(
                state.acceleration,
                config.accelerationLimitMetersPerSecond2,
            )
            state.uncertaintyMeters = (
                state.uncertaintyMeters *
                    uncertaintyMultiplier +
                    dtSeconds * config.uncertaintyGrowthPerSecond
                ).coerceIn(
                    config.uncertaintyFloorMeters,
                    config.uncertaintyCeilingMeters,
                )
            state.confidence = (
                state.confidence * 0.86 *
                    (1.0 - occlusionRisk * 0.22) *
                    synchronizationConfidence
                ).coerceIn(0.0, 1.0)
            state.occlusionFrames += 1
            state.lastTimestampNanos = timestampNanos

            val confidence = (
                state.confidence *
                    (1.0 - cameraMotionRisk * 0.28)
                ).coerceIn(0.0, 1.0)

            val joint = ReconstructionJoint(
                landmarkId = landmarkId,
                position = state.position,
                velocity = state.velocity,
                acceleration = state.acceleration,
                confidence = confidence,
                reprojectionErrorPx = Double.POSITIVE_INFINITY,
                observationCount = 0,
                positionUncertaintyMeters = state.uncertaintyMeters,
            )

            output += joint

            diagnostics += JointDiagnostics(
                landmarkId = landmarkId,
                confidence = confidence,
                innovationMeters = state.residualMeters,
                normalizedInnovation = (
                    state.residualMeters /
                        state.uncertaintyMeters.coerceAtLeast(1e-5)
                    ),
                measurementWeight = 0.0,
                predictionWeight = 1.0,
                velocityMetersPerSecond = state.velocity.norm(),
                accelerationMetersPerSecond2 = state.acceleration.norm(),
                uncertaintyMeters = state.uncertaintyMeters,
                occlusionFrames = state.occlusionFrames,
                measurementAccepted = false,
                predictionOnly = true,
                microMotionMeters = if (preserveMicroMotion) {
                    state.velocity.norm() * 0.001
                } else {
                    0.0
                },
            )
        }
    }

    private fun aggregateDiagnostics(
        timestampNanos: Long,
        input: SkeletonFrame,
        diagnostics: List<JointDiagnostics>,
        synchronizationConfidence: Double,
    ): Diagnostics {
        val innovations = diagnostics
            .map { it.innovationMeters }
            .filter(Double::isFinite)

        val uncertainties = diagnostics
            .map { it.uncertaintyMeters }
            .filter(Double::isFinite)

        val velocities = diagnostics
            .map { it.velocityMetersPerSecond }
            .filter(Double::isFinite)

        val accelerations = diagnostics
            .map { it.accelerationMetersPerSecond2 }
            .filter(Double::isFinite)

        val accepted = diagnostics.count { it.measurementAccepted }
        val rejected = diagnostics.count { !it.measurementAccepted }
        val predictionOnly = diagnostics.count { it.predictionOnly }

        val meanInnovation = meanFiniteOrZero(innovations)
        val p95Innovation = percentile(innovations, 0.95)
        val meanUncertainty = meanFiniteOrZero(uncertainties)
        val meanVelocity = meanFiniteOrZero(velocities)
        val meanAcceleration = meanFiniteOrZero(accelerations)
        val meanTemporalMemoryScore =
            diagnostics
                .map { it.temporalMemoryScore }
                .filter(Double::isFinite)
                .average()
                .coerceIn(0.0, 1.0)

        val microMotionRetention = computeMicroMotionRetention(
            input = input,
            diagnostics = diagnostics,
        )

        val continuityScore = computeContinuityScore(
            accepted = accepted,
            rejected = rejected,
            predictionOnly = predictionOnly,
            total = diagnostics.size,
            synchronizationConfidence = synchronizationConfidence,
        )

        val temporalScore = (
            expDecay(
                meanInnovation,
                config.qualityResidualScaleMeters,
            ) * 0.46 +
                expDecay(
                    meanVelocity,
                    config.qualityVelocityScaleMetersPerSecond,
                ) * 0.20 +
                expDecay(
                    meanAcceleration,
                    config.qualityAccelerationScaleMetersPerSecond2,
                ) * 0.16 +
                microMotionRetention * 0.08 +
                meanTemporalMemoryScore * 0.12 +
                continuityScore * 0.08
            ).coerceIn(0.0, 1.0)

        val confidenceScore = if (diagnostics.isEmpty()) {
            input.globalConfidence.coerceIn(0.0, 1.0)
        } else {
            diagnostics
                .map { it.confidence }
                .average()
                .coerceIn(0.0, 1.0)
        }

        val residualScore = expDecay(
            meanInnovation,
            config.qualityResidualScaleMeters,
        )

        val uncertaintyScore = expDecay(
            meanUncertainty,
            config.uncertaintyFloorMeters * 8.0,
        )

        val rejectionPenalty = if (diagnostics.isEmpty()) {
            0.0
        } else {
            rejected.toDouble() / diagnostics.size.toDouble()
        }

        val finalScore = (
            temporalScore * 0.42 +
                confidenceScore * 0.26 +
                continuityScore * 0.18 +
                uncertaintyScore * 0.08 +
                residualScore * 0.06 -
                rejectionPenalty * 0.10
            ).coerceIn(0.0, 1.0)

        val warnings = buildWarnings(
            diagnostics = diagnostics,
            meanInnovation = meanInnovation,
            p95Innovation = p95Innovation,
            meanUncertainty = meanUncertainty,
            synchronizationConfidence = synchronizationConfidence,
            finalScore = finalScore,
        )

        return Diagnostics(
            timestampNanos = timestampNanos,
            inputJointCount = input.joints.size,
            outputJointCount = diagnostics.size,
            acceptedMeasurements = accepted,
            rejectedMeasurements = rejected,
            predictionOnlyJoints = predictionOnly,
            meanInnovationMeters = meanInnovation,
            p95InnovationMeters = p95Innovation,
            meanUncertaintyMeters = meanUncertainty,
            meanVelocityMetersPerSecond = meanVelocity,
            meanAccelerationMetersPerSecond2 = meanAcceleration,
            microMotionRetention = microMotionRetention,
            continuityScore = continuityScore,
            temporalScore = temporalScore,
            confidenceScore = confidenceScore,
            meanTemporalMemoryScore =
                meanTemporalMemoryScore,
            finalScore = finalScore,
            warnings = warnings,
            jointDiagnostics = diagnostics.sortedBy { it.landmarkId },
        )
    }

    private fun buildWarnings(
        diagnostics: List<JointDiagnostics>,
        meanInnovation: Double,
        p95Innovation: Double,
        meanUncertainty: Double,
        synchronizationConfidence: Double,
        finalScore: Double,
    ): List<String> {
        val warnings = ArrayList<String>()

        val total = diagnostics.size.coerceAtLeast(1)
        val predictionRatio =
            diagnostics.count { it.predictionOnly }.toDouble() / total.toDouble()
        val rejectionRatio =
            diagnostics.count { !it.measurementAccepted }.toDouble() / total.toDouble()

        if (predictionRatio > 0.40) {
            warnings += "high-prediction-ratio"
        }
        if (rejectionRatio > 0.35) {
            warnings += "high-innovation-rejection"
        }
        if (meanInnovation > config.qualityResidualScaleMeters) {
            warnings += "elevated-temporal-residual"
        }
        if (p95Innovation > config.softInnovationGateMeters * 1.5) {
            warnings += "p95-residual-outlier"
        }
        if (meanUncertainty > config.uncertaintyFloorMeters * 7.0) {
            warnings += "elevated-position-uncertainty"
        }
        if (synchronizationConfidence < 0.60) {
            warnings += "weak-synchronization-evidence"
        }
        if (finalScore < 0.45) {
            warnings += "temporal-quality-degraded"
        }

        return warnings.distinct().take(12)
    }

    private fun computeMicroMotionRetention(
        input: SkeletonFrame,
        diagnostics: List<JointDiagnostics>,
    ): Double {
        if (diagnostics.isEmpty()) {
            return 0.0
        }

        val observedJoints = input.joints
            .filter { it.confidence >= config.minimumConfidence }

        if (observedJoints.isEmpty()) {
            return 0.0
        }

        val measured =
            observedJoints.map { joint ->
                val state = states[joint.landmarkId]
                state?.residualMeters ?: 0.0
            }

        val retained =
            meanFiniteOrZero(
                diagnostics
                    .map { it.microMotionMeters }
                    .filter(Double::isFinite),
            )
                

        val source = meanFiniteOrZero(measured)
        if (source <= 1e-6) {
            return if (retained <= 1e-6) 1.0 else 0.75
        }

        return (
            retained / source.coerceAtLeast(1e-6)
            ).coerceIn(0.0, 1.0)
    }

    private fun computeContinuityScore(
        accepted: Int,
        rejected: Int,
        predictionOnly: Int,
        total: Int,
        synchronizationConfidence: Double,
    ): Double {
        if (total <= 0) {
            return 0.0
        }

        val acceptedRatio =
            accepted.toDouble() / total.toDouble()
        val predictedRatio =
            predictionOnly.toDouble() / total.toDouble()
        val rejectedRatio =
            rejected.toDouble() / total.toDouble()

        return (
            acceptedRatio * 0.64 +
                (1.0 - predictedRatio) * 0.20 +
                (1.0 - rejectedRatio) * 0.08 +
                synchronizationConfidence * 0.08
            ).coerceIn(0.0, 1.0)
    }

    private fun confidenceMeasurementWeight(
        confidence: Double,
        stateConfidence: Double,
        synchronizationConfidence: Double,
    ): Double {
        val direct = confidence
            .coerceIn(0.0, 1.0)
        val prior = stateConfidence
            .coerceIn(0.0, 1.0)

        return (
            direct * 0.58 +
                prior * 0.24 +
                synchronizationConfidence * 0.18
            ).coerceIn(0.05, 1.0)
    }

    private fun robustResidualWeight(
        residualMeters: Double,
        softGateMeters: Double,
        hardGateMeters: Double,
    ): Double {
        if (!residualMeters.isFinite()) {
            return 0.0
        }

        if (residualMeters <= softGateMeters) {
            return 1.0
        }

        if (residualMeters >= hardGateMeters) {
            return 0.03
        }

        val normalized = (
            residualMeters - softGateMeters
            ) / (
                hardGateMeters - softGateMeters
            ).coerceAtLeast(1e-6)

        return (
            1.0 -
                normalized * normalized * 0.92
            ).coerceIn(0.03, 1.0)
    }

    private fun limitPositionJump(
        previous: Vec3,
        candidate: Vec3,
        dtSeconds: Double,
        confidence: Double,
        cameraMotionRisk: Double,
    ): Vec3 {
        val delta = candidate - previous
        val magnitude = delta.norm()
        if (!magnitude.isFinite()) {
            return previous
        }

        val confidenceSlack = (
            1.0 +
                (1.0 - confidence) * 0.28
            )
        val cameraSlack = (
            1.0 +
                cameraMotionRisk * 0.22
            )

        val maximumDisplacement = (
            config.velocityLimitMetersPerSecond *
                dtSeconds.coerceIn(
                    1.0 / 480.0,
                    0.20,
                ) *
                confidenceSlack *
                cameraSlack
            ).coerceAtLeast(0.01)

        if (magnitude <= maximumDisplacement) {
            return finiteOrZero(candidate)
        }

        val scale =
            maximumDisplacement /
                magnitude.coerceAtLeast(1e-9)

        return previous + delta * scale
    }

    private fun pruneExpiredStates() {
        val iterator = states.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.value.occlusionFrames >
                config.maximumOcclusionFrames + 2
            ) {
                iterator.remove()
            }
        }
    }

    private fun newState(
        joint: ReconstructionJoint,
        timestampNanos: Long,
    ): JointState {
        return JointState(
            position = finiteOrZero(joint.position),
            velocity = finiteOrZero(joint.velocity),
            acceleration = Vec3(0.0, 0.0, 0.0),
            uncertaintyMeters = uncertaintyFromIncoming(joint),
            lastTimestampNanos = timestampNanos,
            confidence = joint.confidence.coerceIn(0.0, 1.0),
            occlusionFrames = 0,
            residualMeters = 0.0,
            initialized = false,
        )
    }

    private fun sanitizeJoint(
        joint: ReconstructionJoint,
    ): ReconstructionJoint {
        return joint.copy(
            position = finiteOrZero(joint.position),
            velocity = finiteOrZero(joint.velocity),
                        confidence = joint.confidence
                .takeIf(Double::isFinite)
                ?.coerceIn(0.0, 1.0)
                ?: 0.0,
            positionUncertaintyMeters =
                if (joint.positionUncertaintyMeters.isFinite()) {
                    joint.positionUncertaintyMeters.coerceIn(
                        config.uncertaintyFloorMeters,
                        config.uncertaintyCeilingMeters,
                    )
                } else {
                    config.uncertaintyCeilingMeters
                },
        )
    }

    private fun uncertaintyFromIncoming(
        joint: ReconstructionJoint,
    ): Double {
        if (!joint.positionUncertaintyMeters.isFinite()) {
            return config.uncertaintyCeilingMeters
        }

        val confidence = joint.confidence
            .coerceIn(0.05, 1.0)

        return (
            joint.positionUncertaintyMeters +
                config.measurementNoiseMeters *
                (1.0 - confidence)
            ).coerceIn(
                config.uncertaintyFloorMeters,
                config.uncertaintyCeilingMeters,
            )
    }

    private fun computeDeltaSeconds(
        previousTimestampNanos: Long,
        timestampNanos: Long,
    ): Double {
        if (previousTimestampNanos == Long.MIN_VALUE) {
            return 1.0 / 60.0
        }

        return (
            (
                timestampNanos - previousTimestampNanos
            ).toDouble() / 1e9
        ).coerceIn(
            1.0 / 480.0,
            0.20,
        )
    }

    private fun incomingTimestampForState(
        state: JointState,
    ): Long = state.lastTimestampNanos

    private fun lerp(
        a: Vec3,
        b: Vec3,
        factor: Double,
    ): Vec3 {
        val t = factor.coerceIn(0.0, 1.0)
        return a + (
            b - a
            ) * t
    }

    private fun clampMagnitude(
        value: Vec3,
        maximum: Double,
    ): Vec3 {
        val magnitude = value.norm()
        if (!magnitude.isFinite() || magnitude <= maximum) {
            return finiteOrZero(value)
        }
        if (magnitude <= 1e-9) {
            return Vec3(0.0, 0.0, 0.0)
        }
        return value * (
            maximum /
                magnitude
            )
    }

    private fun finiteOrZero(
        value: Vec3,
    ): Vec3 {
        return Vec3(
            if (value.x.isFinite()) value.x else 0.0,
            if (value.y.isFinite()) value.y else 0.0,
            if (value.z.isFinite()) value.z else 0.0,
        )
    }

    private fun finiteNorm(
        value: Vec3,
    ): Double {
        val magnitude = value.norm()
        return if (magnitude.isFinite()) {
            magnitude
        } else {
            0.0
        }
    }

    private fun meanFiniteOrZero(
        values: List<Double>,
    ): Double = values
        .filter(Double::isFinite)
        .let { filtered ->
            if (filtered.isEmpty()) 0.0 else filtered.average()
        }

    private fun List<Double>.meanOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private fun percentile(
        values: List<Double>,
        p: Double,
    ): Double {
        val sorted = values
            .filter(Double::isFinite)
            .sorted()

        if (sorted.isEmpty()) {
            return 0.0
        }
        if (sorted.size == 1) {
            return sorted.first()
        }

        val position = p.coerceIn(0.0, 1.0) *
            (sorted.size - 1)

        val lower = position.toInt()
        val upper = min(
            sorted.lastIndex,
            lower + 1,
        )
        val factor = position - lower

        return sorted[lower] * (1.0 - factor) +
            sorted[upper] * factor
    }

    private fun expDecay(
        value: Double,
        scale: Double,
    ): Double {
        if (!value.isFinite()) {
            return 0.0
        }
        return kotlin.math.exp(
            -abs(value) /
                scale.coerceAtLeast(1e-6),
        ).coerceIn(0.0, 1.0)
    }
}

package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Production motion-dynamics analysis layer.
 *
 * This stage is deliberately downstream of perception and temporal state
 * estimation. It does not fabricate missing joints. It measures the dynamics
 * of the reconstructed state and produces explicit evidence that can be used
 * by quality gates, event segmentation, retargeting, contact solvers, and the
 * operator UI.
 *
 * Signals:
 *  - root linear motion and acceleration;
 *  - per-joint speed/acceleration/jerk;
 *  - lower-body contact stability;
 *  - left/right symmetry;
 *  - micro-motion retention;
 *  - motion phase classification;
 *  - dynamic regime confidence;
 *  - kinematic stress and instability indicators;
 *  - aggregate uncertainty and trusted-joint ratio.
 */
class ProductionMotionDynamicsEngine(
    private var config: Config = Config(),
) {
    enum class MotionPhase {
        UNKNOWN,
        IDLE,
        STARTING,
        LOCOMOTION,
        RUNNING,
        AIRBORNE,
        TURNING,
        GESTURE,
        HIGH_DYNAMICS,
        RECOVERING,
    }

    data class Config(
        val idleSpeedMetersPerSecond: Double = 0.045,
        val gestureSpeedMetersPerSecond: Double = 0.55,
        val locomotionSpeedMetersPerSecond: Double = 1.15,
        val runningSpeedMetersPerSecond: Double = 2.4,
        val airborneAccelerationMetersPerSecond2: Double = 5.0,
        val highDynamicsJerkMetersPerSecond3: Double = 85.0,
        val instabilityJerkMetersPerSecond3: Double = 220.0,
        val contactSpeedMetersPerSecond: Double = 0.08,
        val symmetryToleranceMeters: Double = 0.11,
        val microMotionThresholdMeters: Double = 0.012,
        val maximumUncertaintyMeters: Double = 0.20,
        val smoothingAlpha: Double = 0.24,
        val temporalBandAlpha: Double = 0.18,
        val temporalBandTimeConstantSeconds: Double = 0.08,
        val phaseHoldSamples: Int = 3,
    ) {
        init {
            require(idleSpeedMetersPerSecond > 0.0)
            require(gestureSpeedMetersPerSecond > idleSpeedMetersPerSecond)
            require(locomotionSpeedMetersPerSecond > gestureSpeedMetersPerSecond)
            require(runningSpeedMetersPerSecond > locomotionSpeedMetersPerSecond)
            require(airborneAccelerationMetersPerSecond2 > 0.0)
            require(highDynamicsJerkMetersPerSecond3 > 0.0)
            require(instabilityJerkMetersPerSecond3 >= highDynamicsJerkMetersPerSecond3)
            require(contactSpeedMetersPerSecond > 0.0)
            require(symmetryToleranceMeters > 0.0)
            require(microMotionThresholdMeters > 0.0)
            require(maximumUncertaintyMeters > 0.0)
            require(smoothingAlpha in 0.01..1.0)
            require(temporalBandAlpha in 0.01..1.0)
            require(temporalBandTimeConstantSeconds > 0.0)
            require(phaseHoldSamples >= 1)
        }
    }

    data class JointDynamics(
        val landmarkId: Int,
        val speedMetersPerSecond: Double,
        val accelerationMetersPerSecond2: Double,
        val jerkMetersPerSecond3: Double,
        val uncertaintyMeters: Double,
        val confidence: Double,
        val motionEnergy: Double,
        val highDynamics: Boolean,
        val lowFrequencyMotionEnergy: Double = 0.0,
        val highFrequencyMotionEnergy: Double = 0.0,
    )

    data class Result(
        val timestampNanos: Long,
        val phase: MotionPhase,
        val phaseConfidence: Double,
        val rootSpeedMetersPerSecond: Double,
        val rootAccelerationMetersPerSecond2: Double,
        val rootJerkMetersPerSecond3: Double,
        val meanSpeedMetersPerSecond: Double,
        val meanAccelerationMetersPerSecond2: Double,
        val meanJerkMetersPerSecond3: Double,
        val peakJerkMetersPerSecond3: Double,
        val microMotionRetention: Double,
        val lowerBodyContactStability: Double,
        val symmetryScore: Double,
        val airborneScore: Double,
        val dynamicStressScore: Double,
        val uncertaintyScore: Double,
        val trustedJointRatio: Double,
        val lowFrequencyMotionScore: Double = 0.0,
        val highFrequencyMotionScore: Double = 0.0,
        val globalScore: Double,
        val warnings: List<String>,
        val recommendations: List<String>,
        val joints: List<JointDynamics>,
    )

    private var previousTimestampNanos = Long.MIN_VALUE
    private var previousRootPosition = Vec3Zero
    private var previousRootVelocity = Vec3Zero
    private var previousRootAcceleration = Vec3Zero
    private var previousPhase = MotionPhase.UNKNOWN
    private var phaseStabilitySamples = 0
    private var smoothedScore = 0.0
    private val temporalBands = HashMap<Int, TemporalBandState>()
    
    private data class TemporalBandState(
        var lowPassSpeed: Double = 0.0,
        var highFrequencyEnergy: Double = 0.0,
    )

    fun updateConfig(newConfig: Config) {
        config = newConfig
    }

    fun config(): Config = config

    fun reset() {
        previousTimestampNanos = Long.MIN_VALUE
        previousRootPosition = Vec3Zero
        previousRootVelocity = Vec3Zero
        previousRootAcceleration = Vec3Zero
        previousPhase = MotionPhase.UNKNOWN
        phaseStabilitySamples = 0
        smoothedScore = 0.0
        temporalBands.clear()
    }

    fun analyze(
        skeleton: SkeletonFrame,
    ): Result {
        val dt =
            if (previousTimestampNanos == Long.MIN_VALUE) {
                1.0 / 60.0
            } else {
                (
                    (skeleton.timestampNanos - previousTimestampNanos)
                        .coerceAtLeast(1L) / 1e9
                    ).coerceIn(1.0 / 240.0, 0.25)
            }

        val root =
            skeleton.joints.firstOrNull { it.landmarkId == 23 }
                ?: skeleton.joints.minByOrNull { it.position.y }
                ?: skeleton.joints.firstOrNull()

        val rootPosition =
            root?.position ?: previousRootPosition

        val rootVelocity =
            root?.velocity ?: (
                rootPosition - previousRootPosition
            ) / dt

        val rootAcceleration =
            root?.acceleration ?: (
                rootVelocity - previousRootVelocity
            ) / dt

        val rootJerk =
            (
                rootAcceleration - previousRootAcceleration
                ) / dt

        val jointDynamics =
            skeleton.joints.map {
                dynamicsForJoint(
                    joint = it,
                    dtSeconds = dt,
                )
            }

        val meanSpeed =
            jointDynamics.map { it.speedMetersPerSecond }
                .averageOrZero()

        val meanAcceleration =
            jointDynamics.map { it.accelerationMetersPerSecond2 }
                .averageOrZero()

        val meanJerk =
            jointDynamics.map { it.jerkMetersPerSecond3 }
                .averageOrZero()

        val peakJerk =
            jointDynamics.maxOfOrNull {
                it.jerkMetersPerSecond3
            } ?: 0.0

        val microMotion =
            jointDynamics
                .map { joint ->
                    val speedContribution =
                        (joint.speedMetersPerSecond /
                            config.gestureSpeedMetersPerSecond)
                            .coerceIn(0.0, 1.0)
                    val uncertaintyPenalty =
                        (
                            joint.uncertaintyMeters /
                                config.maximumUncertaintyMeters
                            ).coerceIn(0.0, 1.0)
                    val confidence =
                        joint.confidence.coerceIn(0.0, 1.0)

                    (
                        (1.0 - abs(speedContribution - 0.28)) *
                            confidence *
                            (1.0 - uncertaintyPenalty)
                        ).coerceIn(0.0, 1.0)
                }
                .averageOrZero()

        val lowFrequencyMotionScore =
            jointDynamics.map {
                it.lowFrequencyMotionEnergy
            }.averageOrZero()

        val highFrequencyMotionMean =
            jointDynamics.map {
                it.highFrequencyMotionEnergy
            }.averageOrZero()
        val highFrequencyMotionPeak =
            jointDynamics.maxOfOrNull {
                it.highFrequencyMotionEnergy
            } ?: 0.0
        val highFrequencyMotionScore =
            (
                highFrequencyMotionMean * 0.55 +
                    highFrequencyMotionPeak * 0.45
                ).coerceIn(0.0, 1.0)

        val trustedRatio =
            jointDynamics.count {
                it.confidence >= 0.65 &&
                    it.uncertaintyMeters <=
                    config.maximumUncertaintyMeters
            }.toDouble() /
                jointDynamics.size.coerceAtLeast(1).toDouble()

        val contact =
            contactStability(skeleton)

        val symmetry =
            bilateralSymmetry(skeleton)

        val airborne =
            airborneScore(
                skeleton = skeleton,
                rootAcceleration = rootAcceleration.norm(),
            )

        val stress =
            (
                peakJerk /
                    config.instabilityJerkMetersPerSecond3
                ).coerceIn(0.0, 1.0)

        val uncertainty =
            (
                jointDynamics.map { dynamics ->
                    val normalized =
                        dynamics.uncertaintyMeters /
                            config.maximumUncertaintyMeters
                    val confidencePenalty =
                        1.0 - dynamics.confidence
                    (
                        1.0 -
                            min(
                                1.0,
                                normalized * 0.72 +
                                    confidencePenalty * 0.28,
                            )
                        ).coerceIn(0.0, 1.0)
                }.averageOrZero()
            ).coerceIn(0.0, 1.0)

        val phaseCandidate =
            classifyPhase(
                rootSpeed = rootVelocity.norm(),
                rootAcceleration = rootAcceleration.norm(),
                rootJerk = rootJerk.norm(),
                contact = contact,
                airborne = airborne,
                symmetry = symmetry,
                meanJointSpeed = meanSpeed,
            )

        val phase =
            stabilizePhase(
                candidate = phaseCandidate,
                confidence =
                    phaseConfidence(
                        phase = phaseCandidate,
                        rootSpeed = rootVelocity.norm(),
                        rootAcceleration = rootAcceleration.norm(),
                        rootJerk = rootJerk.norm(),
                        contact = contact,
                        symmetry = symmetry,
                        trustedRatio = trustedRatio,
                    ),
            )

        val phaseConfidence =
            phaseConfidence(
                phase = phase,
                rootSpeed = rootVelocity.norm(),
                rootAcceleration = rootAcceleration.norm(),
                rootJerk = rootJerk.norm(),
                contact = contact,
                symmetry = symmetry,
                trustedRatio = trustedRatio,
            )

        val rawScore =
            (
                skeleton.globalConfidence.coerceIn(0.0, 1.0) * 0.19 +
                    trustedRatio * 0.17 +
                    uncertainty * 0.14 +
                    contact * 0.11 +
                    symmetry * 0.09 +
                    phaseConfidence * 0.10 +
                    lowFrequencyMotionScore * 0.04 +
                    highFrequencyMotionScore * 0.04 +
                    (1.0 - stress) * 0.12
                ).coerceIn(0.0, 1.0)

        smoothedScore =
            if (previousTimestampNanos == Long.MIN_VALUE) {
                rawScore
            } else {
                smoothedScore +
                    (rawScore - smoothedScore) *
                        config.smoothingAlpha
            }

        val warnings =
            buildWarnings(
                phase = phase,
                dynamicStress = stress,
                contact = contact,
                symmetry = symmetry,
                airborne = airborne,
                uncertainty = uncertainty,
                trustedRatio = trustedRatio,
                highFrequencyMotionScore = highFrequencyMotionScore,
            )

        val recommendations =
            buildRecommendations(
                phase = phase,
                dynamicStress = stress,
                contact = contact,
                symmetry = symmetry,
                airborne = airborne,
                uncertainty = uncertainty,
                highFrequencyMotionScore = highFrequencyMotionScore,
            )

        previousTimestampNanos = skeleton.timestampNanos
        previousRootPosition = rootPosition
        previousRootVelocity = rootVelocity
        previousRootAcceleration = rootAcceleration

        return Result(
            timestampNanos = skeleton.timestampNanos,
            phase = phase,
            phaseConfidence = phaseConfidence,
            rootSpeedMetersPerSecond = rootVelocity.norm(),
            rootAccelerationMetersPerSecond2 = rootAcceleration.norm(),
            rootJerkMetersPerSecond3 = rootJerk.norm(),
            meanSpeedMetersPerSecond = meanSpeed,
            meanAccelerationMetersPerSecond2 = meanAcceleration,
            meanJerkMetersPerSecond3 = meanJerk,
            peakJerkMetersPerSecond3 = peakJerk,
            microMotionRetention = microMotion,
            lowerBodyContactStability = contact,
            symmetryScore = symmetry,
            airborneScore = airborne,
            dynamicStressScore = stress,
            uncertaintyScore = uncertainty,
            trustedJointRatio = trustedRatio,
            lowFrequencyMotionScore = lowFrequencyMotionScore,
            highFrequencyMotionScore = highFrequencyMotionScore,
            globalScore = smoothedScore.coerceIn(0.0, 1.0),
            warnings = warnings,
            recommendations = recommendations,
            joints = jointDynamics,
        )
    }

    private fun dynamicsForJoint(
        joint: ReconstructionJoint,
        dtSeconds: Double,
    ): JointDynamics {
        val speed =
            joint.velocity.norm()
        val acceleration =
            joint.acceleration.norm()

        val jerkEstimate =
            acceleration /
                dtSeconds.coerceAtLeast(1e-4)

        val band =
            temporalBands.getOrPut(
                joint.landmarkId,
            ) {
                TemporalBandState(
                    lowPassSpeed = speed,
                )
            }

        val frameAdaptiveAlpha =
            (
                1.0 -
                    exp(
                        -dtSeconds.coerceAtLeast(1.0 / 240.0) /
                            config.temporalBandTimeConstantSeconds
                    )
            ).coerceIn(0.01, 0.92)

        val alpha =
            (
                frameAdaptiveAlpha *
                    config.temporalBandAlpha.coerceIn(0.01, 1.0)
            ).coerceIn(0.01, 0.92)
        band.lowPassSpeed =
            band.lowPassSpeed * (1.0 - alpha) +
                speed * alpha

        val highFrequencyResidual =
            abs(speed - band.lowPassSpeed)
        band.highFrequencyEnergy =
            band.highFrequencyEnergy * (1.0 - alpha) +
                highFrequencyResidual * alpha

        val uncertaintyPenalty =
            (
                joint.positionUncertaintyMeters /
                    config.maximumUncertaintyMeters
            ).coerceIn(0.0, 1.0)

        val confidenceScale =
            joint.confidence.coerceIn(0.0, 1.0) *
                (1.0 - uncertaintyPenalty)

        val lowFrequencyEnergy =
            (
                (
                    band.lowPassSpeed /
                        config.locomotionSpeedMetersPerSecond
                ).coerceIn(0.0, 1.0) *
                    confidenceScale
            ).coerceIn(0.0, 1.0)

        val highFrequencyEnergy =
            (
                (
                    band.highFrequencyEnergy /
                        config.gestureSpeedMetersPerSecond
                ).coerceIn(0.0, 1.0) *
                    confidenceScale
            ).coerceIn(0.0, 1.0)

        val energy =
            (
                speed * 0.30 +
                    (acceleration /
                        10.0.coerceAtLeast(1.0)) *
                        0.25 +
                    (jerkEstimate /
                        config.highDynamicsJerkMetersPerSecond3)
                        .coerceIn(0.0, 1.0) *
                        0.45
                ).coerceIn(0.0, 1.0)

        return JointDynamics(
            landmarkId = joint.landmarkId,
            speedMetersPerSecond = speed,
            accelerationMetersPerSecond2 = acceleration,
            jerkMetersPerSecond3 = jerkEstimate,
            uncertaintyMeters =
                joint.positionUncertaintyMeters
                    .coerceAtLeast(0.0),
            confidence =
                joint.confidence.coerceIn(0.0, 1.0),
            motionEnergy = energy,
            highDynamics =
                jerkEstimate >=
                    config.highDynamicsJerkMetersPerSecond3,
            lowFrequencyMotionEnergy = lowFrequencyEnergy,
            highFrequencyMotionEnergy = highFrequencyEnergy,
        )
    }

    private fun contactStability(
        skeleton: SkeletonFrame,
    ): Double {
        val left =
            skeleton.joints.firstOrNull { it.landmarkId == 31 }
                ?: skeleton.joints.firstOrNull { it.landmarkId == 29 }

        val right =
            skeleton.joints.firstOrNull { it.landmarkId == 32 }
                ?: skeleton.joints.firstOrNull { it.landmarkId == 30 }

        val scores =
            listOfNotNull(left, right).map {
                val speed =
                    it.velocity.norm()
                val confidence =
                    it.confidence.coerceIn(0.0, 1.0)
                val uncertainty =
                    it.positionUncertaintyMeters
                    .coerceAtLeast(0.0)

                (
                    (1.0 -
                        (speed /
                            config.contactSpeedMetersPerSecond)
                            .coerceIn(0.0, 1.0)
                        ) *
                        confidence *
                        (
                            1.0 -
                                uncertainty /
                                    config.maximumUncertaintyMeters
                                    .coerceIn(0.0, 1.0)
                            )
                    ).coerceIn(0.0, 1.0)
            }

        return scores.averageOrZero()
    }

    private fun bilateralSymmetry(
        skeleton: SkeletonFrame,
    ): Double {
        val pairs =
            listOf(
                11 to 12,
                13 to 14,
                15 to 16,
                23 to 24,
                25 to 26,
                27 to 28,
            )

        val scores =
            pairs.mapNotNull { (leftId, rightId) ->
                val left =
                    skeleton.joints.firstOrNull {
                        it.landmarkId == leftId
                    } ?: return@mapNotNull null
                val right =
                    skeleton.joints.firstOrNull {
                        it.landmarkId == rightId
                    } ?: return@mapNotNull null

                val positional =
                    abs(
                        left.position.x +
                            right.position.x,
                    ).coerceAtMost(
                        config.symmetryToleranceMeters * 4.0,
                    )

                val velocity =
                    abs(
                        left.velocity.norm() -
                            right.velocity.norm(),
                    )

                val score =
                    (
                        1.0 -
                            positional /
                                config.symmetryToleranceMeters *
                                0.72 -
                            velocity /
                                3.0 *
                                0.28
                        ).coerceIn(0.0, 1.0)

                val confidence =
                    min(
                        left.confidence,
                        right.confidence,
                    ).coerceIn(0.0, 1.0)

                score * confidence
            }

        return scores.averageOrZero()
    }

    private fun airborneScore(
        skeleton: SkeletonFrame,
        rootAcceleration: Double,
    ): Double {
        val leftFoot =
            skeleton.joints.firstOrNull {
                it.landmarkId == 31
            } ?: return 0.0
        val rightFoot =
            skeleton.joints.firstOrNull {
                it.landmarkId == 32
            } ?: return 0.0

        val footSpeed =
            (
                leftFoot.velocity.norm() +
                    rightFoot.velocity.norm()
                ) * 0.5

        val stableContact =
            1.0 -
                contactStability(
                    skeleton,
                )

        val accelerationEvidence =
            (
                rootAcceleration /
                    config.airborneAccelerationMetersPerSecond2
                ).coerceIn(0.0, 1.0)

        return (
            stableContact * 0.58 +
                accelerationEvidence * 0.24 +
                (footSpeed /
                    1.8).coerceIn(0.0, 1.0) * 0.18
            ).coerceIn(0.0, 1.0)
    }

    private fun classifyPhase(
        rootSpeed: Double,
        rootAcceleration: Double,
        rootJerk: Double,
        contact: Double,
        airborne: Double,
        symmetry: Double,
        meanJointSpeed: Double,
    ): MotionPhase {
        if (airborne > 0.78) {
            return MotionPhase.AIRBORNE
        }

        if (rootJerk >=
            config.instabilityJerkMetersPerSecond3
        ) {
            return MotionPhase.HIGH_DYNAMICS
        }

        if (
            rootSpeed <
                config.idleSpeedMetersPerSecond &&
            meanJointSpeed <
                config.gestureSpeedMetersPerSecond * 0.12 &&
            contact > 0.76
        ) {
            return MotionPhase.IDLE
        }

        if (
            rootSpeed >=
                config.runningSpeedMetersPerSecond
        ) {
            return MotionPhase.RUNNING
        }

        if (
            rootSpeed >=
                config.locomotionSpeedMetersPerSecond
        ) {
            return MotionPhase.LOCOMOTION
        }

        if (
            rootAcceleration >
                config.locomotionSpeedMetersPerSecond &&
            contact > 0.50
        ) {
            return MotionPhase.STARTING
        }

        if (
            meanJointSpeed >=
                config.gestureSpeedMetersPerSecond
        ) {
            return MotionPhase.GESTURE
        }

        if (
            symmetry < 0.42 &&
            rootSpeed >=
                config.idleSpeedMetersPerSecond
        ) {
            return MotionPhase.TURNING
        }

        return MotionPhase.RECOVERING
    }

    private fun stabilizePhase(
        candidate: MotionPhase,
        confidence: Double,
    ): MotionPhase {
        if (
            candidate != previousPhase &&
            confidence >= 0.52
        ) {
            phaseStabilitySamples += 1
        } else if (candidate == previousPhase) {
            phaseStabilitySamples =
                (phaseStabilitySamples + 1)
                    .coerceAtMost(60)
        } else {
            phaseStabilitySamples = 0
        }

        if (
            previousPhase == MotionPhase.UNKNOWN ||
            phaseStabilitySamples >=
                config.phaseHoldSamples
        ) {
            previousPhase = candidate
            phaseStabilitySamples = 0
        }

        return previousPhase
    }

    private fun phaseConfidence(
        phase: MotionPhase,
        rootSpeed: Double,
        rootAcceleration: Double,
        rootJerk: Double,
        contact: Double,
        symmetry: Double,
        trustedRatio: Double,
    ): Double {
        val dynamicEvidence =
            (
                rootSpeed /
                    config.runningSpeedMetersPerSecond
                ).coerceIn(0.0, 1.0) * 0.24 +
                (
                    rootAcceleration /
                        8.0
                    ).coerceIn(0.0, 1.0) * 0.18 +
                (
                    rootJerk /
                        config.instabilityJerkMetersPerSecond3
                    ).coerceIn(0.0, 1.0) * 0.12

        val phasePrior =
            when (phase) {
                MotionPhase.IDLE -> contact * 0.80 + symmetry * 0.20
                MotionPhase.LOCOMOTION -> 0.64
                MotionPhase.RUNNING -> 0.76
                MotionPhase.AIRBORNE -> 0.82
                MotionPhase.GESTURE -> 0.62
                MotionPhase.TURNING -> 0.58 + (1.0 - symmetry) * 0.32
                MotionPhase.STARTING -> 0.62 + dynamicEvidence * 0.22
                MotionPhase.HIGH_DYNAMICS -> 0.76
                MotionPhase.RECOVERING -> 0.50
                MotionPhase.UNKNOWN -> 0.20
            }

        return (
            phasePrior * 0.60 +
                trustedRatio * 0.22 +
                dynamicEvidence * 0.18
            ).coerceIn(0.0, 1.0)
    }

    private fun buildWarnings(
        phase: MotionPhase,
        dynamicStress: Double,
        contact: Double,
        symmetry: Double,
        airborne: Double,
        uncertainty: Double,
        trustedRatio: Double,
        highFrequencyMotionScore: Double,
    ): List<String> = buildList {
        if (dynamicStress > 0.72) {
            add("high-jerk-regime")
        }

        if (contact < 0.34 &&
            phase != MotionPhase.AIRBORNE
        ) {
            add("lower-body-contact-unstable")
        }

        if (symmetry < 0.30) {
            add("bilateral-dynamics-asymmetric")
        }

        if (airborne > 0.75) {
            add("airborne-transition")
        }

        if (uncertainty < 0.42) {
            add("dynamics-uncertainty-high")
        }

        if (trustedRatio < 0.50) {
            add("trusted-joint-ratio-low")
        }
        if (highFrequencyMotionScore > 0.68) {
            add("high-frequency-articulation-burst")
        }
    }

    private fun buildRecommendations(
        phase: MotionPhase,
        dynamicStress: Double,
        contact: Double,
        symmetry: Double,
        airborne: Double,
        uncertainty: Double,
        highFrequencyMotionScore: Double,
    ): List<String> = buildList {
        if (dynamicStress > 0.72) {
            add(
                "Increase temporal observation density and preserve high-confidence residuals.",
            )
        }

        if (
            phase == MotionPhase.LOCOMOTION ||
            phase == MotionPhase.RUNNING
        ) {
            add(
                "Increase foot-contact and pelvis/root weighting for retargeting.",
            )
        }

        if (airborne > 0.72) {
            add(
                "Temporarily relax contact constraints while retaining root trajectory continuity.",
            )
        }

        if (contact < 0.40) {
            add(
                "Raise lower-body observation priority before applying foot lock.",
            )
        }

        if (symmetry < 0.38) {
            add(
                "Increase camera diversity or treat one-sided observations as reduced-confidence evidence.",
            )
        }

        if (uncertainty < 0.50) {
            add(
                "Reduce output trust for joints with elevated position uncertainty.",
            )
        }
        if (highFrequencyMotionScore > 0.68) {
            add(
                "Prioritize short-interval observations for fast articulated joints while preserving the low-frequency body trajectory.",
            )
        }
    }

    companion object {
        private val Vec3Zero =
            com.nivukx.nox.core.model.Vec3(
                0.0,
                0.0,
                0.0,
            )
    }
}

private fun List<Double>.averageOrZero(): Double =
    if (isEmpty()) 0.0 else average()

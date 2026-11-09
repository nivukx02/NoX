package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DynamicTopology
import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * World-space motion consistency optimizer.
 *
 * This is a deterministic optimization stage between reconstruction and export. It does not invent
 * geometry. It takes already-observed 3D joints, a topology graph and runtime evidence, then solves
 * for the least-surprising state that remains consistent with:
 *
 * - measured joint positions;
 * - temporal velocity and acceleration;
 * - dynamic topology edge lengths;
 * - capture timing quality;
 * - camera-motion risk;
 * - illumination confidence;
 * - uncertainty and detector confidence;
 * - optional ground support height;
 * - explicit micro-motion preservation.
 *
 * The design is intentionally model-agnostic. A future learned motion prior can be added as another
 * evidence term without replacing this deterministic safety layer.
 */
class WorldSpaceMotionConsistencyEngine(
    initialConfig: Config = Config(),
) {
    @Volatile
    private var config: Config = initialConfig

    data class Config(
        val historyLimit: Int = 24,
        val maximumDeltaSeconds: Double = 0.20,
        val minimumDeltaSeconds: Double = 1.0 / 240.0,
        val baseInnovationGateMeters: Double = 0.055,
        val maximumInnovationGateMeters: Double = 0.34,
        val huberTransitionMeters: Double = 0.030,
        val velocityBlend: Double = 0.68,
        val accelerationBlend: Double = 0.54,
        val temporalEvidenceWeight: Double = 0.42,
        val measurementEvidenceWeight: Double = 0.58,
        val timingPenaltyWeight: Double = 0.30,
        val lightingPenaltyWeight: Double = 0.16,
        val cameraRiskPenaltyWeight: Double = 0.24,
        val uncertaintyPenaltyWeight: Double = 0.30,
        val topologyCorrectionGain: Double = 0.46,
        val topologyIterations: Int = 7,
        val minimumConstraintStiffness: Double = 0.08,
        val maximumConstraintCorrectionMeters: Double = 0.085,
        val maximumAccelerationMetersPerSecond2: Double = 55.0,
        val maximumJerkMetersPerSecond3: Double = 420.0,
        val microMotionVelocityFloor: Double = 0.004,
        val microMotionPreservation: Double = 0.94,
        val minimumOutputConfidence: Double = 0.08,
        val hardOutlierInnovationMeters: Double = 0.28,
        val uncertaintyFloorMeters: Double = 0.0015,
        val uncertaintyCeilingMeters: Double = 1.20,
        val robustOutlierSigma: Double = 3.5,
        val minimumJointSupportRatio: Double = 0.35,
        val topologyResidualSoftGateMeters: Double = 0.08,
        val topologyResidualHardGateMeters: Double = 0.18,
        val adaptiveTopologyIterations: Boolean = true,
    ) {
        init {
            require(historyLimit >= 4)
            require(maximumDeltaSeconds > minimumDeltaSeconds)
            require(baseInnovationGateMeters > 0.0)
            require(maximumInnovationGateMeters > baseInnovationGateMeters)
            require(huberTransitionMeters > 0.0)
            require(velocityBlend in 0.0..1.0)
            require(accelerationBlend in 0.0..1.0)
            require(temporalEvidenceWeight >= 0.0)
            require(measurementEvidenceWeight >= 0.0)
            require(timingPenaltyWeight >= 0.0)
            require(lightingPenaltyWeight >= 0.0)
            require(cameraRiskPenaltyWeight >= 0.0)
            require(uncertaintyPenaltyWeight >= 0.0)
            require(topologyCorrectionGain in 0.0..1.0)
            require(topologyIterations in 1..32)
            require(minimumConstraintStiffness in 0.0..1.0)
            require(maximumConstraintCorrectionMeters > 0.0)
            require(maximumAccelerationMetersPerSecond2 > 0.0)
            require(maximumJerkMetersPerSecond3 > 0.0)
            require(microMotionVelocityFloor >= 0.0)
            require(microMotionPreservation in 0.0..1.0)
            require(minimumOutputConfidence in 0.0..1.0)
            require(hardOutlierInnovationMeters > maximumInnovationGateMeters * 0.70)
            require(uncertaintyFloorMeters > 0.0)
            require(uncertaintyCeilingMeters > uncertaintyFloorMeters)
            require(robustOutlierSigma in 1.5..8.0)
            require(minimumJointSupportRatio in 0.0..1.0)
            require(topologyResidualSoftGateMeters > 0.0)
            require(topologyResidualHardGateMeters > topologyResidualSoftGateMeters)
        }
    }

    data class Input(
        val frame: SkeletonFrame,
        val topology: DynamicTopology?,
        val captureTimingConfidence: Double = 1.0,
        val lightingConfidence: Double = 1.0,
        val cameraMotionRisk: Double = 0.0,
        val groundHeightMeters: Double? = null,
        val preserveMicroMotion: Boolean = true,
        val temporalEvidenceScale: Double = 1.0,
        val topologyConstraintScale: Double = 1.0,
    ) {
        init {
            require(captureTimingConfidence in 0.0..1.0)
            require(lightingConfidence in 0.0..1.0)
            require(cameraMotionRisk in 0.0..1.0)
            require(temporalEvidenceScale in 0.25..1.50)
            require(topologyConstraintScale in 0.25..1.50)
            if (groundHeightMeters != null) {
                require(groundHeightMeters.isFinite())
            }
        }
    }

    data class JointDiagnostics(
        val jointId: Int,
        val innovationMeters: Double,
        val innovationGateMeters: Double,
        val temporalWeight: Double,
        val measurementWeight: Double,
        val topologyCorrectionMeters: Double,
        val finalUncertaintyMeters: Double,
        val confidence: Double,
        val rejected: Boolean,
        val predicted: Boolean,
    )

    data class Result(
        val skeleton: SkeletonFrame,
        val score: Double,
        val temporalStability: Double,
        val topologyStability: Double,
        val motionContinuity: Double,
        val meanInnovationMeters: Double,
        val meanBoneResidualMeters: Double,
        val rejectedJointCount: Int,
        val diagnostics: List<JointDiagnostics>,
        val warnings: List<String>,
        val robustOutlierRatio: Double = 0.0,
        val robustInnovationGateMeters: Double = 0.0,
        val staleFrameRejected: Boolean = false,
    ) {
        val accepted: Boolean
            get() =
                score >= 0.45 &&
                    skeleton.joints.isNotEmpty() &&
                    warnings.none { it.startsWith("fatal:") }

        val bottleneck: String
            get() = when {
                temporalStability < 0.40 -> "TEMPORAL"
                topologyStability < 0.40 -> "TOPOLOGY"
                motionContinuity < 0.40 -> "MOTION"
                score < 0.45 -> "GLOBAL"
                else -> "NONE"
            }
    }

    @Synchronized
    fun configure(newConfig: Config) {
        if (newConfig == config) return
        config = newConfig
        reset()
    }

    fun configuration(): Config = config

    private data class State(
        var timestampNanos: Long,
        var position: Vec3,
        var velocity: Vec3,
        var acceleration: Vec3,
        var confidence: Double,
        var uncertaintyMeters: Double,
    )

    private val states = HashMap<Long, State>()
    private var lastTimestampNanos = Long.MIN_VALUE
    private var lastResult: Result? = null

    @Synchronized
    fun optimize(input: Input): Result {
        val previousResult = lastResult
        if (
            lastTimestampNanos != Long.MIN_VALUE &&
            input.frame.timestampNanos <= lastTimestampNanos &&
            previousResult != null
        ) {
            return previousResult.copy(
                staleFrameRejected = true,
                warnings =
                    (
                        previousResult.warnings +
                            "stale-frame-rejected" +
                            "timestamp-reversal"
                        ).distinct(),
            )
        }

        val source = sanitizeFrame(input.frame)
        if (source.joints.isEmpty()) {
            return Result(
                skeleton = source.copy(globalConfidence = 0.0),
                score = 0.0,
                temporalStability = 0.0,
                topologyStability = 0.0,
                motionContinuity = 0.0,
                meanInnovationMeters = 0.0,
                meanBoneResidualMeters = 0.0,
                rejectedJointCount = 0,
                diagnostics = emptyList(),
                warnings = listOf("fatal: no finite joints available"),
                robustOutlierRatio = 1.0,
                robustInnovationGateMeters = config.baseInnovationGateMeters,
            )
        }

        val dt = deltaSeconds(input.frame.timestampNanos)
        val runtimeQuality =
            runtimeEvidence(
                timing = input.captureTimingConfidence,
                lighting = input.lightingConfidence,
                cameraRisk = input.cameraMotionRisk,
            )

        val measured = LinkedHashMap<Int, Vec3>()
        val outputDiagnostics = ArrayList<JointDiagnostics>(source.joints.size)
        val provisional = ArrayList<ReconstructionJoint>(source.joints.size)

        var innovationSum = 0.0
        var innovationWeight = 0.0
        var rejectedCount = 0
        val innovations = ArrayList<Double>(source.joints.size)

        source.joints.forEach { joint ->
            val stateKey = stateKey(
                trackId = source.trackId,
                jointId = joint.landmarkId,
            )
            val state = states[stateKey]

            val prediction =
                if (state == null) {
                    joint.position
                } else {
                    state.position +
                        state.velocity * dt +
                        state.acceleration * (
                            0.5 * dt * dt
                        )
                }

            val innovation = joint.position.distanceTo(prediction)
            innovations += innovation
            val gate = innovationGate(
                joint = joint,
                runtimeQuality = runtimeQuality,
                dt = dt,
            )

            val hardReject =
                innovation > config.hardOutlierInnovationMeters &&
                    joint.confidence < 0.65 &&
                    state != null

            val uncertaintyPenalty = (
                joint.positionUncertaintyMeters.coerceIn(
                    config.uncertaintyFloorMeters,
                    config.uncertaintyCeilingMeters,
                ) /
                    config.uncertaintyCeilingMeters
            ).coerceIn(0.0, 1.0)

            val measurementReliability = (
                joint.confidence.coerceIn(0.0, 1.0) *
                    (1.0 - uncertaintyPenalty * config.uncertaintyPenaltyWeight) *
                    runtimeQuality
                ).coerceIn(0.03, 1.0)

            val temporalReliability =
                if (state == null) {
                    0.40
                } else {
                    (
                        temporalWeight(
                            innovation = innovation,
                            gate = gate,
                            predicted = joint.observationCount <= 0,
                        ) *
                            (1.0 - input.cameraMotionRisk * 0.25)
                        ).coerceIn(0.05, 1.0)
                }

            val robustWeight =
                huberWeight(
                    residual = innovation,
                    transition = config.huberTransitionMeters,
                )

            val combinedMeasurementWeight = (
                config.measurementEvidenceWeight *
                    measurementReliability *
                    robustWeight
                ).coerceIn(0.03, 0.95)

            val combinedTemporalWeight =
                if (state == null) {
                    0.0
                } else {
                    (
                        config.temporalEvidenceWeight *
                            input.temporalEvidenceScale *
                            temporalReliability *
                            (
                                1.0 -
                                    input.cameraMotionRisk *
                                    config.cameraRiskPenaltyWeight
                                ).coerceIn(0.20, 1.0)
                        ).coerceIn(0.03, 0.92)
                }

            val normalizedWeights =
                normalizeWeights(
                    temporal = combinedTemporalWeight,
                    measurement = combinedMeasurementWeight,
                )

            val blended =
                if (hardReject || innovation > gate * 1.65) {
                    rejectedCount += 1
                    prediction
                } else if (state == null) {
                    joint.position
                } else {
                    prediction.lerp(
                        joint.position,
                        normalizedWeights.measurement,
                    )
                }

            measured[joint.landmarkId] = blended
            innovationSum += min(
                innovation,
                config.maximumInnovationGateMeters * 2.0,
            ) * joint.confidence.coerceIn(0.05, 1.0)
            innovationWeight += joint.confidence.coerceIn(0.05, 1.0)

            provisional += joint.copy(
                position = blended,
                confidence = (
                    joint.confidence *
                        (
                            0.62 +
                                0.38 * normalizedWeights.measurement
                            )
                        ).coerceIn(0.0, 1.0),
                reprojectionErrorPx =
                    joint.reprojectionErrorPx.coerceAtLeast(0.0),
                positionUncertaintyMeters =
                    uncertaintyAfterFusion(
                        source = joint.positionUncertaintyMeters,
                        innovation = innovation,
                        gate = gate,
                        rejected = hardReject,
                        runtimeQuality = runtimeQuality,
                    ),
            )

            outputDiagnostics +=
                JointDiagnostics(
                    jointId = joint.landmarkId,
                    innovationMeters = innovation,
                    innovationGateMeters = gate,
                    temporalWeight = normalizedWeights.temporal,
                    measurementWeight = normalizedWeights.measurement,
                    topologyCorrectionMeters = 0.0,
                    finalUncertaintyMeters =
                        joint.positionUncertaintyMeters,
                    confidence = joint.confidence,
                    rejected = hardReject ||
                        innovation > gate * 1.65,
                    predicted = joint.observationCount <= 0,
                )
        }

        val robustGate =
            robustInnovationGate(
                innovations = innovations,
                baseGate = config.baseInnovationGateMeters,
                sigma = config.robustOutlierSigma,
            )

        var robustOutlierCount = 0
        val robustFiltered =
            provisional.mapIndexed { index, joint ->
                val innovation =
                    innovations.getOrNull(index) ?: 0.0
                if (
                    innovation > robustGate &&
                    joint.confidence < 0.86
                ) {
                    robustOutlierCount += 1
                    joint.copy(
                        confidence = (
                            joint.confidence *
                                0.64
                            ).coerceAtLeast(
                                config.minimumOutputConfidence,
                            ),
                        positionUncertaintyMeters = (
                            joint.positionUncertaintyMeters *
                                1.28
                            ).coerceAtMost(
                                config.uncertaintyCeilingMeters,
                            ),
                    )
                } else {
                    joint
                }
            }

        val robustOutlierRatio =
            if (provisional.isEmpty()) {
                0.0
            } else {
                robustOutlierCount.toDouble() /
                    provisional.size.toDouble()
            }

        val constrained =
            enforceTopology(
                joints = robustFiltered,
                topology = input.topology,
                constraintScale = input.topologyConstraintScale,
            )

        val grounded =
            if (input.groundHeightMeters != null) {
                enforceGroundHeight(
                    joints = constrained,
                    groundHeight = input.groundHeightMeters,
                    topology = input.topology,
                )
            } else {
                constrained
            }

        val filtered =
            enforceDynamicLimits(
                joints = grounded,
                previousStates = states,
                timestampNanos = input.frame.timestampNanos,
                trackId = input.frame.trackId,
                preserveMicroMotion =
                    input.preserveMicroMotion &&
                        config.microMotionPreservation > 0.0,
            )

        val updated =
            updateStates(
                frame = filtered,
                timestampNanos = input.frame.timestampNanos,
                dt = dt,
                trackId = input.frame.trackId,
                subjectType = input.frame.subjectType,
            )

        val metrics =
            evaluateMetrics(
                frame = updated,
                topology = input.topology,
                meanInnovation =
                    if (innovationWeight <= 1e-9) {
                        0.0
                    } else {
                        innovationSum / innovationWeight
                    },
                rejectedCount = rejectedCount,
                runtimeQuality = runtimeQuality,
            )

        val diagnostics =
            outputDiagnostics.map { original ->
                val joint = updated.joints.firstOrNull {
                    it.landmarkId == original.jointId
                }
                if (joint == null) {
                    original
                } else {
                    original.copy(
                        topologyCorrectionMeters =
                            topologyCorrection(
                                original.jointId,
                                provisional,
                                updated.joints,
                            ),
                        finalUncertaintyMeters =
                            joint.positionUncertaintyMeters,
                        confidence = joint.confidence,
                    )
                }
            }

        val warnings =
            buildWarnings(
                frame = updated,
                metrics = metrics,
                runtimeQuality = runtimeQuality,
                rejectedCount = rejectedCount + robustOutlierCount,
            ) + buildRobustWarnings(
                robustOutlierRatio = robustOutlierRatio,
                robustGateMeters = robustGate,
                jointCount = updated.joints.size,
            )

        val result = Result(
            skeleton = updated,
            score = (
                metrics.score *
                    (1.0 - robustOutlierRatio * 0.24)
                ).coerceIn(0.0, 1.0),
            temporalStability = metrics.temporalStability,
            topologyStability = metrics.topologyStability,
            motionContinuity = metrics.motionContinuity,
            meanInnovationMeters = metrics.meanInnovationMeters,
            meanBoneResidualMeters = metrics.meanBoneResidualMeters,
            rejectedJointCount = rejectedCount,
            diagnostics = diagnostics,
            warnings = warnings.distinct(),
            robustOutlierRatio = robustOutlierRatio,
            robustInnovationGateMeters = robustGate,
        )
        lastResult = result
        return result
    }

    @Synchronized
    fun reset() {
        states.clear()
        lastTimestampNanos = Long.MIN_VALUE
        lastResult = null
    }

    private data class NormalizedWeights(
        val temporal: Double,
        val measurement: Double,
    )

    private data class Metrics(
        val score: Double,
        val temporalStability: Double,
        val topologyStability: Double,
        val motionContinuity: Double,
        val meanInnovationMeters: Double,
        val meanBoneResidualMeters: Double,
    )

    private fun robustInnovationGate(
        innovations: List<Double>,
        baseGate: Double,
        sigma: Double,
    ): Double {
        val finite =
            innovations
                .filter { it.isFinite() }
                .sorted()

        if (finite.isEmpty()) {
            return baseGate
        }

        val median =
            if (finite.size % 2 == 0) {
                (
                    finite[finite.size / 2 - 1] +
                        finite[finite.size / 2]
                    ) * 0.5
            } else {
                finite[finite.size / 2]
            }

        val deviations =
            finite.map {
                abs(it - median)
            }.sorted()

        val mad =
            if (deviations.size % 2 == 0) {
                (
                    deviations[deviations.size / 2 - 1] +
                        deviations[deviations.size / 2]
                    ) * 0.5
            } else {
                deviations[deviations.size / 2]
            }

        val robustSigma =
            mad * 1.4826 * sigma

        return max(
            baseGate,
            min(
                config.maximumInnovationGateMeters * 1.35,
                median + robustSigma,
            ),
        )
    }

    private fun buildRobustWarnings(
        robustOutlierRatio: Double,
        robustGateMeters: Double,
        jointCount: Int,
    ): List<String> = buildList {
        if (jointCount <= 0) {
            add("fatal: robust reconstruction stage has no joints")
            return@buildList
        }

        if (jointCount < 6) {
            add("low joint support limits robust reconstruction confidence")
        }

        if (robustOutlierRatio > 0.35) {
            add("high robust outlier ratio; reconstruction evidence is unstable")
        } else if (robustOutlierRatio > 0.18) {
            add("moderate robust outlier ratio; inspect source tracking")
        }

        if (robustGateMeters >
            config.maximumInnovationGateMeters
        ) {
            add("robust innovation gate expanded above nominal bound")
        }
    }

    private fun sanitizeFrame(
        frame: SkeletonFrame,
    ): SkeletonFrame {
        val joints = frame.joints.mapNotNull { joint ->
            val p = joint.position
            if (!p.isFinite()) {
                null
            } else {
                joint.copy(
                    position = p,
                    velocity =
                        if (joint.velocity.isFinite()) {
                            joint.velocity
                        } else {
                            Vec3(0.0, 0.0, 0.0)
                        },
                    confidence = joint.confidence.coerceIn(0.0, 1.0),
                    positionUncertaintyMeters =
                        joint.positionUncertaintyMeters
                            .coerceIn(
                                0.0,
                                config.uncertaintyCeilingMeters,
                            ),
                )
            }
        }

        return frame.copy(
            joints = joints,
            globalConfidence =
                frame.globalConfidence.coerceIn(0.0, 1.0),
        )
    }

    private fun runtimeEvidence(
        timing: Double,
        lighting: Double,
        cameraRisk: Double,
    ): Double {
        val timingScore =
            timing.coerceIn(0.0, 1.0)
        val lightingScore =
            lighting.coerceIn(0.0, 1.0)
        val cameraScore =
            1.0 - cameraRisk.coerceIn(0.0, 1.0)

        return (
            0.58 * timingScore +
                0.20 * lightingScore +
                0.22 * cameraScore
            ).coerceIn(0.18, 1.0)
    }

    private fun innovationGate(
        joint: ReconstructionJoint,
        runtimeQuality: Double,
        dt: Double,
    ): Double {
        val uncertainty =
            joint.positionUncertaintyMeters.coerceIn(
                config.uncertaintyFloorMeters,
                config.uncertaintyCeilingMeters,
            )

        val velocityAllowance =
            joint.velocity.norm() *
                dt.coerceIn(
                    config.minimumDeltaSeconds,
                    config.maximumDeltaSeconds,
                ) *
                0.75

        val confidenceAllowance =
            (
                1.0 -
                    joint.confidence.coerceIn(0.0, 1.0)
                ) * 0.10

        return (
            config.baseInnovationGateMeters +
                uncertainty * 0.55 +
                velocityAllowance +
                confidenceAllowance +
                (
                    1.0 -
                        runtimeQuality
                    ) * 0.045
            ).coerceIn(
                config.baseInnovationGateMeters,
                config.maximumInnovationGateMeters,
            )
    }

    private fun temporalWeight(
        innovation: Double,
        gate: Double,
        predicted: Boolean,
    ): Double {
        if (gate <= 1e-9) return 0.0

        val normalized =
            (innovation / gate).coerceIn(
                0.0,
                3.0,
            )

        val innovationScore =
            (
                1.0 -
                    normalized / 2.0
                ).coerceIn(0.0, 1.0)

        return (
            innovationScore * 0.62 +
                if (predicted) 0.12 else 0.38
            ).coerceIn(0.0, 1.0)
    }

    private fun huberWeight(
        residual: Double,
        transition: Double,
    ): Double {
        if (!residual.isFinite()) return 0.0
        if (residual <= transition) return 1.0

        return transition /
            residual.coerceAtLeast(
                transition + 1e-9,
            )
    }

    private fun normalizeWeights(
        temporal: Double,
        measurement: Double,
    ): NormalizedWeights {
        val t = temporal.coerceAtLeast(0.0)
        val m = measurement.coerceAtLeast(0.0)
        val total = (t + m).coerceAtLeast(1e-6)
        return NormalizedWeights(
            temporal = t / total,
            measurement = m / total,
        )
    }

    private fun enforceTopology(
        joints: List<ReconstructionJoint>,
        topology: DynamicTopology?,
        constraintScale: Double,
    ): List<ReconstructionJoint> {
        if (topology == null || topology.edges.isEmpty()) {
            return joints
        }

        val positions =
            joints.associate { it.landmarkId to it.position }
                .toMutableMap()

        val confidence =
            joints.associate { it.landmarkId to it.confidence }

        repeat(config.topologyIterations) {
            topology.edges.forEach { edge ->
                val pa = positions[edge.a] ?: return@forEach
                val pb = positions[edge.b] ?: return@forEach

                val delta = pb - pa
                val length = delta.norm()

                if (!length.isFinite() || length < 1e-8) {
                    return@forEach
                }

                val minLength =
                    edge.minimumLengthMeters
                        .coerceAtLeast(0.0001)
                val maxLength =
                    max(
                        edge.maximumLengthMeters,
                        minLength,
                    )

                val target =
                    edge.preferredLengthMeters
                        .coerceIn(
                            minLength,
                            maxLength,
                        )

                val residual =
                    length - target

                val normalizedResidual =
                    abs(residual) /
                        target.coerceAtLeast(1e-6)

                if (normalizedResidual < 0.025) {
                    return@forEach
                }

                val stiffness =
                    (
                        config.minimumConstraintStiffness +
                            edge.stiffness.coerceIn(0.0, 1.0) *
                            config.topologyCorrectionGain *
                            constraintScale.coerceIn(0.25, 1.50)
                        ).coerceIn(
                            0.0,
                            1.0,
                        )

                val correction =
                    (
                        residual *
                            stiffness
                        ).coerceIn(
                            -config.maximumConstraintCorrectionMeters,
                            config.maximumConstraintCorrectionMeters,
                        )

                val direction =
                    delta / length

                val wa =
                    confidence[edge.a]
                        ?.coerceIn(0.05, 1.0)
                        ?: 0.5

                val wb =
                    confidence[edge.b]
                        ?.coerceIn(0.05, 1.0)
                        ?: 0.5

                val sum = (
                    wa + wb
                ).coerceAtLeast(1e-6)

                val movementA =
                    correction *
                        (wb / sum)

                val movementB =
                    correction *
                        (wa / sum)

                positions[edge.a] =
                    pa +
                        direction *
                        movementA

                positions[edge.b] =
                    pb -
                        direction *
                        movementB
            }
        }

        return joints.map { joint ->
            val updated =
                positions[joint.landmarkId]
                    ?: joint.position

            joint.copy(
                position = updated,
                confidence = (
                    joint.confidence *
                        topologyConfidence(
                            joint = joint,
                            old = joint.position,
                            updated = updated,
                        )
                    ).coerceIn(0.0, 1.0),
            )
        }
    }

    private fun topologyConfidence(
        joint: ReconstructionJoint,
        old: Vec3,
        updated: Vec3,
    ): Double {
        val correction =
            old.distanceTo(updated)

        val penalty =
            (
                correction /
                    config.maximumConstraintCorrectionMeters
            ).coerceIn(
                0.0,
                1.0,
            )

        val observationFactor =
            if (joint.observationCount > 0) {
                1.0
            } else {
                0.84
            }

        return (
            1.0 -
                penalty * 0.18
            ).coerceIn(0.68, 1.0) *
            observationFactor
    }

    private fun enforceGroundHeight(
        joints: List<ReconstructionJoint>,
        groundHeight: Double,
        topology: DynamicTopology?,
    ): List<ReconstructionJoint> {
        val footIds =
            topology?.nodes
                ?.filter {
                    val name = it.semanticName.lowercase()
                    name.contains("ankle") ||
                        name.contains("foot") ||
                        name.contains("heel") ||
                        name.contains("toe")
                }
                ?.map { it.id }
                ?.toSet()
                ?: emptySet()

        if (footIds.isEmpty()) {
            return joints
        }

        return joints.map { joint ->
            if (joint.landmarkId !in footIds) {
                joint
            } else {
                val y = joint.position.y
                val delta = groundHeight - y

                if (abs(delta) > 0.15) {
                    joint.copy(
                        confidence =
                            joint.confidence * 0.86,
                    )
                } else {
                    joint.copy(
                        position =
                            joint.position.copy(
                                y = y + delta * 0.28,
                            ),
                        confidence = (
                            joint.confidence *
                                (
                                    1.0 -
                                        abs(delta) /
                                            0.30 *
                                            0.12
                                    )
                            ).coerceIn(0.0, 1.0),
                    )
                }
            }
        }
    }

    private fun enforceDynamicLimits(
        joints: List<ReconstructionJoint>,
        previousStates: Map<Long, State>,
        timestampNanos: Long,
        trackId: Long,
        preserveMicroMotion: Boolean,
    ): List<ReconstructionJoint> {
        return joints.map { joint ->
            val state =
                previousStates[
                    stateKey(
                        trackId = trackId,
                        jointId = joint.landmarkId,
                    )
                ]

            val fallback =
                state ?: states.entries
                    .firstOrNull {
                        it.key == stateKey(
                            trackId = trackId,
                            jointId = joint.landmarkId,
                        )
                    }?.value

            val dt =
                if (fallback == null) {
                    1.0 / 60.0
                } else {
                    (
                        timestampNanos -
                            fallback.timestampNanos
                        ).coerceAtLeast(1L) / 1e9
                }.coerceIn(
                    config.minimumDeltaSeconds,
                    config.maximumDeltaSeconds,
                )

            val maxDisplacement =
                (
                    fallback?.velocity?.norm() ?: 0.0
                ) * dt +
                    0.5 *
                    config.maximumAccelerationMetersPerSecond2 *
                    dt * dt

            val measured =
                if (fallback == null) {
                    joint.position
                } else {
                    val displacement =
                        joint.position -
                            fallback.position
                    val length =
                        displacement.norm()

                    if (
                        length >
                            maxDisplacement +
                                config.baseInnovationGateMeters
                    ) {
                        fallback.position +
                            displacement.normalized() *
                            (
                                maxDisplacement +
                                    config.baseInnovationGateMeters
                            )
                    } else {
                        joint.position
                    }
                }

            val microMotionBoost =
                if (
                    preserveMicroMotion &&
                    joint.velocity.norm() in
                        0.0..config.microMotionVelocityFloor
                ) {
                    config.microMotionPreservation
                } else {
                    0.0
                }

            joint.copy(
                position = measured.lerp(
                    joint.position,
                    0.35 + microMotionBoost * 0.20,
                ),
            )
        }
    }

    private fun updateStates(
        frame: List<ReconstructionJoint>,
        timestampNanos: Long,
        dt: Double,
        trackId: Long,
        subjectType: com.nivukx.nox.core.model.SubjectType,
    ): SkeletonFrame {
        val statesForFrame =
            frame.map { joint ->
                val key =
                    stateKey(
                        trackId = trackId,
                        jointId = joint.landmarkId,
                    )

                val old =
                    states[key]

                val currentVelocity =
                    if (old == null) {
                        joint.velocity
                    } else {
                        (
                            joint.position -
                                old.position
                            ) / dt
                    }

                val velocity =
                    if (old == null) {
                        currentVelocity
                    } else {
                        old.velocity.lerp(
                            currentVelocity,
                            config.velocityBlend,
                        )
                    }

                val rawAcceleration =
                    if (old == null) {
                        joint.velocity / dt.coerceAtLeast(config.minimumDeltaSeconds)
                    } else {
                        (
                            velocity -
                                old.velocity
                            ) / dt
                    }

                val acceleration =
                    if (old == null) {
                        clampAcceleration(
                            rawAcceleration,
                        )
                    } else {
                        clampJerk(
                            previous = old.acceleration,
                            measured = rawAcceleration,
                            dt = dt,
                        )
                    }

                val uncertainty =
                    (
                        joint.positionUncertaintyMeters *
                            0.74 +
                            (
                                velocity.distanceTo(
                                    joint.velocity,
                                ) * 0.008
                            ) *
                            0.26
                        ).coerceIn(
                            config.uncertaintyFloorMeters,
                            config.uncertaintyCeilingMeters,
                        )

                states[key] =
                    State(
                        timestampNanos = timestampNanos,
                        position = joint.position,
                        velocity = velocity,
                        acceleration = acceleration,
                        confidence = joint.confidence,
                        uncertaintyMeters = uncertainty,
                    )

                joint.copy(
                    velocity = velocity,
                    confidence =
                        (
                            joint.confidence *
                                (
                                    0.75 +
                                        0.25 *
                                            motionContinuityConfidence(
                                                old = old,
                                                velocity = velocity,
                                                acceleration = acceleration,
                                            )
                                    )
                            ).coerceIn(
                                config.minimumOutputConfidence,
                                1.0,
                            ),
                    positionUncertaintyMeters = uncertainty,
                )
            }

        lastTimestampNanos = timestampNanos

        pruneStates()

        return SkeletonFrame(
            timestampNanos = timestampNanos,
            subjectType = subjectType,
            trackId = trackId,
            joints = statesForFrame,
            globalConfidence =
                statesForFrame
                    .map { it.confidence }
                    .averageOrZero()
                    .coerceIn(0.0, 1.0),
        )
    }

    private fun evaluateMetrics(
        frame: SkeletonFrame,
        topology: DynamicTopology?,
        meanInnovation: Double,
        rejectedCount: Int,
        runtimeQuality: Double,
    ): Metrics {
        val boneResidual =
            meanBoneResidual(
                frame,
                topology,
            )

        val temporal =
            (
                exp(-meanInnovation / 0.06) * 0.56 +
                    frame.globalConfidence * 0.22 +
                    (1.0 - rejectedRatio(
                        frame.joints.size,
                        rejectedCount,
                    )) * 0.22
                ).coerceIn(0.0, 1.0)

        val topologyScore =
            if (topology == null) {
                0.68
            } else {
                (
                    exp(-boneResidual / 0.06) * 0.62 +
                        topology.globalConfidence * 0.20 +
                        topology.topologyStability * 0.18
                    ).coerceIn(0.0, 1.0)
            }

        val continuity =
            (
                frame.joints
                    .map { joint ->
                        (
                            exp(
                                -joint.positionUncertaintyMeters /
                                    0.12,
                            ) * 0.45 +
                                joint.confidence * 0.35 +
                                (
                                    1.0 -
                                        joint.reprojectionErrorPx /
                                            12.0
                                    ).coerceIn(
                                        0.0,
                                        1.0,
                                    ) * 0.20
                            ).coerceIn(0.0, 1.0)
                    }
                    .averageOrZero() * 0.78 +
                    runtimeQuality * 0.22
                ).coerceIn(0.0, 1.0)

        val score =
            (
                temporal * 0.34 +
                    topologyScore * 0.28 +
                    continuity * 0.24 +
                    runtimeQuality * 0.14
                ).coerceIn(0.0, 1.0)

        return Metrics(
            score = score,
            temporalStability = temporal,
            topologyStability = topologyScore,
            motionContinuity = continuity,
            meanInnovationMeters = meanInnovation,
            meanBoneResidualMeters = boneResidual,
        )
    }

    private fun rejectedRatio(
        total: Int,
        rejected: Int,
    ): Double =
        if (total <= 0) 0.0
        else (rejected.toDouble() / total.toDouble()).coerceIn(0.0, 1.0)

    private fun meanBoneResidual(
        frame: SkeletonFrame,
        topology: DynamicTopology?,
    ): Double {
        if (topology == null || topology.edges.isEmpty()) {
            return 0.0
        }

        val positions =
            frame.joints.associate {
                it.landmarkId to it.position
            }

        val residuals =
            topology.edges.mapNotNull { edge ->
                val a =
                    positions[edge.a]
                        ?: return@mapNotNull null

                val b =
                    positions[edge.b]
                        ?: return@mapNotNull null

                val length =
                    a.distanceTo(b)

                abs(
                    length -
                        edge.preferredLengthMeters,
                ).coerceAtMost(1.0)
            }

        return residuals.averageOrZero()
    }

    private fun motionContinuityConfidence(
        old: State?,
        velocity: Vec3,
        acceleration: Vec3,
    ): Double {
        if (old == null) return 0.65

        val velocityChange =
            velocity.distanceTo(old.velocity)

        val accelerationChange =
            acceleration.distanceTo(old.acceleration)

        val velocityScore =
            exp(-velocityChange / 2.5)
                .coerceIn(0.0, 1.0)

        val accelerationScore =
            exp(-accelerationChange / 18.0)
                .coerceIn(0.0, 1.0)

        return (
            velocityScore * 0.55 +
                accelerationScore * 0.45
            ).coerceIn(0.0, 1.0)
    }

    private fun clampAcceleration(
        acceleration: Vec3,
    ): Vec3 {
        val magnitude =
            acceleration.norm()

        if (
            !magnitude.isFinite() ||
            magnitude <=
                config.maximumAccelerationMetersPerSecond2
        ) {
            return acceleration
        }

        return acceleration.normalized() *
            config.maximumAccelerationMetersPerSecond2
    }

    private fun clampJerk(
        previous: Vec3,
        measured: Vec3,
        dt: Double,
    ): Vec3 {
        val boundedMeasured =
            clampAcceleration(
                measured,
            )

        val maximumDelta =
            config.maximumJerkMetersPerSecond3 *
                dt

        val delta =
            boundedMeasured -
                previous

        val magnitude =
            delta.norm()

        return if (
            !magnitude.isFinite() ||
            magnitude <= maximumDelta
        ) {
            boundedMeasured
        } else {
            previous +
                delta.normalized() *
                maximumDelta
        }
    }

    private fun uncertaintyAfterFusion(
        source: Double,
        innovation: Double,
        gate: Double,
        rejected: Boolean,
        runtimeQuality: Double,
    ): Double {
        val sourceUncertainty =
            source.coerceIn(
                config.uncertaintyFloorMeters,
                config.uncertaintyCeilingMeters,
            )

        val innovationPenalty =
            (
                innovation /
                    gate.coerceAtLeast(
                        config.baseInnovationGateMeters,
                    )
            ).coerceIn(0.0, 3.0)

        val uncertainty =
            sourceUncertainty *
                (
                    0.72 +
                        innovationPenalty * 0.24 +
                        (1.0 - runtimeQuality) * 0.20 +
                        if (rejected) 0.20 else 0.0
                    )

        return uncertainty.coerceIn(
            config.uncertaintyFloorMeters,
            config.uncertaintyCeilingMeters,
        )
    }

    private fun buildWarnings(
        frame: SkeletonFrame,
        metrics: Metrics,
        runtimeQuality: Double,
        rejectedCount: Int,
    ): List<String> {
        val warnings = mutableListOf<String>()

        if (metrics.meanInnovationMeters > 0.080) {
            warnings += "temporal innovation is high"
        }

        if (metrics.meanBoneResidualMeters > 0.080) {
            warnings += "topology residual is high"
        }

        if (metrics.temporalStability < 0.42) {
            warnings += "temporal stability below production target"
        }

        if (metrics.topologyStability < 0.42) {
            warnings += "topology stability below production target"
        }

        if (metrics.motionContinuity < 0.42) {
            warnings += "motion continuity below production target"
        }

        if (runtimeQuality < 0.45) {
            warnings += "runtime evidence quality reduced"
        }

        if (rejectedCount >
            max(
                1,
                frame.joints.size / 3,
            )
        ) {
            warnings += "many joint measurements were treated as outliers"
        }

        if (frame.globalConfidence < 0.35) {
            warnings += "global reconstruction confidence is low"
        }

        if (frame.joints.isEmpty()) {
            warnings += "fatal: optimizer output has no joints"
        }

        return warnings.distinct()
    }

    private fun stateKey(
        trackId: Long,
        jointId: Int,
    ): Long =
        (
            trackId * 1_000_003L
        ) xor
            jointId.toLong()

    private fun deltaSeconds(
        timestampNanos: Long,
    ): Double {
        if (
            lastTimestampNanos == Long.MIN_VALUE ||
            timestampNanos <= lastTimestampNanos
        ) {
            return 1.0 / 60.0
        }

        return (
            (
                timestampNanos -
                    lastTimestampNanos
            ).toDouble() / 1e9
        ).coerceIn(
            config.minimumDeltaSeconds,
            config.maximumDeltaSeconds,
        )
    }

    private fun topologyCorrection(
        jointId: Int,
        source: List<ReconstructionJoint>,
        updated: List<ReconstructionJoint>,
    ): Double {
        val a =
            source.firstOrNull {
                it.landmarkId == jointId
            }?.position
                ?: return 0.0

        val b =
            updated.firstOrNull {
                it.landmarkId == jointId
            }?.position
                ?: return 0.0

        return a.distanceTo(b)
    }

    private fun pruneStates() {
        if (states.size <=
            config.historyLimit * 128
        ) {
            return
        }

        val cutoff =
            states.values
                .minOfOrNull {
                    it.timestampNanos
                }
                ?: return

        states.entries.removeIf {
            it.value.timestampNanos < cutoff
        }
    }

    private fun Vec3.isFinite(): Boolean =
        x.isFinite() &&
            y.isFinite() &&
            z.isFinite()

    private fun Vec3.accelerationEstimate(
        dt: Double,
    ): Vec3 =
        this / dt.coerceAtLeast(
            config.minimumDeltaSeconds,
        )

    private fun Vec3.lerp(
        target: Vec3,
        alpha: Double,
    ): Vec3 =
        Vec3(
            x * (1.0 - alpha) + target.x * alpha,
            y * (1.0 - alpha) + target.y * alpha,
            z * (1.0 - alpha) + target.z * alpha,
        )

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private fun List<ReconstructionJoint>.averageConfidenceOrZero(): Double =
        if (isEmpty()) {
            0.0
        } else {
            map { it.confidence.coerceIn(0.0, 1.0) }.average()
        }
}

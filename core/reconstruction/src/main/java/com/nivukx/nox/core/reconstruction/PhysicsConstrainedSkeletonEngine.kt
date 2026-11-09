package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Physics-aware kinematic projection for metric skeleton reconstruction.
 *
 * This engine is intentionally deterministic and model-agnostic. It does not invent missing joints
 * and it never replaces geometric evidence with a rigid avatar prior. Instead it projects already
 * reconstructed joints toward a constrained, internally consistent state while preserving stronger
 * observations and explicit uncertainty.
 *
 * The design combines:
 * - robust bone-length learning with confidence/uncertainty gating;
 * - bounded iterative edge projection;
 * - velocity and displacement sanity limits;
 * - acceleration and jerk consistency checks;
 * - bilateral limb-length symmetry diagnostics;
 * - support-plane/foot-contact diagnostics when a plausible floor exists;
 * - confidence penalties rather than silent deletion when a constraint is violated;
 * - hysteretic history so one bad frame cannot permanently rewrite the rig.
 *
 * Landmark identifiers follow the common 33-point RGB pose convention by default. Applications may
 * replace the graph with a rig-specific edge table without changing the solver.
 */
class PhysicsConstrainedSkeletonEngine(
    private var config: Config = Config(),
    edges: List<Edge> = defaultEdges(),
) {

    data class Edge(
        val parent: Int,
        val child: Int,
        val group: String,
        val expectedLengthMeters: Double? = null,
        val bilateralKey: String? = null,
    ) {
        init {
            require(parent >= 0)
            require(child >= 0)
            require(parent != child)
            require(group.isNotBlank())
            require(
                expectedLengthMeters == null ||
                    expectedLengthMeters.isFinite() &&
                    expectedLengthMeters > 0.0,
            )
        }
    }

    data class Config(
        val iterations: Int = 5,
        val learningRate: Double = 0.10,
        val projectionStrength: Double = 0.74,
        val maxCorrectionFraction: Double = 0.28,
        val minimumJointConfidence: Double = 0.34,
        val maximumBoneResidualMeters: Double = 0.085,
        val warningBoneResidualMeters: Double = 0.045,
        val baselineLearningRate: Double = 0.055,
        val maximumBaselineUpdateResidualMeters: Double = 0.060,
        val minimumObservationCountForLearning: Int = 2,
        val maximumVelocityMetersPerSecond: Double = 8.0,
        val warningVelocityMetersPerSecond: Double = 4.5,
        val maximumAccelerationMetersPerSecond2: Double = 55.0,
        val warningAccelerationMetersPerSecond2: Double = 28.0,
        val maximumJerkMetersPerSecond3: Double = 420.0,
        val warningJerkMetersPerSecond3: Double = 170.0,
        val symmetryToleranceRatio: Double = 0.30,
        val floorQuantile: Double = 0.14,
        val minimumFloorConfidence: Double = 0.48,
        val contactVerticalVelocityMetersPerSecond: Double = 0.30,
        val contactHeightToleranceMeters: Double = 0.08,
        val blockedPhysicsScore: Double = 0.42,
        val warningPhysicsScore: Double = 0.68,
        val historyAlpha: Double = 0.18,
        val recoveryFrames: Int = 5,
    ) {
        init {
            require(iterations >= 1)
            require(learningRate in 0.0..1.0)
            require(projectionStrength in 0.0..1.0)
            require(maxCorrectionFraction in 0.0..1.0)
            require(minimumJointConfidence in 0.0..1.0)
            require(maximumBoneResidualMeters > warningBoneResidualMeters)
            require(warningBoneResidualMeters > 0.0)
            require(baselineLearningRate in 0.0..1.0)
            require(maximumBaselineUpdateResidualMeters > 0.0)
            require(minimumObservationCountForLearning >= 1)
            require(maximumVelocityMetersPerSecond > warningVelocityMetersPerSecond)
            require(maximumAccelerationMetersPerSecond2 > warningAccelerationMetersPerSecond2)
            require(maximumJerkMetersPerSecond3 > warningJerkMetersPerSecond3)
            require(symmetryToleranceRatio in 0.0..1.0)
            require(floorQuantile in 0.0..1.0)
            require(minimumFloorConfidence in 0.0..1.0)
            require(contactVerticalVelocityMetersPerSecond >= 0.0)
            require(contactHeightToleranceMeters > 0.0)
            require(blockedPhysicsScore in 0.0..warningPhysicsScore)
            require(warningPhysicsScore <= 1.0)
            require(historyAlpha in 0.0..1.0)
            require(recoveryFrames >= 1)
        }
    }

    data class JointPhysics(
        val landmarkId: Int,
        val correctionMeters: Double,
        val velocityMetersPerSecond: Double,
        val accelerationMetersPerSecond2: Double,
        val jerkMetersPerSecond3: Double,
        val confidencePenalty: Double,
        val constraintResidualMeters: Double,
        val corrected: Boolean,
        val warnings: List<String>,
    )

    data class EdgePhysics(
        val parent: Int,
        val child: Int,
        val group: String,
        val observedLengthMeters: Double,
        val targetLengthMeters: Double,
        val residualMeters: Double,
        val normalizedResidual: Double,
        val weight: Double,
        val corrected: Boolean,
        val warnings: List<String>,
    )

    data class Result(
        val frame: SkeletonFrame,
        val accepted: Boolean,
        val physicsScore: Double,
        val confidence: Double,
        val meanBoneResidualMeters: Double,
        val p95BoneResidualMeters: Double,
        val maxBoneResidualMeters: Double,
        val velocityViolationRatio: Double,
        val accelerationViolationRatio: Double,
        val jerkViolationRatio: Double,
        val symmetryScore: Double,
        val floorContactScore: Double,
        val correctedJointCount: Int,
        val constrainedEdgeCount: Int,
        val edges: List<EdgePhysics>,
        val joints: List<JointPhysics>,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val blocked: Boolean
            get() = !accepted || physicsScore < 0.42
    }

    private data class Baseline(
        var lengthMeters: Double,
        var confidence: Double,
        var samples: Int,
    )

    private data class History(
        var score: Double,
        var confidence: Double,
        var warningFrames: Int,
        var blockedFrames: Int,
        var recoveryFrames: Int,
    )

    private var activeEdges: List<Edge> = edges
        .distinctBy { Triple(it.parent, it.child, it.group) }

    private val baselines = LinkedHashMap<String, Baseline>()
    private val previousJoints = LinkedHashMap<Int, ReconstructionJoint>()
    private var previousTimestampNanos: Long = 0L
    private var previousVelocities = LinkedHashMap<Int, Vec3>()
    private var history: History? = null
    private var lastResult: Result? = null

    @Synchronized
    fun configure(
        newConfig: Config = config,
        newEdges: List<Edge> = activeEdges,
    ) {
        config = newConfig
        activeEdges = newEdges
            .distinctBy { Triple(it.parent, it.child, it.group) }
        reset()
    }

    fun currentEdges(): List<Edge> = activeEdges.toList()

    fun currentConfig(): Config = config

    @Synchronized
    fun refine(frame: SkeletonFrame): Result {
        if (frame.joints.isEmpty()) {
            return blocked(
                frame = frame,
                warning = "physics-guard: no reconstructed joints",
            )
        }

        val dt = if (
            previousTimestampNanos <= 0L ||
            frame.timestampNanos <= previousTimestampNanos
        ) {
            1.0 / 30.0
        } else {
            (
                frame.timestampNanos -
                    previousTimestampNanos
                ).toDouble() / 1e9
        }.coerceIn(1.0 / 240.0, 0.25)

        val initial = frame.joints.associateBy { it.landmarkId }.toMutableMap()

        val relevantEdges = activeEdges.filter {
            initial.containsKey(it.parent) &&
                initial.containsKey(it.child)
        }

        val learnedTargets = relevantEdges.associateWith { edge ->
            targetLength(edge, initial)
        }

        val working = initial.toMutableMap()

        repeat(config.iterations) {
            relevantEdges.forEach { edge ->
                val parent = working[edge.parent] ?: return@forEach
                val child = working[edge.child] ?: return@forEach
                val target = learnedTargets[edge] ?: return@forEach

                val delta = child.position - parent.position
                val length = delta.norm()
                if (!length.isFinite() || length <= 1e-8) return@forEach

                val residual = length - target
                val absoluteResidual = abs(residual)
                if (absoluteResidual <= config.warningBoneResidualMeters) {
                    return@forEach
                }

                val direction = delta / length
                val confidenceParent = jointWeight(parent)
                val confidenceChild = jointWeight(child)
                val totalWeight = (
                    confidenceParent +
                        confidenceChild
                    ).coerceAtLeast(1e-6)

                val correction = (
                    residual *
                        config.projectionStrength *
                        (
                            confidenceParent /
                                totalWeight
                            )
                    ).coerceIn(
                        -length * config.maxCorrectionFraction,
                        length * config.maxCorrectionFraction,
                    )

                val parentScale = (
                    1.0 -
                        confidenceParent /
                            totalWeight
                    ).coerceIn(0.0, 1.0)
                val childScale = (
                    confidenceChild /
                        totalWeight
                    ).coerceIn(0.0, 1.0)

                val parentPosition = parent.position +
                    direction *
                    correction *
                    parentScale *
                    config.learningRate

                val childPosition = child.position -
                    direction *
                    correction *
                    childScale *
                    config.learningRate

                working[edge.parent] = parent.copy(
                    position = parentPosition,
                )
                working[edge.child] = child.copy(
                    position = childPosition,
                )
            }
        }

        val velocityChecks = mutableListOf<Double>()
        val accelerationChecks = mutableListOf<Double>()
        val jerkChecks = mutableListOf<Double>()
        val jointDiagnostics = mutableListOf<JointPhysics>()

        val constrainedJoints = working.mapValues { (_, joint) ->
            val previous = previousJoints[joint.landmarkId]
            val previousVelocity = previousVelocities[joint.landmarkId]
            val inferredVelocity = if (previous != null) {
                (joint.position - previous.position) / dt
            } else {
                joint.velocity
            }

            val speed = inferredVelocity.norm()
            val acceleration = if (previousVelocity != null) {
                (inferredVelocity - previousVelocity).norm() / dt
            } else {
                joint.velocity.distanceTo(inferredVelocity) / dt
            }
            val jerk = if (previous != null && previousVelocity != null) {
                val previousAcceleration =
                    (previous.velocity - previousVelocity).norm() / dt
                abs(acceleration - previousAcceleration) / dt
            } else {
                0.0
            }

            val velocityViolation = (
                speed /
                    config.maximumVelocityMetersPerSecond
            ).coerceAtLeast(0.0)

            val accelerationViolation = (
                acceleration /
                    config.maximumAccelerationMetersPerSecond2
            ).coerceAtLeast(0.0)

            val jerkViolation = (
                jerk /
                    config.maximumJerkMetersPerSecond3
            ).coerceAtLeast(0.0)

            velocityChecks += velocityViolation
            accelerationChecks += accelerationViolation
            jerkChecks += jerkViolation

            val speedPenalty = violationPenalty(
                value = speed,
                warning = config.warningVelocityMetersPerSecond,
                maximum = config.maximumVelocityMetersPerSecond,
            )
            val accelerationPenalty = violationPenalty(
                value = acceleration,
                warning = config.warningAccelerationMetersPerSecond2,
                maximum = config.maximumAccelerationMetersPerSecond2,
            )
            val jerkPenalty = violationPenalty(
                value = jerk,
                warning = config.warningJerkMetersPerSecond3,
                maximum = config.maximumJerkMetersPerSecond3,
            )

            val totalPenalty = (
                speedPenalty * 0.40 +
                    accelerationPenalty * 0.36 +
                    jerkPenalty * 0.24
                ).coerceIn(0.0, 0.95)

            val constraintResidual = edgeResidualForJoint(
                landmarkId = joint.landmarkId,
                working = working,
                targets = learnedTargets,
            )

            val corrected =
                joint.position.distanceTo(
                    initial[joint.landmarkId]?.position
                        ?: joint.position,
                ) > 1e-6

            val warnings = buildList {
                if (speed >= config.warningVelocityMetersPerSecond) {
                    add("velocity-elevated")
                }
                if (speed >= config.maximumVelocityMetersPerSecond) {
                    add("velocity-exceeded-production-envelope")
                }
                if (acceleration >= config.warningAccelerationMetersPerSecond2) {
                    add("acceleration-elevated")
                }
                if (acceleration >= config.maximumAccelerationMetersPerSecond2) {
                    add("acceleration-exceeded-production-envelope")
                }
                if (jerk >= config.warningJerkMetersPerSecond3) {
                    add("jerk-elevated")
                }
                if (jerk >= config.maximumJerkMetersPerSecond3) {
                    add("jerk-exceeded-production-envelope")
                }
                if (constraintResidual > config.warningBoneResidualMeters) {
                    add("kinematic-constraint-residual-elevated")
                }
            }.distinct()

            val confidence = (
                joint.confidence *
                    (
                        1.0 -
                            totalPenalty *
                            0.58
                    )
                ).coerceIn(0.0, 1.0)

            jointDiagnostics += JointPhysics(
                landmarkId = joint.landmarkId,
                correctionMeters = joint.position.distanceTo(
                    initial[joint.landmarkId]?.position ?: joint.position,
                ),
                velocityMetersPerSecond = speed,
                accelerationMetersPerSecond2 = acceleration,
                jerkMetersPerSecond3 = jerk,
                confidencePenalty = totalPenalty,
                constraintResidualMeters = constraintResidual,
                corrected = corrected,
                warnings = warnings,
            )

            joint.copy(
                velocity = inferredVelocity,
                confidence = confidence,
                positionUncertaintyMeters = (
                    joint.positionUncertaintyMeters *
                        (
                            1.0 +
                                totalPenalty * 0.45
                        )
                    ).coerceAtLeast(0.0),
            )
        }

        val edgeDiagnostics = relevantEdges.map { edge ->
            val parent = constrainedJoints[edge.parent]!!
            val child = constrainedJoints[edge.child]!!
            val target = learnedTargets[edge]!!
            val observed = parent.position.distanceTo(child.position)
            val residual = abs(observed - target)
            EdgePhysics(
                parent = edge.parent,
                child = edge.child,
                group = edge.group,
                observedLengthMeters = observed,
                targetLengthMeters = target,
                residualMeters = residual,
                normalizedResidual = (
                    residual /
                        config.maximumBoneResidualMeters
                ).coerceIn(0.0, 1.0),
                weight = (
                    jointWeight(parent) *
                        jointWeight(child)
                    ).coerceIn(0.0, 1.0),
                corrected = (
                    parent.position.distanceTo(
                        initial[edge.parent]?.position ?: parent.position,
                    ) > 1e-6 ||
                        child.position.distanceTo(
                            initial[edge.child]?.position ?: child.position,
                        ) > 1e-6
                    ),
                warnings = buildList {
                    if (residual > config.warningBoneResidualMeters) {
                        add("bone-length-residual-elevated")
                    }
                    if (residual > config.maximumBoneResidualMeters) {
                        add("bone-length-residual-exceeded-production-envelope")
                    }
                },
            )
        }

        val residuals = edgeDiagnostics
            .map { it.residualMeters }
            .filter { it.isFinite() }
            .sorted()

        val meanResidual = residuals.averageOrZero()
        val p95Residual = percentile(
            values = residuals,
            fraction = 0.95,
        )
        val maxResidual = residuals.maxOrNull() ?: 0.0

        val velocityViolationRatio = violationRatio(
            values = velocityChecks,
            threshold = 1.0,
        )
        val accelerationViolationRatio = violationRatio(
            values = accelerationChecks,
            threshold = 1.0,
        )
        val jerkViolationRatio = violationRatio(
            values = jerkChecks,
            threshold = 1.0,
        )

        val symmetryScore = symmetryScore(
            constrainedJoints,
        )

        val floorContactScore = floorContactScore(
            constrainedJoints,
        )

        val residualScore = if (residuals.isEmpty()) {
            1.0
        } else {
            (
                1.0 -
                    meanResidual /
                    config.maximumBoneResidualMeters
            ).coerceIn(0.0, 1.0)
        }

        val dynamicsScore = (
            1.0 -
                velocityViolationRatio * 0.34 -
                accelerationViolationRatio * 0.38 -
                jerkViolationRatio * 0.28
            ).coerceIn(0.0, 1.0)

        val confidenceScore = if (constrainedJoints.isEmpty()) {
            0.0
        } else {
            constrainedJoints.values.map { it.confidence }.average()
        }

        val physicsScore = (
            residualScore * 0.34 +
                dynamicsScore * 0.28 +
                symmetryScore * 0.18 +
                floorContactScore * 0.08 +
                confidenceScore * 0.12
            ).coerceIn(0.0, 1.0)

        val warningNow =
            physicsScore < config.warningPhysicsScore ||
                maxResidual >= config.warningBoneResidualMeters ||
                velocityViolationRatio > 0.06 ||
                accelerationViolationRatio > 0.05 ||
                jerkViolationRatio > 0.04

        val blockedNow =
            physicsScore < config.blockedPhysicsScore ||
                maxResidual >= config.maximumBoneResidualMeters * 1.8 ||
                velocityViolationRatio > 0.35 ||
                accelerationViolationRatio > 0.35

        updateHistory(
            warningNow = warningNow,
            blockedNow = blockedNow,
            score = physicsScore,
            confidence = confidenceScore,
        )

        val state = history!!
        val accepted = (
            state.blockedFrames < 3 &&
                physicsScore >= config.blockedPhysicsScore
            )

        val globalConfidence = (
            frame.globalConfidence * 0.72 +
                confidenceScore * 0.18 +
                physicsScore * 0.10
            ).coerceIn(0.0, 1.0)

        val outputFrame = frame.copy(
            joints = constrainedJoints.values
                .sortedBy { it.landmarkId },
            globalConfidence = globalConfidence,
        )

        updateHistoryFromOutput(outputFrame)

        val result = Result(
            frame = outputFrame,
            accepted = accepted,
            physicsScore = physicsScore,
            confidence = globalConfidence,
            meanBoneResidualMeters = meanResidual,
            p95BoneResidualMeters = p95Residual,
            maxBoneResidualMeters = maxResidual,
            velocityViolationRatio = velocityViolationRatio,
            accelerationViolationRatio = accelerationViolationRatio,
            jerkViolationRatio = jerkViolationRatio,
            symmetryScore = symmetryScore,
            floorContactScore = floorContactScore,
            correctedJointCount = jointDiagnostics.count { it.corrected },
            constrainedEdgeCount = edgeDiagnostics.size,
            edges = edgeDiagnostics,
            joints = jointDiagnostics,
            warnings = buildWarnings(
                physicsScore = physicsScore,
                meanResidual = meanResidual,
                maxResidual = maxResidual,
                velocityViolationRatio = velocityViolationRatio,
                accelerationViolationRatio = accelerationViolationRatio,
                jerkViolationRatio = jerkViolationRatio,
                symmetryScore = symmetryScore,
                floorContactScore = floorContactScore,
                history = state,
                edgeCount = edgeDiagnostics.size,
            ),
            recommendations = buildRecommendations(
                physicsScore = physicsScore,
                symmetryScore = symmetryScore,
                floorContactScore = floorContactScore,
                velocityViolationRatio = velocityViolationRatio,
                accelerationViolationRatio = accelerationViolationRatio,
                jerkViolationRatio = jerkViolationRatio,
                maxResidual = maxResidual,
            ),
        )
        lastResult = result
        return result
    }

    @Synchronized
    fun diagnostics(): Result? = lastResult

    @Synchronized
    fun reset() {
        baselines.clear()
        previousJoints.clear()
        previousVelocities.clear()
        previousTimestampNanos = 0L
        history = null
        lastResult = null
    }

    private fun targetLength(
        edge: Edge,
        joints: Map<Int, ReconstructionJoint>,
    ): Double? {
        edge.expectedLengthMeters?.let {
            return it
        }

        val parent = joints[edge.parent] ?: return null
        val child = joints[edge.child] ?: return null
        val observed = parent.position.distanceTo(child.position)

        if (!observed.isFinite() || observed <= 1e-6) {
            return null
        }

        val key = edgeKey(edge)
        val existing = baselines[key]

        val eligible =
            parent.confidence >= config.minimumJointConfidence &&
                child.confidence >= config.minimumJointConfidence &&
                parent.observationCount >= config.minimumObservationCountForLearning &&
                child.observationCount >= config.minimumObservationCountForLearning &&
                abs(observed - (existing?.lengthMeters ?: observed)) <=
                    config.maximumBaselineUpdateResidualMeters

        if (existing == null) {
            if (!eligible) {
                /*
                 * Keep an unqualified first observation as frame-local evidence only. A weak
                 * first frame must never become the metric definition of the rig.
                 */
                return observed
            }

            baselines[key] = Baseline(
                lengthMeters = observed,
                confidence = (
                    jointWeight(parent) *
                        jointWeight(child)
                    ).coerceIn(0.0, 1.0),
                samples = 1,
            )
            return observed
        }

        if (eligible) {
            val alpha = (
                config.baselineLearningRate *
                    (
                        jointWeight(parent) +
                            jointWeight(child)
                    ) * 0.5
                ).coerceIn(0.0, 1.0)
            existing.lengthMeters =
                existing.lengthMeters * (1.0 - alpha) +
                    observed * alpha
            existing.confidence = (
                existing.confidence * 0.96 +
                    (
                        jointWeight(parent) *
                            jointWeight(child)
                        ) * 0.04
                ).coerceIn(0.0, 1.0)
            existing.samples += 1
        }

        return existing.lengthMeters
    }

    private fun edgeResidualForJoint(
        landmarkId: Int,
        working: Map<Int, ReconstructionJoint>,
        targets: Map<Edge, Double?>,
    ): Double {
        val residuals = activeEdges
            .asSequence()
            .filter {
                it.parent == landmarkId ||
                    it.child == landmarkId
            }
            .mapNotNull { edge ->
                val target = targets[edge]
                val parent = working[edge.parent]
                val child = working[edge.child]
                if (target == null || parent == null || child == null) {
                    null
                } else {
                    abs(
                        parent.position.distanceTo(child.position) -
                            target,
                    )
                }
            }
            .toList()

        return residuals.averageOrZero()
    }

    private fun jointWeight(joint: ReconstructionJoint): Double {
        val uncertaintyWeight = if (
            joint.positionUncertaintyMeters.isFinite() &&
            joint.positionUncertaintyMeters > 0.0
        ) {
            1.0 / (
                1.0 +
                    joint.positionUncertaintyMeters / 0.08
            )
        } else {
            0.55
        }

        return (
            joint.confidence.coerceIn(0.0, 1.0) * 0.78 +
                uncertaintyWeight.coerceIn(0.0, 1.0) * 0.22
            ).coerceIn(0.05, 1.0)
    }

    private fun violationPenalty(
        value: Double,
        warning: Double,
        maximum: Double,
    ): Double {
        if (!value.isFinite() || value <= warning) return 0.0
        return (
            (value - warning) /
                (maximum - warning).coerceAtLeast(1e-6)
            ).coerceIn(0.0, 1.0)
    }

    private fun violationRatio(
        values: List<Double>,
        threshold: Double,
    ): Double {
        if (values.isEmpty()) return 0.0
        return values.count { it > threshold }.toDouble() /
            values.size.toDouble()
    }

    private fun symmetryScore(
        joints: Map<Int, ReconstructionJoint>,
    ): Double {
        val segmentPairs = listOf(
            Triple(11 to 13, 12 to 14, "upper-arm"),
            Triple(13 to 15, 14 to 16, "forearm"),
            Triple(23 to 25, 24 to 26, "thigh"),
            Triple(25 to 27, 26 to 28, "shin"),
            Triple(27 to 31, 28 to 32, "foot"),
        )

        val scores = segmentPairs.mapNotNull { (leftEdge, rightEdge, _) ->
            val leftParent = joints[leftEdge.first] ?: return@mapNotNull null
            val leftChild = joints[leftEdge.second] ?: return@mapNotNull null
            val rightParent = joints[rightEdge.first] ?: return@mapNotNull null
            val rightChild = joints[rightEdge.second] ?: return@mapNotNull null

            val leftLength = leftParent.position.distanceTo(leftChild.position)
            val rightLength = rightParent.position.distanceTo(rightChild.position)
            val reference = max(
                max(leftLength, rightLength),
                1e-4,
            )
            val relativeDifference = abs(leftLength - rightLength) / reference
            (
                1.0 -
                    relativeDifference /
                        config.symmetryToleranceRatio.coerceAtLeast(1e-6)
                ).coerceIn(0.0, 1.0)
        }

        if (scores.isEmpty()) return 0.78
        return scores.average().coerceIn(0.0, 1.0)
    }

    private fun floorContactScore(
        joints: Map<Int, ReconstructionJoint>,
    ): Double {
        val candidateIds = listOf(
            27,
            28,
            29,
            30,
            31,
            32,
        )

        val feet = candidateIds
            .mapNotNull(joints::get)
            .filter {
                jointWeight(it) >= config.minimumFloorConfidence
            }

        if (feet.isEmpty()) return 0.72

        val ys = feet.map { it.position.y }.sorted()
        val floor = percentile(
            values = ys,
            fraction = config.floorQuantile,
        )

        val nearFloor = feet.count {
            abs(it.position.y - floor) <=
                config.contactHeightToleranceMeters
        }

        val contactVelocity = feet.map { joint ->
            abs(
                joint.velocity.y,
            ) <= config.contactVerticalVelocityMetersPerSecond
        }.count { it }

        val contactScore = (
            nearFloor.toDouble() /
                feet.size.toDouble() * 0.58 +
                contactVelocity.toDouble() /
                feet.size.toDouble() * 0.42
        ).coerceIn(0.0, 1.0)

        return contactScore
    }

    private fun updateHistory(
        warningNow: Boolean,
        blockedNow: Boolean,
        score: Double,
        confidence: Double,
    ) {
        val current = history ?: History(
            score = score,
            confidence = confidence,
            warningFrames = 0,
            blockedFrames = 0,
            recoveryFrames = 0,
        )

        val alpha = config.historyAlpha
        current.score =
            current.score * (1.0 - alpha) +
                score * alpha
        current.confidence =
            current.confidence * (1.0 - alpha) +
                confidence * alpha

        current.warningFrames = if (warningNow) {
            current.warningFrames + 1
        } else {
            max(0, current.warningFrames - 1)
        }

        current.blockedFrames = if (blockedNow) {
            current.blockedFrames + 1
        } else {
            max(0, current.blockedFrames - 1)
        }

        current.recoveryFrames = if (
            !warningNow &&
                !blockedNow
        ) {
            current.recoveryFrames + 1
        } else {
            0
        }

        history = current
    }

    private fun updateHistoryFromOutput(
        frame: SkeletonFrame,
    ) {
        previousJoints.clear()
        previousVelocities.clear()
        frame.joints.forEach { joint ->
            previousJoints[joint.landmarkId] = joint
            previousVelocities[joint.landmarkId] =
                if (joint.velocity.norm() > 1e-6) {
                    joint.velocity
                } else {
                    Vec3(0.0, 0.0, 0.0)
                }
        }
        previousTimestampNanos = frame.timestampNanos
    }

    private fun buildWarnings(
        physicsScore: Double,
        meanResidual: Double,
        maxResidual: Double,
        velocityViolationRatio: Double,
        accelerationViolationRatio: Double,
        jerkViolationRatio: Double,
        symmetryScore: Double,
        floorContactScore: Double,
        history: History,
        edgeCount: Int,
    ): List<String> = buildList {
        if (edgeCount == 0) {
            add("physics-guard: no configured edges were observable")
        }
        if (physicsScore < config.warningPhysicsScore) {
            add("physics-guard: kinematic consistency is degraded")
        }
        if (physicsScore < config.blockedPhysicsScore) {
            add("physics-guard: metric output should be held")
        }
        if (meanResidual >= config.warningBoneResidualMeters) {
            add("physics-guard: mean bone-length residual elevated")
        }
        if (maxResidual >= config.maximumBoneResidualMeters) {
            add("physics-guard: maximum bone residual exceeds production envelope")
        }
        if (velocityViolationRatio > 0.06) {
            add("physics-guard: velocity envelope violations are persistent")
        }
        if (accelerationViolationRatio > 0.05) {
            add("physics-guard: acceleration envelope violations are persistent")
        }
        if (jerkViolationRatio > 0.04) {
            add("physics-guard: jerk envelope violations are persistent")
        }
        if (symmetryScore < 0.58) {
            add("physics-guard: bilateral limb geometry is inconsistent")
        }
        if (floorContactScore < 0.45) {
            add("physics-guard: lower-body support evidence is unstable")
        }
        if (history.warningFrames >= 4) {
            add("physics-guard: degradation is persistent")
        }
        if (history.blockedFrames >= 3) {
            add("physics-guard: production block is persistent")
        }
    }.distinct()

    private fun buildRecommendations(
        physicsScore: Double,
        symmetryScore: Double,
        floorContactScore: Double,
        velocityViolationRatio: Double,
        accelerationViolationRatio: Double,
        jerkViolationRatio: Double,
        maxResidual: Double,
    ): List<String> = buildList {
        if (maxResidual >= config.warningBoneResidualMeters) {
            add("revalidate camera geometry before accepting a long metric segment")
        }
        if (velocityViolationRatio > 0.06) {
            add("increase temporal evidence and reject isolated pose jumps instead of smoothing them into the track")
        }
        if (accelerationViolationRatio > 0.05 || jerkViolationRatio > 0.04) {
            add("prioritize high-dynamics refinement around fast joints and retain per-joint uncertainty")
        }
        if (symmetryScore < 0.58) {
            add("inspect left/right association and occlusion before applying aggressive retargeting")
        }
        if (floorContactScore < 0.45) {
            add("increase lower-body view coverage and avoid assuming a fixed floor plane")
        }
        if (physicsScore < config.blockedPhysicsScore) {
            add("hold metric output until kinematic consistency returns above the production floor")
        }
    }.distinct()

    private fun blocked(
        frame: SkeletonFrame,
        warning: String,
    ): Result {
        val result = Result(
            frame = frame,
            accepted = false,
            physicsScore = 0.0,
            confidence = 0.0,
            meanBoneResidualMeters = Double.POSITIVE_INFINITY,
            p95BoneResidualMeters = Double.POSITIVE_INFINITY,
            maxBoneResidualMeters = Double.POSITIVE_INFINITY,
            velocityViolationRatio = 1.0,
            accelerationViolationRatio = 1.0,
            jerkViolationRatio = 1.0,
            symmetryScore = 0.0,
            floorContactScore = 0.0,
            correctedJointCount = 0,
            constrainedEdgeCount = 0,
            edges = emptyList(),
            joints = emptyList(),
            warnings = listOf(warning),
            recommendations = listOf(
                "hold metric output until reconstructed joints are available",
            ),
        )
        lastResult = result
        return result
    }

    private fun edgeKey(edge: Edge): String =
        edge.parent.toString() +
            ">" +
            edge.child.toString() +
            ":" +
            edge.group

    private fun percentile(
        values: List<Double>,
        fraction: Double,
    ): Double {
        if (values.isEmpty()) return 0.0
        if (values.size == 1) return values.first()
        val position =
            fraction.coerceIn(0.0, 1.0) *
                (values.size - 1)
        val lower = position.toInt()
        val upper = min(
            lower + 1,
            values.lastIndex,
        )
        val blend = position - lower
        return values[lower] * (1.0 - blend) +
            values[upper] * blend
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    companion object {
        fun defaultEdges(): List<Edge> = listOf(
            Edge(11, 12, "shoulder", bilateralKey = "shoulder"),
            Edge(11, 13, "upper-arm-left", bilateralKey = "upper-arm"),
            Edge(13, 15, "forearm-left", bilateralKey = "forearm"),
            Edge(12, 14, "upper-arm-right", bilateralKey = "upper-arm"),
            Edge(14, 16, "forearm-right", bilateralKey = "forearm"),
            Edge(11, 23, "torso-left", bilateralKey = "torso"),
            Edge(12, 24, "torso-right", bilateralKey = "torso"),
            Edge(23, 24, "pelvis", bilateralKey = "pelvis"),
            Edge(23, 25, "thigh-left", bilateralKey = "thigh"),
            Edge(25, 27, "shin-left", bilateralKey = "shin"),
            Edge(24, 26, "thigh-right", bilateralKey = "thigh"),
            Edge(26, 28, "shin-right", bilateralKey = "shin"),
            Edge(27, 29, "heel-left", bilateralKey = "foot"),
            Edge(28, 30, "heel-right", bilateralKey = "foot"),
            Edge(29, 31, "foot-left", bilateralKey = "foot"),
            Edge(30, 32, "foot-right", bilateralKey = "foot"),
        )
    }
}

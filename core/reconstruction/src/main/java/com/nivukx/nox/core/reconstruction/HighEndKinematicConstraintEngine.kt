package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Deterministic kinematic safety stage for reconstructed motion.
 *
 * It operates on observed 3D joints and enforces production constraints without fabricating
 * unseen geometry. The stage is intentionally model/provider neutral and preserves high-frequency
 * motion unless a constraint is materially violated.
 */
class HighEndKinematicConstraintEngine(
    private var config: Config = Config(),
) {
    data class Config(
        val iterations: Int = 7,
        val minimumConfidence: Double = 0.12,
        val lengthToleranceMeters: Double = 0.004,
        val hardLengthToleranceMeters: Double = 0.085,
        val correctionStrength: Double = 0.78,
        val maximumCorrectionMeters: Double = 0.045,
        val rootMotionLock: Double = 0.78,
        val scalePreservation: Double = 0.94,
        val footContactStrictness: Double = 0.80,
        val symmetryStrength: Double = 0.64,
        val posePriorStrength: Double = 0.84,
        val microMotionPreservation: Double = 0.88,
        val maximumVelocityMetersPerSecond: Double = 12.0,
        val maximumAccelerationMetersPerSecond2: Double = 60.0,
        val maximumJerkMetersPerSecond3: Double = 300.0,
        val minimumDtSeconds: Double = 1.0 / 240.0,
        val maximumDtSeconds: Double = 0.20,
        val rejectionFloor: Double = 0.25,
    ) {
        init {
            require(iterations in 1..20)
            require(minimumConfidence in 0.0..1.0)
            require(lengthToleranceMeters > 0.0)
            require(hardLengthToleranceMeters > lengthToleranceMeters)
            require(correctionStrength in 0.0..1.0)
            require(maximumCorrectionMeters > 0.0)
            require(rootMotionLock in 0.0..1.0)
            require(scalePreservation in 0.0..1.0)
            require(footContactStrictness in 0.0..1.0)
            require(symmetryStrength in 0.0..1.0)
            require(posePriorStrength in 0.0..1.0)
            require(microMotionPreservation in 0.0..1.0)
            require(maximumVelocityMetersPerSecond > 0.0)
            require(maximumAccelerationMetersPerSecond2 > 0.0)
            require(maximumJerkMetersPerSecond3 > 0.0)
            require(minimumDtSeconds > 0.0)
            require(maximumDtSeconds >= minimumDtSeconds)
            require(rejectionFloor in 0.0..1.0)
        }
    }

    data class BoneConstraint(
        val a: Int,
        val b: Int,
        val preferredLengthMeters: Double,
        val minimumLengthMeters: Double,
        val maximumLengthMeters: Double,
        val stiffness: Double = 0.82,
        val semantic: String = "bone",
    ) {
        init {
            require(a != b)
            require(preferredLengthMeters > 0.0)
            require(minimumLengthMeters > 0.0)
            require(maximumLengthMeters >= minimumLengthMeters)
            require(preferredLengthMeters in minimumLengthMeters..maximumLengthMeters)
            require(stiffness in 0.0..1.0)
        }

        companion object {
            fun relaxed(
                a: Int,
                b: Int,
                preferredLengthMeters: Double,
                semantic: String = "bone",
            ): BoneConstraint =
                BoneConstraint(
                    a = a,
                    b = b,
                    preferredLengthMeters = preferredLengthMeters,
                    minimumLengthMeters = preferredLengthMeters * 0.72,
                    maximumLengthMeters = preferredLengthMeters * 1.28,
                    semantic = semantic,
                )
        }
    }

    data class SymmetryConstraint(
        val left: Int,
        val right: Int,
        val root: Int? = null,
        val axis: Axis = Axis.X,
        val stiffness: Double = 0.64,
    ) {
        enum class Axis { X, Y, Z }
        init {
            require(left != right)
            require(stiffness in 0.0..1.0)
        }
    }

    data class FootContact(
        val jointId: Int,
        val floorHeightMeters: Double = 0.0,
        val lockStrength: Double = 0.80,
        val allowedVerticalDriftMeters: Double = 0.018,
        val velocityThresholdMetersPerSecond: Double = 0.065,
    ) {
        init {
            require(lockStrength in 0.0..1.0)
            require(allowedVerticalDriftMeters >= 0.0)
            require(velocityThresholdMetersPerSecond >= 0.0)
        }
    }

    data class Input(
        val skeleton: SkeletonFrame,
        val boneConstraints: List<BoneConstraint> = emptyList(),
        val symmetryConstraints: List<SymmetryConstraint> = emptyList(),
        val footContacts: List<FootContact> = emptyList(),
        val rootJointId: Int? = null,
        val captureFloorHeightMeters: Double? = null,
        val preserveMicroMotion: Boolean = true,
        val floorLocked: Boolean = false,
        val confidenceHint: Double = 1.0,
        val cameraMotionRisk: Double = 0.0,
        val occlusionRisk: Double = 0.0,
        val synchronizationConfidence: Double = 1.0,
    )

    data class Diagnostics(
        val iterations: Int,
        val constrainedJointCount: Int,
        val activeConstraintCount: Int,
        val violatedConstraintCount: Int,
        val hardViolationCount: Int,
        val symmetryCorrectionCount: Int,
        val footCorrectionCount: Int,
        val rootCorrectionCount: Int,
        val velocityViolations: Int,
        val accelerationViolations: Int,
        val jerkViolations: Int,
        val meanLengthResidualMeters: Double,
        val p95LengthResidualMeters: Double,
        val meanCorrectionMeters: Double,
        val maximumCorrectionMeters: Double,
        val temporalScore: Double,
        val geometricScore: Double,
        val contactScore: Double,
        val finalScore: Double,
        val accepted: Boolean,
        val bottleneck: String,
        val warnings: List<String>,
    )

    data class Result(
        val skeleton: SkeletonFrame,
        val diagnostics: Diagnostics,
        val score: Double,
        val accepted: Boolean,
    )

    private data class StateJoint(
        val id: Int,
        var position: Vec3,
        val rawPosition: Vec3,
        val confidence: Double,
        val uncertaintyMeters: Double,
        val velocity: Vec3,
        val acceleration: Vec3,
    )

    private data class History(
        val timestampNanos: Long,
        val positions: Map<Int, Vec3>,
        val velocities: Map<Int, Vec3>,
    )

    private data class TemporalMetrics(
        val score: Double,
        val velocityViolations: Int,
        val accelerationViolations: Int,
        val jerkViolations: Int,
        val velocityResidual: Double,
        val accelerationResidual: Double,
        val jerkResidual: Double,
    )

    private var previous: History? = null
    private var previousPrevious: History? = null

    @Synchronized
    fun configure(value: Config) {
        config = value
    }

    @Synchronized
    fun configuration(): Config = config

    @Synchronized
    fun solve(input: Input): Result {
        val safe = sanitize(input.skeleton)
        if (safe.joints.isEmpty()) {
            return emptyResult(safe, "EMPTY_SKELETON")
        }

        val previousHistory = previous
        val states = safe.joints.associate { joint ->
            val acceleration =
                if (
                    previousHistory != null &&
                    safe.timestampNanos > previousHistory.timestampNanos
                ) {
                    val dt =
                        (
                            safe.timestampNanos -
                                previousHistory.timestampNanos
                            ).toDouble() / 1e9
                    val boundedDt =
                        dt.coerceIn(
                            config.minimumDtSeconds,
                            config.maximumDtSeconds,
                        )
                    previousHistory.velocities[joint.landmarkId]?.let { previousVelocity ->
                        finite(
                            (finite(joint.velocity) - previousVelocity) /
                                boundedDt.coerceAtLeast(config.minimumDtSeconds),
                        )
                    } ?: Vec3(0.0, 0.0, 0.0)
                } else {
                    Vec3(0.0, 0.0, 0.0)
                }

            joint.landmarkId to StateJoint(
                id = joint.landmarkId,
                position = joint.position,
                rawPosition = joint.position,
                confidence = joint.confidence.coerceIn(0.0, 1.0),
                uncertaintyMeters = joint.positionUncertaintyMeters
                    .coerceAtLeast(0.0)
                    .takeIf(Double::isFinite)
                    ?: 1.0,
                velocity = finite(joint.velocity),
                acceleration = acceleration,
            )
        }.toMutableMap()

        val bones = resolveBones(input, states)
        val symmetry = input.symmetryConstraints.filter { it.left in states && it.right in states }
        val feet = input.footContacts.filter { it.jointId in states }
        val rootId = input.rootJointId?.takeIf { it in states } ?: selectRoot(safe)
        val rootStart = rootId?.let { states[it]?.position }

        val residuals = mutableListOf<Double>()
        val correctionMagnitudes = mutableListOf<Double>()
        val warnings = mutableListOf<String>()

        var iterationsUsed = 0
        var violated = 0
        var hardViolations = 0
        var symmetryCorrections = 0
        var footCorrections = 0
        var rootCorrections = 0

        repeat(config.iterations) { pass ->
            iterationsUsed = pass + 1
            var maximumStep = 0.0

            for (bone in bones) {
                val a = states[bone.a] ?: continue
                val b = states[bone.b] ?: continue
                val delta = sub(b.position, a.position)
                val current = magnitude(delta)

                if (!current.isFinite() || current < 1e-8) {
                    violated += 1
                    hardViolations += 1
                    warnings += "degenerate-bone-" + bone.semantic
                    continue
                }

                val target = when {
                    current < bone.minimumLengthMeters -> bone.minimumLengthMeters
                    current > bone.maximumLengthMeters -> bone.maximumLengthMeters
                    else -> bone.preferredLengthMeters
                }

                val residual = abs(current - target)
                residuals += residual
                if (residual <= config.lengthToleranceMeters) continue

                violated += 1
                if (residual >= config.hardLengthToleranceMeters) {
                    hardViolations += 1
                }

                val strength = (
                    config.correctionStrength *
                        bone.stiffness *
                        config.posePriorStrength *
                        confidenceWeight(a, b)
                ).coerceIn(0.0, 1.0)

                val scalar = (
                    (target - current) * strength
                ).coerceIn(
                    -config.maximumCorrectionMeters,
                    config.maximumCorrectionMeters,
                )

                val direction = normalize(delta)
                val confidenceTotal = (
                    a.confidence + b.confidence
                ).coerceAtLeast(1e-6)

                val aShare = b.confidence / confidenceTotal
                val bShare = a.confidence / confidenceTotal
                val aStep = scale(direction, scalar * aShare * 0.5)
                val bStep = scale(direction, -scalar * bShare * 0.5)

                a.position = add(a.position, aStep)
                b.position = add(b.position, bStep)

                val amountA = magnitude(aStep)
                val amountB = magnitude(bStep)
                correctionMagnitudes += amountA
                correctionMagnitudes += amountB
                maximumStep = max(maximumStep, max(amountA, amountB))
            }

            for (rule in symmetry) {
                val left = states[rule.left] ?: continue
                val right = states[rule.right] ?: continue
                val pivot =
                    rule.root?.let { states[it]?.position }
                        ?: rootId?.let { states[it]?.position }
                        ?: midpoint(left.position, right.position)

                val reflected = reflect(
                    sub(left.position, pivot),
                    rule.axis,
                )
                val error = sub(
                    sub(right.position, pivot),
                    reflected,
                )
                val magnitudeError = magnitude(error)

                if (magnitudeError <= config.lengthToleranceMeters) continue

                val strength = (
                    config.symmetryStrength *
                        rule.stiffness *
                        confidenceWeight(left, right)
                ).coerceIn(0.0, 1.0)

                val step = clamp(
                    scale(error, strength * 0.5),
                    config.maximumCorrectionMeters,
                )

                right.position = sub(right.position, step)
                left.position = add(
                    left.position,
                    reflect(step, rule.axis),
                )

                val amount = magnitude(step)
                correctionMagnitudes += amount
                correctionMagnitudes += amount
                maximumStep = max(maximumStep, amount)
                symmetryCorrections += 1
            }

            for (foot in feet) {
                val joint = states[foot.jointId] ?: continue
                val speed = magnitude(joint.velocity)
                if (speed > foot.velocityThresholdMetersPerSecond &&
                    joint.confidence < 0.92
                ) {
                    continue
                }

                val floor = input.captureFloorHeightMeters ?: foot.floorHeightMeters
                val deltaY = floor - joint.position.y
                if (abs(deltaY) <= foot.allowedVerticalDriftMeters) continue

                val strength = (
                    config.footContactStrictness *
                        foot.lockStrength *
                        if (input.floorLocked) 1.0 else 0.72
                ).coerceIn(0.0, 1.0)

                val stepY = (
                    deltaY * strength
                ).coerceIn(
                    -config.maximumCorrectionMeters,
                    config.maximumCorrectionMeters,
                )

                joint.position = add(
                    joint.position,
                    Vec3(0.0, stepY, 0.0),
                )
                val amount = abs(stepY)
                correctionMagnitudes += amount
                maximumStep = max(maximumStep, amount)
                footCorrections += 1
            }

            applyScalePreservation(
                states = states,
                bones = bones,
                rootId = rootId,
                correctionMagnitudes = correctionMagnitudes,
            )

            if (rootId != null && rootStart != null && config.rootMotionLock > 0.0) {
                val root = states[rootId]
                if (root != null) {
                    val drift = sub(root.position, rootStart)
                    val step = clamp(
                        scale(
                            drift,
                            -config.rootMotionLock * 0.35,
                        ),
                        config.maximumCorrectionMeters,
                    )
                    root.position = add(root.position, step)
                    val amount = magnitude(step)
                    if (amount > 1e-6) rootCorrections += 1
                    correctionMagnitudes += amount
                    maximumStep = max(maximumStep, amount)
                }
            }

            applyMicroMotionProtection(
                states = states,
                preserve = input.preserveMicroMotion,
                correctionMagnitudes = correctionMagnitudes,
            )

            if (maximumStep < config.lengthToleranceMeters * 0.12) {
                return@repeat
            }
        }

        val temporal = evaluateTemporal(
            timestampNanos = safe.timestampNanos,
            states = states,
            previous = previous,
            previousPrevious = previousPrevious,
        )

        val finalJoints = safe.joints.map { joint ->
            val state = states[joint.landmarkId] ?: return@map joint
            val delta = sub(state.position, joint.position)
            joint.copy(
                position = add(
                    joint.position,
                    clamp(
                        delta,
                        config.maximumCorrectionMeters,
                    ),
                ),
            )
        }

        val geometricScore = geometricScore(
            residuals = residuals,
            hardViolations = hardViolations,
            meanCorrection = correctionMagnitudes.averageOrZero(),
        )

        val contactScore =
            if (feet.isEmpty()) 1.0
            else (
                1.0 -
                    footCorrections.toDouble() /
                    max(1, feet.size).toDouble() *
                    0.18
            ).coerceIn(0.0, 1.0)

        val globalConfidence = globalConfidence(
            original = safe.globalConfidence,
            temporalScore = temporal.score,
            residuals = residuals,
            meanCorrection = correctionMagnitudes.averageOrZero(),
            hardViolations = hardViolations,
            hint = input.confidenceHint,
            cameraMotionRisk = input.cameraMotionRisk,
            occlusionRisk = input.occlusionRisk,
            synchronizationConfidence = input.synchronizationConfidence,
        )

        val constrained = safe.copy(
            joints = finalJoints,
            globalConfidence = globalConfidence,
        )

        if (hardViolations > 0) warnings += "hard-kinematic-violation"
        if (temporal.velocityViolations > 0) warnings += "velocity-limit"
        if (temporal.accelerationViolations > 0) warnings += "acceleration-limit"
        if (temporal.jerkViolations > 0) warnings += "jerk-limit"
        if (input.cameraMotionRisk > 0.70) warnings += "camera-motion-risk"
        if (input.occlusionRisk > 0.70) warnings += "occlusion-risk"
        if (input.synchronizationConfidence < 0.50) warnings += "sync-confidence-low"

        val finalScore = (
            0.40 * geometricScore +
                0.32 * temporal.score +
                0.16 * contactScore +
                0.12 * globalConfidence
        ).coerceIn(0.0, 1.0)

        val accepted =
            finalScore >= config.rejectionFloor &&
                hardViolations <= max(1, bones.size / 8) &&
                globalConfidence >= 0.18

        val bottleneck = when {
            hardViolations > 0 -> "BONE_CONSTRAINTS"
            temporal.jerkViolations > 0 -> "JERK"
            temporal.accelerationViolations > 0 -> "ACCELERATION"
            temporal.velocityViolations > 0 -> "VELOCITY"
            contactScore < geometricScore -> "FOOT_CONTACT"
            geometricScore < temporal.score -> "GEOMETRY"
            else -> "TEMPORAL"
        }

        val diagnostics = Diagnostics(
            iterations = iterationsUsed,
            constrainedJointCount = countChanged(safe.joints, finalJoints),
            activeConstraintCount = bones.size,
            violatedConstraintCount = violated,
            hardViolationCount = hardViolations,
            symmetryCorrectionCount = symmetryCorrections,
            footCorrectionCount = footCorrections,
            rootCorrectionCount = rootCorrections,
            velocityViolations = temporal.velocityViolations,
            accelerationViolations = temporal.accelerationViolations,
            jerkViolations = temporal.jerkViolations,
            meanLengthResidualMeters = residuals.averageOrZero(),
            p95LengthResidualMeters = percentile(residuals, 0.95),
            meanCorrectionMeters = correctionMagnitudes.averageOrZero(),
            maximumCorrectionMeters = correctionMagnitudes.maxOrNull() ?: 0.0,
            temporalScore = temporal.score,
            geometricScore = geometricScore,
            contactScore = contactScore,
            finalScore = finalScore,
            accepted = accepted,
            bottleneck = bottleneck,
            warnings = warnings.distinct(),
        )

        previousPrevious = previous
        previous = History(
            timestampNanos = safe.timestampNanos,
            positions = states.mapValues { it.value.position },
            velocities = states.mapValues { finite(it.value.velocity) },
        )

        return Result(
            skeleton = constrained,
            diagnostics = diagnostics,
            score = finalScore,
            accepted = accepted,
        )
    }

    @Synchronized
    fun reset() {
        previous = null
        previousPrevious = null
    }

    private fun resolveBones(
        input: Input,
        states: Map<Int, StateJoint>,
    ): List<BoneConstraint> {
        val explicit = input.boneConstraints.filter { it.a in states && it.b in states }
        if (explicit.isNotEmpty()) return explicit

        val history = previous ?: return emptyList()
        return DEFAULT_HUMAN_EDGES.mapNotNull { edge ->
            val currentA = states[edge.first] ?: return@mapNotNull null
            val currentB = states[edge.second] ?: return@mapNotNull null
            val previousA = history.positions[edge.first] ?: return@mapNotNull null
            val previousB = history.positions[edge.second] ?: return@mapNotNull null
            val referenceLength = distance(previousA, previousB)
            if (!referenceLength.isFinite() || referenceLength < 0.015) {
                return@mapNotNull null
            }
            BoneConstraint(
                a = currentA.id,
                b = currentB.id,
                preferredLengthMeters = referenceLength,
                minimumLengthMeters = referenceLength * 0.76,
                maximumLengthMeters = referenceLength * 1.24,
                stiffness = 0.66,
                semantic = "auto-human-reference",
            )
        }
    }

    companion object {
        private val DEFAULT_HUMAN_EDGES = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 4,
            0 to 5, 5 to 6, 6 to 7, 7 to 8,
            9 to 10, 11 to 12,
            11 to 13, 13 to 15, 15 to 17, 15 to 19, 15 to 21,
            12 to 14, 14 to 16, 16 to 18, 16 to 20, 16 to 22,
            11 to 23, 23 to 25, 25 to 27, 27 to 29, 29 to 31,
            12 to 24, 24 to 26, 26 to 28, 28 to 30, 30 to 32,
            23 to 24,
        )
    }

    private fun applyScalePreservation(
        states: MutableMap<Int, StateJoint>,
        bones: List<BoneConstraint>,
        rootId: Int?,
        correctionMagnitudes: MutableList<Double>,
    ) {
        if (config.scalePreservation <= 0.0 || bones.isEmpty()) return

        val target = median(
            bones.map { it.preferredLengthMeters },
        )
        val current = median(
            bones.mapNotNull { bone ->
                val a = states[bone.a]?.position ?: return@mapNotNull null
                val b = states[bone.b]?.position ?: return@mapNotNull null
                magnitude(sub(a, b))
            },
        )

        if (target <= 1e-6 || current <= 1e-6) return
        if (abs(target - current) <= config.lengthToleranceMeters) return

        val pivot = rootId?.let { states[it]?.position }
            ?: centroid(states.values)

        val factor = (
            1.0 +
                ((target / current) - 1.0) *
                config.scalePreservation *
                config.posePriorStrength *
                0.18
        ).coerceIn(0.94, 1.06)

        states.values.forEach { joint ->
            if (joint.id == rootId) return@forEach
            val next = add(
                pivot,
                scale(
                    sub(joint.position, pivot),
                    factor,
                ),
            )
            val step = clamp(
                sub(next, joint.position),
                config.maximumCorrectionMeters,
            )
            joint.position = add(joint.position, step)
            correctionMagnitudes += magnitude(step)
        }
    }

    private fun applyMicroMotionProtection(
        states: MutableMap<Int, StateJoint>,
        preserve: Boolean,
        correctionMagnitudes: MutableList<Double>,
    ) {
        if (!preserve) return
        val factor = (
            1.0 -
                config.microMotionPreservation * 0.22
        ).coerceIn(0.40, 1.0)

        states.values.forEach { joint ->
            val correction = sub(
                joint.position,
                joint.rawPosition,
            )
            if (magnitude(correction) <= 1e-8) return@forEach

            val uncertaintyScale = (
                1.0 /
                    (1.0 + joint.uncertaintyMeters * 2.6)
            ).coerceIn(0.18, 1.0)
            val confidenceScale = max(
                0.35,
                joint.confidence,
            )

            val limited = clamp(
                correction,
                config.maximumCorrectionMeters *
                    factor *
                    confidenceScale *
                    uncertaintyScale,
            )

            joint.position = add(
                joint.rawPosition,
                limited,
            )
            correctionMagnitudes += magnitude(limited)
        }
    }

    private fun evaluateTemporal(
        timestampNanos: Long,
        states: Map<Int, StateJoint>,
        previous: History?,
        previousPrevious: History?,
    ): TemporalMetrics {
        if (previous == null) {
            return TemporalMetrics(
                score = 1.0,
                velocityViolations = 0,
                accelerationViolations = 0,
                jerkViolations = 0,
                velocityResidual = 0.0,
                accelerationResidual = 0.0,
                jerkResidual = 0.0,
            )
        }

        val dt = (
            timestampNanos - previous.timestampNanos
        ).toDouble() / 1e9

        val safeDt = dt.coerceIn(
            config.minimumDtSeconds,
            config.maximumDtSeconds,
        )

        var velocityViolations = 0
        var accelerationViolations = 0
        var jerkViolations = 0

        val velocityResiduals = mutableListOf<Double>()
        val accelerationResiduals = mutableListOf<Double>()
        val jerkResiduals = mutableListOf<Double>()

        states.forEach { (id, state) ->
            val lastPosition = previous.positions[id] ?: return@forEach
            val measuredVelocity = scale(
                sub(state.position, lastPosition),
                1.0 / safeDt,
            )

            velocityResiduals += magnitude(
                sub(measuredVelocity, state.velocity),
            )

            if (magnitude(measuredVelocity) >
                config.maximumVelocityMetersPerSecond
            ) {
                velocityViolations += 1
                state.position = interpolate(
                    lastPosition,
                    state.position,
                    0.58,
                )
            }

            val lastVelocity = previous.velocities[id] ?: return@forEach
            val measuredAcceleration = scale(
                sub(measuredVelocity, lastVelocity),
                1.0 / safeDt,
            )

            accelerationResiduals += magnitude(
                sub(measuredAcceleration, state.acceleration),
            )

            if (magnitude(measuredAcceleration) >
                config.maximumAccelerationMetersPerSecond2
            ) {
                accelerationViolations += 1
                state.position = interpolate(
                    lastPosition,
                    state.position,
                    (
                        1.0 -
                            config.microMotionPreservation * 0.28
                    ).coerceIn(0.55, 0.94),
                )
            }

            val olderVelocity = previousPrevious
                ?.velocities
                ?.get(id)
                ?: return@forEach

            val olderDt = (
                previous.timestampNanos -
                    previousPrevious.timestampNanos
            ).toDouble() / 1e9
            val safeOlderDt = olderDt.coerceIn(
                config.minimumDtSeconds,
                config.maximumDtSeconds,
            )

            val olderAcceleration = scale(
                sub(lastVelocity, olderVelocity),
                1.0 / safeOlderDt,
            )
            val jerk = scale(
                sub(
                    measuredAcceleration,
                    olderAcceleration,
                ),
                1.0 / safeDt,
            )

            val jerkMagnitude = magnitude(jerk)
            jerkResiduals += jerkMagnitude

            if (jerkMagnitude >
                config.maximumJerkMetersPerSecond3
            ) {
                jerkViolations += 1
                state.position = interpolate(
                    lastPosition,
                    state.position,
                    (
                        1.0 -
                            config.microMotionPreservation * 0.20
                    ).coerceIn(0.62, 0.96),
                )
            }
        }

        val velocityResidual = velocityResiduals.averageOrZero()
        val accelerationResidual = accelerationResiduals.averageOrZero()
        val jerkResidual = jerkResiduals.averageOrZero()

        val velocityScore = (
            1.0 -
                velocityViolations.toDouble() /
                max(1, states.size).toDouble()
        ).coerceIn(0.0, 1.0)

        val accelerationScore = (
            1.0 -
                accelerationViolations.toDouble() /
                max(1, states.size).toDouble()
        ).coerceIn(0.0, 1.0)

        val jerkScore = (
            1.0 -
                jerkViolations.toDouble() /
                max(1, states.size).toDouble()
        ).coerceIn(0.0, 1.0)

        val residualScore = (
            1.0 -
                velocityResidual /
                config.maximumVelocityMetersPerSecond * 0.46 -
                accelerationResidual /
                config.maximumAccelerationMetersPerSecond2 * 0.32 -
                jerkResidual /
                config.maximumJerkMetersPerSecond3 * 0.22
        ).coerceIn(0.0, 1.0)

        return TemporalMetrics(
            score = (
                0.42 * velocityScore +
                    0.25 * accelerationScore +
                    0.13 * jerkScore +
                    0.20 * residualScore
            ).coerceIn(0.0, 1.0),
            velocityViolations = velocityViolations,
            accelerationViolations = accelerationViolations,
            jerkViolations = jerkViolations,
            velocityResidual = velocityResidual,
            accelerationResidual = accelerationResidual,
            jerkResidual = jerkResidual,
        )
    }

    private fun globalConfidence(
        original: Double,
        temporalScore: Double,
        residuals: List<Double>,
        meanCorrection: Double,
        hardViolations: Int,
        hint: Double,
        cameraMotionRisk: Double,
        occlusionRisk: Double,
        synchronizationConfidence: Double,
    ): Double {
        val residualPenalty = (
            residuals.averageOrZero() /
                config.hardLengthToleranceMeters
        ).coerceIn(0.0, 1.0) * 0.20

        val correctionPenalty = (
            meanCorrection /
                config.maximumCorrectionMeters
        ).coerceIn(0.0, 1.0) * 0.10

        val hardPenalty = (
            hardViolations.toDouble() /
            max(1, residuals.size).toDouble()
        ).coerceIn(0.0, 1.0) * 0.22

        val environmentPenalty =
            cameraMotionRisk.coerceIn(0.0, 1.0) * 0.12 +
                occlusionRisk.coerceIn(0.0, 1.0) * 0.10 +
                (
                    1.0 -
                        synchronizationConfidence.coerceIn(0.0, 1.0)
                ) * 0.08

        return (
            original.coerceIn(0.0, 1.0) * 0.48 +
                temporalScore.coerceIn(0.0, 1.0) * 0.22 +
                hint.coerceIn(0.0, 1.0) * 0.12 +
                config.posePriorStrength * 0.08 +
                config.scalePreservation * 0.10 -
                residualPenalty -
                correctionPenalty -
                hardPenalty -
                environmentPenalty
        ).coerceIn(0.0, 1.0)
    }

    private fun geometricScore(
        residuals: List<Double>,
        hardViolations: Int,
        meanCorrection: Double,
    ): Double {
        if (residuals.isEmpty()) return 0.92

        val residualScore = (
            1.0 -
                residuals.averageOrZero() /
                config.hardLengthToleranceMeters
        ).coerceIn(0.0, 1.0)

        val hardScore = (
            1.0 -
                hardViolations.toDouble() /
                max(1, residuals.size).toDouble()
        ).coerceIn(0.0, 1.0)

        val correctionScore = (
            1.0 -
                meanCorrection /
                config.maximumCorrectionMeters
        ).coerceIn(0.0, 1.0)

        return (
            0.48 * residualScore +
                0.28 * hardScore +
                0.24 * correctionScore
        ).coerceIn(0.0, 1.0)
    }

    private fun selectRoot(skeleton: SkeletonFrame): Int? {
        val confident = skeleton.joints.filter {
            it.confidence >= config.minimumConfidence
        }

        val pelvisPreference = listOf(23, 24, 11, 12)
            .firstOrNull { id ->
                confident.any { it.landmarkId == id }
            }

        if (pelvisPreference != null) {
            return pelvisPreference
        }

        val medianY = confident
            .map { it.position.y }
            .sorted()
            .let { values ->
                values.getOrNull(values.size / 2)
                    ?: 0.0
            }

        return confident
            .minByOrNull { joint ->
                abs(joint.position.y - medianY) +
                    0.12 * (
                        joint.position.x * joint.position.x +
                            joint.position.z * joint.position.z
                    )
            }
            ?.landmarkId
            ?: skeleton.joints.firstOrNull()?.landmarkId
    }

    private fun sanitize(skeleton: SkeletonFrame): SkeletonFrame =
        skeleton.copy(
            joints = skeleton.joints.map { joint ->
                joint.copy(
                    position = finite(joint.position),
                    velocity = finite(joint.velocity),
                    acceleration = finite(joint.acceleration),
                    confidence = joint.confidence.coerceIn(0.0, 1.0),
                    positionUncertaintyMeters =
                        if (joint.positionUncertaintyMeters.isFinite()) {
                            joint.positionUncertaintyMeters.coerceAtLeast(0.0)
                        } else {
                            1.0
                        },
                )
            },
            globalConfidence =
                if (skeleton.globalConfidence.isFinite()) {
                    skeleton.globalConfidence.coerceIn(0.0, 1.0)
                } else {
                    0.0
                },
        )

    private fun countChanged(
        original: List<ReconstructionJoint>,
        result: List<ReconstructionJoint>,
    ): Int {
        val old = original.associateBy { it.landmarkId }
        return result.count { current ->
            val previous = old[current.landmarkId] ?: return@count false
            distance(previous.position, current.position) > 1e-6
        }
    }

    private fun emptyResult(
        skeleton: SkeletonFrame,
        bottleneck: String,
    ): Result {
        val diagnostics = Diagnostics(
            iterations = 0,
            constrainedJointCount = 0,
            activeConstraintCount = 0,
            violatedConstraintCount = 0,
            hardViolationCount = 0,
            symmetryCorrectionCount = 0,
            footCorrectionCount = 0,
            rootCorrectionCount = 0,
            velocityViolations = 0,
            accelerationViolations = 0,
            jerkViolations = 0,
            meanLengthResidualMeters = 0.0,
            p95LengthResidualMeters = 0.0,
            meanCorrectionMeters = 0.0,
            maximumCorrectionMeters = 0.0,
            temporalScore = 0.0,
            geometricScore = 0.0,
            contactScore = 0.0,
            finalScore = 0.0,
            accepted = false,
            bottleneck = bottleneck,
            warnings = listOf(bottleneck),
        )
        return Result(
            skeleton = skeleton,
            diagnostics = diagnostics,
            score = 0.0,
            accepted = false,
        )
    }

    private fun confidenceWeight(
        a: StateJoint,
        b: StateJoint,
    ): Double = (
        a.confidence * 0.55 +
            b.confidence * 0.45
    ).coerceIn(0.10, 1.0)

    private fun add(a: Vec3, b: Vec3): Vec3 =
        Vec3(a.x + b.x, a.y + b.y, a.z + b.z)

    private fun sub(a: Vec3, b: Vec3): Vec3 =
        Vec3(a.x - b.x, a.y - b.y, a.z - b.z)

    private fun scale(v: Vec3, factor: Double): Vec3 =
        Vec3(v.x * factor, v.y * factor, v.z * factor)

    private fun midpoint(a: Vec3, b: Vec3): Vec3 =
        Vec3(
            (a.x + b.x) * 0.5,
            (a.y + b.y) * 0.5,
            (a.z + b.z) * 0.5,
        )

    private fun centroid(states: Collection<StateJoint>): Vec3 {
        if (states.isEmpty()) return Vec3(0.0, 0.0, 0.0)
        val x = states.sumOf { it.position.x }
        val y = states.sumOf { it.position.y }
        val z = states.sumOf { it.position.z }
        val count = states.size.toDouble()
        return Vec3(x / count, y / count, z / count)
    }

    private fun reflect(
        value: Vec3,
        axis: SymmetryConstraint.Axis,
    ): Vec3 = when (axis) {
        SymmetryConstraint.Axis.X -> Vec3(-value.x, value.y, value.z)
        SymmetryConstraint.Axis.Y -> Vec3(value.x, -value.y, value.z)
        SymmetryConstraint.Axis.Z -> Vec3(value.x, value.y, -value.z)
    }

    private fun normalize(value: Vec3): Vec3 {
        val m = magnitude(value)
        return if (m < 1e-10 || !m.isFinite()) {
            Vec3(0.0, 0.0, 0.0)
        } else {
            scale(value, 1.0 / m)
        }
    }

    private fun clamp(value: Vec3, maximum: Double): Vec3 {
        val m = magnitude(value)
        if (!m.isFinite() || m <= maximum) return value
        if (m < 1e-10) return Vec3(0.0, 0.0, 0.0)
        return scale(value, maximum / m)
    }

    private fun interpolate(
        a: Vec3,
        b: Vec3,
        factor: Double,
    ): Vec3 {
        val t = factor.coerceIn(0.0, 1.0)
        return Vec3(
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t,
            a.z + (b.z - a.z) * t,
        )
    }

    private fun finite(value: Vec3): Vec3 =
        Vec3(
            if (value.x.isFinite()) value.x else 0.0,
            if (value.y.isFinite()) value.y else 0.0,
            if (value.z.isFinite()) value.z else 0.0,
        )

    private fun magnitude(value: Vec3): Double =
        sqrt(
            value.x * value.x +
                value.y * value.y +
                value.z * value.z,
        )

    private fun distance(a: Vec3, b: Vec3): Double =
        magnitude(sub(a, b))

    private fun median(values: List<Double>): Double {
        val sorted = values.filter(Double::isFinite).sorted()
        if (sorted.isEmpty()) return 0.0
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }

    private fun percentile(
        values: List<Double>,
        p: Double,
    ): Double {
        val sorted = values.filter(Double::isFinite).sorted()
        if (sorted.isEmpty()) return 0.0
        if (sorted.size == 1) return sorted.first()
        val position = p.coerceIn(0.0, 1.0) * (sorted.size - 1)
        val lower = position.toInt()
        val upper = min(sorted.lastIndex, lower + 1)
        val fraction = position - lower
        return sorted[lower] * (1.0 - fraction) +
            sorted[upper] * fraction
    }

    private fun format(value: Double): String =
        if (!value.isFinite()) "nan"
        else "%.6f".format(java.util.Locale.US, value)

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}

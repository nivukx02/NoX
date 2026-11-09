package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.ceil
import kotlin.math.sqrt

/**
 * Temporal 3D reconstruction stabilizer for live motion-capture output.
 *
 * This is a geometry-preserving temporal layer, not an image upscaler. It works on measured
 * 3D points with uncertainty and reprojection error, rejects physically implausible jumps,
 * estimates velocity/acceleration, and emits an explicit confidence/acceptance decision.
 */
class TemporalReconstructionStabilizer(
    private val initialConfig: Config = Config(),
) {
    data class Config(
        val maxJointTracks: Int = 256,
        val stateTimeoutNanos: Long = 1_500_000_000L,
        val maxPredictionSeconds: Double = 0.16,
        val maxSpeedMetersPerSecond: Double = 12.0,
        val maxAccelerationMetersPerSecond2: Double = 90.0,
        val maxJerkMetersPerSecond3: Double = 600.0,
        val baseGateMeters: Double = 0.12,
        val uncertaintyGateScale: Double = 2.4,
        val reprojectionGatePixels: Double = 8.0,
        val confidenceFloor: Double = 0.05,
        val confidenceDecaySeconds: Double = 0.44,
        val minimumAcceptedJoints: Int = 5,
        val minimumFrameConfidence: Double = 0.34,
        val maximumMeanUncertaintyMeters: Double = 0.30,
        val maximumMedianReprojectionPixels: Double = 6.0,
        val maximumClockSkewNanos: Long = 25_000_000L,
        val smoothingBaseGain: Double = 0.36,
        val outlierRetention: Double = 0.05,
        val maxLinks: Int = 512,
    ) {
        init {
            require(maxJointTracks in 1..4096)
            require(stateTimeoutNanos > 0L)
            require(maxPredictionSeconds in 0.01..2.0)
            require(maxSpeedMetersPerSecond > 0.0)
            require(maxAccelerationMetersPerSecond2 > 0.0)
            require(maxJerkMetersPerSecond3 > 0.0)
            require(baseGateMeters > 0.0)
            require(uncertaintyGateScale >= 0.0)
            require(reprojectionGatePixels > 0.0)
            require(minimumAcceptedJoints >= 1)
            require(maxLinks >= 0)
        }
    }

    data class Vec3(
        val x: Double,
        val y: Double,
        val z: Double,
    ) {
        operator fun plus(other: Vec3): Vec3 =
            Vec3(x + other.x, y + other.y, z + other.z)

        operator fun minus(other: Vec3): Vec3 =
            Vec3(x - other.x, y - other.y, z - other.z)

        operator fun times(scale: Double): Vec3 =
            Vec3(x * scale, y * scale, z * scale)

        fun norm(): Double =
            sqrt(x * x + y * y + z * z)

        fun distanceTo(other: Vec3): Double =
            (this - other).norm()

        fun finite(): Boolean =
            x.isFinite() && y.isFinite() && z.isFinite()

        companion object {
            val ZERO = Vec3(0.0, 0.0, 0.0)
        }
    }

    data class JointObservation(
        val jointId: Int,
        val positionMeters: Vec3,
        val confidence: Double,
        val uncertaintyMeters: Double,
        val reprojectionErrorPx: Double,
        val viewCount: Int = 2,
        val timestampNanos: Long = 0L,
    )

    data class Link(
        val parentJointId: Int,
        val childJointId: Int,
        val nominalLengthMeters: Double,
        val toleranceMeters: Double = nominalLengthMeters * 0.20,
    )

    data class FrameInput(
        val timestampNanos: Long,
        val joints: List<JointObservation>,
        val camerasUsed: Int,
        val clockSkewNanos: Long,
        val globalConfidence: Double,
        val droppedObservations: Int = 0,
        val links: List<Link> = emptyList(),
    )

    data class JointResult(
        val jointId: Int,
        val positionMeters: Vec3,
        val velocityMetersPerSecond: Vec3,
        val accelerationMetersPerSecond2: Vec3,
        val jerkMetersPerSecond3: Vec3,
        val confidence: Double,
        val uncertaintyMeters: Double,
        val reprojectionErrorPx: Double,
        val predicted: Boolean,
        val rejectedMeasurement: Boolean,
        val temporalStability: Double,
    )

    data class Diagnostics(
        val jointsObserved: Int,
        val jointsAccepted: Int,
        val jointsPredicted: Int,
        val jointsRejected: Int,
        val meanUncertaintyMeters: Double,
        val medianReprojectionErrorPx: Double,
        val maxVelocityMetersPerSecond: Double,
        val maxAccelerationMetersPerSecond2: Double,
        val topologyViolations: Int,
        val temporalStability: Double,
    )

    data class Evaluation(
        val joints: List<JointResult>,
        val diagnostics: Diagnostics,
        val score: Double,
        val accepted: Boolean,
        val reasons: List<String>,
    )

    private data class State(
        val jointId: Int,
        var position: Vec3,
        var velocity: Vec3,
        var acceleration: Vec3,
        var jerk: Vec3,
        var lastTimestampNanos: Long,
        var lastObservedTimestampNanos: Long,
        var confidence: Double,
        var uncertainty: Double,
        var reprojectionErrorPx: Double,
        var stability: Double,
        var missedFrames: Int,
    )

    private val states = LinkedHashMap<Int, State>()

    @Volatile
    private var config: Config = initialConfig

    @Volatile
    private var tuning = TemporalReconstructionTuning()

    /**
     * Applies bounded runtime tuning while preserving the constructor safety envelope.
     *
     * The UI may make the tracker more or less responsive, but it cannot remove the underlying
     * speed, acceleration, uncertainty, clock-skew, joint-count or finite-value guardrails.
     */
    @Synchronized
    fun configure(newTuning: TemporalReconstructionTuning) {
        tuning = newTuning

        config = initialConfig.copy(
            maxPredictionSeconds =
                newTuning.predictionHorizonSeconds.coerceIn(
                    0.02,
                    initialConfig.stateTimeoutNanos / 1e9,
                ),
            baseGateMeters = (
                initialConfig.baseGateMeters *
                    newTuning.innovationGateScale
                ).coerceIn(0.02, 0.75),
            uncertaintyGateScale = (
                initialConfig.uncertaintyGateScale *
                    newTuning.uncertaintyGateScale
                ).coerceIn(0.25, 6.0),
            reprojectionGatePixels = (
                initialConfig.reprojectionGatePixels *
                    newTuning.reprojectionGateScale
                ).coerceIn(1.0, 24.0),
            confidenceDecaySeconds = (
                initialConfig.confidenceDecaySeconds *
                    newTuning.confidenceDecayScale
                ).coerceIn(0.08, 2.5),
            smoothingBaseGain = newTuning.smoothingResponsiveness,
            outlierRetention = newTuning.outlierRetention,
        )
    }

    fun tuning(): TemporalReconstructionTuning = tuning

    @Synchronized
    fun update(input: FrameInput): Evaluation {
        require(input.timestampNanos >= 0L)
        require(input.camerasUsed >= 0)
        require(input.clockSkewNanos >= 0L)
        require(input.globalConfidence.isFinite())

        val observations = input.joints
            .asSequence()
            .filter { it.positionMeters.finite() }
            .filter { it.confidence.isFinite() }
            .filter { it.uncertaintyMeters.isFinite() }
            .filter { it.reprojectionErrorPx.isFinite() }
            .map(::sanitizeObservation)
            .distinctBy { it.jointId }
            .take(config.maxJointTracks)
            .toList()

        val observedIds = HashSet<Int>(observations.size)
        val output = ArrayList<JointResult>(observations.size)

        observations.forEach { observation ->
            observedIds += observation.jointId

            val state = states[observation.jointId]
            if (state == null) {
                ensureTrackCapacity(
                    timestampNanos = input.timestampNanos,
                    requiredFreeSlots = 1,
                )
                val fresh = createState(observation, input.timestampNanos)
                states[observation.jointId] = fresh
                output += toResult(
                    state = fresh,
                    predicted = false,
                    rejected = false,
                )
            } else {
                output += updateState(
                    state = state,
                    observation = observation,
                    timestampNanos = input.timestampNanos,
                )
            }
        }

        val predicted = predictMissing(
            timestampNanos = input.timestampNanos,
            observedIds = observedIds,
        )
        output += predicted

        applyTopologyConstraint(
            results = output,
            links = input.links.take(config.maxLinks),
        )

        prune(
            nowNanos = input.timestampNanos,
            observedIds = observedIds,
        )

        val diagnostics = diagnostics(output)
        val reasons = evaluateReasons(
            input = input,
            diagnostics = diagnostics,
            jointCount = output.count { !it.rejectedMeasurement },
        )
        val score = computeScore(
            input = input,
            diagnostics = diagnostics,
            reasons = reasons,
        )

        return Evaluation(
            joints = output.sortedBy { it.jointId },
            diagnostics = diagnostics,
            score = score,
            accepted = reasons.isEmpty() && score >= 0.55,
            reasons = if (reasons.isEmpty() && score < 0.55) {
                listOf("quality-score-below-threshold")
            } else {
                reasons
            },
        )
    }

    private fun ensureTrackCapacity(
        timestampNanos: Long,
        requiredFreeSlots: Int,
    ) {
        val target =
            (config.maxJointTracks - requiredFreeSlots)
                .coerceAtLeast(0)
        if (states.size <= target) return

        val victims =
            states.values
                .sortedWith(
                    compareBy<State> { state ->
                        when {
                            timestampNanos < state.lastObservedTimestampNanos -> Long.MAX_VALUE
                            else -> timestampNanos - state.lastObservedTimestampNanos
                        }
                    }.thenBy { it.confidence },
                )
                .take(states.size - target)
                .map { it.jointId }

        victims.forEach(states::remove)
    }

    @Synchronized
    fun reset() {
        states.clear()
    }

    @Synchronized
    fun snapshot(nowNanos: Long): List<JointResult> =
        states.values
            .filter {
                nowNanos >= it.lastObservedTimestampNanos &&
                    nowNanos - it.lastObservedTimestampNanos <= config.stateTimeoutNanos
            }
            .map {
                toResult(
                    state = it,
                    predicted = it.missedFrames > 0,
                    rejected = false,
                )
            }
            .sortedBy { it.jointId }

    private fun sanitizeObservation(
        input: JointObservation,
    ): JointObservation =
        input.copy(
            confidence = input.confidence.coerceIn(0.0, 1.0),
            uncertaintyMeters = input.uncertaintyMeters
                .coerceAtLeast(0.0)
                .coerceAtMost(100.0),
            reprojectionErrorPx = input.reprojectionErrorPx
                .coerceAtLeast(0.0)
                .coerceAtMost(10_000.0),
            viewCount = input.viewCount.coerceAtLeast(0),
        )

    private fun createState(
        observation: JointObservation,
        timestampNanos: Long,
    ): State =
        State(
            jointId = observation.jointId,
            position = observation.positionMeters,
            velocity = Vec3.ZERO,
            acceleration = Vec3.ZERO,
            jerk = Vec3.ZERO,
            lastTimestampNanos = timestampNanos,
            lastObservedTimestampNanos = timestampNanos,
            confidence = observation.confidence,
            uncertainty = observation.uncertaintyMeters,
            reprojectionErrorPx = observation.reprojectionErrorPx,
            stability = observation.confidence,
            missedFrames = 0,
        )

    private fun updateState(
        state: State,
        observation: JointObservation,
        timestampNanos: Long,
    ): JointResult {
        val dt = (
            (timestampNanos - state.lastTimestampNanos).toDouble() * 1e-9
        ).coerceIn(1.0 / 240.0, config.maxPredictionSeconds)

        val predicted = predictPosition(state, dt)
        val innovation = observation.positionMeters - predicted

        val gate = (
            config.baseGateMeters +
                config.uncertaintyGateScale *
                    (
                        observation.uncertaintyMeters +
                            state.uncertainty
                    )
        ).coerceAtLeast(config.baseGateMeters)

        val innovationNorm = innovation.norm()
        val velocity = innovation * (1.0 / dt)

        val dynamicGate = gate + (
            state.velocity.norm() *
                dt *
                0.35
        )

        val accepted = innovationNorm <= dynamicGate &&
            velocity.norm() <=
                config.maxSpeedMetersPerSecond * 1.85

        if (!accepted) {
            val outlierEvidence = exp(
                -(
                    innovationNorm * innovationNorm
                ) / (
                    2.0 * dynamicGate * dynamicGate +
                        1e-9
                    )
            ).coerceIn(0.0, 1.0)

            val retainedEvidenceGain = (
                0.22 +
                    0.36 * config.outlierRetention
            ).coerceIn(0.18, 0.36)

            state.position = predicted
            state.velocity = limitNorm(
                state.velocity,
                config.maxSpeedMetersPerSecond,
            )
            state.acceleration = limitNorm(
                state.acceleration,
                config.maxAccelerationMetersPerSecond2,
            )
            state.jerk = limitNorm(
                state.jerk,
                config.maxJerkMetersPerSecond3,
            )
            state.confidence = (
                state.confidence *
                    exp(
                        -dt /
                            config.confidenceDecaySeconds.coerceAtLeast(0.05),
                    )
            ).coerceAtLeast(config.confidenceFloor)
            state.uncertainty = min(
                state.uncertainty +
                    dt * (
                        0.08 *
                            (
                                1.0 -
                                    0.50 *
                                    config.outlierRetention
                            )
                    ) +
                    innovationNorm *
                        config.outlierRetention *
                        0.02,
                5.0,
            )
            state.stability = blend(
                state.stability,
                outlierEvidence,
                retainedEvidenceGain,
            )
            state.reprojectionErrorPx = max(
                state.reprojectionErrorPx,
                observation.reprojectionErrorPx,
            )
            state.stability = blend(
                state.stability,
                0.0,
                0.22,
            )
            state.missedFrames = (
                state.missedFrames + 1
            ).coerceAtMost(Int.MAX_VALUE)

            return toResult(
                state = state,
                predicted = true,
                rejected = true,
            )
        }

        val quality = observationQuality(observation)
        val alpha = (
            config.smoothingBaseGain +
                quality * 0.48
        ).coerceIn(0.12, 0.92)

        val previousVelocity = state.velocity
        val corrected = predicted + (
            innovation * alpha
        )

        val measuredVelocity = (
            corrected - state.position
        ) * (1.0 / dt)

        val blendedVelocity = blendVec(
            current = state.velocity,
            measured = measuredVelocity,
            alpha = (
                0.14 +
                    quality * 0.42
            ).coerceIn(0.10, 0.60),
        )

        val acceleration = (
            blendedVelocity - previousVelocity
        ) * (1.0 / dt)

        val measuredJerk = (
            acceleration - state.acceleration
        ) * (1.0 / dt)

        state.position = corrected
        state.velocity = limitNorm(
            blendedVelocity,
            config.maxSpeedMetersPerSecond,
        )
        state.acceleration = blendVec(
            current = state.acceleration,
            measured = limitNorm(
                acceleration,
                config.maxAccelerationMetersPerSecond2,
            ),
            alpha = 0.24,
        )
        state.jerk = blendVec(
            current = state.jerk,
            measured = limitNorm(
                measuredJerk,
                config.maxJerkMetersPerSecond3,
            ),
            alpha = 0.18,
        )
        state.lastTimestampNanos = timestampNanos
        state.lastObservedTimestampNanos = timestampNanos
        state.confidence = blend(
            state.confidence,
            observation.confidence,
            0.30 + quality * 0.28,
        )
        state.uncertainty = blend(
            state.uncertainty,
            observation.uncertaintyMeters,
            0.34,
        )
        state.reprojectionErrorPx = blend(
            state.reprojectionErrorPx,
            observation.reprojectionErrorPx,
            0.30,
        )

        val innovationScore = exp(
            -innovationNorm * innovationNorm /
                (2.0 * gate * gate + 1e-9),
        ).coerceIn(0.0, 1.0)

        val jerkScore = exp(
            -measuredJerk.norm() * measuredJerk.norm() /
                (
                    2.0 *
                        config.maxJerkMetersPerSecond3 *
                        config.maxJerkMetersPerSecond3 +
                        1e-9
                    ),
        ).coerceIn(0.0, 1.0)

        state.stability = blend(
            state.stability,
            (
                0.44 * innovationScore +
                    0.36 * observation.confidence +
                    0.20 * jerkScore
            ).coerceIn(0.0, 1.0),
            0.28,
        )
        state.missedFrames = 0

        return toResult(
            state = state,
            predicted = false,
            rejected = false,
        )
    }

    private fun predictMissing(
        timestampNanos: Long,
        observedIds: Set<Int>,
    ): List<JointResult> {
        val output = ArrayList<JointResult>()

        states.values.forEach { state ->
            if (state.jointId in observedIds) return@forEach
            val dt = (
                (timestampNanos - state.lastTimestampNanos).toDouble() * 1e-9
            ).coerceIn(0.0, config.maxPredictionSeconds)

            if (dt <= 0.0) return@forEach

            state.position = predictPosition(state, dt)
            state.velocity = limitNorm(
                state.velocity,
                config.maxSpeedMetersPerSecond,
            )
            state.acceleration = limitNorm(
                state.acceleration,
                config.maxAccelerationMetersPerSecond2,
            )
            state.jerk = limitNorm(
                state.jerk,
                config.maxJerkMetersPerSecond3,
            )
            state.lastTimestampNanos = timestampNanos
            state.missedFrames = (
                state.missedFrames + 1
            ).coerceAtMost(Int.MAX_VALUE)

            val decay = exp(
                -dt /
                    config.confidenceDecaySeconds.coerceAtLeast(0.05),
            )
            state.confidence = (
                state.confidence * decay
            ).coerceAtLeast(config.confidenceFloor)
            state.uncertainty = (
                state.uncertainty +
                    dt * (
                        0.04 +
                            state.acceleration.norm() * 0.003
                        )
            ).coerceAtMost(5.0)
            state.stability = blend(
                state.stability,
                0.38 * decay,
                0.18,
            )

            if (
                state.missedFrames <=
                    ceil(config.stateTimeoutNanos * 1e-9 * 60.0)
                        .toInt()
            ) {
                output += toResult(
                    state = state,
                    predicted = true,
                    rejected = false,
                )
            }
        }

        return output
    }

    private fun predictPosition(
        state: State,
        dtSeconds: Double,
    ): Vec3 =
        state.position +
            state.velocity * dtSeconds +
            state.acceleration * (
                0.5 * dtSeconds * dtSeconds
            ) +
            state.jerk * (
                (1.0 / 6.0) *
                    dtSeconds *
                    dtSeconds *
                    dtSeconds
            )

    private fun applyTopologyConstraint(
        results: MutableList<JointResult>,
        links: List<Link>,
    ) {
        if (links.isEmpty() || results.isEmpty()) return

        val byId = results.associateBy { it.jointId }

        links.forEach { link ->
            if (link.parentJointId == link.childJointId) return@forEach
            val parent = byId[link.parentJointId] ?: return@forEach
            val child = byId[link.childJointId] ?: return@forEach

            val delta = child.positionMeters - parent.positionMeters
            val actual = delta.norm()
            if (!actual.isFinite() || actual <= 1e-9) return@forEach

            val tolerance = max(
                link.toleranceMeters,
                (
                    parent.uncertaintyMeters +
                        child.uncertaintyMeters
                ) * 1.5,
            ) * tuning.topologyToleranceScale

            val error = actual - link.nominalLengthMeters
            if (abs(error) <= tolerance) return@forEach

            val direction = delta * (1.0 / actual)
            val correction = (
                error - tolerance.signWithMagnitude(error)
            ).coerceAtMost(actual * 0.5).coerceAtLeast(-actual * 0.5)

            val childState = states[child.jointId]
            val parentState = states[parent.jointId]

            if (childState != null && childState.missedFrames > 0) {
                childState.position = (
                    child.positionMeters -
                        direction * correction
                )
                childState.uncertainty = (
                    childState.uncertainty +
                        abs(error) * 0.08
                ).coerceAtMost(5.0)
            } else if (parentState != null && parentState.missedFrames > 0) {
                parentState.position = (
                    parent.positionMeters +
                        direction * correction
                )
                parentState.uncertainty = (
                    parentState.uncertainty +
                        abs(error) * 0.08
                ).coerceAtMost(5.0)
            }
        }

        results.forEachIndexed { index, result ->
            states[result.jointId]?.let { state ->
                results[index] = toResult(
                    state = state,
                    predicted = result.predicted,
                    rejected = result.rejectedMeasurement,
                )
            }
        }
    }

    private fun toResult(
        state: State,
        predicted: Boolean,
        rejected: Boolean,
    ): JointResult =
        JointResult(
            jointId = state.jointId,
            positionMeters = state.position,
            velocityMetersPerSecond = state.velocity,
            accelerationMetersPerSecond2 = state.acceleration,
            jerkMetersPerSecond3 = state.jerk,
            confidence = state.confidence.coerceIn(0.0, 1.0),
            uncertaintyMeters = state.uncertainty.coerceAtLeast(0.0),
            reprojectionErrorPx = state.reprojectionErrorPx
                .coerceAtLeast(0.0),
            predicted = predicted,
            rejectedMeasurement = rejected,
            temporalStability = state.stability.coerceIn(0.0, 1.0),
        )

    private fun diagnostics(
        results: List<JointResult>,
    ): Diagnostics {
        val valid = results.filter {
            it.positionMeters.finite()
        }

        val uncertainties = valid.map { it.uncertaintyMeters }
            .filter { it.isFinite() }
            .sorted()

        val errors = valid.map { it.reprojectionErrorPx }
            .filter { it.isFinite() }
            .sorted()

        val velocities = valid.map {
            it.velocityMetersPerSecond.norm()
        }
        val accelerations = valid.map {
            it.accelerationMetersPerSecond2.norm()
        }

        val topologyViolations = 0

        return Diagnostics(
            jointsObserved = results.count { !it.predicted },
            jointsAccepted = results.count {
                !it.rejectedMeasurement
            },
            jointsPredicted = results.count { it.predicted },
            jointsRejected = results.count {
                it.rejectedMeasurement
            },
            meanUncertaintyMeters = if (uncertainties.isEmpty()) {
                Double.POSITIVE_INFINITY
            } else {
                uncertainties.average()
            },
            medianReprojectionErrorPx = median(errors),
            maxVelocityMetersPerSecond = velocities.maxOrNull() ?: 0.0,
            maxAccelerationMetersPerSecond2 =
                accelerations.maxOrNull() ?: 0.0,
            topologyViolations = topologyViolations,
            temporalStability = if (valid.isEmpty()) {
                0.0
            } else {
                valid.map { it.temporalStability }.average()
            },
        )
    }

    private fun evaluateReasons(
        input: FrameInput,
        diagnostics: Diagnostics,
        jointCount: Int,
    ): List<String> {
        val reasons = ArrayList<String>()

        if (jointCount < config.minimumAcceptedJoints) {
            reasons += "insufficient-joint-coverage"
        }

        if (input.camerasUsed < 2) {
            reasons += "insufficient-camera-count"
        }

        if (
            !diagnostics.meanUncertaintyMeters.isFinite() ||
            diagnostics.meanUncertaintyMeters >
                config.maximumMeanUncertaintyMeters
        ) {
            reasons += "mean-depth-uncertainty"
        }

        if (
            !diagnostics.medianReprojectionErrorPx.isFinite() ||
            diagnostics.medianReprojectionErrorPx >
                config.maximumMedianReprojectionPixels
        ) {
            reasons += "median-reprojection-error"
        }

        if (input.clockSkewNanos > config.maximumClockSkewNanos) {
            reasons += "clock-skew"
        }

        if (input.globalConfidence < config.minimumFrameConfidence) {
            reasons += "low-global-confidence"
        }

        if (
            diagnostics.jointsRejected >
                max(1, diagnostics.jointsObserved / 2)
        ) {
            reasons += "excess-outlier-rejection"
        }

        if (
            diagnostics.maxVelocityMetersPerSecond >
                config.maxSpeedMetersPerSecond * 1.35
        ) {
            reasons += "velocity-instability"
        }

        if (
            diagnostics.maxAccelerationMetersPerSecond2 >
                config.maxAccelerationMetersPerSecond2 * 1.50
        ) {
            reasons += "acceleration-instability"
        }

        if (input.droppedObservations > jointCount) {
            reasons += "excess-dropped-observations"
        }

        return reasons.distinct()
    }

    private fun computeScore(
        input: FrameInput,
        diagnostics: Diagnostics,
        reasons: List<String>,
    ): Double {
        val uncertaintyScore = if (
            diagnostics.meanUncertaintyMeters.isFinite()
        ) {
            exp(
                -diagnostics.meanUncertaintyMeters /
                    config.maximumMeanUncertaintyMeters.coerceAtLeast(1e-4),
            ).coerceIn(0.0, 1.0)
        } else {
            0.0
        }

        val reprojectionScore = if (
            diagnostics.medianReprojectionErrorPx.isFinite()
        ) {
            exp(
                -diagnostics.medianReprojectionErrorPx /
                    config.maximumMedianReprojectionPixels,
            ).coerceIn(0.0, 1.0)
        } else {
            0.0
        }

        val jointCoverage = (
            diagnostics.jointsAccepted.toDouble() /
                max(
                    config.minimumAcceptedJoints,
                    diagnostics.jointsObserved,
                ).toDouble()
        ).coerceIn(0.0, 1.0)

        val motionStability = (
            diagnostics.temporalStability *
                exp(
                    -(
                        diagnostics.maxAccelerationMetersPerSecond2 /
                            config.maxAccelerationMetersPerSecond2
                    ).coerceAtLeast(0.0)
                )
        ).coerceIn(0.0, 1.0)

        val cameraScore = (
            input.camerasUsed / 4.0
        ).coerceIn(0.0, 1.0)

        val skewScore = (
            1.0 -
                input.clockSkewNanos.toDouble() /
                    config.maximumClockSkewNanos.coerceAtLeast(1L)
        ).coerceIn(0.0, 1.0)

        var score = (
            0.22 * uncertaintyScore +
                0.20 * reprojectionScore +
                0.18 * jointCoverage +
                0.16 * input.globalConfidence.coerceIn(0.0, 1.0) +
                0.10 * motionStability +
                0.07 * cameraScore +
                0.07 * skewScore
        ).coerceIn(0.0, 1.0)

        if (reasons.isNotEmpty()) {
            score *= (
                1.0 -
                    min(0.55, reasons.size * 0.065)
            )
        }

        return score.coerceIn(0.0, 1.0)
    }

    private fun observationQuality(
        observation: JointObservation,
    ): Double {
        val reprojection = exp(
            -observation.reprojectionErrorPx /
                config.reprojectionGatePixels,
        ).coerceIn(0.0, 1.0)

        val uncertainty = exp(
            -observation.uncertaintyMeters /
                config.maximumMeanUncertaintyMeters.coerceAtLeast(1e-4),
        ).coerceIn(0.0, 1.0)

        val viewScore = (
            observation.viewCount / 4.0
        ).coerceIn(0.0, 1.0)

        return (
            0.46 * observation.confidence +
                0.28 * reprojection +
                0.18 * uncertainty +
                0.08 * viewScore
        ).coerceIn(0.0, 1.0)
    }

    private fun blend(
        current: Double,
        measured: Double,
        alpha: Double,
    ): Double {
        val a = alpha.coerceIn(0.0, 1.0)
        return current * (1.0 - a) + measured * a
    }

    private fun blendVec(
        current: Vec3,
        measured: Vec3,
        alpha: Double,
    ): Vec3 =
        Vec3(
            x = blend(current.x, measured.x, alpha),
            y = blend(current.y, measured.y, alpha),
            z = blend(current.z, measured.z, alpha),
        )

    private fun limitNorm(
        value: Vec3,
        limit: Double,
    ): Vec3 {
        val length = value.norm()
        if (!length.isFinite() || length <= limit || length <= 1e-12) {
            return value
        }
        return value * (limit / length)
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return Double.POSITIVE_INFINITY
        val size = values.size
        return if (size % 2 == 1) {
            values[size / 2]
        } else {
            0.5 * (
                values[size / 2 - 1] +
                    values[size / 2]
            )
        }
    }

    private fun Double.signWithMagnitude(
        original: Double,
    ): Double {
        val magnitude = abs(this)
        return if (original < 0.0) -magnitude else magnitude
    }

    private fun prune(
        nowNanos: Long,
        observedIds: Set<Int>,
    ) {
        val iterator = states.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            val state = entry.value

            if (state.jointId in observedIds) {
                continue
            }

            if (
                nowNanos - state.lastObservedTimestampNanos >
                    config.stateTimeoutNanos
            ) {
                iterator.remove()
            }
        }
    }
}


package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * High-end geometric constraint solver for camera-space motion landmarks.
 *
 * This solver performs iterative constraint projection after perception and before avatar
 * retargeting. It preserves observed micro-motion while preventing bone-length and temporal
 * discontinuities from spreading through the rig.
 */
class HighEndKinematicConstraintSolver(
    private val config: Config = Config(),
) {

    data class Config(
        val iterations: Int = 18,
        val temporalBlend: Double = 0.36,
        val edgeStiffness: Double = 0.82,
        val anchorStiffness: Double = 0.90,
        val predictionWeight: Double = 0.68,
        val microMotionPreservation: Double = 0.92,
        val outlierSigma: Double = 3.5,
        val residualScaleMeters: Double = 0.08,
        val velocityLimitMetersPerSecond: Double = 12.0,
        val accelerationLimitMetersPerSecond2: Double = 50.0,
        val hardDistanceToleranceMeters: Double = 0.025,
        val minimumConfidence: Double = 0.05,
    ) {
        init {
            require(iterations in 1..128)
            require(temporalBlend in 0.0..1.0)
            require(edgeStiffness in 0.0..1.0)
            require(anchorStiffness in 0.0..1.0)
            require(predictionWeight in 0.0..1.0)
            require(microMotionPreservation in 0.0..1.0)
            require(outlierSigma > 0.0)
            require(residualScaleMeters > 0.0)
            require(velocityLimitMetersPerSecond > 0.0)
            require(accelerationLimitMetersPerSecond2 > 0.0)
            require(hardDistanceToleranceMeters >= 0.0)
            require(minimumConfidence in 0.0..1.0)
        }
    }

    data class Joint(
        val id: Int,
        val semanticName: String,
        val position: Vec3,
        val confidence: Double,
        val predicted: Boolean = false,
        val visible: Boolean = true,
    )

    data class Edge(
        val a: Int,
        val b: Int,
        val preferredLengthMeters: Double,
        val minimumLengthMeters: Double,
        val maximumLengthMeters: Double,
        val stiffness: Double = 1.0,
    ) {
        init {
            require(preferredLengthMeters >= 0.0)
            require(minimumLengthMeters >= 0.0)
            require(maximumLengthMeters >= minimumLengthMeters)
            require(preferredLengthMeters in minimumLengthMeters..maximumLengthMeters)
            require(stiffness in 0.0..1.0)
        }
    }

    data class JointLimit(
        val id: Int,
        val parentId: Int?,
        val minimumDistanceMeters: Double = 0.0,
        val maximumDistanceMeters: Double = Double.POSITIVE_INFINITY,
        val maximumSpeedMetersPerSecond: Double? = null,
        val maximumAccelerationMetersPerSecond2: Double? = null,
    )

    data class Frame(
        val timestampNanos: Long,
        val subjectType: SubjectType,
        val joints: List<Joint>,
        val edges: List<Edge>,
        val limits: List<JointLimit> = emptyList(),
        val rootId: Int? = null,
    )

    data class SolvedJoint(
        val source: Joint,
        val position: Vec3,
        val velocityMetersPerSecond: Vec3,
        val accelerationMetersPerSecond2: Vec3,
        val residualMeters: Double,
        val confidence: Double,
        val predicted: Boolean,
        val constrained: Boolean,
        val outlier: Boolean,
    )

    data class Result(
        val timestampNanos: Long,
        val subjectType: SubjectType,
        val joints: List<SolvedJoint>,
        val iterations: Int,
        val maxResidualMeters: Double,
        val meanResidualMeters: Double,
        val temporalStability: Double,
        val topologyStability: Double,
        val outlierCount: Int,
        val predictedCount: Int,
        val preservedMicroMotionCount: Int,
        val diagnostics: List<String>,
    ) {
        val globalConfidence: Double
            get() = joints
                .map { it.confidence }
                .let { values ->
                    if (values.isEmpty()) 0.0 else values.average()
                }
                .let {
                    (
                        it *
                            (1.0 - min(0.55, meanResidualMeters / 0.40)) *
                            temporalStability *
                            topologyStability
                        ).coerceIn(0.0, 1.0)
                }
    }

    private data class PreviousFrame(
        val timestampNanos: Long,
        val positions: Map<Int, Vec3>,
        val velocities: Map<Int, Vec3>,
    )

    private var previous: PreviousFrame? = null

    @Synchronized
    fun solve(frame: Frame): Result {
        require(frame.joints.isNotEmpty()) {
            "Kinematic solver requires at least one joint"
        }

        val positions = frame.joints
            .associate { it.id to it.position }
            .toMutableMap()

        val observations = frame.joints.associateBy { it.id }

        val previousFrame = previous
        val dtSeconds = if (
            previousFrame == null ||
            frame.timestampNanos <= previousFrame.timestampNanos
        ) {
            1.0 / 60.0
        } else {
            (
                frame.timestampNanos -
                    previousFrame.timestampNanos
                ).toDouble() / 1e9
        }.coerceIn(1e-4, 0.50)

        val predicted = mutableMapOf<Int, Vec3>()
        observations.forEach { (id, joint) ->
            val old = previousFrame?.positions?.get(id)
            val oldVelocity = previousFrame?.velocities?.get(id)
            if (old != null && oldVelocity != null) {
                predicted[id] = old + oldVelocity * dtSeconds
            }
        }

        observations.forEach { (id, joint) ->
            val target = predicted[id] ?: return@forEach
            val temporalWeight = (
                config.temporalBlend *
                    if (joint.predicted) config.predictionWeight else 1.0
                ).coerceIn(0.0, 1.0)

            positions[id] = lerp(
                positions[id] ?: target,
                target,
                temporalWeight * (
                    1.0 -
                        config.microMotionPreservation *
                        microMotionFactor(
                            joint = joint,
                            target = target,
                            observation = positions[id] ?: target,
                        )
                ),
            )
        }

        repeat(config.iterations) {
            applyAnchorConstraints(
                observations,
                positions,
            )
            applyJointLimits(
                frame.limits,
                observations,
                positions,
            )
            applyEdgeConstraints(
                frame.edges,
                observations,
                positions,
            )
        }

        val residuals = computeResiduals(
            edges = frame.edges,
            positions = positions,
        )

        val robustOutliers = robustOutlierIds(
            observations = observations,
            residuals = residuals,
        )

        val solved = frame.joints.map { original ->
            val solvedPosition = positions[original.id] ?: original.position
            val residual = residuals[original.id] ?: 0.0
            val outlier = robustOutliers.contains(original.id)

            val velocity = velocityFor(
                id = original.id,
                current = solvedPosition,
                previous = previousFrame,
                dtSeconds = dtSeconds,
            )

            val requestedSpeedLimit = frame.limits
                .firstOrNull { it.id == original.id }
                ?.maximumSpeedMetersPerSecond

            val speedLimit =
                requestedSpeedLimit?.coerceAtLeast(0.0)
                    ?: config.velocityLimitMetersPerSecond

            val boundedVelocity = limitVector(
                value = velocity,
                maximumMagnitude = speedLimit,
            )

            val acceleration = accelerationFor(
                id = original.id,
                velocity = boundedVelocity,
                previous = previousFrame,
                dtSeconds = dtSeconds,
            )

            val requestedAccelerationLimit = frame.limits
                .firstOrNull { it.id == original.id }
                ?.maximumAccelerationMetersPerSecond2

            val accelerationLimit =
                requestedAccelerationLimit?.coerceAtLeast(0.0)
                    ?: config.accelerationLimitMetersPerSecond2

            val boundedAcceleration = limitVector(
                value = acceleration,
                maximumMagnitude = accelerationLimit,
            )

            val confidence = (
                original.confidence.coerceIn(0.0, 1.0) *
                    residualConfidence(residual) *
                    if (outlier) 0.38 else 1.0
                ).coerceIn(0.0, 1.0)

            SolvedJoint(
                source = original,
                position = solvedPosition,
                velocityMetersPerSecond = boundedVelocity,
                accelerationMetersPerSecond2 = boundedAcceleration,
                residualMeters = residual,
                confidence = confidence,
                predicted = original.predicted,
                constrained = residual > config.hardDistanceToleranceMeters,
                outlier = outlier,
            )
        }

        val meanResidual = solved
            .map { it.residualMeters }
            .averageOrZero()

        val maxResidual = solved
            .maxOfOrNull { it.residualMeters }
            ?: 0.0

        val temporalStability = temporalStability(
            current = solved,
            previous = previousFrame,
        )

        val topologyStability = topologyStability(
            edges = frame.edges,
            positions = positions,
        )

        val microCount = solved.count {
            val speed = magnitude(it.velocityMetersPerSecond)
            speed >= microMotionFloor && speed <= microMotionCeiling
        }

        val result = Result(
            timestampNanos = frame.timestampNanos,
            subjectType = frame.subjectType,
            joints = solved,
            iterations = config.iterations,
            maxResidualMeters = maxResidual,
            meanResidualMeters = meanResidual,
            temporalStability = temporalStability,
            topologyStability = topologyStability,
            outlierCount = solved.count { it.outlier },
            predictedCount = solved.count { it.predicted },
            preservedMicroMotionCount = microCount,
            diagnostics = buildDiagnostics(
                frame = frame,
                solved = solved,
                temporalStability = temporalStability,
                topologyStability = topologyStability,
                meanResidual = meanResidual,
                maxResidual = maxResidual,
            ),
        )

        previous = PreviousFrame(
            timestampNanos = frame.timestampNanos,
            positions = solved.associate { it.source.id to it.position },
            velocities = solved.associate {
                it.source.id to it.velocityMetersPerSecond
            },
        )

        return result
    }

    @Synchronized
    fun reset() {
        previous = null
    }

    private val microMotionFloor: Double
        get() = 0.003

    private val microMotionCeiling: Double
        get() = 0.50

    private fun applyEdgeConstraints(
        edges: List<Edge>,
        observations: Map<Int, Joint>,
        positions: MutableMap<Int, Vec3>,
    ) {
        edges.forEach { edge ->
            val a = positions[edge.a] ?: return@forEach
            val b = positions[edge.b] ?: return@forEach
            val delta = b - a
            val distance = magnitude(delta)

            if (!distance.isFinite() || distance < 1e-7) return@forEach

            val target = edgeTargetDistance(edge, distance)
            val error = distance - target

            if (abs(error) < config.hardDistanceToleranceMeters) {
                return@forEach
            }

            val direction = delta / distance

            val confidenceA = observations[edge.a]
                ?.confidence
                ?.coerceIn(0.0, 1.0)
                ?: config.minimumConfidence

            val confidenceB = observations[edge.b]
                ?.confidence
                ?.coerceIn(0.0, 1.0)
                ?: config.minimumConfidence

            val weightA = (
                (1.0 - confidenceA) +
                    if (observations[edge.a]?.predicted == true) 0.65 else 0.0
                ).coerceAtLeast(0.05)

            val weightB = (
                (1.0 - confidenceB) +
                    if (observations[edge.b]?.predicted == true) 0.65 else 0.0
                ).coerceAtLeast(0.05)

            val total = weightA + weightB
            val correction = (
                error *
                    config.edgeStiffness *
                    edge.stiffness.coerceIn(0.0, 1.0)
                )

            val correctionA = direction * (
                correction * weightA / total
            )
            val correctionB = direction * (
                correction * weightB / total
            )

            positions[edge.a] = a + correctionA
            positions[edge.b] = b - correctionB
        }
    }

    private fun applyJointLimits(
        limits: List<JointLimit>,
        observations: Map<Int, Joint>,
        positions: MutableMap<Int, Vec3>,
    ) {
        limits.forEach { limit ->
            val parentId = limit.parentId ?: return@forEach
            val parent = positions[parentId] ?: return@forEach
            val child = positions[limit.id] ?: return@forEach

            val delta = child - parent
            val distance = magnitude(delta)
            if (!distance.isFinite() || distance < 1e-7) return@forEach

            val target = when {
                distance < limit.minimumDistanceMeters ->
                    limit.minimumDistanceMeters
                limit.maximumDistanceMeters.isFinite() &&
                    distance > limit.maximumDistanceMeters ->
                    limit.maximumDistanceMeters
                else ->
                    distance
            }

            if (abs(target - distance) < 1e-7) return@forEach

            val direction = delta / distance
            val confidence = observations[limit.id]
                ?.confidence
                ?.coerceIn(0.0, 1.0)
                ?: 0.50

            val correction = (
                target - distance
            ) *
                config.anchorStiffness *
                (0.45 + 0.55 * (1.0 - confidence))

            positions[limit.id] = child + direction * correction
        }
    }

    private fun applyAnchorConstraints(
        observations: Map<Int, Joint>,
        positions: MutableMap<Int, Vec3>,
    ) {
        observations.forEach { (id, observation) ->
            val current = positions[id] ?: return@forEach
            val confidence = observation.confidence
                .coerceIn(config.minimumConfidence, 1.0)

            val weight = config.anchorStiffness *
                if (observation.predicted) {
                    confidence * 0.35
                } else {
                    confidence
                }

            positions[id] = lerp(
                current,
                observation.position,
                weight.coerceIn(0.0, 1.0),
            )
        }
    }

    private fun edgeTargetDistance(
        edge: Edge,
        actualDistance: Double,
    ): Double =
        when {
            actualDistance < edge.minimumLengthMeters ->
                edge.minimumLengthMeters
            actualDistance > edge.maximumLengthMeters ->
                edge.maximumLengthMeters
            else ->
                lerpScalar(
                    actualDistance,
                    edge.preferredLengthMeters,
                    edge.stiffness.coerceIn(0.0, 1.0),
                )
        }

    private fun computeResiduals(
        edges: List<Edge>,
        positions: Map<Int, Vec3>,
    ): Map<Int, Double> {
        val accum = mutableMapOf<Int, MutableList<Double>>()

        edges.forEach { edge ->
            val a = positions[edge.a] ?: return@forEach
            val b = positions[edge.b] ?: return@forEach
            val distance = magnitude(b - a)

            if (!distance.isFinite()) return@forEach

            val target = edgeTargetDistance(edge, distance)
            val residual = abs(distance - target)
            accum.getOrPut(edge.a) { mutableListOf() }.add(residual)
            accum.getOrPut(edge.b) { mutableListOf() }.add(residual)
        }

        return accum.mapValues { (_, values) ->
            values.average()
        }
    }

    private fun robustOutlierIds(
        observations: Map<Int, Joint>,
        residuals: Map<Int, Double>,
    ): Set<Int> {
        if (residuals.isEmpty()) return emptySet()

        val values = residuals.values.sorted()
        val median = if (values.size % 2 == 0) {
            (
                values[values.size / 2 - 1] +
                    values[values.size / 2]
                ) * 0.5
        } else {
            values[values.size / 2]
        }

        val deviations = values
            .map { abs(it - median) }
            .sorted()

        val mad = if (deviations.size % 2 == 0) {
            (
                deviations[deviations.size / 2 - 1] +
                    deviations[deviations.size / 2]
                ) * 0.5
        } else {
            deviations[deviations.size / 2]
        }

        val sigma = max(
            1e-5,
            1.4826 * mad,
        )

        val threshold = median + config.outlierSigma * sigma

        return residuals
            .filter { (id, residual) ->
                observations[id]?.confidence?.let {
                    it < 0.35
                } == true &&
                    residual > threshold &&
                    residual > config.residualScaleMeters
            }
            .keys
    }

    private fun velocityFor(
        id: Int,
        current: Vec3,
        previous: PreviousFrame?,
        dtSeconds: Double,
    ): Vec3 {
        val old = previous?.positions?.get(id)
            ?: return Vec3(0.0, 0.0, 0.0)

        return (current - old) / dtSeconds.coerceAtLeast(1e-4)
    }

    private fun accelerationFor(
        id: Int,
        velocity: Vec3,
        previous: PreviousFrame?,
        dtSeconds: Double,
    ): Vec3 {
        val old = previous?.velocities?.get(id)
            ?: return Vec3(0.0, 0.0, 0.0)

        return (velocity - old) / dtSeconds.coerceAtLeast(1e-4)
    }

    private fun limitVector(
        value: Vec3,
        maximumMagnitude: Double,
    ): Vec3 {
        if (!maximumMagnitude.isFinite() || maximumMagnitude <= 0.0) {
            return Vec3(0.0, 0.0, 0.0)
        }

        val valueMagnitude = magnitude(value)
        if (!valueMagnitude.isFinite() || valueMagnitude <= maximumMagnitude) {
            return value
        }

        return value * (
            maximumMagnitude /
                valueMagnitude.coerceAtLeast(1e-8)
        )
    }

    private fun temporalStability(
        current: List<SolvedJoint>,
        previous: PreviousFrame?,
    ): Double {
        if (current.isEmpty()) return 0.0
        if (previous == null) return 0.55

        val errors = current.mapNotNull { joint ->
            val old = previous.positions[joint.source.id]
                ?: return@mapNotNull null
            val distance = magnitude(joint.position - old)

            if (!distance.isFinite()) null
            else (distance / 0.35).coerceAtMost(1.0)
        }

        if (errors.isEmpty()) return 0.45

        return (
            1.0 - errors.average()
        ).coerceIn(0.0, 1.0)
    }

    private fun topologyStability(
        edges: List<Edge>,
        positions: Map<Int, Vec3>,
    ): Double {
        if (edges.isEmpty()) return 0.55

        val scores = edges.mapNotNull { edge ->
            val a = positions[edge.a]
                ?: return@mapNotNull null
            val b = positions[edge.b]
                ?: return@mapNotNull null

            val distance = magnitude(b - a)
            if (!distance.isFinite()) return@mapNotNull null

            val minPenalty = when {
                distance < edge.minimumLengthMeters ->
                    (
                        edge.minimumLengthMeters - distance
                    ) / edge.maximumLengthMeters.coerceAtLeast(1e-4)
                distance > edge.maximumLengthMeters ->
                    (
                        distance - edge.maximumLengthMeters
                    ) / edge.maximumLengthMeters.coerceAtLeast(1e-4)
                else -> 0.0
            }

            1.0 - minPenalty.coerceIn(0.0, 1.0)
        }

        return scores.averageOrZero()
    }

    private fun microMotionFactor(
        joint: Joint,
        target: Vec3,
        observation: Vec3,
    ): Double {
        val delta = magnitude(observation - target)
        if (!delta.isFinite()) return 0.0
        return (delta / 0.06).coerceIn(0.0, 1.0)
    }

    private fun residualConfidence(residual: Double): Double =
        if (!residual.isFinite()) {
            0.0
        } else {
            kotlin.math.exp(
                -residual /
                    config.residualScaleMeters.coerceAtLeast(1e-4),
            ).coerceIn(0.0, 1.0)
        }

    private fun buildDiagnostics(
        frame: Frame,
        solved: List<SolvedJoint>,
        temporalStability: Double,
        topologyStability: Double,
        meanResidual: Double,
        maxResidual: Double,
    ): List<String> = buildList {
        add("subject=" + frame.subjectType)
        add("joints=" + solved.size)
        add("edges=" + frame.edges.size)
        add("iterations=" + config.iterations)
        add("meanResidualMeters=" + format(meanResidual))
        add("maxResidualMeters=" + format(maxResidual))
        add("temporalStability=" + format(temporalStability))
        add("topologyStability=" + format(topologyStability))
        add("outliers=" + solved.count { it.outlier })
        add("predicted=" + solved.count { it.predicted })
        add("constrained=" + solved.count { it.constrained })
        add(
            "microMotion=" +
                solved.count {
                    magnitude(it.velocityMetersPerSecond) in
                        microMotionFloor..microMotionCeiling
                },
        )
    }

    private fun lerp(
        from: Vec3,
        to: Vec3,
        alpha: Double,
    ): Vec3 {
        val a = alpha.coerceIn(0.0, 1.0)
        return Vec3(
            from.x + (to.x - from.x) * a,
            from.y + (to.y - from.y) * a,
            from.z + (to.z - from.z) * a,
        )
    }

    private fun lerpScalar(
        from: Double,
        to: Double,
        alpha: Double,
    ): Double {
        val a = alpha.coerceIn(0.0, 1.0)
        return from + (to - from) * a
    }

    private fun magnitude(value: Vec3): Double =
        sqrt(
            value.x * value.x +
                value.y * value.y +
                value.z * value.z,
        )

    private fun format(value: Double): String =
        if (!value.isFinite()) {
            "nan"
        } else {
            "%.5f".format(java.util.Locale.US, value)
        }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}

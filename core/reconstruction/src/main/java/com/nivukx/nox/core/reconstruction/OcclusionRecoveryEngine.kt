package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DynamicTopology
import com.nivukx.nox.core.model.MonocularReconstructionFrame
import com.nivukx.nox.core.model.MonocularReconstructionJoint
import com.nivukx.nox.core.model.Vec3
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * High-persistence 3D occlusion recovery.
 *
 * The detector may lose a wrist, ankle, face point or entire limb during self-occlusion. Rather
 * than dropping the node immediately, this engine predicts from velocity, propagates through
 * topology constraints and decays confidence as prediction age increases. Every predicted joint
 * remains marked as predicted so downstream quality gates can decide whether to accept it.
 */
class OcclusionRecoveryEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumPredictionAgeSeconds: Double = 0.65,
        val confidenceHalfLifeSeconds: Double = 0.18,
        val velocityPredictionWeight: Double = 0.82,
        val accelerationPredictionWeight: Double = 0.58,
        val neighborConstraintWeight: Double = 0.36,
        val symmetryWeight: Double = 0.18,
        val maximumPredictionMetersPerSecond: Double = 12.0,
        val maximumAccelerationMetersPerSecond2: Double = 35.0,
        val uncertaintyGrowthMetersPerSecond: Double = 0.42,
        val minimumAcceptedConfidence: Double = 0.08,
        val maximumMissingJointHistory: Int = 512,
        val maximumMissingHistoryAgeSeconds: Double = 2.0,
    )

    data class Result(
        val frame: MonocularReconstructionFrame,
        val recoveredJointCount: Int,
        val staleJointCount: Int,
        val predictionFraction: Double,
    )

    private data class History(
        val joint: MonocularReconstructionJoint,
        val firstMissingNanos: Long,
        var lastSeenNanos: Long,
    )

    private val missing = LinkedHashMap<Int, History>()
    private var runtimeMaximumPredictionAgeSeconds =
        config.maximumPredictionAgeSeconds

    @Synchronized
    fun configureMaximumPredictionAgeFrames(
        frames: Int,
        captureFps: Int,
    ) {
        runtimeMaximumPredictionAgeSeconds =
            frames.coerceIn(0, 240).toDouble() /
                captureFps.coerceIn(1, 240).toDouble()
    }

    @Synchronized
    fun maximumPredictionAgeSeconds(): Double =
        runtimeMaximumPredictionAgeSeconds

    @Synchronized
    fun recover(
        current: MonocularReconstructionFrame,
        topology: DynamicTopology,
        previous: MonocularReconstructionFrame?,
    ): Result {
        evictExpiredHistory(current.timestampNanos)
        if (current.joints.isEmpty()) {
            return Result(
                frame = current,
                recoveredJointCount = 0,
                staleJointCount = 0,
                predictionFraction = 1.0,
            )
        }

        val currentById = current.joints.associateBy { it.landmarkId }
        val previousById = previous?.joints
            ?.associateBy { it.landmarkId }
            .orEmpty()

        val output = LinkedHashMap<Int, MonocularReconstructionJoint>()
        current.joints.forEach { output[it.landmarkId] = it }

        previousById.forEach { (id, prior) ->
            if (id !in currentById) {
                val history = missing[id]
                if (history == null) {
                    enforceHistoryCapacity()
                    missing[id] = History(
                        joint = prior,
                        firstMissingNanos = current.timestampNanos,
                        lastSeenNanos = previous?.timestampNanos
                            ?: current.timestampNanos,
                    )
                }
            }
        }

        val recovered = ArrayList<MonocularReconstructionJoint>()
        var stale = 0

        val recoveryIds =
            (topology.nodes.map { it.id } + missing.keys)
                .distinct()

        recoveryIds.forEach { id ->
            if (id in output) {
                missing.remove(id)
                return@forEach
            }

            val history = missing[id]
                ?: return@forEach

            val ageSeconds =
                (
                    current.timestampNanos -
                        history.lastSeenNanos
                ).coerceAtLeast(0L) / 1e9

            if (ageSeconds > runtimeMaximumPredictionAgeSeconds) {
                stale += 1
                return@forEach
            }

            val predictedPosition = predictPosition(
                history.joint,
                ageSeconds,
                topology,
                output,
                id,
            )

            val speed = history.joint
                .velocityCameraMetersPerSecond
                .let {
                    val norm = it.norm()
                    if (
                        norm >
                            config.maximumPredictionMetersPerSecond
                    ) {
                        it * (
                            config.maximumPredictionMetersPerSecond /
                                norm
                        )
                    } else {
                        it
                    }
                }

            val ageConfidence = exp(
                -ageSeconds /
                    config.confidenceHalfLifeSeconds
                        .coerceAtLeast(0.01),
            )

            val confidence = (
                history.joint.confidence *
                    ageConfidence *
                    config.velocityPredictionWeight +
                    (
                        topology.nodes.firstOrNull { it.id == id }?.confidence
                            ?: history.joint.confidence
                    ) *
                    config.neighborConstraintWeight
            ).coerceIn(0.0, 1.0)

            if (confidence < config.minimumAcceptedConfidence) {
                stale += 1
                return@forEach
            }

            val uncertainty =
                history.joint.uncertaintyMeters +
                    ageSeconds *
                    config.uncertaintyGrowthMetersPerSecond

            recovered += history.joint.copy(
                positionCameraMeters = predictedPosition,
                velocityCameraMetersPerSecond = speed,
                accelerationCameraMetersPerSecond2 =
                    speed * -0.04,
                confidence = confidence,
                uncertaintyMeters = uncertainty,
                predicted = true,
                depthMeters = predictedPosition.z,
                depthConfidence =
                    confidence * ageConfidence,
            )
        }

        recovered.forEach {
            output[it.landmarkId] = it
        }

        val all = output.values.toList()
        val globalConfidence =
            weightedGlobalConfidence(
                current = all,
                recoveredCount = recovered.size,
                staleCount = stale,
            )

        val predictionFraction =
            if (all.isEmpty()) 1.0
            else recovered.size.toDouble() / all.size

        val meanUncertainty =
            if (all.isEmpty()) {
                current.meanUncertaintyMeters
            } else {
                all.map { it.uncertaintyMeters }.average()
            }

        val recoveredFrame = current.copy(
            joints = all.sortedBy { it.landmarkId },
            globalConfidence = globalConfidence,
            meanUncertaintyMeters = meanUncertainty,
        )

        return Result(
            frame = recoveredFrame,
            recoveredJointCount = recovered.size,
            staleJointCount = stale,
            predictionFraction = predictionFraction,
        )
    }

    @Synchronized
    fun clear() {
        missing.clear()
        runtimeMaximumPredictionAgeSeconds =
            config.maximumPredictionAgeSeconds
    }

    private fun evictExpiredHistory(
        timestampNanos: Long,
    ) {
        if (missing.isEmpty()) return

        val maximumAgeNanos =
            (
                config.maximumMissingHistoryAgeSeconds *
                    1e9
            ).toLong().coerceAtLeast(1L)
        val threshold =
            timestampNanos - maximumAgeNanos

        val iterator = missing.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.value.lastSeenNanos < threshold) {
                iterator.remove()
            }
        }
    }

    private fun enforceHistoryCapacity() {
        if (missing.size < config.maximumMissingJointHistory) {
            return
        }

        val overflow =
            missing.size -
                config.maximumMissingJointHistory +
                1

        missing.entries
            .sortedBy { it.value.lastSeenNanos }
            .take(overflow.coerceAtLeast(1))
            .map { it.key }
            .forEach(missing::remove)
    }

    private fun predictPosition(
        history: MonocularReconstructionJoint,
        ageSeconds: Double,
        topology: DynamicTopology,
        observed: Map<Int, MonocularReconstructionJoint>,
        id: Int,
    ): Vec3 {
        val dt =
            ageSeconds.coerceIn(
                0.0,
                config.maximumPredictionAgeSeconds,
            )

        val velocity =
            boundedVector(
                value = history.velocityCameraMetersPerSecond,
                maximumNorm =
                    config.maximumPredictionMetersPerSecond,
            )

        val acceleration =
            boundedVector(
                value = history.accelerationCameraMetersPerSecond2,
                maximumNorm =
                    config.maximumAccelerationMetersPerSecond2,
            )

        var predicted =
            history.positionCameraMeters +
                velocity * dt +
                acceleration *
                    (
                        0.5 *
                            dt *
                            dt *
                            config.accelerationPredictionWeight
                    )

        val neighbors = topology.edges
            .filter { it.a == id || it.b == id }
            .mapNotNull { edge ->
                val otherId =
                    if (edge.a == id) edge.b else edge.a
                observed[otherId]?.let {
                    edge to it
                }
            }

        if (neighbors.isNotEmpty()) {
            var correction = Vec3(
                0.0,
                0.0,
                0.0,
            )
            var weightSum = 0.0

            neighbors.forEach { (edge, neighbor) ->
                val fromNeighbor =
                    neighbor.positionCameraMeters +
                        directionToward(
                            predicted,
                            neighbor.positionCameraMeters,
                            edge.preferredLengthMeters,
                        )

                val weight =
                    edge.stiffness.coerceAtLeast(0.05)

                correction += (
                    fromNeighbor - predicted
                ) * weight
                weightSum += weight
            }

            if (weightSum > 1e-6) {
                predicted += (
                    correction / weightSum
                ) * config.neighborConstraintWeight
            }
        }

        val symmetric = symmetryAnchor(
            id = id,
            topology = topology,
            observed = observed,
        )

        if (symmetric != null) {
            predicted =
                predicted.lerp(
                    symmetric,
                    config.symmetryWeight.coerceIn(0.0, 1.0),
                )
        }

        return predicted
    }

    private fun boundedVector(
        value: Vec3,
        maximumNorm: Double,
    ): Vec3 {
        val norm = value.norm()
        if (!norm.isFinite() || norm <= maximumNorm) {
            return value
        }
        return value *
            (maximumNorm / norm.coerceAtLeast(1e-9))
    }

    private fun directionToward(
        point: Vec3,
        neighbor: Vec3,
        desiredLength: Double,
    ): Vec3 {
        val direction =
            (point - neighbor)

        val length =
            direction.norm()

        if (length < 1e-6) {
            return Vec3(
                0.0,
                0.0,
                desiredLength,
            )
        }

        return direction.normalized() *
            desiredLength
    }

    private fun symmetryAnchor(
        id: Int,
        topology: DynamicTopology,
        observed: Map<Int, MonocularReconstructionJoint>,
    ): Vec3? {
        val mirror =
            when (id) {
                11 -> 12
                12 -> 11
                13 -> 14
                14 -> 13
                15 -> 16
                16 -> 15
                23 -> 24
                24 -> 23
                25 -> 26
                26 -> 25
                27 -> 28
                28 -> 27
                else -> null
            } ?: return null

        val mirrorJoint = observed[mirror]
            ?: return null

        val rootLeft = observed[23]
            ?: return null
        val rootRight = observed[24]
            ?: return null

        val mid =
            rootLeft.positionCameraMeters
                .lerp(
                    rootRight.positionCameraMeters,
                    0.5,
                )

        val relative =
            mirrorJoint.positionCameraMeters -
                mid

        return mid + Vec3(
            -relative.x,
            relative.y,
            relative.z,
        )
    }

    private fun weightedGlobalConfidence(
        current: List<MonocularReconstructionJoint>,
        recoveredCount: Int,
        staleCount: Int,
    ): Double {
        if (current.isEmpty()) return 0.0

        val average =
            current.map { it.confidence }.average()

        val recoveryPenalty =
            (
                recoveredCount.toDouble() /
                    current.size.coerceAtLeast(1)
            ) * 0.14

        val stalePenalty =
            min(
                0.20,
                staleCount.toDouble() * 0.02,
            )

        return max(
            0.0,
            average -
                recoveryPenalty -
                stalePenalty,
        )
    }
}

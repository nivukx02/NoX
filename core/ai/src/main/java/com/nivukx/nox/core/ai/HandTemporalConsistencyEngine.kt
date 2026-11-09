package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Hand3DFrame
import com.nivukx.nox.core.model.Hand3DResult
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp

/**
 * Temporal hand-articulation consistency gate.
 *
 * Multi-view hand geometry is especially sensitive to tiny correspondence errors because a one
 * landmark jump can produce a large visible finger rotation. This engine tracks 3D landmark motion,
 * evaluates velocity continuity, articulation-energy continuity, uncertainty growth and per-hand
 * identity stability, then returns a bounded confidence surface for recording/export.
 */
class HandTemporalConsistencyEngine(
    config: Config = Config(),
) {
    @Volatile
    private var config: Config = config


    data class Config(
        val historyLimit: Int = 16,
        val maximumDeltaSeconds: Double = 0.12,
        val innovationGateMeters: Double = 0.050,
        val hardInnovationMeters: Double = 0.22,
        val velocityScaleMetersPerSecond: Double = 2.50,
        val accelerationScaleMetersPerSecond2: Double = 40.0,
        val uncertaintyScaleMeters: Double = 0.10,
        val minimumConfidence: Double = 0.18,
        val targetConfidence: Double = 0.82,
        val smoothing: Double = 0.72,
        val articulationChangeScale: Double = 1.50,
        val perHandWeight: Double = 1.0,
    ) {
        init {
            require(historyLimit >= 4)
            require(maximumDeltaSeconds > 0.0)
            require(innovationGateMeters > 0.0)
            require(hardInnovationMeters > innovationGateMeters)
            require(velocityScaleMetersPerSecond > 0.0)
            require(accelerationScaleMetersPerSecond2 > 0.0)
            require(uncertaintyScaleMeters > 0.0)
            require(minimumConfidence in 0.0..1.0)
            require(targetConfidence in 0.0..1.0)
            require(targetConfidence >= minimumConfidence)
            require(smoothing in 0.0..1.0)
            require(articulationChangeScale > 0.0)
            require(perHandWeight > 0.0)
        }
    }

    data class HandDiagnostics(
        val handId: Long,
        val confidence: Double,
        val innovationMeters: Double,
        val meanVelocityMetersPerSecond: Double,
        val meanAccelerationMetersPerSecond2: Double,
        val meanUncertaintyMeters: Double,
        val temporalStability: Double,
        val rejectedLandmarks: Int,
        val warnings: List<String>,
    )

    data class Result(
        val timestampNanos: Long,
        val confidence: Double,
        val temporalStability: Double,
        val articulationStability: Double,
        val identityStability: Double,
        val rejectedLandmarks: Int,
        val hands: List<HandDiagnostics>,
        val warnings: List<String>,
    ) {
        val reliable: Boolean
            get() =
                confidence >= 0.55 &&
                    temporalStability >= 0.40 &&
                    warnings.none { it.startsWith("fatal:") }
    }

    private data class LandmarkState(
        var timestampNanos: Long,
        var position: Vec3,
        var velocity: Vec3,
        var acceleration: Vec3,
        var confidence: Double,
    )

    private data class HandState(
        var timestampNanos: Long,
        var centroid: Vec3,
        var confidence: Double,
        var articulationEnergy: Double,
    )

    private val landmarks = HashMap<Pair<Long, Int>, LandmarkState>()
    private val hands = HashMap<Long, HandState>()

    @Synchronized
    fun evaluate(frame: Hand3DFrame): Result {
        if (frame.hands.isEmpty()) {
            prune(frame.timestampNanos)
            return Result(
                timestampNanos = frame.timestampNanos,
                confidence = 0.0,
                temporalStability = 0.0,
                articulationStability = 0.0,
                identityStability = 0.0,
                rejectedLandmarks = 0,
                hands = emptyList(),
                warnings = listOf("hand reconstruction unavailable"),
            )
        }

        var totalWeight = 0.0
        var confidenceAccumulator = 0.0
        var temporalAccumulator = 0.0
        var articulationAccumulator = 0.0
        var rejectedTotal = 0

        val diagnostics = frame.hands.map { hand ->
            val previousHandState = hands[hand.handId]
            val result = evaluateHand(hand)
            val weight =
                config.perHandWeight *
                    hand.confidence.coerceIn(0.05, 1.0)

            totalWeight += weight
            confidenceAccumulator += result.confidence * weight
            temporalAccumulator += result.temporalStability * weight
            articulationAccumulator +=
                articulationStability(
                    hand = hand,
                    previous = previousHandState,
                ) * weight
            rejectedTotal += result.rejectedLandmarks

            result
        }

        val baseConfidence =
            if (totalWeight <= 1e-9) {
                0.0
            } else {
                confidenceAccumulator / totalWeight
            }

        val temporalStability =
            if (totalWeight <= 1e-9) {
                0.0
            } else {
                temporalAccumulator / totalWeight
            }

        val articulationStability =
            if (totalWeight <= 1e-9) {
                0.0
            } else {
                articulationAccumulator / totalWeight
            }

        val identityStability =
            identityStability(frame)

        val warnings =
            buildList {
                if (baseConfidence < 0.45) {
                    add("hand confidence below production target")
                }
                if (temporalStability < 0.45) {
                    add("hand temporal stability below production target")
                }
                if (articulationStability < 0.45) {
                    add("finger articulation continuity reduced")
                }
                if (identityStability < 0.50) {
                    add("hand identity continuity is uncertain")
                }
                if (rejectedTotal >
                    maxOf(
                        1,
                        frame.hands.sumOf { it.landmarks.size } / 4,
                    )
                ) {
                    add("many hand landmarks were treated as temporal outliers")
                }
            }

        val confidence =
            (
                baseConfidence * 0.48 +
                    temporalStability * 0.24 +
                    articulationStability * 0.18 +
                    identityStability * 0.10
                ).coerceIn(0.0, 1.0)

        prune(frame.timestampNanos)

        return Result(
            timestampNanos = frame.timestampNanos,
            confidence = confidence,
            temporalStability = temporalStability,
            articulationStability = articulationStability,
            identityStability = identityStability,
            rejectedLandmarks = rejectedTotal,
            hands = diagnostics,
            warnings = warnings.distinct(),
        )
    }

    @Synchronized
    fun configureStrictness(strictness: Double) {
        val normalized = strictness.coerceIn(0.0, 1.0)
        config = Config(
            historyLimit = 16,
            maximumDeltaSeconds = 0.12,
            innovationGateMeters =
                0.075 - normalized * 0.030,
            hardInnovationMeters =
                0.28 - normalized * 0.090,
            velocityScaleMetersPerSecond =
                2.90 - normalized * 0.55,
            accelerationScaleMetersPerSecond2 =
                52.0 - normalized * 18.0,
            uncertaintyScaleMeters =
                0.13 - normalized * 0.035,
            minimumConfidence =
                0.12 + normalized * 0.12,
            targetConfidence =
                0.74 + normalized * 0.12,
            smoothing =
                0.82 - normalized * 0.16,
            articulationChangeScale =
                1.80 - normalized * 0.35,
            perHandWeight = 1.0,
        )
        landmarks.clear()
        hands.clear()
    }

    @Synchronized
    fun reset() {
        landmarks.clear()
        hands.clear()
    }

    private fun evaluateHand(
        hand: Hand3DResult,
    ): HandDiagnostics {
        var innovationAccumulator = 0.0
        var innovationWeight = 0.0
        var velocityAccumulator = 0.0
        var accelerationAccumulator = 0.0
        var uncertaintyAccumulator = 0.0
        var rejected = 0

        hand.landmarks.forEach { landmark ->
            val key = hand.handId to landmark.index
            val previous = landmarks[key]

            val dt =
                if (previous == null) {
                    1.0 / 60.0
                } else {
                    (
                        (hand.timestampNanos - previous.timestampNanos)
                            .coerceAtLeast(1L)
                            .toDouble() /
                            1e9
                    ).coerceIn(
                        1.0 / 240.0,
                        config.maximumDeltaSeconds,
                    )
                }

            val predicted =
                if (previous == null) {
                    landmark.position
                } else {
                    previous.position +
                        previous.velocity * dt +
                        previous.acceleration * (
                            0.5 * dt * dt
                        )
                }

            val innovation =
                landmark.position.distanceTo(predicted)

            val confidence =
                landmark.confidence.coerceIn(
                    0.0,
                    1.0,
                )

            val uncertainty =
                landmark.uncertaintyMeters
                    .coerceAtLeast(0.0)

            val hardOutlier =
                previous != null &&
                    innovation >
                    config.hardInnovationMeters &&
                    confidence <
                    0.65

            val measurementWeight =
                (
                    confidence *
                        exp(
                            -uncertainty /
                                config.uncertaintyScaleMeters,
                        )
                    ).coerceIn(
                        0.05,
                        1.0,
                    )

            val innovationWeightValue =
                (
                    1.0 -
                        innovation /
                            config.hardInnovationMeters
                    ).coerceIn(
                        0.0,
                        1.0,
                    )

            if (hardOutlier) {
                rejected += 1
            }

            val corrected =
                if (hardOutlier && previous != null) {
                    predicted
                } else if (previous == null) {
                    landmark.position
                } else {
                    predicted.lerp(
                        landmark.position,
                        (
                            config.smoothing *
                                measurementWeight
                                ).coerceIn(
                                    0.08,
                                    0.90,
                                ),
                    )
                }

            val velocity =
                if (previous == null) {
                    Vec3(0.0, 0.0, 0.0)
                } else {
                    (
                        corrected -
                            previous.position
                        ) / dt
                }

            val measuredAcceleration =
                if (previous == null) {
                    Vec3(0.0, 0.0, 0.0)
                } else {
                    (
                        velocity -
                            previous.velocity
                    ) / dt
                }

            val acceleration =
                if (previous == null) {
                    Vec3(0.0, 0.0, 0.0)
                } else {
                    previous.acceleration.lerp(
                        measuredAcceleration.limitNorm(
                            config.accelerationScaleMetersPerSecond2,
                        ),
                        0.28,
                    )
                }

            landmarks[key] =
                LandmarkState(
                    timestampNanos = hand.timestampNanos,
                    position = corrected,
                    velocity = velocity,
                    acceleration = acceleration,
                    confidence =
                        if (hardOutlier) {
                            previous?.confidence?.times(0.82) ?: confidence
                        } else {
                            confidence
                        },
                )

            innovationAccumulator +=
                minOf(
                    innovation,
                    config.hardInnovationMeters * 2.0,
                ) *
                measurementWeight
            innovationWeight += measurementWeight

            velocityAccumulator +=
                velocity.norm() *
                    measurementWeight

            accelerationAccumulator +=
                acceleration.norm() *
                    measurementWeight

            uncertaintyAccumulator +=
                uncertainty *
                    measurementWeight
        }

        val meanInnovation =
            if (innovationWeight <= 1e-9) {
                0.0
            } else {
                innovationAccumulator /
                    innovationWeight
            }

        val meanVelocity =
            if (innovationWeight <= 1e-9) {
                0.0
            } else {
                velocityAccumulator /
                    innovationWeight
            }

        val meanAcceleration =
            if (innovationWeight <= 1e-9) {
                0.0
            } else {
                accelerationAccumulator /
                    innovationWeight
            }

        val meanUncertainty =
            if (innovationWeight <= 1e-9) {
                0.0
            } else {
                uncertaintyAccumulator /
                    innovationWeight
            }

        val temporalStability =
            (
                exp(
                    -meanInnovation /
                        config.innovationGateMeters,
                ) * 0.52 +
                    (
                        1.0 -
                            meanVelocity /
                                config.velocityScaleMetersPerSecond
                        ).coerceIn(0.0, 1.0) * 0.20 +
                    (
                        1.0 -
                            meanAcceleration /
                                config.accelerationScaleMetersPerSecond2
                        ).coerceIn(0.0, 1.0) * 0.09 +
                    (
                        1.0 -
                            meanUncertainty /
                                config.uncertaintyScaleMeters
                        ).coerceIn(0.0, 1.0) * 0.15 +
                    hand.confidence * 0.10
                ).coerceIn(0.0, 1.0)

        val confidence =
            (
                hand.confidence * 0.48 +
                    temporalStability * 0.35 +
                    (
                        1.0 -
                            meanUncertainty /
                                (
                                    config.uncertaintyScaleMeters *
                                        2.0
                                    )
                        ).coerceIn(0.0, 1.0) * 0.17
                ).coerceIn(0.0, 1.0)

        hands[hand.handId] =
            HandState(
                timestampNanos = hand.timestampNanos,
                centroid = centroid(hand),
                confidence = confidence,
                articulationEnergy = articulationEnergy(hand),
            )

        val warnings =
            buildList {
                if (meanInnovation >
                    config.innovationGateMeters * 1.5
                ) {
                    add("high temporal innovation")
                }
                if (meanUncertainty >
                    config.uncertaintyScaleMeters
                ) {
                    add("high landmark uncertainty")
                }
                if (rejected > 0) {
                    add("$rejected landmark outlier(s)")
                }
            }

        return HandDiagnostics(
            handId = hand.handId,
            confidence = confidence,
            innovationMeters = meanInnovation,
            meanVelocityMetersPerSecond = meanVelocity,
            meanAccelerationMetersPerSecond2 = meanAcceleration,
            meanUncertaintyMeters = meanUncertainty,
            temporalStability = temporalStability,
            rejectedLandmarks = rejected,
            warnings = warnings,
        )
    }

    private fun articulationStability(
        hand: Hand3DResult,
        previous: HandState?,
    ): Double {
        if (previous == null) return 0.70

        val currentEnergy =
            articulationEnergy(hand)

        val delta =
            abs(
                currentEnergy -
                    previous.articulationEnergy,
            )

        return (
            1.0 -
                delta /
                    config.articulationChangeScale
            ).coerceIn(0.0, 1.0)
    }

    private fun identityStability(
        frame: Hand3DFrame,
    ): Double {
        if (frame.hands.isEmpty()) return 0.0

        val currentIds =
            frame.hands.map { it.handId }.toSet()

        if (hands.isEmpty()) return 0.70

        val retained =
            currentIds.count {
                hands.containsKey(it)
            }

        return (
            retained.toDouble() /
                currentIds.size.coerceAtLeast(1)
        ).coerceIn(0.0, 1.0)
    }

    private fun centroid(
        hand: Hand3DResult,
    ): Vec3 {
        if (hand.landmarks.isEmpty()) {
            return Vec3(0.0, 0.0, 0.0)
        }

        var sum = Vec3(0.0, 0.0, 0.0)
        var weight = 0.0

        hand.landmarks.forEach { landmark ->
            val w =
                landmark.confidence.coerceIn(
                    0.05,
                    1.0,
                )
            sum += landmark.position * w
            weight += w
        }

        return sum /
            weight.coerceAtLeast(
                1e-6,
            )
    }

    private fun articulationEnergy(
        hand: Hand3DResult,
    ): Double {
        if (hand.landmarks.size < 2) return 0.0

        val mean =
            hand.landmarks
                .zipWithNext()
                .map {
                    it.first.position
                        .distanceTo(it.second.position)
                }
                .average()

        return (
            mean /
                config.articulationChangeScale
            ).coerceIn(0.0, 1.0)
    }

    private fun prune(
        nowNanos: Long,
    ) {
        if (landmarks.size <=
            config.historyLimit * 128 &&
            hands.size <= config.historyLimit * 8
        ) {
            return
        }

        val landmarkCutoff =
            nowNanos -
                (
                    config.historyLimit *
                        config.maximumDeltaSeconds *
                        1_000_000_000.0
                    ).toLong()

        landmarks.entries.removeIf {
            it.value.timestampNanos <
                landmarkCutoff
        }

        hands.entries.removeIf {
            it.value.timestampNanos <
                landmarkCutoff
        }
    }

    private fun Vec3.limitNorm(maximum: Double): Vec3 {
        val norm = norm()
        if (!norm.isFinite() || maximum <= 0.0) {
            return Vec3(0.0, 0.0, 0.0)
        }
        if (norm <= maximum) return this
        val scale = maximum / norm
        return Vec3(
            x * scale,
            y * scale,
            z * scale,
        )
    }

    private fun Vec3.lerp(
        target: Vec3,
        alpha: Double,
    ): Vec3 =
        Vec3(
            x * (1.0 - alpha) +
                target.x * alpha,
            y * (1.0 - alpha) +
                target.y * alpha,
            z * (1.0 - alpha) +
                target.z * alpha,
        )
}

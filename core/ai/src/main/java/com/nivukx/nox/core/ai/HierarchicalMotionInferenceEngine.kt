package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.model.WorldSpaceProposal
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Hierarchical world-space motion estimator.
 *
 * Production behavior:
 *  - observed multi-view geometry remains authoritative when its evidence is strong;
 *  - the temporal model predicts a bounded position from velocity, acceleration and jerk;
 *  - new measurements are admitted through a robust innovation gate instead of replacing a stable
 *    trajectory blindly;
 *  - prediction-only recovery is explicitly marked and confidence decays with time;
 *  - all track state is bounded so crowd-scale capture cannot grow memory without limit.
 */
class HierarchicalMotionInferenceEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        val maximumTracks: Int = 4_096,
        val maximumHypothesesPerTrack: Int = 3,
        val maximumPredictionGapNanos: Long = 180_000_000L,
        val maximumSpeedMetersPerSecond: Double = 8.0,
        val maximumAccelerationMetersPerSecond2: Double = 80.0,
        val maximumJerkMetersPerSecond3: Double = 400.0,
        val velocityBlend: Double = 0.72,
        val accelerationBlend: Double = 0.60,
        val confidenceHalfLifeMillis: Double = 160.0,
        /** Innovation threshold, expressed in robust model sigma. */
        val innovationGateSigma: Double = 4.5,
        /** Minimum contribution of a fresh measurement before robust down-weighting. */
        val minimumMeasurementBlend: Double = 0.35,
    ) {
        init {
            require(maximumTracks > 0)
            require(maximumHypothesesPerTrack in 1..8)
            require(maximumPredictionGapNanos > 0L)
            require(maximumSpeedMetersPerSecond > 0.0)
            require(maximumAccelerationMetersPerSecond2 > 0.0)
            require(maximumJerkMetersPerSecond3 > 0.0)
            require(velocityBlend in 0.0..1.0)
            require(accelerationBlend in 0.0..1.0)
            require(confidenceHalfLifeMillis > 0.0)
            require(innovationGateSigma in 1.0..12.0)
            require(minimumMeasurementBlend in 0.05..0.95)
        }
    }

    data class State(
        val actorId: Int,
        val pointId: Int,
        val timestampNanos: Long,
        val positionMeters: Vec3,
        val velocityMetersPerSecond: Vec3,
        val accelerationMetersPerSecond2: Vec3,
        val jerkMetersPerSecond3: Vec3,
        val confidence: Double,
        val uncertaintyMeters: Double,
        val origin: EvidenceOrigin,
        val hypothesisRank: Int,
        val sourceObservationCount: Int,
        /** Normalized model-vs-measurement innovation diagnostic. */
        val innovationSigmaMeters: Double = 0.0,
        /** Robust innovation likelihood, 1 is a strong model-consistent update. */
        val innovationScore: Double = 1.0,
        /** Fraction of the incoming measured position admitted into the state. */
        val measurementBlend: Double = 1.0,
    )

    data class Result(
        val timestampNanos: Long,
        val states: List<State>,
        val recoveredTrackCount: Int,
        val activeTrackCount: Int,
        val droppedProposalCount: Int,
        val warnings: List<String>,
        val meanInnovationScore: Double = 1.0,
        val robustOutlierCount: Int = 0,
    )

    private data class Track(
        val actorId: Int,
        val pointId: Int,
        var timestampNanos: Long,
        var position: Vec3,
        var velocity: Vec3,
        var acceleration: Vec3,
        var jerk: Vec3,
        var confidence: Double,
        var uncertaintyMeters: Double,
    )

    private val tracks = LinkedHashMap<Long, Track>()
    private var closed = false

    @Synchronized
    fun process(
        timestampNanos: Long,
        proposals: List<WorldSpaceProposal>,
    ): Result {
        check(!closed) { "HierarchicalMotionInferenceEngine is closed" }
        require(timestampNanos >= 0L)
        while (tracks.size > config.maximumTracks) {
            pruneOldest()
        }

        val grouped =
            proposals
                .filter {
                    it.timestampNanos == timestampNanos &&
                        finiteProposal(it)
                }
                .groupBy {
                    key(it.actorId, it.pointId)
                }

        var dropped =
            proposals.count {
                it.timestampNanos != timestampNanos ||
                    !finiteProposal(it)
            }
        val outputs =
            ArrayList<State>(
                grouped.size +
                    min(
                        tracks.size,
                        config.maximumTracks,
                    ),
            )
        var recovered = 0
        var innovationScoreSum = 0.0
        var innovationCount = 0
        var robustOutlierCount = 0

        for ((trackKey, group) in grouped) {
            val fused = fuse(group)
            val actorId = (trackKey shr 32).toInt()
            val pointId = trackKey.toInt()
            val track = tracks[trackKey]

            if (track == null) {
                val created =
                    Track(
                        actorId = actorId,
                        pointId = pointId,
                        timestampNanos = timestampNanos,
                        position = fused.position,
                        velocity = Vec3(0.0, 0.0, 0.0),
                        acceleration = Vec3(0.0, 0.0, 0.0),
                        jerk = Vec3(0.0, 0.0, 0.0),
                        confidence = fused.confidence,
                        uncertaintyMeters =
                            fused.uncertaintyMeters
                                .coerceIn(0.001, 2.0),
                    )
                tracks[trackKey] = created
                outputs +=
                    state(
                        track = created,
                        origin =
                            EvidenceOrigin.GEOMETRICALLY_CONSTRAINED,
                        rank = 0,
                        count = group.size,
                        innovationSigmaMeters = 0.0,
                        innovationScore = 1.0,
                        measurementBlend = 1.0,
                    )
                continue
            }

            val dt =
                (
                    timestampNanos -
                        track.timestampNanos
                    ).toDouble() / 1e9

            if (dt <= 0.0) {
                dropped += group.size
                outputs +=
                    state(
                        track = track,
                        origin = EvidenceOrigin.TEMPORALLY_PROPAGATED,
                        rank = 0,
                        count = 0,
                        innovationSigmaMeters =
                            track.uncertaintyMeters,
                        innovationScore = 1.0,
                        measurementBlend = 0.0,
                    )
                continue
            }

            if (dt * 1e9 > config.maximumPredictionGapNanos) {
                track.position = fused.position
                track.velocity =
                    Vec3(0.0, 0.0, 0.0)
                track.acceleration =
                    Vec3(0.0, 0.0, 0.0)
                track.jerk =
                    Vec3(0.0, 0.0, 0.0)
                track.timestampNanos = timestampNanos
                track.confidence =
                    (
                        fused.confidence *
                            0.75
                        ).coerceIn(0.0, 1.0)
                track.uncertaintyMeters =
                    fused.uncertaintyMeters
                        .coerceIn(0.001, 2.0)
                outputs +=
                    state(
                        track = track,
                        origin =
                            EvidenceOrigin.GEOMETRICALLY_CONSTRAINED,
                        rank = 0,
                        count = group.size,
                        innovationSigmaMeters =
                            track.uncertaintyMeters,
                        innovationScore = 1.0,
                        measurementBlend = 1.0,
                    )
                continue
            }

            val predicted =
                predict(
                    track = track,
                    dt = dt,
                )

            val innovation =
                fused.position -
                    predicted
            val innovationMagnitude =
                innovation.norm()
            val dynamicAllowance =
                max(
                    0.003,
                    config.maximumSpeedMetersPerSecond *
                        dt *
                        0.10 +
                        config.maximumAccelerationMetersPerSecond2 *
                            dt *
                            dt *
                            0.05,
                )
            val innovationSigma =
                max(
                    0.003,
                    (
                        track.uncertaintyMeters +
                            fused.uncertaintyMeters +
                            dynamicAllowance +
                            0.5 *
                                track.acceleration.norm() *
                                dt *
                                dt +
                            (1.0 / 6.0) *
                                track.jerk.norm() *
                                dt *
                                dt *
                                dt
                        ).coerceAtMost(2.0),
                )

            val normalizedInnovation =
                innovationMagnitude /
                    innovationSigma.coerceAtLeast(1e-6)
            val innovationScore =
                exp(
                    -0.5 *
                        (
                            normalizedInnovation /
                                config.innovationGateSigma
                        ) *
                        (
                            normalizedInnovation /
                                config.innovationGateSigma
                        ),
                ).coerceIn(0.0, 1.0)

            val normalizedByGate =
                normalizedInnovation /
                    config.innovationGateSigma
            val robustWeight =
                if (
                    normalizedInnovation <=
                        config.innovationGateSigma
                ) {
                    1.0
                } else {
                    1.0 /
                        (
                            1.0 +
                                normalizedByGate *
                                    normalizedByGate
                            )
                }

            val measurementBlend =
                (
                    config.minimumMeasurementBlend +
                        (
                            1.0 -
                                config.minimumMeasurementBlend
                            ) *
                            fused.confidence
                    ) *
                    robustWeight
                ).coerceIn(
                    0.05,
                    1.0,
                )

            if (robustWeight < 0.35) {
                robustOutlierCount += 1
            }

            innovationScoreSum += innovationScore
            innovationCount += 1

            val correctedPosition =
                predicted +
                    innovation *
                    measurementBlend

            val measuredVelocity =
                (
                    correctedPosition -
                        track.position
                    ) / dt
            val boundedVelocity =
                measuredVelocity.clampNorm(
                    config.maximumSpeedMetersPerSecond,
                )
            val measuredAcceleration =
                (
                    boundedVelocity -
                        track.velocity
                    ) / dt
            val boundedAcceleration =
                measuredAcceleration.clampNorm(
                    config.maximumAccelerationMetersPerSecond2,
                )
            val newAcceleration =
                track.acceleration.lerp(
                    boundedAcceleration,
                    config.accelerationBlend,
                )
            val measuredJerk =
                (
                    newAcceleration -
                        track.acceleration
                    ) / dt
            val newJerk =
                measuredJerk.clampNorm(
                    config.maximumJerkMetersPerSecond3,
                )

            track.velocity =
                track.velocity
                    .lerp(
                        boundedVelocity,
                        config.velocityBlend,
                    )
                    .clampNorm(
                        config.maximumSpeedMetersPerSecond,
                    )
            track.acceleration =
                newAcceleration
            track.jerk =
                newJerk
            track.position =
                correctedPosition
            track.timestampNanos =
                timestampNanos
            track.confidence =
                (
                    0.60 *
                        fused.confidence +
                        0.40 *
                        track.confidence
                    ).coerceIn(0.0, 1.0)

            val residualUncertainty =
                innovationMagnitude *
                    (
                        1.0 -
                            measurementBlend
                    ) *
                    0.25
            track.uncertaintyMeters =
                max(
                    0.001,
                    min(
                        2.0,
                        max(
                            fused.uncertaintyMeters,
                            residualUncertainty +
                                fused.uncertaintyMeters *
                                    (
                                        1.0 +
                                            (
                                                1.0 -
                                                    innovationScore
                                            ) *
                                            0.50
                                    ),
                        ),
                    ),
                )

            outputs +=
                state(
                    track = track,
                    origin = EvidenceOrigin.OBSERVED,
                    rank = 0,
                    count = group.size,
                    innovationSigmaMeters =
                        innovationSigma,
                    innovationScore =
                        innovationScore,
                    measurementBlend =
                        measurementBlend,
                )
        }

        val missingKeys =
            tracks.keys
                .filter {
                    it !in grouped.keys
                }
                .take(config.maximumTracks)

        for (trackKey in missingKeys) {
            val track =
                tracks[trackKey]
                    ?: continue
            val dt =
                (
                    timestampNanos -
                        track.timestampNanos
                    ).toDouble() / 1e9

            if (dt <= 0.0) continue
            if (
                dt * 1e9 >
                    config.maximumPredictionGapNanos
            ) {
                continue
            }

            val decay =
                exp(
                    -dt *
                        1000.0 /
                        config.confidenceHalfLifeMillis,
                )
            val predicted =
                predict(
                    track = track,
                    dt = dt,
                ).clampStep(
                    track.position,
                    config.maximumSpeedMetersPerSecond *
                        dt,
                )
            val originalPosition =
                track.position

            track.position = predicted
            track.timestampNanos =
                timestampNanos
            track.confidence =
                (
                    track.confidence *
                        decay *
                        0.92
                    ).coerceIn(
                        0.0,
                        1.0,
                    )
            track.uncertaintyMeters =
                (
                    track.uncertaintyMeters +
                        track.velocity.norm() *
                        dt *
                        0.10
                    ).coerceAtMost(2.0)

            recovered += 1
            outputs +=
                state(
                    track = track,
                    origin =
                        EvidenceOrigin.TEMPORALLY_PROPAGATED,
                    rank = 0,
                    count = 0,
                    innovationSigmaMeters =
                        track.uncertaintyMeters,
                    innovationScore =
                        decay.coerceIn(
                            0.0,
                            1.0,
                        ),
                    measurementBlend = 0.0,
                )

            for (
                rank in 1 until
                    config.maximumHypothesesPerTrack
            ) {
                val sign =
                    if (rank % 2 == 0) {
                        -1.0
                    } else {
                        1.0
                    }
                val hypothesisOffset =
                    track.acceleration.normalized() *
                        sign *
                        min(
                            0.05 * rank,
                            track.uncertaintyMeters,
                        )

                outputs +=
                    state(
                        track =
                            track.copy(
                                position =
                                    predicted +
                                        hypothesisOffset,
                                confidence =
                                    (
                                        track.confidence *
                                            (
                                                1.0 -
                                                    rank *
                                                    0.18
                                            )
                                        ).coerceAtLeast(
                                            0.0,
                                        ),
                                uncertaintyMeters =
                                    track.uncertaintyMeters +
                                        0.01 * rank,
                            ),
                        origin =
                            EvidenceOrigin.TEMPORALLY_PROPAGATED,
                        rank = rank,
                        count = 0,
                        innovationSigmaMeters =
                            track.uncertaintyMeters +
                                0.01 * rank,
                        innovationScore =
                            decay.coerceIn(
                                0.0,
                                1.0,
                            ),
                        measurementBlend = 0.0,
                    )
            }

            if (
                predicted.distanceTo(
                    originalPosition,
                ) >
                    config.maximumSpeedMetersPerSecond *
                        dt +
                    1e-6
            ) {
                dropped += 1
            }
        }

        val meanInnovationScore =
            if (innovationCount == 0) {
                1.0
            } else {
                (
                    innovationScoreSum /
                        innovationCount.toDouble()
                    ).coerceIn(0.0, 1.0)
            }

        val warnings =
            buildList {
                if (recovered > 0) {
                    add("temporal-recovery-active")
                }
                if (robustOutlierCount > 0) {
                    add(
                        "robust-innovation-downweighting-active",
                    )
                }
                if (
                    meanInnovationScore < 0.55 &&
                    innovationCount > 0
                ) {
                    add(
                        "motion-model-measurement-disagreement-high",
                    )
                }
            }

        return Result(
            timestampNanos = timestampNanos,
            states =
                outputs.sortedWith(
                    compareBy<State> {
                        it.actorId
                    }
                        .thenBy { it.pointId }
                        .thenBy { it.hypothesisRank },
                ),
            recoveredTrackCount = recovered,
            activeTrackCount = tracks.size,
            droppedProposalCount = dropped,
            warnings = warnings,
            meanInnovationScore =
                meanInnovationScore,
            robustOutlierCount =
                robustOutlierCount,
        )
    }

    private data class Fused(
        val position: Vec3,
        val confidence: Double,
        val uncertaintyMeters: Double,
    )

    private fun predict(
        track: Track,
        dt: Double,
    ): Vec3 =
        (
            track.position +
                track.velocity * dt +
                track.acceleration *
                    (0.5 * dt * dt) +
                track.jerk *
                    (
                        (1.0 / 6.0) *
                            dt *
                            dt *
                            dt
                    )
            ).clampStep(
                track.position,
                config.maximumSpeedMetersPerSecond *
                    dt +
                    0.5 *
                        config.maximumAccelerationMetersPerSecond2 *
                        dt *
                        dt,
            )

    private fun fuse(
        group: List<WorldSpaceProposal>,
    ): Fused {
        var weightSum = 0.0
        var position =
            Vec3(0.0, 0.0, 0.0)
        var confidence = 0.0

        for (proposal in group) {
            val denominator =
                max(
                    proposal.uncertainty.xxMeters2 +
                        proposal.uncertainty.yyMeters2 +
                        proposal.uncertainty.zzMeters2,
                    1e-9,
                )
            val cameraSupport =
                (
                    proposal.independentCameraCount.toDouble() /
                        4.0
                ).coerceIn(0.25, 1.0)
            val evidenceAuthority =
                (
                    proposal.geometricSupport.coerceIn(0.0, 1.0) *
                        0.42 +
                        proposal.temporalSupport.coerceIn(0.0, 1.0) *
                        0.18 +
                        proposal.viewDiversity.coerceIn(0.0, 1.0) *
                        0.22 +
                        cameraSupport * 0.18
                    ).coerceIn(0.0, 1.0)
            val weight =
                max(
                    1e-6,
                    proposal.confidence *
                        (
                            0.35 +
                                0.65 *
                                    evidenceAuthority
                        ) /
                        denominator,
                )

            position +=
                proposal.positionMeters *
                    weight
            weightSum += weight
            confidence +=
                proposal.confidence *
                    (
                        0.55 +
                            0.45 *
                                evidenceAuthority
                    ) *
                    weight
        }

        val safeWeight =
            weightSum.coerceAtLeast(1e-9)

        return Fused(
            position =
                position /
                    safeWeight,
            confidence =
                (
                    confidence /
                        safeWeight
                    ).coerceIn(
                        0.0,
                        1.0,
                    ),
            uncertaintyMeters =
                (
                    1.0 /
                        sqrtSafe(
                            safeWeight,
                        )
                    ).coerceIn(
                        0.001,
                        2.0,
                    ),
        )
    }

    private fun state(
        track: Track,
        origin: EvidenceOrigin,
        rank: Int,
        count: Int,
        innovationSigmaMeters: Double = 0.0,
        innovationScore: Double = 1.0,
        measurementBlend: Double = 1.0,
    ): State =
        State(
            actorId = track.actorId,
            pointId = track.pointId,
            timestampNanos =
                track.timestampNanos,
            positionMeters = track.position,
            velocityMetersPerSecond =
                track.velocity,
            accelerationMetersPerSecond2 =
                track.acceleration,
            jerkMetersPerSecond3 =
                track.jerk,
            confidence =
                track.confidence.coerceIn(
                    0.0,
                    1.0,
                ),
            uncertaintyMeters =
                track.uncertaintyMeters.coerceAtLeast(
                    0.0,
                ),
            origin = origin,
            hypothesisRank = rank,
            sourceObservationCount = count,
            innovationSigmaMeters =
                innovationSigmaMeters
                    .coerceAtLeast(0.0),
            innovationScore =
                innovationScore
                    .coerceIn(0.0, 1.0),
            measurementBlend =
                measurementBlend
                    .coerceIn(0.0, 1.0),
        )

    private fun finiteProposal(
        proposal: WorldSpaceProposal,
    ): Boolean =
        proposal.positionMeters.finite() &&
            proposal.velocityMetersPerSecond.finite() &&
            proposal.accelerationMetersPerSecond2.finite() &&
            proposal.confidence.isFinite() &&
            proposal.confidence in 0.0..1.0

    private fun key(
        actorId: Int,
        pointId: Int,
    ): Long =
        (actorId.toLong() shl 32) xor
            (
                pointId.toLong() and
                    0xffffffffL
            )

    private fun pruneOldest() {
        val oldest =
            tracks.entries
                .minByOrNull {
                    it.value.timestampNanos
                }?.key
                ?: return
        tracks.remove(oldest)
    }

    override fun close() {
        synchronized(this) {
            tracks.clear()
            closed = true
        }
    }
}

private fun Vec3.finite(): Boolean =
    x.isFinite() &&
        y.isFinite() &&
        z.isFinite()

private fun Vec3.clampNorm(
    maxNorm: Double,
): Vec3 {
    val n = norm()
    return if (
        n <= maxNorm ||
        n < 1e-12
    ) {
        this
    } else {
        this * (
            maxNorm /
                n
            )
    }
}

private fun Vec3.clampStep(
    previous: Vec3,
    maxDistance: Double,
): Vec3 {
    val delta =
        this - previous
    val n = delta.norm()
    return if (
        n <= maxDistance ||
        n < 1e-12
    ) {
        this
    } else {
        previous +
            delta *
            (
                maxDistance /
                    n
            )
    }
}

private fun sqrtSafe(
    value: Double,
): Double =
    kotlin.math.sqrt(
        value.coerceAtLeast(0.0),
    )

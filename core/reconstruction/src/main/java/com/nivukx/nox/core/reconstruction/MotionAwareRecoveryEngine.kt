package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.max
import kotlin.math.min

/**
 * Multi-track motion recovery coordinator.
 *
 * This is the temporal recovery layer between measured 3D reconstruction and long-lived
 * dense tracks. It reuses the existing AdaptiveKalman3DFilter as the numerical state estimator
 * and adds production policy around:
 *  - bounded per-track ownership;
 *  - measurement-quality gating;
 *  - robust measurement/prediction blending;
 *  - short occlusion prediction;
 *  - recovery confidence decay;
 *  - deterministic stale eviction.
 *
 * The layer is intentionally detector-agnostic. It can therefore stabilize body, hand, face,
 * eye-region, prop, or category-specific 3D points without creating a separate identity model.
 */
class MotionAwareRecoveryEngine(
    private val config: Config = Config.production(),
) : AutoCloseable {

    enum class State {
        OBSERVED,
        FUSED,
        PREDICTED,
        STALE,
        REJECTED,
    }

    data class Config(
        val maximumTracks: Int = 4_096,
        val maximumGapMillis: Double = 100.0,
        val staleTimeoutMillis: Double = 1_000.0,
        val predictionConfidenceFloor: Double = 0.18,
        val predictionDecayPerSecond: Double = 0.72,
        val minimumMeasurementConfidence: Double = 0.12,
        val predictionBlendWhenWeak: Double = 0.78,
        val measurementBlendWhenStrong: Double = 0.82,
        val maximumInnovationMeters: Double = 0.12,
        val holdInnovationMeters: Double = 0.045,
    ) {
        init {
            require(maximumTracks in 32..16_384)
            require(maximumGapMillis > 0.0)
            require(staleTimeoutMillis >= maximumGapMillis)
            require(predictionConfidenceFloor in 0.0..1.0)
            require(predictionDecayPerSecond in 0.0..4.0)
            require(minimumMeasurementConfidence in 0.0..1.0)
            require(predictionBlendWhenWeak in 0.0..1.0)
            require(measurementBlendWhenStrong in 0.0..1.0)
            require(maximumInnovationMeters > 0.0)
            require(holdInnovationMeters > 0.0)
        }

        companion object {
            fun production(): Config = Config()

            fun precision(): Config = Config(
                maximumGapMillis = 80.0,
                staleTimeoutMillis = 850.0,
                predictionConfidenceFloor = 0.22,
                predictionDecayPerSecond = 0.58,
                minimumMeasurementConfidence = 0.16,
                predictionBlendWhenWeak = 0.70,
                measurementBlendWhenStrong = 0.88,
                maximumInnovationMeters = 0.08,
                holdInnovationMeters = 0.030,
            )
        }
    }

    data class Observation(
        val trackId: Int,
        val positionMeters: Vec3,
        val confidence: Double,
        val reprojectionErrorPx: Double,
        val timestampNanos: Long,
        val occlusionRisk: Double = 0.0,
        val temporalConfidence: Double = 1.0,
    ) {
        init {
            require(trackId >= 0)
            require(positionMeters.x.isFinite() && positionMeters.y.isFinite() && positionMeters.z.isFinite())
            require(confidence in 0.0..1.0)
            require(reprojectionErrorPx.isFinite() && reprojectionErrorPx >= 0.0)
            require(timestampNanos >= 0L)
            require(occlusionRisk in 0.0..1.0)
            require(temporalConfidence in 0.0..1.0)
        }
    }

    data class Prediction(
        val trackId: Int,
        val positionMeters: Vec3,
        val velocityMetersPerSecond: Vec3,
        val uncertaintyMeters: Double,
        val confidence: Double,
        val innovationMeters: Double,
        val measurementWeight: Double,
        val ageMillis: Double,
        val state: State,
        val warnings: List<String>,
    )

    private data class Track(
        val filter: AdaptiveKalman3DFilter,
        var lastObservedTimestampNanos: Long,
        var confidence: Double,
    )

    private val tracks =
        LinkedHashMap<Int, Track>()

    @Synchronized
    fun update(
        observation: Observation,
    ): Prediction {
        prune(observation.timestampNanos)

        val track =
            tracks[observation.trackId]
                ?: createTrack(
                    observation.timestampNanos,
                ).also {
                    tracks[observation.trackId] = it
                }

        val previousTimestamp =
            track.lastObservedTimestampNanos

        val rawGapNanos =
            observation.timestampNanos -
                previousTimestamp

        if (
            previousTimestamp != 0L &&
                rawGapNanos <= 0L
        ) {
            tracks.remove(observation.trackId)
            val resetTrack =
                createTrack(
                    observation.timestampNanos,
                )
            tracks[observation.trackId] = resetTrack
            return updateInternal(
                observation,
                resetTrack,
                timestampReset = true,
            )
        }

        if (
            observation.confidence <
                config.minimumMeasurementConfidence
        ) {
            val forecast =
                track.filter.forecast(
                    id = observation.trackId,
                    timestampNanos =
                        observation.timestampNanos,
                )

            if (forecast != null) {
                val decay =
                    (
                        1.0 -
                            (
                                observation.occlusionRisk *
                                    0.55 +
                                    (
                                        1.0 -
                                            observation.temporalConfidence
                                    ) * 0.45
                            )
                        ).coerceIn(
                            config.predictionConfidenceFloor,
                            1.0,
                        )

                return Prediction(
                    trackId = observation.trackId,
                    positionMeters = forecast.position,
                    velocityMetersPerSecond = forecast.velocity,
                    uncertaintyMeters =
                        forecast.uncertaintyMeters +
                            0.025,
                    confidence =
                        (
                            track.confidence *
                                decay *
                                0.82
                            ).coerceIn(
                                config.predictionConfidenceFloor,
                                1.0,
                            ),
                    innovationMeters = 0.0,
                    measurementWeight = 0.0,
                    ageMillis = 0.0,
                    state = State.PREDICTED,
                    warnings = listOf(
                        "low-confidence-measurement-held-out",
                        "motion-state-preserved-from-prior-evidence",
                    ),
                )
            }
        }

        val state =
            track.filter.update(
                id = observation.trackId,
                measurement = observation.positionMeters,
                measurementConfidence =
                    observation.confidence,
                reprojectionErrorPx =
                    observation.reprojectionErrorPx,
                timestampNanos =
                    observation.timestampNanos,
            )

        track.lastObservedTimestampNanos =
            observation.timestampNanos

        val observationTrust =
            (
                observation.confidence * 0.45 +
                    observation.temporalConfidence * 0.25 +
                    (1.0 - observation.occlusionRisk) * 0.20 +
                    state.measurementWeight * 0.10
                ).coerceIn(0.0, 1.0)

        track.confidence =
            (
                track.confidence * 0.28 +
                    observationTrust * 0.72
            ).coerceIn(0.0, 1.0)

        val blend =
            when {
                state.innovationMeters <= config.holdInnovationMeters ->
                    config.measurementBlendWhenStrong
                observationTrust < 0.45 ->
                    1.0 - config.predictionBlendWhenWeak
                else ->
                    (
                        0.50 +
                            0.50 * observationTrust
                        ).coerceIn(
                            0.45,
                            config.measurementBlendWhenStrong,
                        )
            }

        val fusedPosition =
            state.position * blend +
                observation.positionMeters * (1.0 - blend)

        val warnings = buildList {
            if (state.innovationMeters > config.holdInnovationMeters) {
                add("motion-measurement-innovation-elevated")
            }
            if (state.innovationMeters > config.maximumInnovationMeters) {
                add("motion-measurement-innovation-rejected-envelope")
            }
            if (observation.occlusionRisk > 0.60) {
                add("motion-update-under-occlusion")
            }
            if (state.measurementWeight < 0.40) {
                add("motion-filter-downweighted-measurement")
            }
            if (state.timestampReset) {
                add("motion-filter-timestamp-reset")
            }
        }.distinct()

        return Prediction(
            trackId = observation.trackId,
            positionMeters = fusedPosition,
            velocityMetersPerSecond =
                state.velocity,
            uncertaintyMeters =
                state.uncertaintyMeters,
            confidence =
                (
                    track.confidence *
                        (
                            0.72 +
                                0.28 *
                                    state.measurementWeight
                            )
                    ).coerceIn(0.0, 1.0),
            innovationMeters =
                state.innovationMeters,
            measurementWeight =
                state.measurementWeight,
            ageMillis = 0.0,
            state =
                if (state.measurementWeight >= 0.55) {
                    State.FUSED
                } else {
                    State.OBSERVED
                },
            warnings = warnings,
        )
    }

    @Synchronized
    fun predict(
        trackId: Int,
        timestampNanos: Long,
    ): Prediction? {
        require(trackId >= 0)
        require(timestampNanos >= 0L)

        val track = tracks[trackId] ?: return null
        val ageNanos =
            timestampNanos -
                track.lastObservedTimestampNanos

        if (ageNanos <= 0L) {
            return null
        }

        val ageMillis =
            ageNanos.toDouble() / 1_000_000.0

        if (ageMillis > config.maximumGapMillis) {
            if (
                ageMillis >
                    config.staleTimeoutMillis
            ) {
                tracks.remove(trackId)
            }
            return null
        }

        val snapshot =
            track.filter.snapshot()[trackId]
                ?: return null

        val forecast =
            track.filter.forecast(
                id = trackId,
                timestampNanos = timestampNanos,
            )
                ?: return null

        val normalizedAge =
            (
                ageMillis /
                    config.maximumGapMillis
            ).coerceIn(0.0, 1.0)

        val decay =
            (
                1.0 -
                    normalizedAge *
                        config.predictionDecayPerSecond
            ).coerceIn(
                config.predictionConfidenceFloor,
                1.0,
            )

        val confidence =
            (
                track.confidence *
                    decay *
                    0.88
                ).coerceIn(
                    config.predictionConfidenceFloor,
                    1.0,
                )

        val warnings =
            buildList {
                add("motion-state-temporally-predicted")
                if (forecast.processGrowthMeters > 0.015) {
                    add("motion-prediction-uncertainty-growth")
                }
                if (ageMillis >
                    config.maximumGapMillis * 0.65
                ) {
                    add("motion-prediction-near-gap-limit")
                }
                if (confidence <
                    config.predictionConfidenceFloor * 1.5
                ) {
                    add("motion-prediction-confidence-low")
                }
            }

        return Prediction(
            trackId = trackId,
            positionMeters =
                forecast.position,
            velocityMetersPerSecond =
                forecast.velocity,
            uncertaintyMeters =
                max(
                    snapshot.uncertaintyMeters,
                    forecast.uncertaintyMeters,
                ) +
                    normalizedAge * 0.020,
            confidence = confidence,
            innovationMeters =
                snapshot.innovationMeters,
            measurementWeight = 0.0,
            ageMillis = ageMillis,
            state = State.PREDICTED,
            warnings = warnings,
        )
    }

    @Synchronized
    fun remove(trackId: Int) {
        tracks.remove(trackId)
    }

    @Synchronized
    fun activeTrackCount(): Int =
        tracks.size

    @Synchronized
    fun reset() {
        tracks.clear()
    }

    override fun close() {
        reset()
    }

    private fun updateInternal(
        observation: Observation,
        track: Track,
        timestampReset: Boolean,
    ): Prediction {
        val state =
            track.filter.update(
                id = observation.trackId,
                measurement = observation.positionMeters,
                measurementConfidence =
                    observation.confidence,
                reprojectionErrorPx =
                    observation.reprojectionErrorPx,
                timestampNanos =
                    observation.timestampNanos,
            )
        track.lastObservedTimestampNanos =
            observation.timestampNanos
        track.confidence =
            observation.confidence.coerceIn(0.0, 1.0)

        return Prediction(
            trackId = observation.trackId,
            positionMeters =
                observation.positionMeters,
            velocityMetersPerSecond =
                state.velocity,
            uncertaintyMeters =
                state.uncertaintyMeters,
            confidence =
                track.confidence,
            innovationMeters =
                state.innovationMeters,
            measurementWeight =
                state.measurementWeight,
            ageMillis = 0.0,
            state =
                if (timestampReset) {
                    State.REJECTED
                } else {
                    State.OBSERVED
                },
            warnings =
                if (timestampReset) {
                    listOf(
                        "motion-track-reinitialized-after-timestamp-reversal",
                    )
                } else {
                    emptyList()
                },
        )
    }

    private fun createTrack(
        timestampNanos: Long,
    ): Track {
        ensureCapacity(timestampNanos)
        return Track(
            filter =
                AdaptiveKalman3DFilter(
                    maximumTrackedStates = 32,
                ),
            lastObservedTimestampNanos = 0L,
            confidence = 0.0,
        )
    }

    private fun ensureCapacity(
        nowNanos: Long,
    ) {
        if (tracks.size < config.maximumTracks) return

        prune(nowNanos)
        if (tracks.size < config.maximumTracks) return

        val victim =
            tracks.entries.minWithOrNull(
                compareBy<Map.Entry<Int, Track>>(
                    { it.value.confidence },
                    { it.value.lastObservedTimestampNanos },
                    { it.key },
                ),
            ) ?: return

        tracks.remove(victim.key)
    }

    private fun prune(
        nowNanos: Long,
    ) {
        val staleNanos =
            (
                config.staleTimeoutMillis *
                    1_000_000.0
                ).toLong()

        val iterator =
            tracks.entries.iterator()

        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (
                nowNanos -
                    entry.value.lastObservedTimestampNanos >
                    staleNanos
            ) {
                iterator.remove()
            }
        }
    }
}

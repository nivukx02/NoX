package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Bounded, deterministic temporal fusion for markerless motion observations.
 *
 * Observation agreement is separated from temporal prediction: prediction is always time-bounded
 * and confidence-decayed, so prolonged occlusion cannot become unbounded fabricated motion.
 */
class HighEndTemporalFusionEngine(
    initialConfig: Config = Config(),
) : AutoCloseable {

    enum class TrackState { WARMING, TRACKING, OCCLUDED, LOST }

    data class Config(
        val maxTracks: Int = 128,
        val maxObservationsPerFrame: Int = 2_048,
        val maxPredictionMilliseconds: Double = 80.0,
        val trackRetentionMilliseconds: Double = 2_000.0,
        val hardOutlierDistance: Double = 1.75,
        val huberDelta: Double = 1.25,
        val minimumObservationConfidence: Double = 0.05,
        val trackingConfidenceThreshold: Double = 0.38,
        val occludedConfidenceFloor: Double = 0.06,
        val confidenceHalfLifeMilliseconds: Double = 160.0,
        val maximumVelocity: Double = 12.0,
        val maximumAcceleration: Double = 120.0,
        val measurementGain: Double = 0.72,
        val velocityGain: Double = 0.38,
        val accelerationGain: Double = 0.18,
    ) {
        init {
            require(maxTracks > 0)
            require(maxObservationsPerFrame > 0)
            require(maxPredictionMilliseconds > 0.0)
            require(trackRetentionMilliseconds >= maxPredictionMilliseconds)
            require(hardOutlierDistance > 0.0)
            require(huberDelta > 0.0)
            require(minimumObservationConfidence in 0.0..1.0)
            require(trackingConfidenceThreshold in 0.0..1.0)
            require(occludedConfidenceFloor in 0.0..1.0)
            require(confidenceHalfLifeMilliseconds > 0.0)
            require(maximumVelocity > 0.0)
            require(maximumAcceleration > 0.0)
            require(measurementGain in 0.0..1.0)
            require(velocityGain in 0.0..1.0)
            require(accelerationGain in 0.0..1.0)
        }
    }

    data class Point3(
        val x: Double,
        val y: Double,
        val z: Double,
    ) {
        init {
            require(x.isFinite() && y.isFinite() && z.isFinite())
        }

        operator fun plus(other: Point3) =
            Point3(x + other.x, y + other.y, z + other.z)

        operator fun minus(other: Point3) =
            Point3(x - other.x, y - other.y, z - other.z)

        operator fun times(scale: Double) =
            Point3(x * scale, y * scale, z * scale)

        operator fun div(scale: Double): Point3 {
            require(scale.isFinite() && abs(scale) > 1e-12)
            return Point3(x / scale, y / scale, z / scale)
        }

        fun norm(): Double =
            sqrt(x * x + y * y + z * z)

        fun clampNorm(limit: Double): Point3 {
            val length = norm()
            if (!length.isFinite() || length <= limit || length <= 1e-12) {
                return this
            }
            return this * (limit / length)
        }
    }

    data class Observation(
        val trackId: Int,
        val timestampNanos: Long,
        val point: Point3,
        val confidence: Double,
        val cameraId: String,
        val visible: Boolean = true,
        val occluded: Boolean = false,
        val covarianceTrace: Double = 0.0,
    ) {
        init {
            require(trackId >= 0)
            require(timestampNanos >= 0L)
            require(confidence.isFinite() && confidence in 0.0..1.0)
            require(cameraId.isNotBlank())
            require(covarianceTrace.isFinite() && covarianceTrace >= 0.0)
        }
    }

    data class TrackSnapshot(
        val trackId: Int,
        val state: TrackState,
        val position: Point3,
        val velocity: Point3,
        val acceleration: Point3,
        val confidence: Double,
        val ageFrames: Long,
        val missingFrames: Int,
        val predictedMilliseconds: Double,
        val inlierCount: Long,
        val rejectedCount: Long,
        val supportingCameraCount: Int,
        val lastTimestampNanos: Long,
    )

    data class FrameSummary(
        val timestampNanos: Long,
        val activeTracks: Int,
        val trackingTracks: Int,
        val occludedTracks: Int,
        val lostTracks: Int,
        val acceptedObservations: Int,
        val rejectedObservations: Int,
        val duplicateObservations: Int,
        val meanInnovation: Double,
        val p95Innovation: Double,
        val jitterRms: Double,
        val temporalConsistency: Double,
        val meanConfidence: Double,
        val predictedMilliseconds: Double,
        val warnings: List<String>,
    )

    private data class FusedObservation(
        val point: Point3,
        val confidence: Double,
        val covarianceTrace: Double,
        val supportCount: Int,
        val cameras: Set<String>,
        val rejectedCount: Int,
        val acceptedCount: Int,
    )

    private data class Track(
        val trackId: Int,
        var state: TrackState,
        var position: Point3,
        var velocity: Point3,
        var acceleration: Point3,
        var confidence: Double,
        var ageFrames: Long,
        var missingFrames: Int,
        var predictedMilliseconds: Double,
        var inlierCount: Long,
        var rejectedCount: Long,
        var supportingCameraCount: Int,
        var lastTimestampNanos: Long,
        var lastObservedTimestampNanos: Long,
    )

    private data class Prediction(
        val position: Point3,
        val velocity: Point3,
        val acceleration: Point3,
        val uncertainty: Double,
    )

    @Volatile
    private var config: Config = initialConfig

    private val tracks = LinkedHashMap<Int, Track>()
    private var lastTimestampNanos: Long = -1L
    private var closed = false
    private var lastFrameSummary = FrameSummary(
        timestampNanos = 0L,
        activeTracks = 0,
        trackingTracks = 0,
        occludedTracks = 0,
        lostTracks = 0,
        acceptedObservations = 0,
        rejectedObservations = 0,
        duplicateObservations = 0,
        meanInnovation = 0.0,
        p95Innovation = 0.0,
        jitterRms = 0.0,
        temporalConsistency = 1.0,
        meanConfidence = 0.0,
        predictedMilliseconds = 0.0,
        warnings = emptyList(),
    )

    @Synchronized
    fun configure(next: Config) {
        check(!closed) { "Temporal fusion engine is closed" }
        config = next
    }

    @Synchronized
    fun update(timestampNanos: Long, observations: List<Observation>): FrameSummary {
        check(!closed) { "Temporal fusion engine is closed" }
        require(timestampNanos >= 0L)
        if (lastTimestampNanos >= 0L) {
            require(timestampNanos > lastTimestampNanos) {
                "Frame timestamps must be strictly increasing"
            }
        }

        val bounded = observations.asSequence()
            .filter { it.timestampNanos == timestampNanos }
            .filter { it.confidence >= config.minimumObservationConfidence }
            .take(config.maxObservationsPerFrame)
            .toList()
        val byTrack = bounded.groupBy { it.trackId }
        val seen = byTrack.keys
        var accepted = 0
        var rejected = observations.size - bounded.size
        var duplicates = 0
        val innovations = ArrayList<Double>()
        val jitterSamples = ArrayList<Double>()
        val warnings = LinkedHashSet<String>()

        for ((trackId, group) in byTrack) {
            if (group.size > 1) duplicates += group.size - 1
            val existing = tracks[trackId]

            if (existing == null) {
                if (tracks.size >= config.maxTracks) {
                    val victim = tracks.values.minByOrNull(::trackPriority)
                    if (victim != null && trackPriority(victim) < trackPriorityForNew(group)) {
                        tracks.remove(victim.trackId)
                    } else {
                        rejected += group.size
                        warnings += "track-capacity-reached"
                        continue
                    }
                }

                val seed = robustFuse(group, null)
                if (seed == null) {
                    rejected += group.size
                    continue
                }
                tracks[trackId] = Track(
                    trackId = trackId,
                    state = TrackState.WARMING,
                    position = seed.point,
                    velocity = Point3(0.0, 0.0, 0.0),
                    acceleration = Point3(0.0, 0.0, 0.0),
                    confidence = seed.confidence,
                    ageFrames = 1L,
                    missingFrames = 0,
                    predictedMilliseconds = 0.0,
                    inlierCount = seed.acceptedCount.toLong(),
                    rejectedCount = seed.rejectedCount.toLong(),
                    supportingCameraCount = seed.cameras.size,
                    lastTimestampNanos = timestampNanos,
                    lastObservedTimestampNanos = timestampNanos,
                )
                accepted += seed.acceptedCount
                rejected += seed.rejectedCount
                if (seed.supportCount == 1) warnings += "single-view-track-$trackId"
                continue
            }

            val dt = ((timestampNanos - existing.lastTimestampNanos).toDouble() / 1_000_000_000.0)
                .coerceIn(1e-6, 1.0)
            val prediction = predict(existing, dt)
            val fused = robustFuse(group, prediction.position)
            if (fused == null) {
                applyPredictionOnly(existing, timestampNanos, dt)
                rejected += group.size
                continue
            }

            val normalizedInnovation = normalizedInnovation(
                prediction.position,
                fused.point,
                prediction.uncertainty,
                fused.covarianceTrace,
            )
            if (normalizedInnovation > config.hardOutlierDistance) {
                applyPredictionOnly(existing, timestampNanos, dt)
                existing.rejectedCount += group.size
                rejected += group.size
                warnings += "hard-outlier-track-$trackId"
                innovations += normalizedInnovation
                continue
            }

            val robust = huberWeight(normalizedInnovation)
            val measurementWeight = (
                robust *
                    fused.confidence *
                    (1.0 / (1.0 + fused.covarianceTrace))
                ).coerceIn(0.0, 1.0)
            val positionGain = (config.measurementGain * measurementWeight)
                .coerceIn(0.05, 0.92)
            val velocityObservation = (fused.point - existing.position) / dt
            val velocityGain = (config.velocityGain * measurementWeight)
                .coerceIn(0.03, 0.75)
            val accelerationObservation = (velocityObservation - existing.velocity) / dt
            val accelerationGain = (config.accelerationGain * measurementWeight)
                .coerceIn(0.02, 0.50)
            val previous = existing.position

            existing.position = lerp(prediction.position, fused.point, positionGain)
            existing.velocity = lerp(
                prediction.velocity,
                velocityObservation,
                velocityGain,
            ).clampNorm(config.maximumVelocity)
            existing.acceleration = lerp(
                prediction.acceleration,
                accelerationObservation,
                accelerationGain,
            ).clampNorm(config.maximumAcceleration)
            existing.confidence = confidenceAfterMeasurement(
                previous = existing.confidence,
                measurement = fused.confidence,
                robustWeight = robust,
                dtSeconds = dt,
            )
            existing.ageFrames += 1L
            existing.missingFrames = 0
            existing.predictedMilliseconds = 0.0
            existing.inlierCount += fused.acceptedCount.toLong()
            existing.rejectedCount += fused.rejectedCount.toLong()
            existing.supportingCameraCount = fused.cameras.size
            existing.lastTimestampNanos = timestampNanos
            existing.lastObservedTimestampNanos = timestampNanos
            existing.state = if (
                existing.confidence >= config.trackingConfidenceThreshold
            ) {
                TrackState.TRACKING
            } else {
                TrackState.WARMING
            }

            jitterSamples += distance(
                existing.position,
                previous + existing.velocity * dt,
            )
            innovations += normalizedInnovation
            accepted += fused.acceptedCount
            rejected += fused.rejectedCount
        }

        for (track in tracks.values.toList()) {
            if (track.trackId in seen) continue
            val dt = ((timestampNanos - track.lastTimestampNanos).toDouble() / 1_000_000_000.0)
                .coerceIn(1e-6, 1.0)
            applyPredictionOnly(track, timestampNanos, dt)
            track.state = if (
                track.predictedMilliseconds > config.maxPredictionMilliseconds
            ) {
                TrackState.LOST
            } else {
                TrackState.OCCLUDED
            }
            if (track.predictedMilliseconds >
                config.maxPredictionMilliseconds * 0.65
            ) {
                warnings += "prediction-horizon-near-limit"
            }
            if (track.predictedMilliseconds >
                config.maxPredictionMilliseconds
            ) {
                warnings += "prediction-budget-exhausted"
            }
        }

        evictExpired(timestampNanos)
        lastTimestampNanos = timestampNanos
        val snapshots = tracks.values.map(::snapshot)
        val meanInnovation = mean(innovations)
        val jitterRms = rms(jitterSamples)
        val meanConfidence = mean(snapshots.map { it.confidence })
        val temporalConsistency =
            if (snapshots.isEmpty()) {
                0.0
            } else {
                (
                    1.0 - (
                        0.52 * normalizedLowerBetter(
                            meanInnovation,
                            config.hardOutlierDistance,
                        ) +
                            0.28 * normalizedLowerBetter(
                                jitterRms,
                                config.hardOutlierDistance,
                            ) +
                            0.20 * (1.0 - meanConfidence)
                        )
                    ).coerceIn(0.0, 1.0)
            }

        lastFrameSummary = FrameSummary(
            timestampNanos = timestampNanos,
            activeTracks = snapshots.size,
            trackingTracks = snapshots.count { it.state == TrackState.TRACKING },
            occludedTracks = snapshots.count { it.state == TrackState.OCCLUDED },
            lostTracks = snapshots.count { it.state == TrackState.LOST },
            acceptedObservations = accepted,
            rejectedObservations = rejected,
            duplicateObservations = duplicates,
            meanInnovation = meanInnovation,
            p95Innovation = percentile(innovations, 0.95),
            jitterRms = jitterRms,
            temporalConsistency = temporalConsistency,
            meanConfidence = meanConfidence,
            predictedMilliseconds = snapshots.maxOfOrNull {
                it.predictedMilliseconds
            } ?: 0.0,
            warnings = warnings.toList(),
        )
        return lastFrameSummary
    }

    @Synchronized
    fun snapshot(trackId: Int): TrackSnapshot? =
        tracks[trackId]?.let(::snapshot)

    @Synchronized
    fun snapshots(): List<TrackSnapshot> =
        tracks.values.map(::snapshot)

    @Synchronized
    fun lastSummary(): FrameSummary = lastFrameSummary

    @Synchronized
    fun reset() {
        check(!closed) { "Temporal fusion engine is closed" }
        tracks.clear()
        lastTimestampNanos = -1L
        lastFrameSummary = lastFrameSummary.copy(
            timestampNanos = 0L,
            activeTracks = 0,
            trackingTracks = 0,
            occludedTracks = 0,
            lostTracks = 0,
            acceptedObservations = 0,
            rejectedObservations = 0,
            duplicateObservations = 0,
            meanInnovation = 0.0,
            p95Innovation = 0.0,
            jitterRms = 0.0,
            temporalConsistency = 1.0,
            meanConfidence = 0.0,
            predictedMilliseconds = 0.0,
            warnings = emptyList(),
        )
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        tracks.clear()
    }

    private fun robustFuse(
        observations: List<Observation>,
        center: Point3?,
    ): FusedObservation? {
        if (observations.isEmpty()) return null
        val candidates = observations
            .filter { it.visible || it.occluded }
            .sortedWith(
                compareByDescending<Observation> { it.confidence }
                    .thenBy { it.cameraId },
            )
            .distinctBy { it.cameraId }
        if (candidates.isEmpty()) return null

        var pivot = center ?: weightedMean(candidates)
        repeat(2) {
            val weights = candidates.map { observation ->
                val residual = distance(observation.point, pivot)
                val robust = huberWeight(
                    residual / max(config.huberDelta, 1e-6),
                )
                val visibility = if (observation.occluded) 0.35 else 1.0
                observation.confidence * robust * visibility
            }
            pivot = weightedMean(candidates, weights)
        }

        val accepted = candidates.filter {
            distance(it.point, pivot) <=
                config.hardOutlierDistance * max(1.0, 0.5 + it.confidence)
        }
        val rejected = candidates.size - accepted.size
        if (accepted.isEmpty()) return null

        val point = weightedMean(accepted)
        return FusedObservation(
            point = point,
            confidence = accepted.map { it.confidence }.average()
                .coerceIn(0.0, 1.0),
            covarianceTrace = accepted.map { it.covarianceTrace }.average()
                .coerceAtLeast(0.0),
            supportCount = accepted.size,
            cameras = accepted.map { it.cameraId }.toSet(),
            rejectedCount = rejected,
            acceptedCount = accepted.size,
        )
    }

    private fun predict(track: Track, dt: Double): Prediction {
        val position = track.position +
            track.velocity * dt +
            track.acceleration * (0.5 * dt * dt)
        val velocity = (
            track.velocity + track.acceleration * dt
            ).clampNorm(config.maximumVelocity)
        val uncertainty = (
            0.001 +
                (1.0 - track.confidence) * 0.35 +
                dt * dt * 0.12
            ).coerceIn(0.001, 2.0)
        return Prediction(
            position = position,
            velocity = velocity,
            acceleration = track.acceleration,
            uncertainty = uncertainty,
        )
    }

    private fun applyPredictionOnly(
        track: Track,
        timestampNanos: Long,
        dt: Double,
    ) {
        val prediction = predict(track, dt)
        track.position = prediction.position
        track.velocity = prediction.velocity
        track.acceleration = (
            track.acceleration * 0.995
            ).clampNorm(config.maximumAcceleration)
        track.ageFrames += 1L
        track.missingFrames += 1
        track.predictedMilliseconds += dt * 1000.0
        val decay = exp(
            -dt * 1000.0 /
                config.confidenceHalfLifeMilliseconds *
                0.6931471805599453,
        )
        track.confidence = max(
            config.occludedConfidenceFloor * 0.40,
            track.confidence * decay,
        ).coerceIn(0.0, 1.0)
        track.supportingCameraCount = 0
        track.lastTimestampNanos = timestampNanos
    }

    private fun confidenceAfterMeasurement(
        previous: Double,
        measurement: Double,
        robustWeight: Double,
        dtSeconds: Double,
    ): Double {
        val recovery = previous +
            (measurement - previous) *
            (0.34 + 0.42 * robustWeight)
        val timeDecay = exp(-dtSeconds * 0.20)
        return (
            recovery * 0.78 +
                measurement * 0.22
            ).coerceIn(0.0, 1.0) * timeDecay +
            (1.0 - timeDecay) * measurement
    }

    private fun evictExpired(timestampNanos: Long) {
        val retentionNanos =
            (config.trackRetentionMilliseconds * 1_000_000.0).toLong()
        val iterator = tracks.entries.iterator()
        while (iterator.hasNext()) {
            val track = iterator.next().value
            if (track.state != TrackState.LOST) continue
            if (timestampNanos - track.lastObservedTimestampNanos >
                retentionNanos
            ) {
                iterator.remove()
            }
        }
    }

    private fun trackPriority(track: Track): Double =
        track.confidence * 0.70 +
            min(track.ageFrames.toDouble(), 120.0) / 120.0 * 0.20 +
            if (track.state == TrackState.TRACKING) 0.10 else 0.0

    private fun trackPriorityForNew(
        observations: List<Observation>,
    ): Double =
        observations.maxOfOrNull { it.confidence } ?: 0.0

    private fun normalizedInnovation(
        predicted: Point3,
        measured: Point3,
        predictionUncertainty: Double,
        measurementUncertainty: Double,
    ): Double {
        val residual = distance(predicted, measured)
        val scale = sqrt(
            max(
                1e-6,
                predictionUncertainty +
                    measurementUncertainty +
                    0.0025,
            ),
        )
        return residual / scale
    }

    private fun huberWeight(normalizedResidual: Double): Double {
        val magnitude = abs(normalizedResidual)
        if (magnitude <= config.huberDelta) return 1.0
        return config.huberDelta / magnitude
    }

    private fun weightedMean(observations: List<Observation>): Point3 =
        weightedMean(
            observations,
            observations.map {
                it.confidence.coerceAtLeast(1e-6)
            },
        )

    private fun weightedMean(
        observations: List<Observation>,
        weights: List<Double>,
    ): Point3 {
        require(observations.isNotEmpty())
        require(observations.size == weights.size)
        val total = weights.sum().coerceAtLeast(1e-9)
        var x = 0.0
        var y = 0.0
        var z = 0.0
        for (index in observations.indices) {
            val weight = weights[index].coerceAtLeast(0.0)
            x += observations[index].point.x * weight
            y += observations[index].point.y * weight
            z += observations[index].point.z * weight
        }
        return Point3(x / total, y / total, z / total)
    }

    private fun lerp(a: Point3, b: Point3, weight: Double): Point3 {
        val w = weight.coerceIn(0.0, 1.0)
        return a * (1.0 - w) + b * w
    }

    private fun snapshot(track: Track): TrackSnapshot =
        TrackSnapshot(
            trackId = track.trackId,
            state = track.state,
            position = track.position,
            velocity = track.velocity,
            acceleration = track.acceleration,
            confidence = track.confidence,
            ageFrames = track.ageFrames,
            missingFrames = track.missingFrames,
            predictedMilliseconds = track.predictedMilliseconds.coerceIn(
                0.0,
                config.trackRetentionMilliseconds,
            ),
            inlierCount = track.inlierCount,
            rejectedCount = track.rejectedCount,
            supportingCameraCount = track.supportingCameraCount,
            lastTimestampNanos = track.lastTimestampNanos,
        )

    private fun distance(a: Point3, b: Point3): Double =
        (a - b).norm()

    private fun mean(values: List<Double>): Double =
        values.filter(Double::isFinite)
            .takeIf { it.isNotEmpty() }
            ?.average() ?: 0.0

    private fun rms(values: List<Double>): Double {
        val finite = values.filter(Double::isFinite)
        if (finite.isEmpty()) return 0.0
        return sqrt(
            finite.sumOf { it * it } /
                finite.size.toDouble(),
        )
    }

    private fun percentile(
        values: List<Double>,
        probability: Double,
    ): Double {
        val finite = values.filter(Double::isFinite).sorted()
        if (finite.isEmpty()) return 0.0
        if (finite.size == 1) return finite[0]
        val p = probability.coerceIn(0.0, 1.0) *
            (finite.size - 1)
        val lower = p.toInt()
        val upper = min(lower + 1, finite.lastIndex)
        return finite[lower] +
            (finite[upper] - finite[lower]) *
            (p - lower)
    }

    private fun normalizedLowerBetter(
        value: Double,
        ceiling: Double,
    ): Double =
        if (!value.isFinite() || ceiling <= 0.0) {
            1.0
        } else {
            (value / ceiling).coerceIn(0.0, 1.0)
        }
}

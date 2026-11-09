package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Geometry-first persistent dense-point correspondence.
 * Appearance descriptors are optional secondary evidence; temporal/world continuity is primary.
 */
class DenseWorldPointCorrespondenceEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        val maximumTracks: Int = 16_384,
        val maximumCandidatesPerTrack: Int = 8,
        val maximumGapNanos: Long = 250_000_000L,
        val minimumConfidence: Double = 0.20,
        val spatialGateMeters: Double = 0.45,
        val temporalWeight: Double = 0.36,
        val geometryWeight: Double = 0.42,
        val appearanceWeight: Double = 0.22,
    ) {
        init {
            require(maximumTracks > 0)
            require(maximumCandidatesPerTrack >= 1)
            require(maximumGapNanos > 0L)
            require(minimumConfidence in 0.0..1.0)
            require(spatialGateMeters > 0.0)
            require(temporalWeight >= 0.0 && geometryWeight >= 0.0 && appearanceWeight >= 0.0)
            require(temporalWeight + geometryWeight + appearanceWeight > 0.0)
        }
    }

    data class Observation(
        val actorId: Int,
        val localPointId: Int,
        val timestampNanos: Long,
        val cameraId: String,
        val imagePosition: Vec2,
        val worldPositionMeters: Vec3?,
        val velocityMetersPerSecond: Vec3?,
        val confidence: Double,
        val appearanceSignature: DoubleArray = doubleArrayOf(),
        val visibility: Double = 1.0,
    ) {
        init {
            require(actorId >= 0)
            require(localPointId >= 0)
            require(timestampNanos >= 0L)
            require(cameraId.isNotBlank())
            require(imagePosition.x.isFinite() && imagePosition.y.isFinite())
            worldPositionMeters?.let { require(finite(it)) }
            velocityMetersPerSecond?.let { require(finite(it)) }
            require(confidence in 0.0..1.0)
            require(visibility in 0.0..1.0)
            require(appearanceSignature.all(Double::isFinite))
        }
    }

    data class Track(
        val persistentId: Long,
        val actorId: Int,
        val timestampNanos: Long,
        val worldPositionMeters: Vec3?,
        val cameraCount: Int,
        val confidence: Double,
        val temporalContinuity: Double,
        val geometricContinuity: Double,
        val appearanceContinuity: Double,
        val sourceCameraIds: Set<String>,
    )

    data class Result(
        val tracks: List<Track>,
        val newTrackCount: Int,
        val reusedTrackCount: Int,
        val rejectedObservationCount: Int,
        val warnings: List<String>,
    )

    private data class MutableTrack(
        val persistentId: Long,
        var actorId: Int,
        var timestampNanos: Long,
        var world: Vec3?,
        var velocity: Vec3?,
        var confidence: Double,
        var appearance: DoubleArray,
        var cameras: MutableSet<String>,
        var temporalContinuity: Double = 1.0,
    )

    private val tracks = LinkedHashMap<Long, MutableTrack>()
    private var nextId = 1L
    private var closed = false

    @Synchronized
    fun update(observations: List<Observation>, timestampNanos: Long): Result {
        check(!closed)
        require(timestampNanos >= 0L)
        val current = observations
            .asSequence()
            .filter { it.timestampNanos == timestampNanos && it.confidence >= config.minimumConfidence }
            .take(config.maximumTracks)
            .toList()
        val rejected = observations.size - current.size
        val assignments = HashMap<Int, Long>()
        val claimed = HashSet<Long>()
        var reused = 0
        var created = 0

        for ((index, observation) in current.withIndex()) {
            val candidates = tracks.values
                .asSequence()
                .filter { it.actorId == observation.actorId }
                .filter { timestampNanos - it.timestampNanos in 0L..config.maximumGapNanos }
                .filter { candidate ->
                    val observedWorld = observation.worldPositionMeters
                    val trackedWorld = candidate.world
                    observedWorld == null || trackedWorld == null || observedWorld.distanceTo(trackedWorld) <= config.spatialGateMeters
                }
                .map { candidate ->
                    Triple(candidate, score(candidate, observation, timestampNanos), candidate.persistentId)
                }
                .sortedByDescending { it.second }
                .take(config.maximumCandidatesPerTrack)
                .toList()

            val best = candidates.firstOrNull { it.first.persistentId !in claimed && it.second > 0.30 }
            if (best != null) {
                val track = best.first
                track.temporalContinuity =
                    (1.0 - (timestampNanos - track.timestampNanos).toDouble() /
                config.maximumGapNanos.toDouble()).coerceIn(0.0, 1.0)
                claimed += track.persistentId
                assignments[index] = track.persistentId
                reused++
                updateTrack(track, observation)
            } else {
                val id = nextId++
                tracks[id] = MutableTrack(
                    persistentId = id,
                    actorId = observation.actorId,
                    timestampNanos = observation.timestampNanos,
                    world = observation.worldPositionMeters,
                    velocity = observation.velocityMetersPerSecond,
                    confidence = observation.confidence,
                    appearance = observation.appearanceSignature.copyOf(),
                    cameras = mutableSetOf(observation.cameraId),
                )
                assignments[index] = id
                claimed += id
                created++
            }
        }

        prune(timestampNanos)
        val resultTracks = current.indices.mapNotNull { index ->
            val id = assignments[index] ?: return@mapNotNull null
            tracks[id]?.let { track ->
                Track(
                    track.persistentId,
                    track.actorId,
                    track.timestampNanos,
                    track.world,
                    track.cameras.size,
                    track.confidence,
                    track.temporalContinuity,
                    geometricContinuity(track, current[index]),
                    appearanceContinuity(track.appearance, current[index].appearanceSignature),
                    track.cameras.toSet(),
                )
            }
        }

        return Result(
            tracks = resultTracks.sortedBy { it.persistentId },
            newTrackCount = created,
            reusedTrackCount = reused,
            rejectedObservationCount = rejected,
            warnings = buildList {
                if (reused == 0 && created > 0) add("dense-point-track-reinitialization")
                if (tracks.size >= config.maximumTracks) add("dense-point-track-capacity-reached")
            },
        )
    }

    private fun score(
        track: MutableTrack,
        observation: Observation,
        now: Long,
    ): Double {
        val dt = max(0.0, (now - track.timestampNanos).toDouble() / 1e9)
        val temporal = (1.0 - dt / (config.maximumGapNanos.toDouble() / 1e9)).coerceIn(0.0, 1.0)
        val geometry = when {
            observation.worldPositionMeters != null && track.world != null -> {
                (1.0 - observation.worldPositionMeters.distanceTo(track.world!!) / config.spatialGateMeters).coerceIn(0.0, 1.0)
            }
            else -> 0.35
        }
        val appearance = appearanceContinuity(track.appearance, observation.appearanceSignature)
        return (
            config.temporalWeight * temporal +
                config.geometryWeight * geometry +
                config.appearanceWeight * appearance
            ) * observation.confidence * observation.visibility
    }

    private fun updateTrack(track: MutableTrack, observation: Observation) {
        if (observation.worldPositionMeters != null) {
            track.world = when (val old = track.world) {
                null -> observation.worldPositionMeters
                else -> old.lerp(observation.worldPositionMeters, observation.confidence.coerceIn(0.15, 0.85))
            }
        }
        track.velocity = observation.velocityMetersPerSecond ?: track.velocity
        track.timestampNanos = observation.timestampNanos
        track.confidence = (track.confidence * 0.35 + observation.confidence * 0.65).coerceIn(0.0, 1.0)
        if (observation.appearanceSignature.isNotEmpty()) track.appearance = observation.appearanceSignature.copyOf()
        track.cameras += observation.cameraId
    }


    private fun geometricContinuity(track: MutableTrack, observation: Observation): Double =
        if (track.world != null && observation.worldPositionMeters != null) {
            (1.0 - track.world!!.distanceTo(observation.worldPositionMeters) / config.spatialGateMeters).coerceIn(0.0, 1.0)
        } else 0.35

    private fun appearanceContinuity(a: DoubleArray, b: DoubleArray): Double {
        if (a.isEmpty() || b.isEmpty() || a.size != b.size) return 0.35
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        if (na < 1e-12 || nb < 1e-12) return 0.35
        return ((dot / kotlin.math.sqrt(na * nb) + 1.0) * 0.5).coerceIn(0.0, 1.0)
    }

    private fun prune(now: Long) {
        if (tracks.size <= config.maximumTracks) return
        tracks.entries.sortedBy { it.value.timestampNanos }.take(tracks.size - config.maximumTracks).forEach { tracks.remove(it.key) }
    }

    override fun close() {
        synchronized(this) {
            tracks.clear()
            closed = true
        }
    }

    companion object {
        private fun finite(v: Vec3): Boolean =
            v.x.isFinite() && v.y.isFinite() && v.z.isFinite()
    }
}

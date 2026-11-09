package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Vec2
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Temporal identity authority for dense markerless multi-person pose detection.
 *
 * Detector ordering is not an identity. This tracker maintains a bounded local identity graph and
 * matches new observations against predicted tracks using center, scale, overlap, motion continuity
 * and confidence. Tracks survive short occlusions and expire deterministically.
 *
 * It is deliberately detector-agnostic so a future neural re-identification embedding can be added
 * without changing the lifecycle contract.
 */
class TemporalPoseIdentityTracker(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumTracks: Int = 64,
        val maximumPredictionFrames: Int = 18,
        val maximumMatchDistanceNormalized: Double = 0.22,
        val maxAssignmentDistanceNormalized: Double =
            maximumMatchDistanceNormalized,
        val maximumCenterGateNormalized: Double = 0.30,
        val minimumMatchScore: Double = 0.34,
        val reIdentificationScore: Double = 0.68,
        val velocitySmoothing: Double = 0.42,
        val confidenceSmoothing: Double = 0.28,
        val occlusionConfidenceDecay: Double = 0.93,
        val globalAssignmentItems: Int = 64,
        val predictionMaximumSpeed: Double = 5.0,
        val predictionMaximumAcceleration: Double = 30.0,
        val predictionMaximumJerk: Double = 140.0,
        val maximumDtSeconds: Double = 0.20,
    ) {
        init {
            require(maximumTracks in 1..128)
            require(maximumPredictionFrames >= 1)
            require(maximumPredictionFrames <= 120)
            require(maximumPredictionFrames >= 6)
            require(maximumMatchDistanceNormalized > 0.0)
            require(maximumCenterGateNormalized >= maximumMatchDistanceNormalized)
            require(minimumMatchScore in 0.0..1.0)
            require(reIdentificationScore >= minimumMatchScore)
            require(velocitySmoothing in 0.0..1.0)
            require(confidenceSmoothing in 0.0..1.0)
            require(occlusionConfidenceDecay in 0.50..1.0)
            require(globalAssignmentItems in 1..128)
            require(predictionMaximumSpeed > 0.0)
            require(predictionMaximumAcceleration > 0.0)
            require(predictionMaximumJerk > 0.0)
            require(maximumDtSeconds > 0.0)
        }
    }

    data class Observation(
        val center: Vec2,
        val width: Double,
        val height: Double,
        val confidence: Double,
        val detectorIndex: Int,
        val shapeSignature: List<Double> = emptyList(),
        val appearanceSignature: List<Double> = emptyList(),
        val appearanceQuality: Double = 0.0,
    ) {
        init {
            require(center.x.isFinite() && center.y.isFinite())
            require(width > 0.0 && height > 0.0)
            require(confidence in 0.0..1.0)
            require(detectorIndex >= 0)
            require(shapeSignature.size <= 128)
            require(shapeSignature.all { it.isFinite() })
            require(appearanceSignature.size <= 2048)
            require(appearanceSignature.all { it.isFinite() })
            require(appearanceQuality in 0.0..1.0)
        }

        val area: Double
            get() = width * height
    }

    data class AssignedObservation(
        val localTrackId: Long,
        val observation: Observation,
        val score: Double,
        val newlyCreated: Boolean,
        val reidentified: Boolean,
        val predictionUsed: Boolean,
    )

    data class Result(
        val timestampNanos: Long,
        val observations: List<AssignedObservation>,
        val activeTrackCount: Int,
        val createdTrackCount: Int,
        val reidentifiedTrackCount: Int,
        val predictedTrackCount: Int,
        val droppedObservationCount: Int,
        val identityContinuityScore: Double,
        val predictionConfidence: Double = 0.0,
        val meanInnovationNormalized: Double = 0.0,
        val acceleratedTrackCount: Int = 0,
        val globalAssignmentBounded: Boolean = false,
        val warnings: List<String>,
    )

    private data class Track(
        val id: Long,
        var center: Vec2,
        var width: Double,
        var height: Double,
        var velocity: Vec2,
        var confidence: Double,
        val predictor: AdaptiveMotionPredictionEngine,
        var lastTimestampNanos: Long,
        var missedFrames: Int,
        var totalAssignments: Long,
        var reidentifications: Long,
        var lastDetectorIndex: Int,
        var shapeSignature: List<Double>,
        var appearanceSignature: List<Double>,
        var appearanceQuality: Double,
    )

    private var nextTrackId = 1L
    private val tracks = LinkedHashMap<Long, Track>()
    private val assignmentEngine =
        BoundedGlobalAssignmentEngine(
            maxItems = config.globalAssignmentItems,
        )

    @Synchronized
    fun update(
        timestampNanos: Long,
        detections: List<Observation>,
    ): Result {
        require(timestampNanos >= 0L)

        val predicted = tracks.values
            .sortedBy { it.id }
            .map { track ->
                track to track.predictor.predict(timestampNanos)
            }

        val matches = globalAssign(
            tracks = predicted,
            detections = detections,
            timestampNanos = timestampNanos,
        )

        val matchedTrackIds = HashSet<Long>()
        val matchedDetectionIndices = HashSet<Int>()
        val assigned = ArrayList<AssignedObservation>(detections.size)

        matches.forEach { match ->
            val track = match.track
            val detection = detections[match.detectionIndex]
            val predictedCenter = match.predictedCenter
            val predictionUpdate =
                track.predictor.update(
                    measurement = detection.center,
                    timestampNanos = timestampNanos,
                    observationConfidence = detection.confidence,
                )

            track.velocity =
                predictionUpdate.state.velocity *
                    config.velocitySmoothing +
                    track.velocity *
                        (1.0 - config.velocitySmoothing)
            track.center =
                predictionUpdate.state.position
            track.width =
                track.width * 0.62 +
                    detection.width * 0.38
            track.height =
                track.height * 0.62 +
                    detection.height * 0.38
            track.confidence =
                track.confidence * (
                    1.0 - config.confidenceSmoothing
                ) +
                    detection.confidence *
                    config.confidenceSmoothing
            track.lastTimestampNanos = timestampNanos
            track.missedFrames = 0
            track.totalAssignments += 1L
            track.lastDetectorIndex = detection.detectorIndex
            track.shapeSignature =
                blendSignature(
                    previous = track.shapeSignature,
                    current = detection.shapeSignature,
                    alpha = 0.34,
                )
            if (
                detection.appearanceQuality >= 0.20 &&
                    detection.appearanceSignature.isNotEmpty()
            ) {
                track.appearanceSignature =
                    blendSignature(
                        previous = track.appearanceSignature,
                        current = detection.appearanceSignature,
                        alpha =
                            (
                                0.16 +
                                    detection.appearanceQuality * 0.36
                            ).coerceIn(0.16, 0.52),
                    )
                track.appearanceQuality =
                    (
                        track.appearanceQuality * 0.56 +
                            detection.appearanceQuality * 0.44
                        ).coerceIn(0.0, 1.0)
            } else {
                track.appearanceQuality =
                    (
                        track.appearanceQuality * 0.97
                    ).coerceIn(0.0, 1.0)
            }

            matchedTrackIds += track.id
            matchedDetectionIndices += match.detectionIndex

            assigned += AssignedObservation(
                localTrackId = track.id,
                observation = detection,
                score = match.score,
                newlyCreated = false,
                reidentified = false,
                predictionUsed = true,
            )
        }

        /*
         * Unmatched observations first try deterministic re-identification against recently lost
         * tracks. The regular assignment gate is stricter than re-identification to prevent identity
         * teleportation across a crowded scene.
         */
        var reidentifiedCount = 0
        detections.forEachIndexed { index, detection ->
            if (index in matchedDetectionIndices) return@forEachIndexed

            val candidate = tracks.values
                .filter {
                    it.id !in matchedTrackIds &&
                        it.missedFrames in 1..config.maximumPredictionFrames
                }
                .mapNotNull { track ->
                    val predictedCenter = predictCenter(
                        track,
                        timestampNanos,
                    )
                    val score = score(
                        track = track,
                        predictedCenter = predictedCenter,
                        detection = detection,
                        reIdentification = true,
                    )
                    if (score >= config.reIdentificationScore) {
                        Pair(track, score)
                    } else {
                        null
                    }
                }
                .sortedWith(
                    compareByDescending<Pair<Track, Double>> { it.second }
                        .thenBy { it.first.id },
                )
                .firstOrNull()

            if (candidate != null) {
                val track = candidate.first
                val predictedCenter =
                    track.predictor.predict(timestampNanos).position
                val predictionUpdate =
                    track.predictor.update(
                        measurement = detection.center,
                        timestampNanos = timestampNanos,
                        observationConfidence = detection.confidence,
                    )
                track.center =
                    predictionUpdate.state.position
                track.width = detection.width
                track.height = detection.height
                track.confidence =
                    max(
                        track.confidence,
                        detection.confidence * 0.84,
                    )
                track.velocity =
                    predictionUpdate.state.velocity
                track.lastTimestampNanos = timestampNanos
                track.missedFrames = 0
                track.totalAssignments += 1L
                track.reidentifications += 1L
                track.lastDetectorIndex = detection.detectorIndex
                track.shapeSignature =
                    detection.shapeSignature
                        .toList()

                matchedTrackIds += track.id
                matchedDetectionIndices += index
                reidentifiedCount += 1

                assigned += AssignedObservation(
                    localTrackId = track.id,
                    observation = detection,
                    score = candidate.second,
                    newlyCreated = false,
                    reidentified = true,
                    predictionUsed = true,
                )
            }
        }

        var createdCount = 0
        detections.forEachIndexed { index, detection ->
            if (index in matchedDetectionIndices) return@forEachIndexed
            if (tracks.size >= config.maximumTracks) return@forEachIndexed

            val id = nextTrackId++
            val predictor =
                AdaptiveMotionPredictionEngine(
                    AdaptiveMotionPredictionEngine.Config(
                        velocityGain =
                            config.velocitySmoothing
                                .coerceIn(0.05, 0.90),
                        maximumDtSeconds =
                            config.maximumDtSeconds,
                        maximumSpeedNormalizedPerSecond =
                            config.predictionMaximumSpeed,
                        maximumAccelerationNormalizedPerSecond2 =
                            config.predictionMaximumAcceleration,
                        maximumJerkNormalizedPerSecond3 =
                            config.predictionMaximumJerk,
                    ),
                )
            predictor.initialize(
                measurement = detection.center,
                timestampNanos = timestampNanos,
                observationConfidence = detection.confidence,
            )
            tracks[id] = Track(
                id = id,
                center = detection.center,
                width = detection.width,
                height = detection.height,
                velocity = Vec2(0.0, 0.0),
                confidence = detection.confidence,
                predictor = predictor,
                lastTimestampNanos = timestampNanos,
                missedFrames = 0,
                totalAssignments = 1L,
                reidentifications = 0L,
                lastDetectorIndex = detection.detectorIndex,
                shapeSignature = detection.shapeSignature.toList(),
            appearanceSignature =
                detection.appearanceSignature.toList(),
            appearanceQuality =
                detection.appearanceQuality
                    .coerceIn(0.0, 1.0),
            )
            matchedTrackIds += id
            matchedDetectionIndices += index
            createdCount += 1

            assigned += AssignedObservation(
                localTrackId = id,
                observation = detection,
                score = 1.0,
                newlyCreated = true,
                reidentified = false,
                predictionUsed = false,
            )
        }

        var predictedCount = 0
        val expired = ArrayList<Long>()
        tracks.values.forEach { track ->
            if (track.id in matchedTrackIds) return@forEach

            val predictedState =
                track.predictor.propagateOcclusion(timestampNanos)
            track.center = predictedState.position
            track.velocity = predictedState.velocity
            track.missedFrames += 1
            track.confidence =
                (
                    track.confidence *
                        config.occlusionConfidenceDecay *
                        (
                            0.86 +
                                0.14 *
                                    predictedState.confidence
                        )
                    ).coerceIn(0.0, 1.0)
            predictedCount += 1

            if (
                track.missedFrames >
                    config.maximumPredictionFrames ||
                track.confidence < 0.08
            ) {
                expired += track.id
            }
        }
        expired.forEach(tracks::remove)

        val assignmentConfidence =
            assigned.map { it.score.coerceIn(0.0, 1.0) }
        val predictionConfidence =
            tracks.values
                .map { it.predictor.snapshot().confidence }
                .ifEmpty { listOf(0.0) }
                .average()
                .coerceIn(0.0, 1.0)
        val meanInnovationNormalized =
            if (matches.isEmpty()) {
                0.0
            } else {
                matches.map { match ->
                    (
                        match.predictedCenter.distanceTo(
                            detections[match.detectionIndex].center,
                        ) / config.maxAssignmentDistanceNormalized
                    ).coerceIn(0.0, 2.0)
                }.average()
            }

        val acceleratedTrackCount =
            tracks.values.count {
                val snapshot = it.predictor.snapshot()
                snapshot.acceleration.norm() >
                    config.predictionMaximumAcceleration * 0.20
            }

        val continuity =
            if (assigned.isEmpty()) {
                0.0
            } else {
                (
                    assigned.map {
                        it.score.coerceIn(0.0, 1.0)
                    }.average() * 0.72 +
                        (
                            1.0 -
                                assigned.count { it.newlyCreated }.toDouble() /
                                    assigned.size.toDouble()
                            ).coerceIn(0.0, 1.0) * 0.18 +
                        (
                            1.0 -
                                reidentifiedCount.toDouble() /
                                    assigned.size.toDouble()
                            ).coerceIn(0.0, 1.0) * 0.10
                    ).coerceIn(0.0, 1.0)
            }

        val warnings = buildList {
            if (detections.size > config.maximumTracks) {
                add("TEMPORAL_IDENTITY_TRACK_CEILING_EXCEEDED")
            }
            if (predictedCount > 0) {
                add("TEMPORAL_IDENTITY_OCCLUSION_PREDICTION_ACTIVE")
            }
            if (reidentifiedCount > 0) {
                add("TEMPORAL_IDENTITY_REIDENTIFICATION_ACTIVE")
            }
            if (continuity < 0.55 && assigned.isNotEmpty()) {
                add("TEMPORAL_IDENTITY_CONTINUITY_DEGRADED")
            }
            if (
                meanInnovationNormalized >
                    0.75
            ) {
                add("TEMPORAL_IDENTITY_MOTION_INNOVATION_ELEVATED")
            }
            if (acceleratedTrackCount > 0) {
                add("TEMPORAL_IDENTITY_ACCELERATION_MODEL_ACTIVE")
            }
            if (
                predicted.size >
                    config.globalAssignmentItems ||
                detections.size >
                    config.globalAssignmentItems
            ) {
                add("TEMPORAL_IDENTITY_ASSIGNMENT_BOUNDED")
            }
        }

        return Result(
            timestampNanos = timestampNanos,
            observations = assigned.sortedBy { it.localTrackId },
            activeTrackCount = tracks.size,
            createdTrackCount = createdCount,
            reidentifiedTrackCount = reidentifiedCount,
            predictedTrackCount = predictedCount,
            droppedObservationCount =
                detections.size -
                    matchedDetectionIndices.size,
            identityContinuityScore = continuity,
            predictionConfidence = predictionConfidence,
            meanInnovationNormalized =
                meanInnovationNormalized.coerceIn(0.0, 2.0),
            acceleratedTrackCount = acceleratedTrackCount,
            globalAssignmentBounded =
                matches.isNotEmpty() &&
                    (
                        predicted.size >
                            config.globalAssignmentItems ||
                        detections.size >
                            config.globalAssignmentItems
                    ),
            warnings = warnings.distinct(),
        )
    }

    @Synchronized
    fun reset() {
        tracks.clear()
        nextTrackId = 1L
    }

    private data class Match(
        val track: Track,
        val detectionIndex: Int,
        val score: Double,
        val predictedCenter: Vec2,
    )

    private fun globalAssign(
        tracks: List<Pair<Track, AdaptiveMotionPredictionEngine.Prediction>>,
        detections: List<Observation>,
        timestampNanos: Long,
    ): List<Match> {
        if (tracks.isEmpty() || detections.isEmpty()) return emptyList()

        val edges = ArrayList<BoundedGlobalAssignmentEngine.Edge>()
        tracks.forEachIndexed { trackIndex, (track, prediction) ->
            detections.forEachIndexed { detectionIndex, detection ->
                val score = score(
                    track = track,
                    predictedCenter = prediction.position,
                    detection = detection,
                    reIdentification = false,
                )
                if (score >= config.minimumMatchScore) {
                    edges +=
                        BoundedGlobalAssignmentEngine.Edge(
                            row = trackIndex,
                            column = detectionIndex,
                            score = score,
                        )
                }
            }
        }

        val result =
            assignmentEngine.assign(
                rowCount = tracks.size,
                columnCount = detections.size,
                edges = edges,
                minimumScore = config.minimumMatchScore,
            )

        return result.matches.mapNotNull { match ->
            val entry = tracks.getOrNull(match.row)
                ?: return@mapNotNull null
            Match(
                track = entry.first,
                detectionIndex = match.column,
                score = match.score,
                predictedCenter = entry.second.position,
            )
        }
    }

    private fun score(
        track: Track,
        predictedCenter: Vec2,
        detection: Observation,
        reIdentification: Boolean,
    ): Double {
        val distance =
            predictedCenter.distanceTo(detection.center)
        val gate =
            if (reIdentification) {
                config.maximumCenterGateNormalized * 1.35
            } else {
                config.maximumCenterGateNormalized
            }

        if (distance > gate) return 0.0

        val distanceScore =
            exp(
                -distance /
                    config.maximumMatchDistanceNormalized
                        .coerceAtLeast(1e-6),
            ).coerceIn(0.0, 1.0)

        val trackArea =
            (
                track.width * track.height
            ).coerceAtLeast(1e-9)
        val detectionArea =
            detection.area.coerceAtLeast(1e-9)
        val scaleRatio =
            min(trackArea, detectionArea) /
                max(trackArea, detectionArea)
        val scaleScore =
            scaleRatio.coerceIn(0.0, 1.0)

        val confidenceScore =
            (
                0.35 +
                    0.65 *
                        (
                            track.confidence *
                                detection.confidence
                        ).coerceIn(0.0, 1.0)
                ).coerceIn(0.0, 1.0)

        val shapeSimilarity =
            signatureSimilarity(
                track.shapeSignature,
                detection.shapeSignature,
            )

        val temporalPenalty =
            (
                1.0 -
                    track.missedFrames.toDouble() /
                        (config.maximumPredictionFrames + 1)
                            .toDouble()
            ).coerceIn(0.20, 1.0)

        val appearanceEvidence =
            if (
                track.appearanceSignature.isEmpty() ||
                detection.appearanceSignature.isEmpty()
            ) {
                0.50
            } else {
                signatureSimilarity(
                    track.appearanceSignature,
                    detection.appearanceSignature,
                )
            }

        val appearanceReliability =
            (
                track.appearanceQuality *
                    detection.appearanceQuality
            ).coerceIn(0.0, 1.0)

        return (
            distanceScore * 0.44 +
                scaleScore * 0.15 +
                confidenceScore * 0.14 +
                shapeSimilarity * 0.09 +
                appearanceEvidence * (0.06 + appearanceReliability * 0.04) +
                temporalPenalty * 0.08
            ).coerceIn(0.0, 1.0)
    }

    private fun signatureSimilarity(
        left: List<Double>,
        right: List<Double>,
    ): Double {
        if (left.isEmpty() || right.isEmpty()) return 0.5
        val n = minOf(left.size, right.size)
        if (n == 0) return 0.5

        var dot = 0.0
        var leftNorm = 0.0
        var rightNorm = 0.0

        for (index in 0 until n) {
            val a = left[index]
            val b = right[index]
            dot += a * b
            leftNorm += a * a
            rightNorm += b * b
        }

        if (leftNorm <= 1e-12 || rightNorm <= 1e-12) {
            return 0.5
        }

        return (
            dot /
                kotlin.math.sqrt(
                    leftNorm * rightNorm,
                ).coerceAtLeast(1e-12)
            ).coerceIn(-1.0, 1.0) * 0.5 + 0.5
    }

    private fun blendSignature(
        previous: List<Double>,
        current: List<Double>,
        alpha: Double,
    ): List<Double> {
        if (previous.isEmpty()) return current.toList()
        if (current.isEmpty()) return previous
        val n = minOf(previous.size, current.size)
        return buildList(n) {
            for (index in 0 until n) {
                add(
                    previous[index] * (1.0 - alpha) +
                        current[index] * alpha,
                )
            }
        }
    }

    private fun predictCenter(
        track: Track,
        timestampNanos: Long,
    ): Vec2 =
        track.predictor.predict(timestampNanos).position
}

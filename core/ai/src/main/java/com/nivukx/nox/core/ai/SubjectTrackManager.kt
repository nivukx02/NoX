package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TrackedObject
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Production temporal subject association.
 *
 * The detector's tracking ID is treated as evidence, not as ground truth. NoX combines it with
 * motion prediction, IoU, center distance and semantic agreement, then solves the bounded global
 * assignment instead of greedily accepting the first locally-best pair. This prevents identity
 * swaps when two subjects cross or briefly occlude one another.
 */
class SubjectTrackManager(
    private val maxMissedFrames: Int = 8,
    private val iouGate: Double = 0.08,
    private val centerGateFraction: Double = 0.75,
    private val velocitySmoothing: Double = 0.45,
    private val maximumRetainedTracks: Int = 64,
    /**
     * Dense assignment is polynomial and therefore can cover the complete production actor
     * envelope. The older recursive search was intentionally removed because its worst-case cost
     * grew factorially and silently reduced association quality above eight subjects.
     */
    private val maxGlobalAssignmentItems: Int = 64,
) {
    data class Track(
        val id: Long,
        val subjectType: SubjectType,
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float,
        val velocityX: Float,
        val velocityY: Float,
        val confidence: Double,
        val ageFrames: Long,
        val missedFrames: Int,
        val lastTimestampNanos: Long,
        val detectorTrackingId: Int = -1,
    ) {
        val centerX: Float
            get() = (left + right) * 0.5f

        val centerY: Float
            get() = (top + bottom) * 0.5f

        val width: Float
            get() = (right - left).coerceAtLeast(0f)

        val height: Float
            get() = (bottom - top).coerceAtLeast(0f)
    }

    private data class MutableTrack(
        var id: Long,
        var subjectType: SubjectType,
        var left: Float,
        var top: Float,
        var right: Float,
        var bottom: Float,
        var velocityX: Float,
        var velocityY: Float,
        var confidence: Double,
        var ageFrames: Long,
        var missedFrames: Int,
        var lastTimestampNanos: Long,
        var detectorTrackingId: Int,
    ) {
        val centerX: Float
            get() = (left + right) * 0.5f

        val centerY: Float
            get() = (top + bottom) * 0.5f
    }

    private data class Prediction(
        val trackId: Long,
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float,
        val centerX: Float,
        val centerY: Float,
    )

    private data class Candidate(
        val trackId: Long,
        val detectionIndex: Int,
        val cost: Double,
    )

    private val tracks =
        LinkedHashMap<Long, MutableTrack>()

    private var nextId = 1L
    private var lastUpdateTimestampNanos =
        Long.MIN_VALUE

    init {
        require(maxMissedFrames >= 0)
        require(iouGate in 0.0..1.0)
        require(centerGateFraction > 0.0)
        require(velocitySmoothing in 0.0..1.0)
        require(maximumRetainedTracks in 1..64)
        require(maxGlobalAssignmentItems in 1..maximumRetainedTracks)
    }

    @Synchronized
    fun update(
        detections: List<TrackedObject>,
        timestampNanos: Long,
    ): List<Track> {
        require(timestampNanos >= 0L)

        if (
            lastUpdateTimestampNanos !=
                Long.MIN_VALUE &&
            timestampNanos <=
                lastUpdateTimestampNanos
        ) {
            // A capture/replay timestamp regression invalidates velocity integration. Reset the
            // temporal map rather than allowing a reversed interval to corrupt prediction.
            tracks.clear()
            lastUpdateTimestampNanos =
                timestampNanos

            if (detections.isEmpty()) {
                return emptyList()
            }
        }

        lastUpdateTimestampNanos =
            timestampNanos

        if (detections.isEmpty()) {
            advanceMissedTracks(timestampNanos)
            expireTracks()
            return tracks.values.map(::snapshot)
        }

        val predictions =
            tracks.values.associate { track ->
                track.id to predict(
                    track,
                    timestampNanos,
                )
            }

        val trackList =
            predictions.values.toList()

        val detectionLimit =
            min(
                detections.size,
                maxGlobalAssignmentItems,
            )

        val trackLimit =
            min(
                trackList.size,
                maxGlobalAssignmentItems,
            )

        val assignment =
            if (
                detectionLimit > 0 &&
                trackLimit > 0
            ) {
                globalAssignment(
                    tracks = trackList.take(trackLimit),
                    detections = detections.take(detectionLimit),
                )
            } else {
                emptyList()
            }

        val matchedTrackIds =
            assignment
                .map { it.trackId }
                .toHashSet()

        val matchedDetectionIndices =
            assignment
                .map { it.detectionIndex }
                .toHashSet()

        assignment.forEach { match ->
            val track =
                tracks[match.trackId]
                    ?: return@forEach

            val detection =
                detections[match.detectionIndex]

            updateTrack(
                track = track,
                detection = detection,
                timestampNanos = timestampNanos,
            )
        }

        tracks.keys
            .toList()
            .filter { it !in matchedTrackIds }
            .forEach { trackId ->
                val track =
                    tracks[trackId]
                        ?: return@forEach

                val prediction =
                    predictions[trackId]
                        ?: return@forEach

                track.left = prediction.left
                track.top = prediction.top
                track.right = prediction.right
                track.bottom = prediction.bottom
                track.confidence =
                    (
                        track.confidence *
                            0.82
                        ).coerceIn(
                            0.0,
                            1.0,
                        )
                track.missedFrames += 1
                track.lastTimestampNanos =
                    timestampNanos
            }

        detections.indices
            .filter { it !in matchedDetectionIndices }
            .forEach { index ->
                createTrack(
                    detections[index],
                    timestampNanos,
                )
            }

        expireTracks()
        enforceTrackCapacity()

        return tracks.values
            .map(::snapshot)
            .sortedBy { it.id }
    }

    @Synchronized
    fun current(): List<Track> =
        tracks.values
            .map(::snapshot)
            .sortedBy { it.id }

    @Synchronized
    fun snapshot(): List<Track> =
        tracks.values
            .map(::snapshot)
            .sortedBy { it.id }

    @Synchronized
    fun clear() {
        tracks.clear()
        nextId = 1L
        lastUpdateTimestampNanos =
            Long.MIN_VALUE
    }

    private fun globalAssignment(
        tracks: List<Prediction>,
        detections: List<TrackedObject>,
    ): List<Candidate> {
        if (tracks.isEmpty() || detections.isEmpty()) {
            return emptyList()
        }

        val boundedTracks =
            tracks.take(maxGlobalAssignmentItems)
        val boundedDetections =
            detections.take(maxGlobalAssignmentItems)

        val unmatchedScore = -0.72
        val forbiddenScore = -1_000_000.0
        val n =
            max(
                boundedTracks.size,
                boundedDetections.size,
            )

        val matrix =
            Array(n) {
                DoubleArray(n) {
                    unmatchedScore
                }
            }

        for (row in boundedTracks.indices) {
            val prediction =
                boundedTracks[row]

            for (column in boundedDetections.indices) {
                val detection =
                    boundedDetections[column]

                if (
                    detection.subjectType !=
                        tracksSubjectType(
                            prediction.trackId,
                        )
                ) {
                    matrix[row][column] =
                        forbiddenScore
                    continue
                }

                val score =
                    matchScore(
                        prediction = prediction,
                        detection = detection,
                    )

                matrix[row][column] =
                    if (score >= 0.10) {
                        score
                    } else {
                        forbiddenScore
                    }
            }
        }

        val assignment =
            maximumWeightAssignment(
                matrix = matrix,
                forbiddenScore = forbiddenScore,
            )

        return buildList {
            boundedTracks.indices.forEach { row ->
                val column =
                    assignment[row]

                if (
                    column !in
                    boundedDetections.indices
                ) {
                    return@forEach
                }

                val score =
                    matrix[row][column]

                if (
                    score <=
                        forbiddenScore
                ) {
                    return@forEach
                }

                if (score < 0.10) {
                    return@forEach
                }

                add(
                    Candidate(
                        trackId =
                            boundedTracks[row].trackId,
                        detectionIndex = column,
                        cost = 1.0 - score,
                    ),
                )
            }
        }
    }

    /**
     * Dense Hungarian maximization with explicit dummy rows/columns. The matrix remains bounded by
     * maxGlobalAssignmentItems, so 50-person association is deterministic and polynomial.
     */
    private fun maximumWeightAssignment(
        matrix: Array<DoubleArray>,
        forbiddenScore: Double,
    ): IntArray {
        val n = matrix.size
        if (n == 0) return IntArray(0)

        val u =
            DoubleArray(n + 1)
        val v =
            DoubleArray(n + 1)
        val p =
            IntArray(n + 1)
        val way =
            IntArray(n + 1)

        for (i in 1..n) {
            p[0] = i
            var j0 = 0
            val minv =
                DoubleArray(
                    n + 1,
                ) {
                    Double.POSITIVE_INFINITY
                }
            val used =
                BooleanArray(n + 1)

            do {
                used[j0] = true
                val i0 = p[j0]
                var delta =
                    Double.POSITIVE_INFINITY
                var j1 = 0

                for (j in 1..n) {
                    if (used[j]) continue

                    val weight =
                        matrix[i0 - 1][j - 1]

                    val cost =
                        if (
                            weight <=
                                forbiddenScore
                        ) {
                            1e12
                        } else {
                            -weight
                        }

                    val current =
                        cost -
                            u[i0] -
                            v[j]

                    if (
                        current <
                            minv[j]
                    ) {
                        minv[j] =
                            current
                        way[j] =
                            j0
                    }

                    if (
                        minv[j] <
                            delta
                    ) {
                        delta =
                            minv[j]
                        j1 =
                            j
                    }
                }

                if (
                    !delta.isFinite()
                ) {
                    break
                }

                for (j in 0..n) {
                    if (used[j]) {
                        u[p[j]] += delta
                        v[j] -= delta
                    } else {
                        minv[j] -= delta
                    }
                }

                j0 = j1
            } while (
                j0 != 0 &&
                p[j0] != 0
            )

            if (
                j0 == 0 &&
                p[j0] != 0
            ) {
                continue
            }

            do {
                val j1 =
                    way[j0]
                p[j0] =
                    p[j1]
                j0 =
                    j1
            } while (j0 != 0)
        }

        val rowToColumn =
            IntArray(n) { -1 }

        for (j in 1..n) {
            if (p[j] != 0) {
                rowToColumn[
                    p[j] - 1
                ] =
                    j - 1
            }
        }

        return rowToColumn
    }

    private fun matchScore(
        prediction: Prediction,
        detection: TrackedObject,
    ): Double {
        val iou =
            iou(
                prediction,
                detection,
            )

        val distance =
            sqrt(
                (
                    prediction.centerX -
                        (
                            detection.left +
                                detection.right
                            ) * 0.5f
                    ).toDouble()
                        .let { it * it } +
                        (
                            prediction.centerY -
                                (
                                    detection.top +
                                        detection.bottom
                                    ) * 0.5f
                            ).toDouble()
                            .let { it * it },
            )

        val diagonal =
            sqrt(
                (
                    prediction.right -
                        prediction.left
                    ).toDouble()
                        .let { it * it } +
                        (
                            prediction.bottom -
                                prediction.top
                            ).toDouble()
                            .let { it * it },
            ).coerceAtLeast(
                1.0,
            )

        val centerScore =
            1.0 -
                (
                    distance /
                        diagonal
                    ).coerceIn(
                        0.0,
                        1.0,
                    )

        val predictedTrack =
            tracks[prediction.trackId]

        val detectorIdBonus =
            if (
                predictedTrack != null &&
                detection.trackingId >= 0 &&
                predictedTrack.detectorTrackingId ==
                    detection.trackingId
            ) {
                0.25
            } else {
                0.0
            }

        val semanticAgreement =
            if (
                predictedTrack?.subjectType ==
                    detection.subjectType
            ) {
                1.0
            } else {
                0.0
            }

        return (
            0.44 * iou +
                0.26 * centerScore +
                0.12 *
                detection.confidence
                    .coerceIn(
                        0.0,
                        1.0,
                    ) +
                0.10 *
                semanticAgreement +
                detectorIdBonus
            ).coerceIn(
                0.0,
                1.0,
            )
    }

    private fun tracksSubjectType(
        trackId: Long,
    ): SubjectType? =
        tracks[trackId]
            ?.subjectType

    private fun predict(
        track: MutableTrack,
        timestampNanos: Long,
    ): Prediction {
        val dt =
            (
                (
                    timestampNanos -
                        track.lastTimestampNanos
                    ).coerceAtLeast(
                        1L,
                    ) / 1e9
                ).coerceIn(
                    1.0 / 240.0,
                    0.5,
                )

        val dx =
            track.velocityX *
                dt.toFloat()

        val dy =
            track.velocityY *
                dt.toFloat()

        return Prediction(
            trackId = track.id,
            left = track.left + dx,
            top = track.top + dy,
            right = track.right + dx,
            bottom = track.bottom + dy,
            centerX =
                track.centerX + dx,
            centerY =
                track.centerY + dy,
        )
    }

    private fun updateTrack(
        track: MutableTrack,
        detection: TrackedObject,
        timestampNanos: Long,
    ) {
        val dt =
            (
                (
                    timestampNanos -
                        track.lastTimestampNanos
                    ).coerceAtLeast(
                        1L,
                    ) / 1e9
                ).coerceIn(
                    1.0 / 240.0,
                    0.5,
                )

        val measuredVx =
            (
                (
                    detection.left +
                        detection.right
                    ) * 0.5f -
                    track.centerX
                ) /
                dt.toFloat()

        val measuredVy =
            (
                (
                    detection.top +
                        detection.bottom
                    ) * 0.5f -
                    track.centerY
                ) /
                dt.toFloat()

        track.velocityX =
            (
                track.velocityX *
                    (1.0 -
                        velocitySmoothing) +
                    measuredVx *
                    velocitySmoothing
                ).toFloat()

        track.velocityY =
            (
                track.velocityY *
                    (1.0 -
                        velocitySmoothing) +
                    measuredVy *
                    velocitySmoothing
                ).toFloat()

        val gain =
            if (
                track.missedFrames == 0
            ) {
                0.78f
            } else {
                0.92f
            }

        track.left +=
            (
                detection.left -
                    track.left
                ) * gain
        track.top +=
            (
                detection.top -
                    track.top
                ) * gain
        track.right +=
            (
                detection.right -
                    track.right
                ) * gain
        track.bottom +=
            (
                detection.bottom -
                    track.bottom
                ) * gain

        track.confidence =
            (
                track.confidence *
                    0.35 +
                    detection.confidence
                        .coerceIn(
                            0.0,
                            1.0,
                        ) *
                    0.65
                ).coerceIn(
                    0.0,
                    1.0,
                )

        track.ageFrames += 1
        track.missedFrames = 0
        track.lastTimestampNanos =
            timestampNanos
        track.detectorTrackingId =
            detection.trackingId
    }

    private fun createTrack(
        detection: TrackedObject,
        timestampNanos: Long,
    ) {
        val id =
            if (
                detection.trackingId >= 0 &&
                tracks.values.none {
                    it.detectorTrackingId ==
                        detection.trackingId
                }
            ) {
                detection.trackingId.toLong()
            } else {
                nextId++
            }

        val resolvedId =
            if (
                tracks.containsKey(id)
            ) {
                nextId++
            } else {
                id
            }

        tracks[resolvedId] =
            MutableTrack(
                id = resolvedId,
                subjectType =
                    detection.subjectType,
                left = detection.left,
                top = detection.top,
                right = detection.right,
                bottom = detection.bottom,
                velocityX = 0f,
                velocityY = 0f,
                confidence =
                    detection.confidence
                        .coerceIn(
                            0.0,
                            1.0,
                        ),
                ageFrames = 1L,
                missedFrames = 0,
                lastTimestampNanos =
                    timestampNanos,
                detectorTrackingId =
                    detection.trackingId,
            )
    }

    private fun advanceMissedTracks(
        timestampNanos: Long,
    ) {
        tracks.values.forEach { track ->
            val prediction =
                predict(
                    track,
                    timestampNanos,
                )
            track.left = prediction.left
            track.top = prediction.top
            track.right = prediction.right
            track.bottom = prediction.bottom
            track.confidence *= 0.80
            track.missedFrames += 1
            track.lastTimestampNanos =
                timestampNanos
        }
    }

    private fun enforceTrackCapacity() {
        while (
            tracks.size >
                maximumRetainedTracks
        ) {
            val victim =
                tracks.values
                    .minWithOrNull(
                        compareBy<MutableTrack> {
                            it.confidence
                        }.thenBy {
                            it.lastTimestampNanos
                        }.thenBy {
                            it.id
                        },
                    )
                    ?.id
                    ?: break

            tracks.remove(victim)
        }
    }

    private fun expireTracks() {
        tracks.values
            .filter {
                it.missedFrames >
                    maxMissedFrames
            }
            .map { it.id }
            .forEach(tracks::remove)
    }

    private fun iou(
        prediction: Prediction,
        detection: TrackedObject,
    ): Double {
        val left =
            max(
                prediction.left,
                detection.left,
            )
        val top =
            max(
                prediction.top,
                detection.top,
            )
        val right =
            min(
                prediction.right,
                detection.right,
            )
        val bottom =
            min(
                prediction.bottom,
                detection.bottom,
            )

        val intersection =
            (
                right - left
                ).coerceAtLeast(0f) *
                (
                    bottom - top
                    ).coerceAtLeast(0f)

        val predictionArea =
            (
                prediction.right -
                    prediction.left
                ).coerceAtLeast(0f) *
                (
                    prediction.bottom -
                        prediction.top
                    ).coerceAtLeast(0f)

        val detectionArea =
            (
                detection.right -
                    detection.left
                ).coerceAtLeast(0f) *
                (
                    detection.bottom -
                        detection.top
                    ).coerceAtLeast(0f)

        val union =
            predictionArea +
                detectionArea -
                intersection

        return if (
            union <= 1e-6f
        ) {
            0.0
        } else {
            (
                intersection /
                    union
                ).toDouble()
        }
    }

    private fun snapshot(
        track: MutableTrack,
    ): Track =
        Track(
            id = track.id,
            subjectType =
                track.subjectType,
            left = track.left,
            top = track.top,
            right = track.right,
            bottom = track.bottom,
            velocityX =
                track.velocityX,
            velocityY =
                track.velocityY,
            confidence =
                track.confidence,
            ageFrames =
                track.ageFrames,
            missedFrames =
                track.missedFrames,
            lastTimestampNanos =
                track.lastTimestampNanos,
            detectorTrackingId =
                track.detectorTrackingId,
        )
}

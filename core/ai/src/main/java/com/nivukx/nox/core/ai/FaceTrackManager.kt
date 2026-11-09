package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.FaceMeshPoint
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Global face association engine.
 *
 * Faces are matched with a small Hungarian assignment rather than a greedy nearest-neighbour pass.
 * The cost combines predicted position, box overlap, scale, head pose, detector identity and the
 * normalized facial landmark signature. Short detector dropouts are predicted forward, while
 * confidence is decayed so a stale face can never masquerade as a fresh detection indefinitely.
 */
class FaceTrackManager(
    private val maxMissedFrames: Int = 8,
    private val minimumMatchScore: Double = 0.34,
    private val maximumTracks: Int = 128,
) {
    init {
        require(maxMissedFrames >= 1)
        require(minimumMatchScore in 0.0..1.0)
        require(maximumTracks >= 1)
    }
    data class Track(
        val id: Long,
        val frame: FaceFrame,
        val velocityX: Float,
        val velocityY: Float,
        val scaleVelocity: Float,
        val yawVelocity: Float,
        val rollVelocity: Float,
        val ageFrames: Long,
        val missedFrames: Int,
        val confidence: Double,
        val identityConfidence: Double,
    )

    private data class MutableTrack(
        var id: Long,
        var frame: FaceFrame,
        var velocityX: Float,
        var velocityY: Float,
        var scaleVelocity: Float,
        var yawVelocity: Float,
        var rollVelocity: Float,
        var ageFrames: Long,
        var missedFrames: Int,
        var confidence: Double,
        var identityConfidence: Double,
        var lastTimestampNanos: Long,
    )

    private val tracks = LinkedHashMap<Long, MutableTrack>()
    private var nextId = 1L

    @Synchronized
    fun update(frames: List<FaceFrame>): List<Track> {
        if (frames.isEmpty()) {
            predictAll(
                timestampNanos = tracks.values.maxOfOrNull {
                    it.lastTimestampNanos
                } ?: 0L,
            )
            expire()
            return tracks.values.map(::snapshot)
        }

        val timestampNanos = frames.maxOf { it.timestampNanos }
        val trackList = tracks.values.toList()
        val scoreMatrix = Array(trackList.size) { DoubleArray(frames.size) { -1.0 } }

        trackList.forEachIndexed { trackIndex, track ->
            val predicted = predictedFrame(track, timestampNanos)
            frames.forEachIndexed { frameIndex, frame ->
                scoreMatrix[trackIndex][frameIndex] =
                    associationScore(
                        track = track,
                        predicted = predicted,
                        detection = frame,
                    )
            }
        }

        val assignments = hungarianMaximum(scoreMatrix)
        val matchedTrackIds = HashSet<Long>()
        val matchedFrames = HashSet<Int>()

        assignments.forEach { (trackIndex, frameIndex) ->
            if (trackIndex !in trackList.indices ||
                frameIndex !in frames.indices
            ) {
                return@forEach
            }

            val score = scoreMatrix[trackIndex][frameIndex]
            if (score < minimumMatchScore) return@forEach

            val track = trackList[trackIndex]
            updateTrack(track, frames[frameIndex], score)
            matchedTrackIds += track.id
            matchedFrames += frameIndex
        }

        trackList
            .filter { it.id !in matchedTrackIds }
            .forEach { predictTrack(it, timestampNanos) }

        frames.indices
            .filterNot { it in matchedFrames }
            .forEach { index ->
                createTrack(frames[index])
            }

        expire()
        enforceTrackCapacity()
        return tracks.values.map(::snapshot)
    }

    @Synchronized
    fun snapshot(): List<Track> =
        tracks.values
            .map(::snapshot)
            .sortedBy { it.id }

    @Synchronized
    fun clear() {
        tracks.clear()
        nextId = 1L
    }

    private fun createTrack(frame: FaceFrame) {
        val detectorId = frame.trackingId?.toLong()
        val id = if (
            detectorId != null &&
            detectorId > 0 &&
            !tracks.containsKey(detectorId)
        ) {
            detectorId
        } else {
            while (tracks.containsKey(nextId)) nextId++
            nextId++
        }

        tracks[id] = MutableTrack(
            id = id,
            frame = frame,
            velocityX = 0f,
            velocityY = 0f,
            scaleVelocity = 0f,
            yawVelocity = 0f,
            rollVelocity = 0f,
            ageFrames = 1L,
            missedFrames = 0,
            confidence = frameConfidence(frame),
            identityConfidence = if (detectorId != null) 0.85 else 0.50,
            lastTimestampNanos = frame.timestampNanos,
        )
    }

    private fun enforceTrackCapacity() {
        if (tracks.size <= maximumTracks) return

        val overflow =
            tracks.size - maximumTracks

        tracks.entries
            .sortedWith(
                compareBy<MutableMap.MutableEntry<Long, MutableTrack>> {
                    it.value.confidence
                }.thenBy {
                    it.value.lastTimestampNanos
                },
            )
            .take(overflow)
            .map { it.key }
            .forEach(tracks::remove)
    }

    private fun associationScore(
        track: MutableTrack,
        predicted: FaceFrame,
        detection: FaceFrame,
    ): Double {
        val overlap = iou(predicted, detection)
        val distance = centerDistance(predicted, detection)
        val predictedSize = max(
            predicted.boundsRight - predicted.boundsLeft,
            predicted.boundsBottom - predicted.boundsTop,
        ).coerceAtLeast(1f)

        val centerScore = expFalloff(
            distance / predictedSize.toDouble(),
            1.8,
        )

        val predictedScale = faceScale(predicted)
        val detectedScale = faceScale(detection)
        val scaleError = abs(
            kotlin.math.ln(
                (detectedScale / predictedScale)
                    .coerceIn(0.25f, 4f)
                    .toDouble(),
            ),
        )
        val scaleScore = expFalloff(scaleError, 1.5)

        val yawScore = 1.0 - (
            abs(detection.yawDegrees - predicted.yawDegrees) / 60.0
            ).coerceIn(0.0, 1.0)

        val rollScore = 1.0 - (
            abs(detection.rollDegrees - predicted.rollDegrees) / 45.0
            ).coerceIn(0.0, 1.0)

        val shapeScore = shapeSimilarity(
            predicted.points,
            detection.points,
        )

        val detectorTrackingId = detection.trackingId
        val detectorScore = if (
            detectorTrackingId != null &&
            detectorTrackingId.toLong() == track.id
        ) 1.0 else 0.0

        return (
            0.30 * overlap +
                0.22 * centerScore +
                0.12 * scaleScore +
                0.12 * yawScore +
                0.06 * rollScore +
                0.13 * shapeScore +
                0.05 * detectorScore
            ).coerceIn(0.0, 1.0)
    }

    private fun updateTrack(
        track: MutableTrack,
        detection: FaceFrame,
        associationScore: Double,
    ) {
        val previous = track.frame
        val dt = (
            (detection.timestampNanos - track.lastTimestampNanos)
                .coerceAtLeast(1L) / 1e9
            ).coerceIn(1.0 / 240.0, 0.5)

        val previousCenterX = centerX(previous)
        val previousCenterY = centerY(previous)
        val detectionCenterX = centerX(detection)
        val detectionCenterY = centerY(detection)

        val predicted = predictedFrame(track, detection.timestampNanos)
        val innovationX = detectionCenterX - centerX(predicted)
        val innovationY = detectionCenterY - centerY(predicted)

        val measuredVelocityX = innovationX / dt.toFloat()
        val measuredVelocityY = innovationY / dt.toFloat()

        track.velocityX += (
            measuredVelocityX - track.velocityX
            ) * 0.46f
        track.velocityY += (
            measuredVelocityY - track.velocityY
            ) * 0.46f

        val previousScale = faceScale(previous)
        val currentScale = faceScale(detection)
        val measuredScaleVelocity = (
            (currentScale - previousScale) / dt.toFloat()
            )

        track.scaleVelocity += (
            measuredScaleVelocity - track.scaleVelocity
            ) * 0.35f

        track.yawVelocity += (
            (detection.yawDegrees - previous.yawDegrees) / dt.toFloat() -
                track.yawVelocity
            ) * 0.38f

        track.rollVelocity += (
            (detection.rollDegrees - previous.rollDegrees) / dt.toFloat() -
                track.rollVelocity
            ) * 0.38f

        val gain = when {
            track.missedFrames == 0 -> 0.72f
            track.missedFrames <= 2 -> 0.88f
            else -> 0.95f
        }

        track.frame = blendFace(
            previous = predicted,
            current = detection,
            gain = gain,
        )

        track.ageFrames += 1L
        track.missedFrames = 0
        track.lastTimestampNanos = detection.timestampNanos

        val detectionConfidence = frameConfidence(detection)
        track.confidence = (
            track.confidence * 0.34 +
                detectionConfidence * 0.46 +
                associationScore * 0.20
            ).coerceIn(0.0, 1.0)

        track.identityConfidence = (
            track.identityConfidence * 0.70 +
                associationScore * 0.30
            ).coerceIn(0.0, 1.0)
    }

    private fun predictAll(timestampNanos: Long) {
        tracks.values.forEach {
            predictTrack(it, timestampNanos)
        }
    }

    private fun predictTrack(
        track: MutableTrack,
        timestampNanos: Long,
    ) {
        val prediction = predictedFrame(track, timestampNanos)
        track.frame = prediction
        track.lastTimestampNanos = timestampNanos
        track.missedFrames += 1
        track.confidence *= 0.86
        track.identityConfidence *= 0.94
    }

    private fun predictedFrame(
        track: MutableTrack,
        timestampNanos: Long,
    ): FaceFrame {
        val dt = (
            (timestampNanos - track.lastTimestampNanos)
                .coerceAtLeast(0L) / 1e9
            ).coerceIn(0.0, 0.5)

        val dx = track.velocityX * dt.toFloat()
        val dy = track.velocityY * dt.toFloat()
        val scaleDelta = track.scaleVelocity * dt.toFloat()

        val left = track.frame.boundsLeft + dx - scaleDelta * 0.5f
        val right = track.frame.boundsRight + dx + scaleDelta * 0.5f
        val top = track.frame.boundsTop + dy - scaleDelta * 0.5f
        val bottom = track.frame.boundsBottom + dy + scaleDelta * 0.5f

        return track.frame.copy(
            boundsLeft = left,
            boundsTop = top,
            boundsRight = right,
            boundsBottom = bottom,
            yawDegrees = track.frame.yawDegrees +
                track.yawVelocity * dt.toFloat(),
            rollDegrees = track.frame.rollDegrees +
                track.rollVelocity * dt.toFloat(),
            timestampNanos = timestampNanos,
        )
    }

    private fun blendFace(
        previous: FaceFrame,
        current: FaceFrame,
        gain: Float,
    ): FaceFrame {
        fun blend(a: Float, b: Float) = a + (b - a) * gain

        val previousPoints = previous.points.associateBy { it.index }
        val currentPoints = current.points.associateBy { it.index }

        val blendedPoints = current.points.map { point ->
            val old = previousPoints[point.index]
                ?: return@map point

            FaceMeshPoint(
                index = point.index,
                point = old.point.lerp(point.point, gain.toDouble()),
                confidence = (
                    old.confidence * (1.0 - gain) +
                        point.confidence * gain
                    ).coerceIn(0.0, 1.0),
            )
        }

        return current.copy(
            boundsLeft = blend(previous.boundsLeft, current.boundsLeft),
            boundsTop = blend(previous.boundsTop, current.boundsTop),
            boundsRight = blend(previous.boundsRight, current.boundsRight),
            boundsBottom = blend(previous.boundsBottom, current.boundsBottom),
            yawDegrees = blend(previous.yawDegrees, current.yawDegrees),
            rollDegrees = blend(previous.rollDegrees, current.rollDegrees),
            points = blendedPoints,
            leftEyeCenter = blendOptional(
                previous.leftEyeCenter,
                current.leftEyeCenter,
                gain,
            ),
            rightEyeCenter = blendOptional(
                previous.rightEyeCenter,
                current.rightEyeCenter,
                gain,
            ),
        )
    }

    private fun blendOptional(
        previous: com.nivukx.nox.core.model.Vec2?,
        current: com.nivukx.nox.core.model.Vec2?,
        gain: Float,
    ): com.nivukx.nox.core.model.Vec2? = when {
        previous == null -> current
        current == null -> previous
        else -> previous.lerp(current, gain.toDouble())
    }

    private fun shapeSimilarity(
        a: List<FaceMeshPoint>,
        b: List<FaceMeshPoint>,
    ): Double {
        if (a.isEmpty() || b.isEmpty()) return 0.0

        val aa = a.associateBy { it.index }
        val bb = b.associateBy { it.index }
        val common = aa.keys.intersect(bb.keys)
        if (common.size < 6) return 0.0

        val aBounds = bounds(a)
        val bBounds = bounds(b)
        val aw = (aBounds[2] - aBounds[0]).coerceAtLeast(1f)
        val ah = (aBounds[3] - aBounds[1]).coerceAtLeast(1f)
        val bw = (bBounds[2] - bBounds[0]).coerceAtLeast(1f)
        val bh = (bBounds[3] - bBounds[1]).coerceAtLeast(1f)

        var total = 0.0
        var count = 0

        common.take(80).forEach { index ->
            val ap = aa.getValue(index).point
            val bp = bb.getValue(index).point

            val ax = (ap.x - aBounds[0]) / aw
            val ay = (ap.y - aBounds[1]) / ah
            val bx = (bp.x - bBounds[0]) / bw
            val by = (bp.y - bBounds[1]) / bh

            total += expFalloff(
                sqrt(
                    (ax - bx) * (ax - bx) +
                        (ay - by) * (ay - by),
                ),
                0.25,
            )
            count += 1
        }

        return if (count == 0) 0.0
        else (total / count).coerceIn(0.0, 1.0)
    }

    private fun bounds(
        points: List<FaceMeshPoint>,
    ): FloatArray {
        return floatArrayOf(
            points.minOf { it.point.x.toFloat() },
            points.minOf { it.point.y.toFloat() },
            points.maxOf { it.point.x.toFloat() },
            points.maxOf { it.point.y.toFloat() },
        )
    }

    private fun faceScale(frame: FaceFrame): Float =
        max(
            frame.boundsRight - frame.boundsLeft,
            frame.boundsBottom - frame.boundsTop,
        ).coerceAtLeast(1f)

    private fun centerX(frame: FaceFrame): Float =
        (frame.boundsLeft + frame.boundsRight) * 0.5f

    private fun centerY(frame: FaceFrame): Float =
        (frame.boundsTop + frame.boundsBottom) * 0.5f

    private fun centerDistance(
        a: FaceFrame,
        b: FaceFrame,
    ): Double {
        val dx = centerX(a) - centerX(b)
        val dy = centerY(a) - centerY(b)
        return sqrt(dx.toDouble() * dx + dy.toDouble() * dy)
    }

    private fun iou(
        a: FaceFrame,
        b: FaceFrame,
    ): Double {
        val left = max(a.boundsLeft, b.boundsLeft)
        val top = max(a.boundsTop, b.boundsTop)
        val right = min(a.boundsRight, b.boundsRight)
        val bottom = min(a.boundsBottom, b.boundsBottom)

        val intersection = (
            (right - left).coerceAtLeast(0f) *
                (bottom - top).coerceAtLeast(0f)
            )

        val areaA = (
            (a.boundsRight - a.boundsLeft).coerceAtLeast(0f) *
                (a.boundsBottom - a.boundsTop).coerceAtLeast(0f)
            )
        val areaB = (
            (b.boundsRight - b.boundsLeft).coerceAtLeast(0f) *
                (b.boundsBottom - b.boundsTop).coerceAtLeast(0f)
            )

        val union = areaA + areaB - intersection
        return if (union <= 1e-6f) 0.0
        else (intersection / union).toDouble()
    }

    private fun frameConfidence(
        frame: FaceFrame,
    ): Double {
        val pointConfidence = if (frame.points.isEmpty()) {
            0.0
        } else {
            frame.points.map { it.confidence }.average()
        }

        val eyeConfidence = (
            if (frame.leftEyeCenter != null) 0.5 else 0.0
        ) + (
            if (frame.rightEyeCenter != null) 0.5 else 0.0
        )

        return (
            0.70 * pointConfidence +
                0.20 * eyeConfidence +
                0.10 * (
                    1.0 - (
                        abs(frame.yawDegrees.toDouble()) / 90.0
                    ).coerceIn(0.0, 1.0)
                )
            ).coerceIn(0.0, 1.0)
    }

    private fun expFalloff(
        error: Double,
        scale: Double,
    ): Double =
        kotlin.math.exp(
            -(
                error.coerceAtLeast(0.0) /
                    scale.coerceAtLeast(1e-6)
                ),
        ).coerceIn(0.0, 1.0)

    /**
     * Maximum-weight bipartite assignment for small face populations.
     * Faces are normally <= 4 on a phone, so cubic complexity is preferable to a greedy identity
     * decision that can cascade into track swaps.
     */
    private fun hungarianMaximum(
        score: Array<DoubleArray>,
    ): List<Pair<Int, Int>> {
        if (score.isEmpty() || score[0].isEmpty()) return emptyList()

        val rows = score.size
        val columns = score[0].size
        val n = max(rows, columns)
        val cost = Array(n) { DoubleArray(n) { 1.0 } }

        for (r in 0 until rows) {
            for (c in 0 until columns) {
                cost[r][c] = 1.0 - score[r][c].coerceIn(0.0, 1.0)
            }
        }

        val u = DoubleArray(n + 1)
        val v = DoubleArray(n + 1)
        val p = IntArray(n + 1)
        val way = IntArray(n + 1)

        for (i in 1..n) {
            p[0] = i
            var j0 = 0
            val minv = DoubleArray(n + 1) { Double.POSITIVE_INFINITY }
            val used = BooleanArray(n + 1)

            do {
                used[j0] = true
                val i0 = p[j0]
                var delta = Double.POSITIVE_INFINITY
                var j1 = 0

                for (j in 1..n) {
                    if (used[j]) continue
                    val current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if (current < minv[j]) {
                        minv[j] = current
                        way[j] = j0
                    }
                    if (minv[j] < delta) {
                        delta = minv[j]
                        j1 = j
                    }
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
            } while (p[j0] != 0)

            do {
                val j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while (j0 != 0)
        }

        val result = ArrayList<Pair<Int, Int>>()
        for (j in 1..n) {
            val row = p[j] - 1
            val column = j - 1
            if (row in 0 until rows && column in 0 until columns) {
                result += row to column
            }
        }
        return result
    }

    private fun expire() {
        tracks.values
            .filter { it.missedFrames > maxMissedFrames }
            .map { it.id }
            .forEach(tracks::remove)
    }

    private fun snapshot(track: MutableTrack): Track =
        Track(
            id = track.id,
            frame = track.frame,
            velocityX = track.velocityX,
            velocityY = track.velocityY,
            scaleVelocity = track.scaleVelocity,
            yawVelocity = track.yawVelocity,
            rollVelocity = track.rollVelocity,
            ageFrames = track.ageFrames,
            missedFrames = track.missedFrames,
            confidence = track.confidence,
            identityConfidence = track.identityConfidence,
        )
}

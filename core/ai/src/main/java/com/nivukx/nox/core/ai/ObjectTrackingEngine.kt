package com.nivukx.nox.core.ai

import android.graphics.Rect
import android.media.Image
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.DetectedObject
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TrackedObject
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min

/**
 * General semantic subject front-end.
 *
 * The detector remains responsible for visual recognition while this layer owns:
 * - broad human/animal/object normalization;
 * - temporal box stabilization;
 * - confidence calibration;
 * - stale-track expiration;
 * - malformed-box rejection;
 * - bounded track memory.
 *
 * NoX deliberately keeps the subject representation generic so the same downstream motion,
 * attention and reconstruction graph can follow people, animals and ordinary moving objects.
 */
class ObjectTrackingEngine(
    private val minimumDetectorConfidence: Double = 0.05,
    private val smoothingAlpha: Double = 0.34,
    private val staleAfterNanos: Long = 360_000_000L,
    private val maximumTracks: Int = 64,
) : AutoCloseable {

    private data class TrackState(
        val id: Int,
        var subjectType: SubjectType,
        var left: Float,
        var top: Float,
        var right: Float,
        var bottom: Float,
        var confidence: Double,
        var lastTimestampNanos: Long,
        var seenFrames: Long,
        var missedFrames: Int,
    )

    private val closed = AtomicBoolean(false)
    private val lock = Any()

    private val animalLabels = setOf(
        "animal", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "rabbit", "mouse", "hamster", "fish",
        "tiger", "lion", "leopard", "deer", "monkey", "goat", "pig", "chicken",
        "duck", "goose", "penguin", "squirrel", "fox", "wolf", "buffalo",
        "camel", "donkey", "insect", "butterfly", "bee", "spider", "reptile",
        "snake", "lizard", "turtle", "frog", "fish", "shark",
    )

    private val humanLabels = setOf(
        "person", "people", "human", "man", "woman", "child", "boy", "girl",
    )

    private val tracks = LinkedHashMap<Int, TrackState>(32, 0.75f, true)

    private val detector = ObjectDetection.getClient(
        ObjectDetectorOptions.Builder()
            .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
            .enableMultipleObjects()
            .enableClassification()
            .build(),
    )

    @Synchronized
    fun process(
        image: Image,
        rotationDegrees: Int,
        timestampNanos: Long,
    ): Task<List<TrackedObject>> {
        check(!closed.get()) { "ObjectTrackingEngine is closed" }
        require(timestampNanos >= 0L) { "timestampNanos must be non-negative" }

        val input = InputImage.fromMediaImage(image, rotationDegrees)

        return detector.process(input).continueWith { task ->
            if (!task.isSuccessful) {
                throw task.exception
                    ?: IllegalStateException("Object detection failed")
            }

            val observations = task.result
                .mapNotNull { mapObject(it, timestampNanos) }

            synchronized(lock) {
                fuseTemporalObservations(
                    observations = observations,
                    timestampNanos = timestampNanos,
                )
            }
        }
    }

    fun snapshot(
        referenceTimestampNanos: Long,
    ): List<TrackedObject> =
        synchronized(lock) {
            tracks.values
                .filter {
                    timestampIsFresh(
                        it.lastTimestampNanos,
                        referenceTimestampNanos,
                    )
                }
                .map { state ->
                    TrackedObject(
                        trackingId = state.id,
                        subjectType = state.subjectType,
                        left = state.left,
                        top = state.top,
                        right = state.right,
                        bottom = state.bottom,
                        confidence = state.confidence,
                        timestampNanos = state.lastTimestampNanos,
                    )
                }
        }

    fun reset() {
        synchronized(lock) {
            tracks.clear()
        }
    }

    private fun fuseTemporalObservations(
        observations: List<TrackedObject>,
        timestampNanos: Long,
    ): List<TrackedObject> {
        val currentIds = HashSet<Int>(observations.size)

        observations.forEach { observation ->
            if (observation.trackingId < 0) return@forEach

            currentIds += observation.trackingId
            val existing = tracks[observation.trackingId]

            if (existing == null) {
                tracks[observation.trackingId] = TrackState(
                    id = observation.trackingId,
                    subjectType = observation.subjectType,
                    left = observation.left,
                    top = observation.top,
                    right = observation.right,
                    bottom = observation.bottom,
                    confidence = observation.confidence,
                    lastTimestampNanos = timestampNanos,
                    seenFrames = 1L,
                    missedFrames = 0,
                )
                return@forEach
            }

            val alpha = adaptiveAlpha(
                previousConfidence = existing.confidence,
                currentConfidence = observation.confidence,
                previousType = existing.subjectType,
                currentType = observation.subjectType,
            )

            existing.left = lerp(existing.left, observation.left, alpha)
            existing.top = lerp(existing.top, observation.top, alpha)
            existing.right = lerp(existing.right, observation.right, alpha)
            existing.bottom = lerp(existing.bottom, observation.bottom, alpha)
            existing.confidence = (
                existing.confidence * (1.0 - alpha) +
                    observation.confidence * alpha
                ).coerceIn(0.0, 1.0)
            existing.subjectType = resolveType(
                existing.subjectType,
                observation.subjectType,
                observation.confidence,
            )
            existing.lastTimestampNanos = timestampNanos
            existing.seenFrames += 1L
            existing.missedFrames = 0
        }

        tracks.values.forEach { state ->
            if (state.id !in currentIds) {
                state.missedFrames += 1
                state.confidence *= 0.88
            }
        }

        val expiryCutoff = timestampNanos - staleAfterNanos
        tracks.entries.removeIf { (_, state) ->
            state.lastTimestampNanos < expiryCutoff ||
                state.missedFrames > 12
        }

        enforceTrackLimit()

        return observations
            .mapNotNull { observation ->
                if (observation.trackingId < 0) {
                    observation
                } else {
                    tracks[observation.trackingId]?.let { state ->
                        TrackedObject(
                            trackingId = state.id,
                            subjectType = state.subjectType,
                            left = state.left,
                            top = state.top,
                            right = state.right,
                            bottom = state.bottom,
                            confidence = state.confidence,
                            timestampNanos = state.lastTimestampNanos,
                        )
                    }
                }
            }
            .distinctBy { it.trackingId }
    }

    private fun mapObject(
        value: DetectedObject,
        timestampNanos: Long,
    ): TrackedObject? {
        val box = sanitizeBounds(value.boundingBox) ?: return null

        val labels = value.labels
            .mapNotNull { label ->
                val normalized = normalizeLabel(label.text)
                if (normalized.isEmpty()) null
                else normalized to label.confidence.toDouble().coerceIn(0.0, 1.0)
            }

        val strongestLabel = labels.maxByOrNull { it.second }
        val type = classify(labels)

        val classifierConfidence = strongestLabel?.second ?: 0.0
        val detectorConfidence = when {
            classifierConfidence >= minimumDetectorConfidence ->
                classifierConfidence
            else ->
                boxQuality(box)
        }

        return TrackedObject(
            trackingId = value.trackingId ?: -1,
            subjectType = type,
            left = box.left.toFloat(),
            top = box.top.toFloat(),
            right = box.right.toFloat(),
            bottom = box.bottom.toFloat(),
            confidence = detectorConfidence.coerceIn(0.0, 1.0),
            timestampNanos = timestampNanos,
        )
    }

    private fun classify(
        labels: List<Pair<String, Double>>,
    ): SubjectType {
        if (labels.isEmpty()) return SubjectType.UNKNOWN

        val humanScore = labels
            .filter { tokenMatches(it.first, humanLabels) }
            .maxOfOrNull { it.second }
            ?: 0.0

        val animalScore = labels
            .filter { tokenMatches(it.first, animalLabels) }
            .maxOfOrNull { it.second }
            ?: 0.0

        return when {
            humanScore >= animalScore &&
                humanScore >= minimumDetectorConfidence ->
                SubjectType.HUMAN

            animalScore >= minimumDetectorConfidence ->
                SubjectType.ANIMAL

            else ->
                SubjectType.OBJECT
        }
    }

    private fun tokenMatches(
        label: String,
        vocabulary: Set<String>,
    ): Boolean {
        if (label in vocabulary) return true
        return label
            .split(' ', '/', ':')
            .any { it in vocabulary }
    }

    private fun normalizeLabel(text: String?): String {
        return text
            ?.trim()
            ?.lowercase(Locale.US)
            ?.replace('_', ' ')
            ?.replace('-', ' ')
            ?.replace(Regex("\\s+"), " ")
            ?.take(80)
            .orEmpty()
    }

    private fun sanitizeBounds(rect: Rect): Rect? {
        val left = rect.left
        val top = rect.top
        val right = rect.right
        val bottom = rect.bottom

        if (left < 0 && right <= 0) return null
        if (top < 0 && bottom <= 0) return null
        if (right <= left || bottom <= top) return null
        return Rect(
            left,
            top,
            right,
            bottom,
        )
    }

    private fun boxQuality(rect: Rect): Double {
        val area = (
            max(0, rect.width()) *
                max(0, rect.height())
            ).toDouble()
        if (area <= 0.0) return 0.0

        val linearQuality = (
            kotlin.math.sqrt(area) / 480.0
            ).coerceIn(0.0, 1.0)

        return 0.15 + 0.35 * linearQuality
    }

    private fun adaptiveAlpha(
        previousConfidence: Double,
        currentConfidence: Double,
        previousType: SubjectType,
        currentType: SubjectType,
    ): Double {
        var alpha = smoothingAlpha

        if (currentConfidence > previousConfidence + 0.20) {
            alpha += 0.18
        }

        if (currentType != previousType) {
            alpha *= 0.55
        }

        return alpha.coerceIn(0.12, 0.78)
    }

    private fun resolveType(
        previous: SubjectType,
        current: SubjectType,
        confidence: Double,
    ): SubjectType {
        if (previous == SubjectType.UNKNOWN) return current
        if (current == SubjectType.UNKNOWN) return previous
        if (previous == current) return previous

        return if (confidence >= 0.82) current else previous
    }

    private fun enforceTrackLimit() {
        while (tracks.size > maximumTracks) {
            val victim = tracks.values.minWithOrNull(
                compareBy<TrackState>(
                    { it.confidence },
                    { it.seenFrames },
                    { it.lastTimestampNanos },
                ),
            ) ?: break
            tracks.remove(victim.id)
        }
    }

    private fun timestampIsFresh(
        timestampNanos: Long,
        nowNanos: Long,
    ): Boolean {
        if (timestampNanos <= 0L) return false
        val age = nowNanos - timestampNanos
        return age in 0L..staleAfterNanos
    }

    private fun lerp(
        a: Float,
        b: Float,
        alpha: Double,
    ): Float = (
        a.toDouble() * (1.0 - alpha) +
            b.toDouble() * alpha
        ).toFloat()

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            synchronized(lock) {
                tracks.clear()
            }
            detector.close()
        }
    }
}

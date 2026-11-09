package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.FaceMeshFrame
import kotlin.math.hypot
import kotlin.math.max

/**
 * Adaptive transport gate for dense facial surface packets.
 *
 * Dense mesh geometry is expensive compared with ordinary pose landmarks. This policy preserves
 * keyframes, sends deformation updates when the surface materially changes, and backs off with
 * increasing peer count. Local capture and session recording remain full-rate; only the network
 * broadcast cadence is reduced.
 *
 * The policy is deterministic and stateful per capture take.
 */
class FaceMeshTransportPolicyEngine(
    private val minimumConfidence: Double = 0.34,
    private val baseIntervalMillis: Long = 50L,
    private val maximumKeyframeIntervalMillis: Long = 450L,
    private val deformationThresholdPixels: Double = 6.5,
    private val centerThresholdPixels: Double = 4.0,
    private val areaChangeThreshold: Double = 0.08,
) {

    data class Decision(
        val send: Boolean,
        val keyframe: Boolean,
        val reason: String,
        val intervalMillis: Long,
        val deformationPixels: Double,
        val centerDeltaPixels: Double,
        val areaChangeRatio: Double,
    )

    private var lastSentTimestampNanos = Long.MIN_VALUE
    private var lastKeyframeTimestampNanos = Long.MIN_VALUE
    private var lastSent: FaceMeshFrame? = null
    private var previousObserved: FaceMeshFrame? = null
    private var previousTrackId: Int? = null
    private var previousPeerTopologyKey: String? = null

    @Synchronized
    fun evaluate(
        frame: FaceMeshFrame,
        peerCount: Int,
        peerTopologyKey: String = "peers:$peerCount",
    ): Decision {
        if (frame.confidence < minimumConfidence || frame.points.isEmpty()) {
            return Decision(
                send = false,
                keyframe = false,
                reason = "low-confidence",
                intervalMillis = 0L,
                deformationPixels = 0.0,
                centerDeltaPixels = 0.0,
                areaChangeRatio = 0.0,
            )
        }

        val interval =
            adaptiveIntervalMillis(peerCount)

        if (
            previousPeerTopologyKey != null &&
            previousPeerTopologyKey != peerTopologyKey &&
            lastSent != null
        ) {
            accept(
                frame = frame,
                keyframe = true,
                peerTopologyKey = peerTopologyKey,
            )
            return Decision(
                send = true,
                keyframe = true,
                reason = "peer-topology-change",
                intervalMillis = interval,
                deformationPixels = 0.0,
                centerDeltaPixels = 0.0,
                areaChangeRatio = 0.0,
            )
        }

        if (lastSentTimestampNanos == Long.MIN_VALUE) {
            accept(
                frame = frame,
                keyframe = true,
                peerTopologyKey = peerTopologyKey,
            )
            return Decision(
                send = true,
                keyframe = true,
                reason = "initial-keyframe",
                intervalMillis = interval,
                deformationPixels = 0.0,
                centerDeltaPixels = 0.0,
                areaChangeRatio = 0.0,
            )
        }

        val elapsedMillis =
            (
                frame.timestampNanos -
                    lastSentTimestampNanos
                ).coerceAtLeast(0L) / 1_000_000L

        val keyframeAgeMillis =
            (
                frame.timestampNanos -
                    lastKeyframeTimestampNanos
                ).coerceAtLeast(0L) / 1_000_000L

        val lastSentFrame = lastSent

        if (
            lastSentFrame == null ||
            frame.trackingId != previousTrackId
        ) {
            accept(
                frame = frame,
                keyframe = true,
                peerTopologyKey = peerTopologyKey,
            )
            return Decision(
                send = true,
                keyframe = true,
                reason = "track-change",
                intervalMillis = interval,
                deformationPixels = 0.0,
                centerDeltaPixels = 0.0,
                areaChangeRatio = 0.0,
            )
        }

        val deformation =
            deformationPixels(
                previous = lastSentFrame,
                current = frame,
            )
        val centerDelta =
            centerDeltaPixels(
                previous = lastSentFrame,
                current = frame,
            )
        val areaChange =
            areaChangeRatio(
                previous = lastSentFrame,
                current = frame,
            )

        val keyframe =
            keyframeAgeMillis >= maximumKeyframeIntervalMillis ||
                deformation >= deformationThresholdPixels * 2.0 ||
                centerDelta >= centerThresholdPixels * 2.0 ||
                areaChange >= areaChangeThreshold * 2.0

        if (keyframe) {
            accept(
                frame = frame,
                keyframe = true,
                peerTopologyKey = peerTopologyKey,
            )
            return Decision(
                send = true,
                keyframe = true,
                reason = "keyframe-horizon-or-large-change",
                intervalMillis = interval,
                deformationPixels = deformation,
                centerDeltaPixels = centerDelta,
                areaChangeRatio = areaChange,
            )
        }

        if (elapsedMillis < interval) {
            previousObserved = frame
            previousTrackId = frame.trackingId
            return Decision(
                send = false,
                keyframe = false,
                reason = "interval-gate",
                intervalMillis = interval,
                deformationPixels = deformation,
                centerDeltaPixels = centerDelta,
                areaChangeRatio = areaChange,
            )
        }

        val materiallyChanged =
            deformation >= deformationThresholdPixels ||
                centerDelta >= centerThresholdPixels ||
                areaChange >= areaChangeThreshold

        if (materiallyChanged) {
            accept(
                frame = frame,
                keyframe = false,
                peerTopologyKey = peerTopologyKey,
            )
            return Decision(
                send = true,
                keyframe = false,
                reason = "surface-change",
                intervalMillis = interval,
                deformationPixels = deformation,
                centerDeltaPixels = centerDelta,
                areaChangeRatio = areaChange,
            )
        }

        previousObserved = frame
        previousTrackId = frame.trackingId

        return Decision(
            send = false,
            keyframe = false,
            reason = "stable-surface",
            intervalMillis = interval,
            deformationPixels = deformation,
            centerDeltaPixels = centerDelta,
            areaChangeRatio = areaChange,
        )
    }

    @Synchronized
    fun reset() {
        lastSentTimestampNanos = Long.MIN_VALUE
        lastKeyframeTimestampNanos = Long.MIN_VALUE
        lastSent = null
        previousObserved = null
        previousTrackId = null
        previousPeerTopologyKey = null
    }

    private fun accept(
        frame: FaceMeshFrame,
        keyframe: Boolean,
        peerTopologyKey: String = "peers:unknown",
    ) {
        lastSentTimestampNanos =
            frame.timestampNanos

        if (keyframe) {
            lastKeyframeTimestampNanos =
                frame.timestampNanos
        }

        lastSent = frame
        previousObserved = frame
        previousTrackId = frame.trackingId
        previousPeerTopologyKey = peerTopologyKey
    }

    private fun adaptiveIntervalMillis(
        peerCount: Int,
    ): Long {
        val multiplier =
            when {
                peerCount <= 1 -> 0.70
                peerCount == 2 -> 1.0
                peerCount <= 4 -> 1.45
                else -> 2.10
            }

        return max(
            20L,
            (
                baseIntervalMillis * multiplier
            ).toLong(),
        )
    }

    private fun deformationPixels(
        previous: FaceMeshFrame,
        current: FaceMeshFrame,
    ): Double {
        val previousPoints =
            previous.pointMap()
        val currentPoints =
            current.pointMap()

        val shared =
            previousPoints.keys
                .intersect(currentPoints.keys)
                .sorted()

        if (shared.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }

        val stride =
            max(
                1,
                shared.size / 32,
            )

        val samples =
            shared
                .filterIndexed { index, _ ->
                    index % stride == 0
                }
                .map { id ->
                    val a =
                        previousPoints.getValue(id)
                    val b =
                        currentPoints.getValue(id)
                    hypot(
                        a.x - b.x,
                        a.y - b.y,
                    )
                }

        return samples
            .filter(Double::isFinite)
            .average()
            .takeIf(Double::isFinite)
            ?: Double.POSITIVE_INFINITY
    }

    private fun centerDeltaPixels(
        previous: FaceMeshFrame,
        current: FaceMeshFrame,
    ): Double {
        val ax =
            (previous.boundsLeft +
                previous.boundsRight) * 0.5
        val ay =
            (previous.boundsTop +
                previous.boundsBottom) * 0.5
        val bx =
            (current.boundsLeft +
                current.boundsRight) * 0.5
        val by =
            (current.boundsTop +
                current.boundsBottom) * 0.5

        return hypot(
            (ax - bx).toDouble(),
            (ay - by).toDouble(),
        )
    }

    private fun areaChangeRatio(
        previous: FaceMeshFrame,
        current: FaceMeshFrame,
    ): Double {
        val previousArea =
            (
                (previous.boundsRight - previous.boundsLeft)
                    .coerceAtLeast(0f) *
                    (previous.boundsBottom - previous.boundsTop)
                        .coerceAtLeast(0f)
                ).toDouble()

        val currentArea =
            (
                (current.boundsRight - current.boundsLeft)
                    .coerceAtLeast(0f) *
                    (current.boundsBottom - current.boundsTop)
                        .coerceAtLeast(0f)
                ).toDouble()

        if (previousArea <= 1e-6) {
            return if (currentArea > 0.0) 1.0 else 0.0
        }

        return kotlin.math.abs(
            currentArea - previousArea,
        ) / previousArea
    }
}

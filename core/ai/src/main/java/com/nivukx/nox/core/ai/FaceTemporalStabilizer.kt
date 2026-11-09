package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.FaceMeshPoint
import com.nivukx.nox.core.model.Vec2
import kotlin.math.sqrt

/**
 * Temporal face landmark stabilizer for high-resolution face-camera capture.
 *
 * Bounding-box tracking alone is not sufficient for animation-quality face data. This engine keeps
 * per-track landmark state, estimates landmark velocity, adapts the smoothing gain to motion, and
 * rejects implausible one-frame jumps. Eye centers are kept on the same temporal state so the eye
 * solver receives coherent geometry.
 */
class FaceTemporalStabilizer(
    private val maximumJumpFraction: Double = 0.22,
    private val minimumGain: Double = 0.28,
    private val maximumGain: Double = 0.90,
    private val missedFrameRetentionNanos: Long = 250_000_000L,
    private val maximumRetainedTracks: Int = 128,
) : AutoCloseable {
    init {
        require(maximumJumpFraction > 0.0)
        require(minimumGain in 0.0..1.0)
        require(maximumGain in minimumGain..1.0)
        require(missedFrameRetentionNanos > 0L)
        require(maximumRetainedTracks >= 16)
    }

    private data class State(
        val points: MutableMap<Int, PointState>,
        var timestampNanos: Long,
    )

    private data class PointState(
        var position: Vec2,
        var velocity: Vec2,
        var acceleration: Vec2,
        var jerk: Vec2,
        var confidence: Double,
    )

    private val tracks = HashMap<Int, State>()

    @Synchronized
    fun refine(frames: List<FaceFrame>): List<FaceFrame> =
        frames.map { refine(it) }

    @Synchronized
    fun refine(frame: FaceFrame): FaceFrame {
        val trackId = frame.trackingId ?: return frame
        val existing = tracks[trackId]
        if (
            existing != null &&
            frame.timestampNanos <= existing.timestampNanos
        ) {
            val stalePoints =
                frame.points.map { point ->
                    existing.points[point.index]?.let { state ->
                        point.copy(
                            point = state.position,
                            confidence = state.confidence.coerceIn(0.0, 1.0),
                        )
                    } ?: point
                }
            return frame.copy(
                points = stalePoints,
                leftEyeCenter = existing.points[-1]?.position
                    ?: frame.leftEyeCenter,
                rightEyeCenter = existing.points[-2]?.position
                    ?: frame.rightEyeCenter,
            )
        }
        val state = existing ?: tracks.getOrPut(trackId) {
            State(
                points = HashMap(),
                timestampNanos = frame.timestampNanos,
            )
        }

        val dt = (
            (frame.timestampNanos - state.timestampNanos)
                .coerceAtLeast(1L) / 1e9
            ).coerceIn(1.0 / 240.0, 0.25)

        val faceSize = sqrt(
            (
                (frame.boundsRight - frame.boundsLeft)
                    .toDouble()
            ).let { it * it } +
                (
                    (frame.boundsBottom - frame.boundsTop)
                        .toDouble()
                ).let { it * it },
        ).coerceAtLeast(16.0)

        val resultPoints = frame.points.map { point ->
            val previous = state.points[point.index]
            val measurement = point.point

            if (previous == null) {
                state.points[point.index] = PointState(
                    position = measurement,
                    velocity = Vec2(0.0, 0.0),
                    acceleration = Vec2(0.0, 0.0),
                    jerk = Vec2(0.0, 0.0),
                    confidence = point.confidence,
                )
                point
            } else {
                val accelerationLimit =
                    (
                        faceSize * 24.0
                    ).coerceIn(2_400.0, 12_000.0)

                val predicted = predictPosition(
                    previous,
                    dt,
                )

                val jump = predicted.distanceTo(measurement)
                val normalizedJump = jump / faceSize
                val jumpGate = (
                    maximumJumpFraction.coerceAtLeast(0.01) /
                        normalizedJump.coerceAtLeast(1e-6)
                    ).coerceIn(0.05, 1.0)

                val confidence = point.confidence
                    .coerceIn(0.0, 1.0)
                val motion = jump / dt
                val motionGate = (
                    1.0 -
                        expLike(-motion / 900.0)
                    ).coerceIn(0.0, 1.0)

                val gain = (
                    minimumGain +
                        (maximumGain - minimumGain) *
                            confidence *
                            (0.38 + 0.62 * motionGate) *
                            jumpGate
                    ).coerceIn(
                        minimumGain,
                        maximumGain,
                    )

                val refined = predicted.lerp(
                    measurement,
                    gain,
                )

                val measuredVelocity =
                    (refined - previous.position) / dt

                val velocityGain =
                    (0.24 + 0.52 * confidence)
                        .coerceIn(0.15, 0.78)

                val measuredAcceleration =
                    (
                        measuredVelocity -
                            previous.velocity
                    ) / dt
                val measuredJerk =
                    (
                        measuredAcceleration -
                            previous.acceleration
                    ) / dt

                previous.velocity = previous.velocity.lerp(
                    measuredVelocity,
                    velocityGain,
                )
                previous.acceleration =
                    previous.acceleration.lerp(
                        measuredAcceleration.limitNorm(
                            accelerationLimit,
                        ),
                        0.26,
                    )
                previous.jerk =
                    previous.jerk.lerp(
                        measuredJerk.limitNorm(
                            accelerationLimit * 8.0,
                        ),
                        0.20,
                    )
                previous.position = refined
                previous.confidence =
                    previous.confidence * 0.25 +
                        confidence * 0.75

                point.copy(
                    point = refined,
                    confidence = previous.confidence
                        .coerceIn(0.0, 1.0),
                )
            }
        }

        val leftEye = smoothOptionalEye(
            state,
            key = -1,
            value = frame.leftEyeCenter,
            faceSize = faceSize,
            dt = dt,
        )

        val rightEye = smoothOptionalEye(
            state,
            key = -2,
            value = frame.rightEyeCenter,
            faceSize = faceSize,
            dt = dt,
        )

        state.timestampNanos = frame.timestampNanos
        prune(frame.timestampNanos)
        enforceCapacity()

        return frame.copy(
            points = resultPoints,
            leftEyeCenter = leftEye,
            rightEyeCenter = rightEye,
        )
    }

    @Synchronized
    fun clear() {
        tracks.clear()
    }

    override fun close() {
        clear()
    }

    @Synchronized
    private fun prune(nowNanos: Long) {
        val cutoff = (nowNanos - missedFrameRetentionNanos)
            .coerceAtLeast(0L)
        val iterator = tracks.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.value.timestampNanos < cutoff) {
                iterator.remove()
            }
        }
    }

    @Synchronized
    private fun enforceCapacity() {
        if (tracks.size <= maximumRetainedTracks) return
        val overflow = tracks.size - maximumRetainedTracks
        tracks.entries
            .sortedBy { it.value.timestampNanos }
            .take(overflow)
            .forEach { tracks.remove(it.key) }
    }

    private fun smoothOptionalEye(
        state: State,
        key: Int,
        value: Vec2?,
        faceSize: Double,
        dt: Double,
    ): Vec2? {
        if (value == null) return null

        val previous = state.points[key]
        if (previous == null) {
            state.points[key] = PointState(
                position = value,
                velocity = Vec2(0.0, 0.0),
                acceleration = Vec2(0.0, 0.0),
                jerk = Vec2(0.0, 0.0),
                confidence = 1.0,
            )
            return value
        }

        val predicted = predictPosition(
            previous,
            dt,
        )
        val jump = predicted.distanceTo(value)
        val normalizedJump =
            jump / faceSize.coerceAtLeast(1.0)
        val gate = (
            maximumJumpFraction /
                normalizedJump.coerceAtLeast(1e-6)
            ).coerceIn(0.08, 1.0)

        val gain = (
            0.42 + 0.46 * gate
        ).coerceIn(0.32, 0.92)

        val refined = predicted.lerp(value, gain)
        val velocity = (refined - previous.position) / dt

        val measuredAcceleration =
            (velocity - previous.velocity) / dt
        val measuredJerk =
            (measuredAcceleration - previous.acceleration) /
                dt

        previous.velocity = previous.velocity.lerp(
            velocity,
            0.55,
        )
        previous.acceleration =
            previous.acceleration.lerp(
                measuredAcceleration.limitNorm(
                    (
                        faceSize * 20.0
                    ).coerceIn(1_800.0, 10_000.0),
                ),
                0.22,
            )
        previous.jerk =
            previous.jerk.lerp(
                measuredJerk.limitNorm(
                    (
                        faceSize * 160.0
                    ).coerceIn(14_400.0, 80_000.0),
                ),
                0.20,
            )
        previous.position = refined

        return refined
    }

    private fun predictPosition(
        state: PointState,
        dtSeconds: Double,
    ): Vec2 {
        val t = dtSeconds.coerceAtLeast(0.0)
        val t2 = t * t
        val t3 = t2 * t
        return (
            state.position +
                state.velocity * t +
                state.acceleration * (0.5 * t2) +
                state.jerk * (t3 / 6.0)
        )
    }

    private fun Vec2.limitNorm(maximum: Double): Vec2 {
        val norm =
            sqrt(
                x * x +
                    y * y,
            )
        if (!norm.isFinite() || maximum <= 0.0) {
            return Vec2(0.0, 0.0)
        }
        if (norm <= maximum) return this
        val scale = maximum / norm
        return Vec2(
            x * scale,
            y * scale,
        )
    }

    private fun Vec2.lerp(other: Vec2, alpha: Double): Vec2 =
        Vec2(
            x * (1.0 - alpha) + other.x * alpha,
            y * (1.0 - alpha) + other.y * alpha,
        )

    private fun Vec2.distanceTo(other: Vec2): Double {
        val dx = x - other.x
        val dy = y - other.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun expLike(value: Double): Double =
        kotlin.math.exp(value)
}

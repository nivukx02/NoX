package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.HandLandmark
import com.nivukx.nox.core.model.HandObservation
import com.nivukx.nox.core.model.HandSide
import com.nivukx.nox.core.model.HandTrackingFrame
import com.nivukx.nox.core.model.Vec3
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Compact, bounded transport for hand landmarks.
 *
 * A maximum of two hands x 21 landmarks is serialized as fixed-width floats so the payload remains
 * deterministic and far below the NoX 32 KiB control packet ceiling. Normalized coordinates are
 * canonical; optional metric world coordinates are preserved when the local inference provider
 * exposes them.
 */
object HandWireCodec {
    private const val VERSION = 1
    private const val HEADER_BYTES = 24
    private const val HAND_HEADER_BYTES = 24
    private const val LANDMARK_BYTES = 28
    private const val MAX_HANDS = 4
    private const val LANDMARKS_PER_HAND = 21

    data class Frame(
        val timestampNanos: Long,
        val imageWidth: Int,
        val imageHeight: Int,
        val hands: List<HandObservation>,
    )

    fun encode(frame: HandTrackingFrame): ByteArray {
        return encode(
            Frame(
                timestampNanos = frame.timestampNanos,
                imageWidth = frame.imageWidth,
                imageHeight = frame.imageHeight,
                hands = frame.hands,
            ),
        )
    }

    fun encode(frame: Frame): ByteArray {
        val hands = frame.hands.take(MAX_HANDS)
        val bytes = ByteBuffer.allocate(
            HEADER_BYTES +
                hands.size * (
                    HAND_HEADER_BYTES +
                        LANDMARKS_PER_HAND * LANDMARK_BYTES
                    ),
        ).order(ByteOrder.BIG_ENDIAN)

        bytes.putInt(VERSION)
        bytes.putLong(frame.timestampNanos)
        bytes.putInt(frame.imageWidth.coerceAtLeast(1))
        bytes.putInt(frame.imageHeight.coerceAtLeast(1))
        bytes.putShort(hands.size.toShort())
        bytes.putShort(0)

        hands.forEach { hand ->
            bytes.putLong(hand.handId)
            bytes.put(hand.side.ordinal.toByte())
            bytes.put(0)
            bytes.putShort(hand.landmarks.size.coerceAtMost(21).toShort())
            bytes.putFloat(hand.handednessConfidence.coerceIn(0.0, 1.0).toFloat())
            bytes.putFloat(hand.confidence.coerceIn(0.0, 1.0).toFloat())
            bytes.putFloat(hand.temporalStability.coerceIn(0.0, 1.0).toFloat())

            val landmarks = hand.landmarks
                .sortedBy { it.index }
                .take(LANDMARKS_PER_HAND)
                .associateBy { it.index }

            repeat(LANDMARKS_PER_HAND) { index ->
                val landmark = landmarks[index]
                if (landmark == null) {
                    repeat(7) { bytes.putFloat(Float.NaN) }
                } else {
                    bytes.putFloat(landmark.normalizedX.toFloat())
                    bytes.putFloat(landmark.normalizedY.toFloat())
                    bytes.putFloat(landmark.normalizedZ.toFloat())

                    val world = landmark.worldPositionMeters
                    if (world == null) {
                        bytes.putFloat(Float.NaN)
                        bytes.putFloat(Float.NaN)
                        bytes.putFloat(Float.NaN)
                    } else {
                        bytes.putFloat(world.x.toFloat())
                        bytes.putFloat(world.y.toFloat())
                        bytes.putFloat(world.z.toFloat())
                    }

                    bytes.putFloat(landmark.confidence.coerceIn(0.0, 1.0).toFloat())
                }
            }
        }

        return bytes.array()
    }

    fun decode(bytes: ByteArray): Frame {
        require(bytes.size >= HEADER_BYTES) { "Hand payload too small" }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)

        require(buffer.int == VERSION) {
            "Unsupported hand payload version"
        }

        val timestamp = buffer.long
        val width = buffer.int
        val height = buffer.int
        val handCount = buffer.short.toInt() and 0xFFFF
        buffer.short
        require(handCount in 0..MAX_HANDS) {
            "Invalid hand count"
        }

        val hands = ArrayList<HandObservation>(handCount)
        repeat(handCount) {
            require(buffer.remaining() >= HAND_HEADER_BYTES) {
                "Truncated hand header"
            }

            val id = buffer.long
            val side = HandSide.entries.getOrNull(
                buffer.get().toInt(),
            ) ?: HandSide.UNKNOWN
            buffer.get()
            val landmarkCount = buffer.short.toInt() and 0xFFFF
            val handedness = buffer.float.toDouble().coerceIn(0.0, 1.0)
            val confidence = buffer.float.toDouble().coerceIn(0.0, 1.0)
            val stability = buffer.float.toDouble().coerceIn(0.0, 1.0)

            require(landmarkCount in 0..LANDMARKS_PER_HAND)

            val landmarks = ArrayList<HandLandmark>(
                landmarkCount,
            )

            repeat(LANDMARKS_PER_HAND) { index ->
                require(buffer.remaining() >= LANDMARK_BYTES) {
                    "Truncated hand landmark block"
                }

                val x = buffer.float.toDouble()
                val y = buffer.float.toDouble()
                val z = buffer.float.toDouble()
                val wx = buffer.float.toDouble()
                val wy = buffer.float.toDouble()
                val wz = buffer.float.toDouble()
                val landmarkConfidence =
                    buffer.float.toDouble().coerceIn(0.0, 1.0)

                if (
                    index < landmarkCount &&
                    x.isFinite() &&
                    y.isFinite() &&
                    z.isFinite()
                ) {
                    val world =
                        if (wx.isFinite() && wy.isFinite() && wz.isFinite()) {
                            Vec3(wx, wy, wz)
                        } else {
                            null
                        }

                    landmarks += HandLandmark(
                        index = index,
                        normalizedX = x.coerceIn(0.0, 1.0),
                        normalizedY = y.coerceIn(0.0, 1.0),
                        normalizedZ = z,
                        worldPositionMeters = world,
                        confidence = landmarkConfidence,
                    )
                }
            }

            hands += HandObservation(
                handId = id,
                side = side,
                handednessConfidence = handedness,
                landmarks = landmarks,
                confidence = confidence,
                timestampNanos = timestamp,
                temporalStability = stability,
            )
        }

        require(buffer.remaining() == 0) {
            "Trailing bytes in hand payload"
        }

        return Frame(
            timestampNanos = timestamp,
            imageWidth = width,
            imageHeight = height,
            hands = hands,
        )
    }
}

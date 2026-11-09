package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MultiActorPoseWireCodecTest {

    @Test
    fun roundTripPreservesCrowdPoseGraph() {
        val frame =
            MultiActorPoseWireCodec.Frame(
                timestampNanos = 9_000_000_000L,
                sourceDeviceId = "camera-alpha",
                inputWidth = 3840,
                inputHeight = 2160,
                actors =
                    (1L..50L).map { actorId ->
                        MultiActorPoseWireCodec.Actor(
                            actorId = actorId,
                            confidence = 0.91f,
                            predicted = actorId % 7L == 0L,
                            missedFrames = 2,
                            centerX = 0.25f,
                            centerY = 0.50f,
                            boundsLeft = 0.10f,
                            boundsTop = 0.20f,
                            boundsRight = 0.40f,
                            boundsBottom = 0.85f,
                            velocityX = 0.10f,
                            velocityY = -0.02f,
                            joints =
                                (0..32).map { jointId ->
                                    MultiActorPoseWireCodec.Joint(
                                        landmarkId = jointId,
                                        x = jointId / 32.0f,
                                        y = 0.5f,
                                        confidence = 0.88f,
                                    )
                                },
                        )
                    },
            )

        val encoded =
            MultiActorPoseWireCodec.encode(frame)
        assertTrue(
            encoded.size <
                NoXWireProtocol.MAX_PACKET_BYTES,
        )

        val decoded =
            MultiActorPoseWireCodec.decode(encoded)

        assertEquals(frame.timestampNanos, decoded.timestampNanos)
        assertEquals(frame.sourceDeviceId, decoded.sourceDeviceId)
        assertEquals(50, decoded.actors.size)
        assertEquals(
            frame.actors.first().joints.size,
            decoded.actors.first().joints.size,
        )
    }

    @Test
    fun roundTripPreservesCompactAppearanceFingerprint() {
        val fingerprint =
            (0 until 16).map {
                it / 15.0f
            }

        val frame =
            MultiActorPoseWireCodec.Frame(
                timestampNanos = 10_000_000_000L,
                sourceDeviceId = "camera-beta",
                inputWidth = 1920,
                inputHeight = 1080,
                actors =
                    listOf(
                        MultiActorPoseWireCodec.Actor(
                            actorId = 77L,
                            confidence = 0.97f,
                            predicted = false,
                            missedFrames = 0,
                            centerX = 0.51f,
                            centerY = 0.46f,
                            boundsLeft = 0.35f,
                            boundsTop = 0.12f,
                            boundsRight = 0.66f,
                            boundsBottom = 0.94f,
                            velocityX = 0.02f,
                            velocityY = -0.01f,
                            joints = emptyList(),
                            appearanceFingerprint =
                                fingerprint,
                            appearanceQuality = 0.93f,
                        ),
                    ),
            )

        val decoded =
            MultiActorPoseWireCodec.decode(
                MultiActorPoseWireCodec.encode(frame),
            )
        val actor = decoded.actors.single()

        assertEquals(
            16,
            actor.appearanceFingerprint.size,
        )
        assertTrue(
            actor.appearanceQuality > 0.90f,
        )
        assertTrue(
            actor.appearanceFingerprint.zip(fingerprint).all {
                (it.first - it.second) < 0.01f
            },
        )
    }

    @Test
    fun maximumFiftyActorPayloadWithAppearanceRemainsBounded() {
        val frame =
            MultiActorPoseWireCodec.Frame(
                timestampNanos = 11L,
                sourceDeviceId = "camera-max",
                inputWidth = 3840,
                inputHeight = 2160,
                actors =
                    (1L..50L).map { actorId ->
                        MultiActorPoseWireCodec.Actor(
                            actorId = actorId,
                            confidence = 0.90f,
                            predicted = false,
                            missedFrames = 0,
                            centerX = 0.5f,
                            centerY = 0.5f,
                            boundsLeft = 0.1f,
                            boundsTop = 0.1f,
                            boundsRight = 0.9f,
                            boundsBottom = 0.9f,
                            velocityX = 0.0f,
                            velocityY = 0.0f,
                            joints =
                                (0..32).map { jointId ->
                                    MultiActorPoseWireCodec.Joint(
                                        landmarkId = jointId,
                                        x = 0.5f,
                                        y = 0.5f,
                                        confidence = 0.85f,
                                    )
                                },
                            worldJoints =
                                (0..32).map { jointId ->
                                    MultiActorPoseWireCodec.WorldJoint(
                                        landmarkId = jointId,
                                        xMeters = 0.0f,
                                        yMeters = jointId / 100.0f,
                                        zMeters = 1.0f,
                                        confidence = 0.80f,
                                    )
                                },
                            appearanceFingerprint =
                                List(16) { 0.5f },
                            appearanceQuality = 0.90f,
                        )
                    },
            )

        val bytes =
            MultiActorPoseWireCodec.encode(frame)

        assertTrue(
            bytes.size <=
                NoXWireProtocol.MAX_PACKET_BYTES - 64,
        )
        assertEquals(
            50,
            MultiActorPoseWireCodec.decode(bytes)
                .actors
                .size,
        )
    }

    @Test
    fun legacyVersionTwoPayloadRemainsDecodableAfterV3Extension() {
        val frame =
            MultiActorPoseWireCodec.Frame(
                timestampNanos = 12L,
                sourceDeviceId = "legacy-camera",
                inputWidth = 1280,
                inputHeight = 720,
                actors =
                    listOf(
                        MultiActorPoseWireCodec.Actor(
                            actorId = 9L,
                            confidence = 0.89f,
                            predicted = false,
                            missedFrames = 0,
                            centerX = 0.50f,
                            centerY = 0.50f,
                            boundsLeft = 0.25f,
                            boundsTop = 0.10f,
                            boundsRight = 0.75f,
                            boundsBottom = 0.90f,
                            velocityX = 0.0f,
                            velocityY = 0.0f,
                            joints = emptyList(),
                            worldJoints = emptyList(),
                            appearanceFingerprint =
                                List(16) { 0.5f },
                            appearanceQuality = 0.80f,
                        ),
                    ),
            )

        val current =
            MultiActorPoseWireCodec.encode(frame)
        val legacy =
            current.copyOf(
                current.size - 17,
            )

        java.nio.ByteBuffer
            .wrap(legacy)
            .putInt(0, 2)

        val decoded =
            MultiActorPoseWireCodec.decode(
                legacy,
            )

        assertEquals(1, decoded.actors.size)
        assertTrue(
            decoded.actors.single().appearanceFingerprint.isEmpty(),
        )
        assertEquals(
            0f,
            decoded.actors.single().appearanceQuality,
            1e-6f,
        )
    }

    @Test
    fun rejectsTrailingBytes() {
        val frame =
            MultiActorPoseWireCodec.Frame(
                timestampNanos = 1L,
                sourceDeviceId = "camera",
                inputWidth = 1920,
                inputHeight = 1080,
                actors = emptyList(),
            )

        val encoded =
            MultiActorPoseWireCodec.encode(frame)
        val malformed =
            encoded + byteArrayOf(7)

        var failed = false
        runCatching {
            MultiActorPoseWireCodec.decode(malformed)
        }.onFailure {
            failed = true
        }

        assertTrue(failed)
    }
}

package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SecureRemoteEvidenceCodecTest {

    private val key =
        ByteArray(32) { index ->
            (index * 7 + 11).toByte()
        }

    @Test
    fun authenticatedRemotePacketRoundTrips() {
        val plaintext =
            "dense-evidence-payload".toByteArray()

        val packet =
            SecureRemoteEvidenceCodec.encode(
                origin = DeviceId("cam-a"),
                destination = DeviceId("master"),
                sequence = 42L,
                timestampNanos = 16_000L,
                plaintext = plaintext,
                key = key,
            )

        val decoded =
            SecureRemoteEvidenceCodec.decode(
                packet,
                key,
            )

        assertTrue(
            plaintext.contentEquals(
                decoded.plaintext,
            ),
        )
        assertTrue(
            decoded.origin ==
                DeviceId("cam-a"),
        )
        assertTrue(
            decoded.destination ==
                DeviceId("master"),
        )
    }

    @Test
    fun tamperedCiphertextFailsAuthentication() {
        val packet =
            SecureRemoteEvidenceCodec.encode(
                origin = DeviceId("cam-b"),
                destination = DeviceId("master"),
                sequence = 7L,
                timestampNanos = 17_000L,
                plaintext = byteArrayOf(1, 2, 3, 4),
                key = key,
            )

        packet[packet.lastIndex] =
            (packet.last().toInt() xor 0x55).toByte()

        assertFailsWith<IllegalArgumentException> {
            SecureRemoteEvidenceCodec.decode(
                packet,
                key,
            )
        }
    }

    @Test
    fun invalidKeyLengthIsRejected() {
        assertFailsWith<IllegalArgumentException> {
            SecureRemoteEvidenceCodec.encode(
                origin = DeviceId("cam-c"),
                destination = DeviceId("master"),
                sequence = 1L,
                timestampNanos = 18_000L,
                plaintext = byteArrayOf(9),
                key = ByteArray(12),
            )
        }
    }

    @Test
    fun packetRemainsInsideNoXTransportLimit() {
        val packet =
            SecureRemoteEvidenceCodec.encode(
                origin = DeviceId("cam-d"),
                destination = DeviceId("master"),
                sequence = 10L,
                timestampNanos = 19_000L,
                plaintext =
                    ByteArray(60_000) { 1 },
                key = key,
            )

        assertTrue(
            packet.size <=
                NoXWireProtocol.MAX_PACKET_BYTES,
        )
    }
}

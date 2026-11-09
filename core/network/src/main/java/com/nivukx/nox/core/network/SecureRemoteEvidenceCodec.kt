package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.GeneralSecurityException
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

/**
 * Authenticated encryption envelope for remote evidence links.
 *
 * AES-GCM is used only as a packet primitive. Route selection, replay protection, key provisioning,
 * and the actual socket/QUIC/TLS transport remain separate concerns.
 *
 * The encrypted payload includes arbitrary NoX evidence, including a distributed route envelope.
 * Origin, destination, packet sequence, and timestamp are bound as additional authenticated data
 * so metadata substitution cannot redirect or replay a packet without detection.
 */
object SecureRemoteEvidenceCodec {

    private const val VERSION = 1
    private const val NONCE_BYTES = 12
    private const val TAG_BITS = 128
    private const val HEADER_BYTES = 28
    private const val MAX_DEVICE_ID_BYTES = 96
    private const val MAX_PLAINTEXT_BYTES =
        NoXWireProtocol.MAX_PACKET_BYTES -
            36 -
            HEADER_BYTES -
            2 * MAX_DEVICE_ID_BYTES -
            16

    private val nonceGenerator =
        SecureRandom()

    data class Decoded(
        val origin: DeviceId,
        val destination: DeviceId,
        val sequence: Long,
        val timestampNanos: Long,
        val plaintext: ByteArray,
    )

    fun encode(
        origin: DeviceId,
        destination: DeviceId,
        sequence: Long,
        timestampNanos: Long,
        plaintext: ByteArray,
        key: ByteArray,
    ): ByteArray {
        validateInputs(
            origin = origin,
            destination = destination,
            sequence = sequence,
            timestampNanos = timestampNanos,
            plaintext = plaintext,
            key = key,
        )

        val nonce =
            ByteArray(NONCE_BYTES)
                .also(nonceGenerator::nextBytes)

        val ciphertext =
            encrypt(
                plaintext = plaintext,
                key = key,
                nonce = nonce,
                aad =
                    aad(
                        origin = origin,
                        destination = destination,
                        sequence = sequence,
                        timestampNanos = timestampNanos,
                    ),
            )

        val originBytes =
            origin.value.toByteArray(Charsets.UTF_8)
        val destinationBytes =
            destination.value.toByteArray(Charsets.UTF_8)

        val body =
            ByteBuffer
                .allocate(
                    HEADER_BYTES +
                        originBytes.size +
                        destinationBytes.size +
                        NONCE_BYTES +
                        ciphertext.size,
                )
                .order(ByteOrder.BIG_ENDIAN)
                .apply {
                    putInt(VERSION)
                    putShort(originBytes.size.toShort())
                    putShort(destinationBytes.size.toShort())
                    putLong(sequence)
                    put(nonce)
                    put(originBytes)
                    put(destinationBytes)
                    put(ciphertext)
                }
                .array()

        require(
            body.size <=
                NoXWireProtocol.MAX_PACKET_BYTES - 36,
        ) {
            "Secure remote evidence packet exceeds transport limit"
        }

        return NoXWireProtocol.encode(
            type =
                NoXWireProtocol.Type.SECURE_REMOTE_EVIDENCE_FRAME,
            sequence = sequence,
            timestampNanos = timestampNanos,
            payload = body,
        )
    }

    fun decode(
        packetBytes: ByteArray,
        key: ByteArray,
    ): Decoded {
        require(key.size == 16 || key.size == 24 || key.size == 32) {
            "AES-GCM key must be 128, 192 or 256 bits"
        }

        val packet =
            runCatching {
                NoXWireProtocol.decode(
                    packetBytes,
                )
            }.getOrElse {
                throw IllegalArgumentException(
                    "Invalid secure remote evidence packet",
                    it,
                )
            }

        require(
            packet.type ==
                NoXWireProtocol.Type.SECURE_REMOTE_EVIDENCE_FRAME,
        )

        val buffer =
            ByteBuffer
                .wrap(packet.payload)
                .order(ByteOrder.BIG_ENDIAN)

        require(
            buffer.remaining() >=
                HEADER_BYTES +
                NONCE_BYTES,
        )

        require(buffer.int == VERSION)

        val originLength =
            buffer.short.toInt() and 0xFFFF
        val destinationLength =
            buffer.short.toInt() and 0xFFFF

        val sequence =
            buffer.long

        require(sequence == packet.sequence) {
            "Secure packet sequence mismatch"
        }

        require(originLength in 1..MAX_DEVICE_ID_BYTES)
        require(destinationLength in 1..MAX_DEVICE_ID_BYTES)

        val nonce =
            ByteArray(NONCE_BYTES)
                .also(buffer::get)

        val origin =
            DeviceId(
                ByteArray(originLength)
                    .also(buffer::get)
                    .toString(Charsets.UTF_8),
            )

        val destination =
            DeviceId(
                ByteArray(destinationLength)
                    .also(buffer::get)
                    .toString(Charsets.UTF_8),
            )

        require(origin.value.isNotBlank())
        require(destination.value.isNotBlank())

        val ciphertext =
            ByteArray(buffer.remaining())
                .also(buffer::get)

        require(
            ciphertext.size >
                TAG_BITS / 8,
        )

        val plaintext =
            decrypt(
                ciphertext = ciphertext,
                key = key,
                nonce = nonce,
                aad =
                    aad(
                        origin = origin,
                        destination = destination,
                        sequence = sequence,
                        timestampNanos =
                            packet.timestampNanos,
                    ),
            )

        require(
            plaintext.size <=
                MAX_PLAINTEXT_BYTES,
        )

        return Decoded(
            origin = origin,
            destination = destination,
            sequence = sequence,
            timestampNanos =
                packet.timestampNanos,
            plaintext = plaintext,
        )
    }

    private fun validateInputs(
        origin: DeviceId,
        destination: DeviceId,
        sequence: Long,
        timestampNanos: Long,
        plaintext: ByteArray,
        key: ByteArray,
    ) {
        require(origin.value.isNotBlank())
        require(destination.value.isNotBlank())
        require(sequence >= 0L)
        require(timestampNanos > 0L)
        require(
            origin.value.toByteArray(Charsets.UTF_8).size <=
                MAX_DEVICE_ID_BYTES,
        )
        require(
            destination.value.toByteArray(Charsets.UTF_8).size <=
                MAX_DEVICE_ID_BYTES,
        )
        require(
            plaintext.isNotEmpty() &&
                plaintext.size <=
                MAX_PLAINTEXT_BYTES,
        )
        require(
            key.size == 16 ||
                key.size == 24 ||
                key.size == 32,
        ) {
            "AES-GCM key must be 128, 192 or 256 bits"
        }
    }

    private fun aad(
        origin: DeviceId,
        destination: DeviceId,
        sequence: Long,
        timestampNanos: Long,
    ): ByteArray {
        val originBytes =
            origin.value.toByteArray(Charsets.UTF_8)
        val destinationBytes =
            destination.value.toByteArray(Charsets.UTF_8)

        return ByteBuffer
            .allocate(
                4 +
                    2 +
                    originBytes.size +
                    2 +
                    destinationBytes.size +
                    8 +
                    8,
            )
            .order(ByteOrder.BIG_ENDIAN)
            .apply {
                putInt(VERSION)
                putShort(originBytes.size.toShort())
                put(originBytes)
                putShort(destinationBytes.size.toShort())
                put(destinationBytes)
                putLong(sequence)
                putLong(timestampNanos)
            }
            .array()
    }

    private fun encrypt(
        plaintext: ByteArray,
        key: ByteArray,
        nonce: ByteArray,
        aad: ByteArray,
    ): ByteArray =
        runCatching {
            Cipher
                .getInstance("AES/GCM/NoPadding")
                .apply {
                    init(
                        Cipher.ENCRYPT_MODE,
                        SecretKeySpec(
                            key,
                            "AES",
                        ),
                        GCMParameterSpec(
                            TAG_BITS,
                            nonce,
                        ),
                    )
                    updateAAD(aad)
                }
                .doFinal(plaintext)
        }.getOrElse {
            throw IllegalArgumentException(
                "Unable to encrypt secure remote evidence",
                it,
            )
        }

    private fun decrypt(
        ciphertext: ByteArray,
        key: ByteArray,
        nonce: ByteArray,
        aad: ByteArray,
    ): ByteArray =
        try {
            Cipher
                .getInstance("AES/GCM/NoPadding")
                .apply {
                    init(
                        Cipher.DECRYPT_MODE,
                        SecretKeySpec(
                            key,
                            "AES",
                        ),
                        GCMParameterSpec(
                            TAG_BITS,
                            nonce,
                        ),
                    )
                    updateAAD(aad)
                }
                .doFinal(ciphertext)
        } catch (error: GeneralSecurityException) {
            throw IllegalArgumentException(
                "Secure remote evidence authentication failed",
                error,
            )
        }
}

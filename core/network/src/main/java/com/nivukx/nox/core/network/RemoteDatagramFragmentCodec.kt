package com.nivukx.nox.core.network

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32

/**
 * Generic MTU-safe fragmentation for already-authenticated remote evidence packets.
 *
 * The secure NoX packet is fragmented after encryption, so fragment metadata never needs to expose
 * plaintext evidence. Reassembly is required before AES-GCM verification. A deliberately conservative
 * 1200-byte datagram budget avoids IP fragmentation across common IPv4/IPv6 paths.
 */
object RemoteDatagramFragmentCodec {

    const val MAX_DATAGRAM_BYTES = 1200
    private const val VERSION = 1
    private const val HEADER_BYTES = 32
    private const val MAX_FRAGMENTS = 128
    private const val MAX_PAYLOAD_BYTES =
        MAX_DATAGRAM_BYTES - HEADER_BYTES

    data class Fragment(
        val messageSequence: Long,
        val fragmentIndex: Int,
        val fragmentCount: Int,
        val payloadLength: Int,
        val totalPayloadBytes: Int,
        val payloadCrc32: Long,
        val payload: ByteArray,
    ) {
        init {
            require(messageSequence >= 0L)
            require(fragmentCount in 1..MAX_FRAGMENTS)
            require(fragmentIndex in 0 until fragmentCount)
            require(payloadLength == payload.size)
            require(payloadLength in 1..MAX_PAYLOAD_BYTES)
            require(totalPayloadBytes >= payloadLength)
            require(totalPayloadBytes <= MAX_PAYLOAD_BYTES * MAX_FRAGMENTS)
            require(payloadCrc32 in 0L..0xFFFF_FFFFL)
        }
    }

    fun fragment(
        payload: ByteArray,
        messageSequence: Long,
    ): List<ByteArray> {
        require(payload.isNotEmpty())
        require(messageSequence >= 0L)

        val fragmentCount =
            (
                payload.size +
                    MAX_PAYLOAD_BYTES -
                    1
            ) /
                MAX_PAYLOAD_BYTES

        require(fragmentCount in 1..MAX_FRAGMENTS) {
            "Remote evidence message exceeds UDP fragment envelope"
        }

        return (0 until fragmentCount).map { index ->
            val from =
                index * MAX_PAYLOAD_BYTES
            val to =
                minOf(
                    payload.size,
                    from + MAX_PAYLOAD_BYTES,
                )
            val part =
                payload.copyOfRange(from, to)
            val crc =
                CRC32().apply {
                    update(part)
                }.value

            ByteBuffer
                .allocate(
                    HEADER_BYTES +
                        part.size,
                )
                .order(ByteOrder.BIG_ENDIAN)
                .apply {
                    putInt(VERSION)
                    putInt(fragmentCount)
                    putInt(index)
                    putLong(messageSequence)
                    putInt(part.size)
                    putInt(crc.toInt())
                    putInt(payload.size)
                    put(part)
                }
                .array()
        }
    }

    fun decode(
        datagram: ByteArray,
    ): Fragment {
        require(
            datagram.size in
                HEADER_BYTES + 1..MAX_DATAGRAM_BYTES,
        )

        val buffer =
            ByteBuffer
                .wrap(datagram)
                .order(ByteOrder.BIG_ENDIAN)

        require(buffer.int == VERSION)

        val fragmentCount =
            buffer.int
        val fragmentIndex =
            buffer.int
        val messageSequence =
            buffer.long
        val payloadLength =
            buffer.int
        val crc32 =
            buffer.int.toLong() and
                0xFFFF_FFFFL
        val totalPayloadBytes =
            buffer.int

        require(fragmentCount in 1..MAX_FRAGMENTS)
        require(fragmentIndex in 0 until fragmentCount)
        require(messageSequence >= 0L)
        require(payloadLength in 1..MAX_PAYLOAD_BYTES)
        require(totalPayloadBytes >= payloadLength)
        require(
            buffer.remaining() ==
                payloadLength,
        )

        val payload =
            ByteArray(payloadLength)
                .also(buffer::get)

        val actualCrc =
            CRC32()
                .apply {
                    update(payload)
                }
                .value

        require(actualCrc == crc32) {
            "Remote evidence fragment CRC mismatch"
        }

        return Fragment(
            messageSequence = messageSequence,
            fragmentIndex = fragmentIndex,
            fragmentCount = fragmentCount,
            payloadLength = payloadLength,
            totalPayloadBytes = totalPayloadBytes,
            payloadCrc32 = crc32,
            payload = payload,
        )
    }

    fun reassemble(
        datagrams: List<ByteArray>,
    ): ByteArray =
        reassemble(
            datagrams.map(::decode),
        )

    fun reassemble(
        fragments: List<Fragment>,
    ): ByteArray {
        require(fragments.isNotEmpty())

        val first =
            fragments.first()

        require(
            fragments.size ==
                first.fragmentCount,
        ) {
            "Incomplete remote evidence fragment set"
        }

        require(
            fragments.map { it.fragmentIndex }.toSet() ==
                (0 until first.fragmentCount).toSet(),
        ) {
            "Remote evidence fragments are incomplete or duplicated"
        }

        fragments.forEach { fragment ->
            require(
                fragment.messageSequence ==
                    first.messageSequence,
            )
            require(
                fragment.fragmentCount ==
                    first.fragmentCount,
            )
            require(
                fragment.totalPayloadBytes ==
                    first.totalPayloadBytes,
            )
        }

        val total =
            fragments.sumOf {
                it.payloadLength
            }

        require(
            total ==
                first.totalPayloadBytes,
        ) {
            "Remote evidence total payload mismatch"
        }

        val output =
            ByteArray(total)
        var offset = 0

        fragments
            .sortedBy { it.fragmentIndex }
            .forEach { fragment ->
                fragment.payload.copyInto(
                    destination = output,
                    destinationOffset = offset,
                )
                offset +=
                    fragment.payloadLength
            }

        return output
    }
}

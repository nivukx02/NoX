package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RemoteDatagramReassemblerTest {

    @Test
    fun outOfOrderDatagramsReassembleAndExpireCleanly() {
        val payload =
            ByteArray(8_000) { index ->
                (index * 13 and 0xFF).toByte()
            }

        val datagrams =
            RemoteDatagramFragmentCodec.fragment(
                payload = payload,
                messageSequence = 17L,
            )

        val reassembler =
            RemoteDatagramReassembler(
                RemoteDatagramReassembler.Config(
                    maximumInFlightAssemblies = 4,
                    maximumInFlightBytes = 128 * 1024,
                    maximumAssemblyAgeNanos = 100L,
                ),
            )

        var result: ByteArray? = null

        datagrams
            .reversed()
            .forEachIndexed { index, datagram ->
                result =
                    reassembler.accept(
                        source = "10.0.0.2:9100",
                        datagram = datagram,
                        nowNanos = 1_000L + index,
                    ) ?: result
            }

        assertTrue(payload.contentEquals(result))
        assertEquals(0, reassembler.inFlightBytes())
        assertEquals(0, reassembler.inFlightAssemblyCount())

        val second =
            RemoteDatagramFragmentCodec.fragment(
                payload = payload,
                messageSequence = 18L,
            )

        reassembler.accept(
            source = "10.0.0.3:9100",
            datagram = second.first(),
            nowNanos = 2_000L,
        )

        reassembler.expire(2_101L)

        assertEquals(
            0,
            reassembler.inFlightAssemblyCount(),
        )
        assertEquals(
            0,
            reassembler.inFlightBytes(),
        )
    }
}

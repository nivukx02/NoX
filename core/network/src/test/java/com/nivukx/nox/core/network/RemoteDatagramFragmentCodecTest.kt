package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RemoteDatagramFragmentCodecTest {

    @Test
    fun securePacketRoundTripsThroughMtuSizedFragments() {
        val packet =
            ByteArray(9_000) { index ->
                (index * 31 and 0xFF).toByte()
            }

        val fragments =
            RemoteDatagramFragmentCodec.fragment(
                payload = packet,
                messageSequence = 42L,
            )

        assertTrue(fragments.size >= 8)
        assertTrue(
            fragments.all {
                it.size <=
                    RemoteDatagramFragmentCodec.MAX_DATAGRAM_BYTES
            },
        )

        val decoded =
            RemoteDatagramFragmentCodec.reassemble(
                fragments
                    .reversed(),
            )

        assertTrue(packet.contentEquals(decoded))
    }

    @Test
    fun fragmentMetadataMustBeComplete() {
        val packet =
            ByteArray(3_000) { 7 }

        val fragments =
            RemoteDatagramFragmentCodec.fragment(
                payload = packet,
                messageSequence = 9L,
            )

        kotlin.test.assertFailsWith<IllegalArgumentException> {
            RemoteDatagramFragmentCodec.reassemble(
                fragments.drop(1),
            )
        }
    }

    @Test
    fun duplicateFragmentMustNotBeSilentlyAccepted() {
        val packet =
            ByteArray(4_000) { 11 }

        val fragments =
            RemoteDatagramFragmentCodec.fragment(
                payload = packet,
                messageSequence = 11L,
            )

        kotlin.test.assertFailsWith<IllegalArgumentException> {
            RemoteDatagramFragmentCodec.reassemble(
                fragments +
                    fragments.first(),
            )
        }
        assertEquals(
            fragments.first().messageSequence,
            11L,
        )
    }
}

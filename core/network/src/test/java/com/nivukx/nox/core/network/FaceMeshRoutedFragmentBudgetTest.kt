package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertTrue

class FaceMeshRoutedFragmentBudgetTest {

    @Test
    fun maximumRouteKeepsEveryFaceMeshPacketWithinEnvelopeBudget() {
        val origin = DeviceId("origin")
        val destination = DeviceId("destination")
        val route =
            (0 until 10).map { DeviceId("relay-$it") } +
                destination

        val routeBudget =
            DistributedTransportEnvelopeCodec
                .maxPayloadBytesForRoute(
                    origin = origin,
                    destination = destination,
                    route = route,
                )

        val face =
            com.nivukx.nox.core.model.FaceMeshFrame(
                timestampNanos = 15_000_000_000L,
                trackId = 42L,
                vertices =
                    FloatArray(120_000) { index ->
                        (index % 17) / 17f
                    },
                triangles =
                    IntArray(60_000) { index ->
                        index % 120_000
                    },
                confidence = 0.95,
                reprojectionErrorPixels = 0.20,
            )

        val streamId = "x".repeat(96)
        val fragmentBudget =
            routeBudget -
                24 -
                streamId.toByteArray().size

        val chunks =
            FaceMeshChunkedWireCodec.encode(
                frame = face,
                streamId = streamId,
                frameSequence = 77L,
                maximumBytesPerChunk = 45 * 1024,
                maximumPayloadBytes = fragmentBudget,
            )

        assertTrue(chunks.isNotEmpty())
        assertTrue(
            chunks.all {
                it.payload.size <=
                    fragmentBudget
            },
        )

        chunks.forEachIndexed { index, chunk ->
            val packet =
                FaceMeshFragmentWireCodec.encode(
                    chunk = chunk,
                    packetSequence = 100L + index,
                )
            assertTrue(
                packet.size <=
                    NoXWireProtocol.MAX_PACKET_BYTES,
            )
        }
    }
}

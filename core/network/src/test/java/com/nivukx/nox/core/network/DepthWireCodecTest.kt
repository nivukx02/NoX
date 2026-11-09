package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DepthFrame
import com.nivukx.nox.core.model.DepthSource
import com.nivukx.nox.core.model.DeviceId
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DepthWireCodecTest {

    @Test
    fun roundTripsCompressedMetricDepthAndPreservesNetworkSource() {
        val frame =
            DepthFrame(
                timestampNanos = 123_456_789L,
                source = DepthSource.MULTI_VIEW,
                width = 4,
                height = 2,
                depthMeters = floatArrayOf(
                    0.5f,
                    0.75f,
                    1.0f,
                    1.25f,
                    0.0f,
                    1.50f,
                    1.75f,
                    2.0f,
                ),
            )

        val encoded =
            DepthWireCodec.encode(
                deviceId = DeviceId("camera-front"),
                frame = frame,
            )

        assertTrue(encoded != null)

        val wire =
            DepthWireCodec.decode(
                requireNotNull(encoded),
            )

        val decoded =
            DepthWireCodec.toDepthFrame(wire)

        assertEquals(
            frame.timestampNanos,
            decoded.timestampNanos,
        )
        assertEquals(
            DepthSource.NETWORK_COMPRESSED,
            decoded.source,
        )
        assertEquals(frame.width, decoded.width)
        assertEquals(frame.height, decoded.height)

        frame.depthMeters.indices.forEach { index ->
            assertEquals(
                frame.depthMeters[index].toDouble(),
                decoded.depthMeters[index].toDouble(),
                0.0011,
            )
        }
    }
}

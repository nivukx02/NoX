package com.nivukx.nox.core.network

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class NoXWireProtocolTest {
    @Test
    fun roundTripsPayloadAndMetadata() {
        val payload = ByteArray(513) { (it * 31).toByte() }
        val encoded = NoXWireProtocol.encode(
            type = NoXWireProtocol.Type.CALIBRATION_FRAME,
            sequence = 77L,
            timestampNanos = 99887766L,
            payload = payload,
        )

        val decoded = NoXWireProtocol.decode(encoded)

        assertEquals(NoXWireProtocol.Type.CALIBRATION_FRAME, decoded.type)
        assertEquals(77L, decoded.sequence)
        assertEquals(99887766L, decoded.timestampNanos)
        assertArrayEquals(payload, decoded.payload)
    }
}

package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertEquals as assertEq
import kotlin.test.assertTrue

class TimingBeaconObservationWireCodecTest {

    @Test
    fun timingBeaconObservationRoundTrips() {
        val input =
            TimingBeaconObservationWireCodec.Observation(
                deviceId = DeviceId("cam-17"),
                frameSequence = 440L,
                deviceTimestampNanos = 10_000_000_000L,
                beaconTimestampNanos = 10_002_500_000L,
                exposureTimeNanos = 2_000_000L,
                rollingShutterSkewNanos = 800_000L,
                normalizedRow = 0.63,
                decodeConfidence = 0.98,
            )

        val encoded =
            TimingBeaconObservationWireCodec.encode(
                input,
                packetSequence = 990L,
            )
        val decoded =
            TimingBeaconObservationWireCodec.decode(
                NoXWireProtocol.decode(encoded),
            )

        assertEquals(input.deviceId, decoded.deviceId)
        assertEq(input.frameSequence, decoded.frameSequence)
        assertEq(
            input.deviceTimestampNanos,
            decoded.deviceTimestampNanos,
        )
        assertEq(
            input.beaconTimestampNanos,
            decoded.beaconTimestampNanos,
        )
        assertTrue(
            kotlin.math.abs(
                input.normalizedRow -
                    decoded.normalizedRow,
            ) < 1e-6,
        )
        assertTrue(
            kotlin.math.abs(
                input.decodeConfidence -
                    decoded.decodeConfidence,
            ) < 1e-5,
        )
    }

    @Test
    fun invalidBeaconConfidenceFailsClosed() {
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            TimingBeaconObservationWireCodec.Observation(
                deviceId = DeviceId("cam"),
                frameSequence = 1L,
                deviceTimestampNanos = 10L,
                beaconTimestampNanos = 20L,
                exposureTimeNanos = 1_000L,
                rollingShutterSkewNanos = 0L,
                normalizedRow = 0.5,
                decodeConfidence = 2.0,
            )
        }
    }
}

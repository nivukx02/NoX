package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PeerTransportSheddingPolicyTest {

    @Test
    fun criticalControlTrafficSurvivesCongestion() {
        val policy = PeerTransportSheddingPolicy()

        assertFalse(
            policy.shouldShed(
                NoXWireProtocol.Type.PING,
                packetBytes = 30_000,
                outstandingPayloads = 128,
            ),
        )
        assertFalse(
            policy.shouldShed(
                NoXWireProtocol.Type.TIMING_BEACON_OBSERVATION,
                packetBytes = 30_000,
                outstandingPayloads = 128,
            ),
        )
    }

    @Test
    fun denseEvidenceIsShedBeforeControlTraffic() {
        val policy = PeerTransportSheddingPolicy()

        assertTrue(
            policy.shouldShed(
                NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                packetBytes = 20_000,
                outstandingPayloads = 96,
            ),
        )
        assertFalse(
            policy.shouldShed(
                NoXWireProtocol.Type.CALIBRATION,
                packetBytes = 20_000,
                outstandingPayloads = 96,
            ),
        )
    }

    @Test
    fun smallPacketsAvoidCongestionShedding() {
        val policy = PeerTransportSheddingPolicy()

        assertFalse(
            policy.shouldShed(
                NoXWireProtocol.Type.DEPTH_FRAME,
                packetBytes = 3_500,
                outstandingPayloads = 512,
            ),
        )
    }
    @Test
    fun reconstructionCriticalEvidenceIsNeverShedByOutstandingCount() {
        val policy =
            PeerTransportSheddingPolicy(
                maximumNormalOutstanding = 1,
            )

        assertFalse(
            policy.shouldShed(
                type =
                    NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                packetBytes = 8 * 1024,
                outstandingPayloads = 1000,
                reconstructionCritical = true,
            ),
        )
    }

    @Test
    fun normalEvidenceStillShedsUnderConfiguredPressure() {
        val policy =
            PeerTransportSheddingPolicy(
                maximumNormalOutstanding = 2,
            )

        assertTrue(
            policy.shouldShed(
                type =
                    NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                packetBytes = 8 * 1024,
                outstandingPayloads = 2,
            ),
        )
    }

    @Test
    fun timingBeaconRemainsUnshedAtExtremeCongestion() {
        val policy =
            PeerTransportSheddingPolicy(
                maximumCriticalOutstanding = 1,
                maximumHighOutstanding = 1,
            )

        assertFalse(
            policy.shouldShed(
                NoXWireProtocol.Type.TIMING_BEACON_OBSERVATION,
                packetBytes = 30_000,
                outstandingPayloads = 1000,
            ),
        )
    }


}

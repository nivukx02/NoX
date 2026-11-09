package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PeerTransportCongestionControllerTest {

    @Test
    fun immediate_failure_can_refund_reserved_budget() {
        val controller =
            PeerTransportCongestionController(
                PeerTransportCongestionController.Config(
                    bootstrapThroughputMbps = 10.0,
                    burstSeconds = 0.05,
                ),
            )

        val now = 1_000_000_000L
        val bytes = 6_000

        assertTrue(
            controller.tryAcquire(
                endpoint = "peer",
                type = NoXWireProtocol.Type.POSE_FRAME,
                packetBytes = bytes,
                measuredThroughputMbps = 10.0,
                nowNanos = now,
            ),
        )
        assertTrue(
            controller.tryAcquire(
                endpoint = "peer",
                type = NoXWireProtocol.Type.POSE_FRAME,
                packetBytes = bytes,
                measuredThroughputMbps = 10.0,
                nowNanos = now,
            ),
        )
        assertTrue(
            !controller.tryAcquire(
                endpoint = "peer",
                type = NoXWireProtocol.Type.POSE_FRAME,
                packetBytes = bytes,
                measuredThroughputMbps = 10.0,
                nowNanos = now,
            ),
        )

        controller.refund(
            endpoint = "peer",
            type = NoXWireProtocol.Type.POSE_FRAME,
            packetBytes = bytes,
        )

        assertTrue(
            controller.tryAcquire(
                endpoint = "peer",
                type = NoXWireProtocol.Type.POSE_FRAME,
                packetBytes = bytes,
                measuredThroughputMbps = 10.0,
                nowNanos = now,
            ),
        )
    }

    @Test
    fun criticalTrafficHasIndependentReservedBudget() {
        val controller =
            PeerTransportCongestionController(
                PeerTransportCongestionController.Config(
                    bootstrapThroughputMbps = 10.0,
                    burstSeconds = 0.05,
                    criticalReserveFraction = 0.50,
                ),
            )

        val first =
            controller.tryAcquire(
                endpoint = "cam-a",
                type = NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                packetBytes = 100_000,
                measuredThroughputMbps = 10.0,
                reconstructionCritical = true,
                nowNanos = 1_000_000_000L,
            )

        assertTrue(first)

        repeat(10) {
            controller.tryAcquire(
                endpoint = "cam-a",
                type = NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                packetBytes = 100_000,
                measuredThroughputMbps = 10.0,
                reconstructionCritical = false,
                nowNanos = 1_000_000_000L,
            )
        }

        assertTrue(
            controller.tryAcquire(
                endpoint = "cam-a",
                type = NoXWireProtocol.Type.TIMING_BEACON_OBSERVATION,
                packetBytes = 4_000,
                measuredThroughputMbps = 10.0,
                reconstructionCritical = true,
                nowNanos = 1_000_000_000L,
            ),
        )
    }

    @Test
    fun normalTrafficIsRejectedWhenItsByteBudgetIsExhausted() {
        val controller =
            PeerTransportCongestionController(
                PeerTransportCongestionController.Config(
                    bootstrapThroughputMbps = 5.0,
                    burstSeconds = 0.02,
                    criticalReserveFraction = 0.50,
                ),
            )

        assertTrue(
            controller.tryAcquire(
                endpoint = "cam-b",
                type = NoXWireProtocol.Type.POSE_FRAME,
                packetBytes = 10_000,
                measuredThroughputMbps = 5.0,
                nowNanos = 2_000_000_000L,
            ),
        )

        assertFalse(
            controller.tryAcquire(
                endpoint = "cam-b",
                type = NoXWireProtocol.Type.POSE_FRAME,
                packetBytes = 100_000,
                measuredThroughputMbps = 5.0,
                nowNanos = 2_000_000_000L,
            ),
        )
    }

    @Test
    fun tokensRecoverAfterTimeAdvances() {
        val controller =
            PeerTransportCongestionController(
                PeerTransportCongestionController.Config(
                    bootstrapThroughputMbps = 20.0,
                    burstSeconds = 0.01,
                    criticalReserveFraction = 0.30,
                    highShare = 0.20,
                    normalShare = 0.40,
                    lowShare = 0.10,
                ),
            )

        controller.tryAcquire(
            endpoint = "cam-c",
            type = NoXWireProtocol.Type.POSE_FRAME,
            packetBytes = 40_000,
            measuredThroughputMbps = 20.0,
            nowNanos = 3_000_000_000L,
        )

        assertTrue(
            controller.tryAcquire(
                endpoint = "cam-c",
                type = NoXWireProtocol.Type.POSE_FRAME,
                packetBytes = 40_000,
                measuredThroughputMbps = 20.0,
                nowNanos = 3_100_000_000L,
            ),
        )
    }
}

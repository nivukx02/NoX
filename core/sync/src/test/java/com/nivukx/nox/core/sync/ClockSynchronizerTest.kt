package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DeviceId
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ClockSynchronizerTest {
    @Test
    fun convertsBothClockDirections() {
        val sync = ClockSynchronizer()
        val device = DeviceId("peer")
        val offset = 4_000_000L
        val t1 = 1_000_000_000L
        val t4 = t1 + 2_000_000L
        val t2 = t1 + offset + 500_000L
        val t3 = t2 + 100_000L

        sync.observe(
            device,
            ClockSynchronizer.Sample(
                t1 = t1,
                t2 = t2,
                t3 = t3,
                t4 = t4 + 100_000L,
            ),
        )

        val estimate = sync.estimate(device)
        val masterTime = 2_000_000_000L
        val peerTime = sync.convertMasterToPeer(device, masterTime)
        val roundTripMaster = sync.convertPeerToMaster(device, peerTime)

        assertTrue(
            kotlin.math.abs(estimate.offsetNanos - (peerTime - masterTime)) <= 1_500_000L,
        )
        assertTrue(
            kotlin.math.abs(masterTime - roundTripMaster) <= 1_500_000L,
        )
    }
    @Test
    fun robust_offset_estimation_rejects_single_extreme_rtt_outlier() {
        val sync = ClockSynchronizer(
            alpha = 0.65,
            windowSize = 16,
        )
        val device = DeviceId("robust-peer")
        val trueOffset = 3_750_000L
        val base = 2_000_000_000L

        repeat(11) { index ->
            val t1 = base + index * 20_000_000L
            val t2 = t1 + trueOffset + 700_000L
            val t3 = t2 + 200_000L
            val t4 = t1 + 1_600_000L
            sync.observe(
                device,
                ClockSynchronizer.Sample(
                    t1 = t1,
                    t2 = t2,
                    t3 = t3,
                    t4 = t4,
                ),
            )
        }

        sync.observe(
            device,
            ClockSynchronizer.Sample(
                t1 = base + 220_000_000L,
                t2 = base + 420_000_000L,
                t3 = base + 425_000_000L,
                t4 = base + 1_200_000_000L,
            ),
        )

        val estimate = sync.estimate(device)

        assertTrue(
            kotlin.math.abs(
                estimate.offsetNanos - trueOffset,
            ) < 900_000L,
        )
        assertTrue(estimate.confidence > 0.35)
    }

}

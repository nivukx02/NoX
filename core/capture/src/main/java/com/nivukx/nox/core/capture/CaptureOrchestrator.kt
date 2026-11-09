package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.sync.ClockSynchronizer
import kotlin.math.max

class CaptureOrchestrator(
    private val clock: ClockSynchronizer,
    private val prepareLeadNanos: Long = 1_500_000_000L,
) {
    data class ScheduledCapture(
        val masterStartNanos: Long,
        val peerStartNanos: Map<DeviceId, Long>,
        val createdAtNanos: Long,
        val requestedLeadNanos: Long,
    )

    fun schedule(
        peers: Collection<DeviceId>,
        nowMasterNanos: Long = System.nanoTime(),
        requestedLeadNanos: Long = prepareLeadNanos,
    ): ScheduledCapture {
        val lead = max(400_000_000L, requestedLeadNanos)
        val masterStart = nowMasterNanos + lead
        val peerStart = peers.associateWith { peer ->
            clock.convertMasterToPeer(peer, masterStart)
        }
        return ScheduledCapture(
            masterStartNanos = masterStart,
            peerStartNanos = peerStart,
            createdAtNanos = nowMasterNanos,
            requestedLeadNanos = lead,
        )
    }

    fun readyForCapture(
        schedule: ScheduledCapture,
        nowMasterNanos: Long = System.nanoTime(),
    ): Boolean =
        nowMasterNanos >= schedule.masterStartNanos - 30_000_000L
}

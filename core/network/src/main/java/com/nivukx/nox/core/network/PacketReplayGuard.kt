package com.nivukx.nox.core.network

import java.util.ArrayDeque
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

/**
 * Bounded duplicate/replay guard for NoX control and evidence packets.
 *
 * This is deliberately not a cryptographic authentication layer. Its job is to prevent
 * transport-level duplicate delivery from feeding the same packet into the orchestration
 * graph repeatedly. Session security remains a separate responsibility.
 */
class PacketReplayGuard(
    private val capacityPerPeer: Int = 512,
    private val maximumPeers: Int = 256,
) {
    init {
        require(capacityPerPeer >= 32)
        require(maximumPeers >= 1)
    }

    private data class PeerState(
        val order: ArrayDeque<Long> = ArrayDeque(),
        val seen: HashSet<Long> = HashSet(),
    )

    private val peers = ConcurrentHashMap<String, PeerState>()
    private val peerOrder = ArrayDeque<String>()
    private val peerLifecycleLock = Any()
    private val accepted = AtomicLong(0L)
    private val rejected = AtomicLong(0L)

    fun accept(
        endpointId: String,
        packet: ByteArray,
    ): Boolean {
        if (packet.isEmpty()) return true

        val fingerprint = fingerprint(packet)

        val state =
            synchronized(peerLifecycleLock) {
                peers[endpointId] ?: run {
                    while (peerOrder.size >= maximumPeers) {
                        val expired =
                            if (peerOrder.isEmpty()) null else peerOrder.removeFirst()
                        if (expired != null) {
                            peers.remove(expired)
                        } else {
                            break
                        }
                    }
                    val created = PeerState()
                    peers[endpointId] = created
                    peerOrder.addLast(endpointId)
                    created
                }
            }

        synchronized(state) {
            if (!state.seen.add(fingerprint)) {
                rejected.incrementAndGet()
                return false
            }

            state.order.addLast(fingerprint)
            while (state.order.size > capacityPerPeer) {
                val expired = state.order.removeFirst()
                state.seen.remove(expired)
            }
        }

        accepted.incrementAndGet()
        return true
    }

    fun clear(endpointId: String) {
        synchronized(peerLifecycleLock) {
            peers.remove(endpointId)
            peerOrder.removeIf { it == endpointId }
        }
    }

    fun clearAll() {
        synchronized(peerLifecycleLock) {
            peers.clear()
            peerOrder.clear()
        }
    }

    data class Diagnostics(
        val peers: Int,
        val accepted: Long,
        val rejected: Long,
    )

    fun diagnostics(): Diagnostics =
        Diagnostics(
            peers = peers.size,
            accepted = accepted.get(),
            rejected = rejected.get(),
        )

    private fun fingerprint(packet: ByteArray): Long {
        var hash = 0xcbf29ce484222325UL
        val sampleCount = minOf(packet.size, 96)

        for (index in 0 until sampleCount) {
            hash = hash xor (packet[index].toLong().and(0xFFL).toULong())
            hash *= 0x100000001B3UL
        }

        hash = hash xor packet.size.toULong()
        hash *= 0x100000001B3UL
        return hash.toLong()
    }
}

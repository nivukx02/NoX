package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Route-aware envelope carried inside the versioned NoX wire protocol.
 *
 * The envelope contains an immutable route selected by DistributedTransportCoordinator. Each relay
 * may only advance the hop when it is the exact next node in that route. TTL and hop bounds make
 * malformed or looping routes fail closed.
 */
object DistributedTransportEnvelopeCodec {

    private const val VERSION = 1
    private const val HEADER_BYTES = 36
    private const val MAX_ROUTE_HOPS = 12
    private const val MAX_DEVICE_ID_BYTES = 96
    private const val MAX_PAYLOAD_BYTES =
        NoXWireProtocol.MAX_PACKET_BYTES - 36 - HEADER_BYTES

    data class Envelope(
        val origin: DeviceId,
        val destination: DeviceId,
        val route: List<DeviceId>,
        val hopIndex: Int,
        val ttl: Int,
        val packetSequence: Long,
        val timestampNanos: Long,
        val payload: ByteArray,
        val reconstructionCritical: Boolean = false,
    ) {
        init {
            require(origin.value.isNotBlank())
            require(destination.value.isNotBlank())
            require(route.isNotEmpty())
            require(route.size <= MAX_ROUTE_HOPS)
            require(
                route.distinct().size ==
                    route.size,
            ) {
                "Distributed route contains duplicate nodes"
            }
            require(hopIndex in 0..route.lastIndex)
            require(ttl in 0..route.size)
            require(packetSequence >= 0L)
            require(timestampNanos > 0L)
            require(payload.isNotEmpty())
            require(
                payload.size <=
                    MAX_PAYLOAD_BYTES,
            )
            require(
                route.last() ==
                    destination,
            )
        }
    }

    fun maxPayloadBytesForRoute(
        origin: DeviceId,
        destination: DeviceId,
        route: List<DeviceId>,
    ): Int {
        require(origin.value.isNotBlank())
        require(destination.value.isNotBlank())
        require(route.isNotEmpty())
        require(route.size <= MAX_ROUTE_HOPS)
        require(route.last() == destination)
        require(route.distinct().size == route.size)

        val originBytes =
            origin.value.toByteArray(Charsets.UTF_8)
        val destinationBytes =
            destination.value.toByteArray(Charsets.UTF_8)
        require(originBytes.size <= MAX_DEVICE_ID_BYTES)
        require(destinationBytes.size <= MAX_DEVICE_ID_BYTES)

        val routeEncodedBytes =
            route.sumOf {
                2 +
                    it.value.toByteArray(
                        Charsets.UTF_8,
                    ).size
            }

        return (
            NoXWireProtocol.MAX_PACKET_BYTES -
                36 -
                HEADER_BYTES -
                originBytes.size -
                destinationBytes.size -
                routeEncodedBytes
            ).coerceAtLeast(0)
    }

    fun encode(
        origin: DeviceId,
        destination: DeviceId,
        route: List<DeviceId>,
        hopIndex: Int,
        ttl: Int,
        packetSequence: Long,
        timestampNanos: Long,
        payload: ByteArray,
        reconstructionCritical: Boolean = false,
    ): ByteArray {
        val originBytes =
            origin.value.toByteArray(Charsets.UTF_8)
        val destinationBytes =
            destination.value.toByteArray(Charsets.UTF_8)

        require(originBytes.isNotEmpty())
        require(destinationBytes.isNotEmpty())
        require(originBytes.size <= MAX_DEVICE_ID_BYTES)
        require(destinationBytes.size <= MAX_DEVICE_ID_BYTES)
        require(route.isNotEmpty())
        require(route.size <= MAX_ROUTE_HOPS)
        require(
            route.last() == destination,
        )

        val routeBytes =
            route.map {
                it.value.toByteArray(Charsets.UTF_8)
            }

        require(
            routeBytes.all {
                it.isNotEmpty() &&
                    it.size <= MAX_DEVICE_ID_BYTES
            },
        )
        require(
            route.distinct().size ==
                route.size,
        )
        require(hopIndex in 0 until route.size)
        require(ttl in 0..route.size)
        require(packetSequence >= 0L)
        require(timestampNanos > 0L)
        require(
            payload.isNotEmpty() &&
                payload.size <=
                    maxPayloadBytesForRoute(
                        origin = origin,
                        destination = destination,
                        route = route,
                    ),
        )

        val routeEncodedBytes =
            routeBytes.sumOf {
                2 + it.size
            }

        val bodySize =
            HEADER_BYTES +
                originBytes.size +
                destinationBytes.size +
                routeEncodedBytes +
                payload.size

        require(
            bodySize <=
                NoXWireProtocol.MAX_PACKET_BYTES - 36,
        ) {
            "Distributed route envelope exceeds NoX packet budget"
        }

        val body =
            ByteBuffer
                .allocate(bodySize)
                .order(ByteOrder.BIG_ENDIAN)
                .apply {
                    putInt(VERSION)
                    putShort(originBytes.size.toShort())
                    putShort(destinationBytes.size.toShort())
                    putShort(route.size.toShort())
                    putShort(hopIndex.toShort())
                    putShort(ttl.toShort())
                    putShort(
                        if (reconstructionCritical) 1 else 0,
                    )
                    putLong(packetSequence)
                    putInt(payload.size)
                    putInt(0)
                    put(originBytes)
                    put(destinationBytes)

                    routeBytes.forEach { id ->
                        putShort(id.size.toShort())
                        put(id)
                    }

                    put(payload)
                }
                .array()

        return NoXWireProtocol.encode(
            type =
                NoXWireProtocol.Type.ROUTED_EVIDENCE_FRAME,
            sequence = packetSequence,
            timestampNanos = timestampNanos,
            payload = body,
        )
    }

    fun decode(
        packet: NoXWireProtocol.Packet,
    ): Envelope {
        require(
            packet.type ==
                NoXWireProtocol.Type.ROUTED_EVIDENCE_FRAME,
        )

        val buffer =
            ByteBuffer
                .wrap(packet.payload)
                .order(ByteOrder.BIG_ENDIAN)

        require(buffer.remaining() >= HEADER_BYTES)

        require(buffer.int == VERSION)

        val originLength =
            buffer.short.toInt() and 0xFFFF
        val destinationLength =
            buffer.short.toInt() and 0xFFFF
        val routeCount =
            buffer.short.toInt() and 0xFFFF
        val hopIndex =
            buffer.short.toInt() and 0xFFFF
        val ttl =
            buffer.short.toInt() and 0xFFFF
        val flags =
            buffer.short.toInt() and 0xFFFF
        require(flags and 0xFFFE == 0) {
            "Unsupported distributed route flags"
        }
        val reconstructionCritical =
            flags and 0x0001 != 0
        val packetSequence = buffer.long
        val payloadLength = buffer.int
        buffer.int

        require(originLength in 1..MAX_DEVICE_ID_BYTES)
        require(destinationLength in 1..MAX_DEVICE_ID_BYTES)
        require(routeCount in 1..MAX_ROUTE_HOPS)
        require(hopIndex in 0 until routeCount)
        require(ttl in 0..routeCount)
        require(packetSequence >= 0L)
        require(
            payloadLength in
                1..MAX_PAYLOAD_BYTES,
        )
        require(buffer.remaining() >= originLength + destinationLength)

        val origin =
            DeviceId(
                ByteArray(originLength)
                    .also(buffer::get)
                    .toString(Charsets.UTF_8),
            )
        val destination =
            DeviceId(
                ByteArray(destinationLength)
                    .also(buffer::get)
                    .toString(Charsets.UTF_8),
            )

        require(origin.value.isNotBlank())
        require(destination.value.isNotBlank())

        val route =
            ArrayList<DeviceId>(routeCount)

        repeat(routeCount) {
            require(buffer.remaining() >= 2)
            val length =
                buffer.short.toInt() and 0xFFFF
            require(length in 1..MAX_DEVICE_ID_BYTES)
            require(buffer.remaining() >= length)
            route +=
                DeviceId(
                    ByteArray(length)
                        .also(buffer::get)
                        .toString(Charsets.UTF_8),
                )
        }

        require(
            route.distinct().size ==
                route.size,
        )
        require(
            route.last() == destination,
        )
        require(
            buffer.remaining() ==
                payloadLength,
        )

        val payload =
            ByteArray(payloadLength)
                .also(buffer::get)

        return Envelope(
            origin = origin,
            destination = destination,
            route = route,
            hopIndex = hopIndex,
            ttl = ttl,
            packetSequence = packetSequence,
            timestampNanos = packet.timestampNanos,
            payload = payload,
            reconstructionCritical =
                reconstructionCritical,
        )
    }

    fun canAdvance(
        envelope: Envelope,
        localDeviceId: DeviceId,
    ): Boolean {
        if (
            localDeviceId ==
                envelope.destination
        ) {
            return false
        }

        if (envelope.ttl <= 0) {
            return false
        }

        if (
            envelope.hopIndex >=
                envelope.route.lastIndex
        ) {
            return false
        }

        return envelope.route[
            envelope.hopIndex
        ] == localDeviceId
    }

    fun advance(
        envelope: Envelope,
        localDeviceId: DeviceId,
    ): Envelope {
        require(
            canAdvance(
                envelope,
                localDeviceId,
            ),
        ) {
            "Local device is not the next hop for distributed route"
        }

        val nextHop =
            envelope.hopIndex + 1
        val nextTtl =
            envelope.ttl - 1

        require(nextTtl >= 0)

        return envelope.copy(
            hopIndex = nextHop,
            ttl = nextTtl,
        )
    }
}

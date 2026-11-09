package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId

/**
 * Encrypts a complete distributed route envelope for long-haul evidence transport.
 *
 * The underlying route codec remains authoritative for hop/TTL semantics. AES-GCM protects the full
 * route, origin, destination, and inner evidence from passive inspection or tampering.
 */
object SecureDistributedRouteEnvelopeCodec {

    data class Decoded(
        val origin: DeviceId,
        val destination: DeviceId,
        val route: List<DeviceId>,
        val hopIndex: Int,
        val ttl: Int,
        val packetSequence: Long,
        val timestampNanos: Long,
        val payload: ByteArray,
    )

    fun encode(
        origin: DeviceId,
        destination: DeviceId,
        route: List<DeviceId>,
        hopIndex: Int,
        ttl: Int,
        packetSequence: Long,
        timestampNanos: Long,
        payload: ByteArray,
        key: ByteArray,
    ): ByteArray {
        val routed =
            DistributedTransportEnvelopeCodec.encode(
                origin = origin,
                destination = destination,
                route = route,
                hopIndex = hopIndex,
                ttl = ttl,
                packetSequence = packetSequence,
                timestampNanos = timestampNanos,
                payload = payload,
            )

        return SecureRemoteEvidenceCodec.encode(
            origin = origin,
            destination = destination,
            sequence = packetSequence,
            timestampNanos = timestampNanos,
            plaintext = routed,
            key = key,
        )
    }

    fun decode(
        packetBytes: ByteArray,
        key: ByteArray,
    ): Decoded {
        val secure =
            SecureRemoteEvidenceCodec.decode(
                packetBytes = packetBytes,
                key = key,
            )

        val routedPacket =
            NoXWireProtocol.decode(
                secure.plaintext,
            )

        val envelope =
            DistributedTransportEnvelopeCodec.decode(
                routedPacket,
            )

        require(
            envelope.origin ==
                secure.origin,
        ) {
            "Secure routed origin mismatch"
        }
        require(
            envelope.destination ==
                secure.destination,
        ) {
            "Secure routed destination mismatch"
        }
        require(
            envelope.packetSequence ==
                secure.sequence,
        ) {
            "Secure routed sequence mismatch"
        }
        require(
            envelope.timestampNanos ==
                secure.timestampNanos,
        ) {
            "Secure routed timestamp mismatch"
        }

        return Decoded(
            origin = envelope.origin,
            destination = envelope.destination,
            route = envelope.route,
            hopIndex = envelope.hopIndex,
            ttl = envelope.ttl,
            packetSequence = envelope.packetSequence,
            timestampNanos = envelope.timestampNanos,
            payload = envelope.payload,
        )
    }

    fun advance(
        decoded: Decoded,
        localDeviceId: DeviceId,
    ): Decoded {
        val advanced =
            DistributedTransportEnvelopeCodec.advance(
                DistributedTransportEnvelopeCodec.Envelope(
                    origin = decoded.origin,
                    destination = decoded.destination,
                    route = decoded.route,
                    hopIndex = decoded.hopIndex,
                    ttl = decoded.ttl,
                    packetSequence = decoded.packetSequence,
                    timestampNanos = decoded.timestampNanos,
                    payload = decoded.payload,
                ),
                localDeviceId,
            )

        return Decoded(
            origin = advanced.origin,
            destination = advanced.destination,
            route = advanced.route,
            hopIndex = advanced.hopIndex,
            ttl = advanced.ttl,
            packetSequence = advanced.packetSequence,
            timestampNanos = advanced.timestampNanos,
            payload = advanced.payload,
        )
    }
}

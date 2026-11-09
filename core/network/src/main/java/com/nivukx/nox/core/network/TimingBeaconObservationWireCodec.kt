package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import java.nio.ByteBuffer
import java.nio.ByteOrder

object TimingBeaconObservationWireCodec {

    private const val VERSION = 1
    private const val MAX_DEVICE_ID_BYTES = 96
    private const val BODY_BYTES = 4 + 2 + 8 + 8 + 8 + 8 + 8 + 8 + 4

    data class Observation(
        val deviceId: DeviceId,
        val frameSequence: Long,
        val deviceTimestampNanos: Long,
        val beaconTimestampNanos: Long,
        val exposureTimeNanos: Long,
        val rollingShutterSkewNanos: Long,
        val normalizedRow: Double,
        val decodeConfidence: Double,
    ) {
        init {
            require(deviceId.value.isNotBlank())
            require(
                deviceId.value.toByteArray(Charsets.UTF_8).size in
                    1..MAX_DEVICE_ID_BYTES,
            )
            require(frameSequence >= 0L)
            require(deviceTimestampNanos >= 0L)
            require(beaconTimestampNanos >= 0L)
            require(exposureTimeNanos >= 0L)
            require(rollingShutterSkewNanos >= 0L)
            require(normalizedRow.isFinite() && normalizedRow in 0.0..1.0)
            require(
                decodeConfidence.isFinite() &&
                    decodeConfidence in 0.0..1.0,
            )
        }
    }

    fun encode(
        observation: Observation,
        packetSequence: Long,
    ): ByteArray {
        require(packetSequence >= 0L)

        val deviceBytes =
            observation.deviceId.value
                .toByteArray(Charsets.UTF_8)

        val payload =
            ByteBuffer
                .allocate(
                    BODY_BYTES +
                        deviceBytes.size,
                )
                .order(ByteOrder.BIG_ENDIAN)
                .apply {
                    putInt(VERSION)
                    putShort(deviceBytes.size.toShort())
                    putLong(observation.frameSequence)
                    putLong(observation.deviceTimestampNanos)
                    putLong(observation.beaconTimestampNanos)
                    putLong(observation.exposureTimeNanos)
                    putLong(observation.rollingShutterSkewNanos)
                    putDouble(observation.normalizedRow)
                    putFloat(observation.decodeConfidence.toFloat())
                    put(deviceBytes)
                }
                .array()

        return NoXWireProtocol.encode(
            type =
                NoXWireProtocol.Type.TIMING_BEACON_OBSERVATION,
            sequence = packetSequence,
            timestampNanos =
                observation.deviceTimestampNanos,
            payload = payload,
        )
    }

    fun decode(
        packet: NoXWireProtocol.Packet,
    ): Observation {
        require(
            packet.type ==
                NoXWireProtocol.Type.TIMING_BEACON_OBSERVATION,
        )

        val buffer =
            ByteBuffer
                .wrap(packet.payload)
                .order(ByteOrder.BIG_ENDIAN)

        require(buffer.remaining() >= 6)
        require(buffer.int == VERSION)

        val deviceLength =
            buffer.short.toInt() and 0xFFFF

        require(
            deviceLength in 1..MAX_DEVICE_ID_BYTES,
        )
        require(buffer.remaining() >= deviceLength + 8 + 8 + 8 + 8 + 8 + 4 + 8)

        val frameSequence =
            buffer.long
        val deviceTimestampNanos =
            buffer.long
        val beaconTimestampNanos =
            buffer.long
        val exposureTimeNanos =
            buffer.long
        val rollingShutterSkewNanos =
            buffer.long
        val normalizedRow =
            buffer.double
        val decodeConfidence =
            buffer.float.toDouble()

        require(buffer.remaining() == deviceLength)

        val deviceId =
            DeviceId(
                ByteArray(deviceLength)
                    .also(buffer::get)
                    .toString(Charsets.UTF_8),
            )

        return Observation(
            deviceId = deviceId,
            frameSequence = frameSequence,
            deviceTimestampNanos = deviceTimestampNanos,
            beaconTimestampNanos = beaconTimestampNanos,
            exposureTimeNanos = exposureTimeNanos,
            rollingShutterSkewNanos = rollingShutterSkewNanos,
            normalizedRow = normalizedRow,
            decodeConfidence = decodeConfidence,
        )
    }
}

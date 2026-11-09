package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceCapabilities
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Compact versioned peer capability handshake.
 *
 * Version 1 remains decodable for older nodes. Version 2 adds high-end distributed-capture
 * capabilities without weakening the existing compatibility path.
 */
object PeerCapabilitiesWireCodec {
    private const val MAGIC = 0x4E585043
    private const val VERSION_1 = 1
    private const val VERSION_2 = 2
    private const val BYTES_V1 = 24
    private const val BYTES_V2 = 40

    private const val FLAG_HARDWARE_READOUT = 1
    private const val FLAG_CALIBRATION_UNCERTAINTY = 2
    private const val FLAG_GLOBAL_IDENTITY = 4
    private const val FLAG_DENSE_FRAGMENTATION = 8
    private const val FLAG_ACTOR_FRAGMENTATION = 16
    private const val FLAG_VISUAL_TIMING_BEACON = 32
    private const val FLAG_DEPTH_FRAGMENTATION = 64
    private const val FLAG_DENSE_TIMING_METADATA = 128

    data class Capabilities(
        val protocolVersion: Int = 2,
        val calibrationSchemaVersion: Int = 1,
        val multiActorPoseSchemaVersion: Int = 1,
        val supportsHardwareReadoutTelemetry: Boolean = false,
        val supportsCalibrationUncertainty: Boolean = false,
        val supportsGlobalIdentityAssignment: Boolean = false,
        val supportsDenseLandmarkFragmentation: Boolean = false,
        val supportsDenseLandmarkTimingMetadata: Boolean = false,
        val supportsMultiActorPoseFragmentation: Boolean = false,
        val supportsVisualTimingBeacon: Boolean = false,
        val supportsDepthFragmentation: Boolean = false,
        val maximumSubjects: Int = 1,
        val maximumVerifiedConcurrentCameras: Int = 1,
        val timingAuthorityScore: Double = 0.0,
        val timingResidualP95Milliseconds: Double = 0.0,
    ) {
        init {
            require(protocolVersion in 1..16)
            require(calibrationSchemaVersion in 1..16)
            require(multiActorPoseSchemaVersion in 1..16)
            require(maximumSubjects in 1..64)
            require(maximumVerifiedConcurrentCameras in 1..128)
            require(timingAuthorityScore.isFinite() && timingAuthorityScore in 0.0..1.0)
            require(timingResidualP95Milliseconds.isFinite() && timingResidualP95Milliseconds >= 0.0)
        }
    }

    fun localDefaults(): Capabilities =
        Capabilities(
            protocolVersion = 2,
            calibrationSchemaVersion = 2,
            multiActorPoseSchemaVersion = 3,
            supportsHardwareReadoutTelemetry = false,
            supportsCalibrationUncertainty = true,
            supportsGlobalIdentityAssignment = true,
            supportsDenseLandmarkFragmentation = true,
            supportsDenseLandmarkTimingMetadata = true,
            supportsMultiActorPoseFragmentation = true,
            supportsVisualTimingBeacon = false,
            supportsDepthFragmentation = true,
            maximumSubjects = 64,
            maximumVerifiedConcurrentCameras = 1,
            timingAuthorityScore = 0.0,
            timingResidualP95Milliseconds = 0.0,
        )

    /**
     * Builds a peer capability advertisement from actual Camera2/device evidence.
     *
     * Timing authority is caller-supplied because only a live timing-quality measurement can
     * establish its value. The codec never infers hardware timing from Android API level alone.
     */
    fun fromDeviceCapabilities(
        capabilities: DeviceCapabilities,
        timingAuthorityScore: Double = 0.0,
        timingResidualP95Milliseconds: Double = 0.0,
        supportsVisualTimingBeacon: Boolean = false,
        supportsDepthFragmentation: Boolean = true,
    ): Capabilities {
        require(timingAuthorityScore in 0.0..1.0)
        require(
            timingResidualP95Milliseconds.isFinite() &&
                timingResidualP95Milliseconds >= 0.0,
        )

        return Capabilities(
            protocolVersion = 2,
            calibrationSchemaVersion = 2,
            multiActorPoseSchemaVersion = 3,
            supportsHardwareReadoutTelemetry =
                capabilities.supportsHardwareReadoutTimestamp,
            supportsCalibrationUncertainty = true,
            supportsGlobalIdentityAssignment = true,
            supportsDenseLandmarkFragmentation = true,
            supportsDenseLandmarkTimingMetadata = true,
            supportsMultiActorPoseFragmentation = true,
            supportsVisualTimingBeacon =
                supportsVisualTimingBeacon,
            supportsDepthFragmentation =
                supportsDepthFragmentation,
            maximumSubjects = 64,
            maximumVerifiedConcurrentCameras =
                capabilities.maximumVerifiedConcurrentCameras
                    .coerceIn(1, 128),
            timingAuthorityScore =
                if (capabilities.supportsHardwareReadoutTimestamp) {
                    timingAuthorityScore
                } else {
                    timingAuthorityScore.coerceAtMost(0.90)
                },
            timingResidualP95Milliseconds =
                timingResidualP95Milliseconds,
        )
    }

    fun encode(
        capabilities: Capabilities,
    ): ByteArray {
        var flags = 0

        if (capabilities.supportsHardwareReadoutTelemetry) {
            flags = flags or FLAG_HARDWARE_READOUT
        }
        if (capabilities.supportsCalibrationUncertainty) {
            flags = flags or FLAG_CALIBRATION_UNCERTAINTY
        }
        if (capabilities.supportsGlobalIdentityAssignment) {
            flags = flags or FLAG_GLOBAL_IDENTITY
        }
        if (capabilities.supportsDenseLandmarkFragmentation) {
            flags = flags or FLAG_DENSE_FRAGMENTATION
        }
        if (capabilities.supportsMultiActorPoseFragmentation) {
            flags = flags or FLAG_ACTOR_FRAGMENTATION
        }
        if (capabilities.supportsVisualTimingBeacon) {
            flags = flags or FLAG_VISUAL_TIMING_BEACON
        }
        if (capabilities.supportsDepthFragmentation) {
            flags = flags or FLAG_DEPTH_FRAGMENTATION
        }
        if (capabilities.supportsDenseLandmarkTimingMetadata) {
            flags = flags or FLAG_DENSE_TIMING_METADATA
        }

        return ByteBuffer.allocate(BYTES_V2)
            .order(ByteOrder.BIG_ENDIAN)
            .apply {
                putInt(MAGIC)
                putInt(VERSION_2)
                putInt(capabilities.protocolVersion)
                putInt(capabilities.calibrationSchemaVersion)
                putInt(capabilities.multiActorPoseSchemaVersion)
                putInt(flags)
                putInt(capabilities.maximumSubjects)
                putInt(capabilities.maximumVerifiedConcurrentCameras)
                putFloat(capabilities.timingAuthorityScore.toFloat())
                putFloat(capabilities.timingResidualP95Milliseconds.toFloat())
            }
            .array()
    }

    fun decode(
        bytes: ByteArray,
    ): Capabilities? =
        runCatching {
            require(
                bytes.size == BYTES_V1 ||
                    bytes.size == BYTES_V2,
            )

            val buffer =
                ByteBuffer
                    .wrap(bytes)
                    .order(ByteOrder.BIG_ENDIAN)

            require(buffer.int == MAGIC)

            val version = buffer.int

            when (version) {
                VERSION_1 ->
                    decodeV1Body(buffer)

                VERSION_2 ->
                    decodeV2Body(buffer)

                else ->
                    error("Unsupported peer capability handshake version")
            }
        }.getOrNull()

    /**
     * Test-only encoder retained as a source-level compatibility hook for v1 fixtures.
     */
    fun encodeV1ForTesting(
        capabilities: Capabilities,
    ): ByteArray {
        var flags = 0
        if (capabilities.supportsHardwareReadoutTelemetry) {
            flags = flags or FLAG_HARDWARE_READOUT
        }
        if (capabilities.supportsCalibrationUncertainty) {
            flags = flags or FLAG_CALIBRATION_UNCERTAINTY
        }
        if (capabilities.supportsGlobalIdentityAssignment) {
            flags = flags or FLAG_GLOBAL_IDENTITY
        }

        return ByteBuffer.allocate(BYTES_V1)
            .order(ByteOrder.BIG_ENDIAN)
            .apply {
                putInt(MAGIC)
                putInt(VERSION_1)
                putInt(capabilities.protocolVersion)
                putInt(capabilities.calibrationSchemaVersion)
                putInt(capabilities.multiActorPoseSchemaVersion)
                putInt(flags)
            }
            .array()
    }

    fun legacyFallback(): Capabilities =
        Capabilities(
            protocolVersion = 2,
            calibrationSchemaVersion = 1,
            multiActorPoseSchemaVersion = 1,
            supportsHardwareReadoutTelemetry = false,
            supportsCalibrationUncertainty = false,
            supportsGlobalIdentityAssignment = false,
            supportsDenseLandmarkFragmentation = false,
            supportsMultiActorPoseFragmentation = false,
            supportsVisualTimingBeacon = false,
            supportsDepthFragmentation = false,
            maximumSubjects = 1,
            maximumVerifiedConcurrentCameras = 1,
            timingAuthorityScore = 0.0,
            timingResidualP95Milliseconds = 0.0,
        )

    private fun decodeV1Body(
        buffer: ByteBuffer,
    ): Capabilities {
        val protocolVersion = buffer.int
        val calibrationSchema = buffer.int
        val multiActorSchema = buffer.int
        val flags = buffer.int

        require(!buffer.hasRemaining())

        return Capabilities(
            protocolVersion = protocolVersion,
            calibrationSchemaVersion = calibrationSchema,
            multiActorPoseSchemaVersion = multiActorSchema,
            supportsHardwareReadoutTelemetry =
                flags and FLAG_HARDWARE_READOUT != 0,
            supportsCalibrationUncertainty =
                flags and FLAG_CALIBRATION_UNCERTAINTY != 0,
            supportsGlobalIdentityAssignment =
                flags and FLAG_GLOBAL_IDENTITY != 0,
        )
    }

    private fun decodeV2Body(
        buffer: ByteBuffer,
    ): Capabilities {
        val protocolVersion = buffer.int
        val calibrationSchema = buffer.int
        val multiActorSchema = buffer.int
        val flags = buffer.int
        val maximumSubjects = buffer.int
        val maximumConcurrentCameras = buffer.int
        val timingAuthorityScore =
            buffer.float.toDouble()
        val timingResidualP95Milliseconds =
            buffer.float.toDouble()

        require(!buffer.hasRemaining())

        return Capabilities(
            protocolVersion = protocolVersion,
            calibrationSchemaVersion = calibrationSchema,
            multiActorPoseSchemaVersion = multiActorSchema,
            supportsHardwareReadoutTelemetry =
                flags and FLAG_HARDWARE_READOUT != 0,
            supportsCalibrationUncertainty =
                flags and FLAG_CALIBRATION_UNCERTAINTY != 0,
            supportsGlobalIdentityAssignment =
                flags and FLAG_GLOBAL_IDENTITY != 0,
            supportsDenseLandmarkFragmentation =
                flags and FLAG_DENSE_FRAGMENTATION != 0,
            supportsDenseLandmarkTimingMetadata =
                flags and FLAG_DENSE_TIMING_METADATA != 0,
            supportsMultiActorPoseFragmentation =
                flags and FLAG_ACTOR_FRAGMENTATION != 0,
            supportsVisualTimingBeacon =
                flags and FLAG_VISUAL_TIMING_BEACON != 0,
            supportsDepthFragmentation =
                flags and FLAG_DEPTH_FRAGMENTATION != 0,
            maximumSubjects = maximumSubjects,
            maximumVerifiedConcurrentCameras =
                maximumConcurrentCameras,
            timingAuthorityScore = timingAuthorityScore,
            timingResidualP95Milliseconds =
                timingResidualP95Milliseconds,
        )
    }
}

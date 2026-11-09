package com.nivukx.nox.core.network

import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.Test

class PeerCapabilitiesWireCodecTest {

    @Test
    fun roundTripsCurrentCapabilities() {
        val input = PeerCapabilitiesWireCodec.Capabilities(
            protocolVersion = 2,
            calibrationSchemaVersion = 2,
            multiActorPoseSchemaVersion = 1,
            supportsHardwareReadoutTelemetry = true,
            supportsCalibrationUncertainty = true,
            supportsGlobalIdentityAssignment = true,
        )

        val decoded =
            PeerCapabilitiesWireCodec.decode(
                PeerCapabilitiesWireCodec.encode(input),
            )

        assertEquals(input, decoded)
    }

    @Test
    fun malformedOrLegacyEmptyPayloadDoesNotPretendSupport() {
        assertEquals(
            null,
            PeerCapabilitiesWireCodec.decode(ByteArray(0)),
        )

        val legacy =
            PeerCapabilitiesWireCodec.legacyFallback()
        assertEquals(1, legacy.calibrationSchemaVersion)
        assertFalse(legacy.supportsCalibrationUncertainty)
        assertFalse(legacy.supportsHardwareReadoutTelemetry)
    }

    @Test
    fun flagsRemainIndependent() {
        val input =
            PeerCapabilitiesWireCodec.Capabilities(
                supportsHardwareReadoutTelemetry = true,
                supportsCalibrationUncertainty = false,
                supportsGlobalIdentityAssignment = true,
            )
        val decoded =
            PeerCapabilitiesWireCodec.decode(
                PeerCapabilitiesWireCodec.encode(input),
            )!!

        assertTrue(decoded.supportsHardwareReadoutTelemetry)
        assertFalse(decoded.supportsCalibrationUncertainty)
        assertTrue(decoded.supportsGlobalIdentityAssignment)
    }
    @Test
    fun currentCapabilitiesRoundTripHighEndDistributedFeatures() {
        val input =
            PeerCapabilitiesWireCodec.Capabilities(
                protocolVersion = 2,
                calibrationSchemaVersion = 2,
                multiActorPoseSchemaVersion = 3,
                supportsHardwareReadoutTelemetry = true,
                supportsCalibrationUncertainty = true,
                supportsGlobalIdentityAssignment = true,
                supportsDenseLandmarkFragmentation = true,
                supportsDenseLandmarkTimingMetadata = true,
                supportsMultiActorPoseFragmentation = true,
                supportsVisualTimingBeacon = true,
                maximumSubjects = 64,
                maximumVerifiedConcurrentCameras = 4,
                timingAuthorityScore = 0.97,
                timingResidualP95Milliseconds = 0.75,
            )

        val decoded =
            PeerCapabilitiesWireCodec.decode(
                PeerCapabilitiesWireCodec.encode(input),
            )

        assertEquals(input, decoded)
    }

    @Test
    fun versionOnePayloadStillDecodesWithSafeDefaults() {
        val legacy =
            PeerCapabilitiesWireCodec.encodeV1ForTesting(
                PeerCapabilitiesWireCodec.Capabilities(
                    protocolVersion = 2,
                    calibrationSchemaVersion = 1,
                    multiActorPoseSchemaVersion = 1,
                    supportsHardwareReadoutTelemetry = true,
                    supportsCalibrationUncertainty = false,
                    supportsGlobalIdentityAssignment = false,
                ),
            )

        val decoded =
            PeerCapabilitiesWireCodec.decode(legacy)

        assertTrue(decoded != null)
        assertTrue(
            decoded?.supportsDenseLandmarkFragmentation == false,
        )
        assertFalse(
            decoded?.supportsDenseLandmarkTimingMetadata ?: true,
        )
        assertTrue(
            decoded?.maximumSubjects ?: 0 >= 1,
        )
    }

    @Test
    fun depthFragmentationCapabilityRoundTripsWithoutSchemaExpansion() {
        val capabilities =
            PeerCapabilitiesWireCodec.Capabilities(
                supportsDepthFragmentation = true,
                maximumSubjects = 64,
                maximumVerifiedConcurrentCameras = 128,
                timingAuthorityScore = 0.95,
                timingResidualP95Milliseconds = 0.8,
            )

        val decoded =
            requireNotNull(
                PeerCapabilitiesWireCodec.decode(
                    PeerCapabilitiesWireCodec.encode(
                        capabilities,
                    ),
                ),
            )

        assertTrue(
            decoded.supportsDepthFragmentation,
        )
    }

}


class PeerCapabilitiesDenseTimingVersionContractTest {
    @Test
    fun fragmentCapableV2PeerDoesNotImplicitlyAdvertiseV4TimingMetadata() {
        val oldPeer =
            PeerCapabilitiesWireCodec.Capabilities(
                protocolVersion = 2,
                supportsDenseLandmarkFragmentation = true,
                supportsDenseLandmarkTimingMetadata = false,
            )
        val decoded =
            requireNotNull(
                PeerCapabilitiesWireCodec.decode(
                    PeerCapabilitiesWireCodec.encode(oldPeer),
                ),
            )

        assertTrue(decoded.supportsDenseLandmarkFragmentation)
        assertFalse(decoded.supportsDenseLandmarkTimingMetadata)
    }
}

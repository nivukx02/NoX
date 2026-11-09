package com.nivukx.nox.core.sync

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TimingAuthorityFusionEngineTest {

    @Test
    fun hardwareAndBeaconAgreementProducesReferenceGradeAuthority() {
        val result =
            TimingAuthorityFusionEngine().evaluate(
                TimingAuthorityFusionEngine.Input(
                    hardwareReadoutAvailable = true,
                    hardwareReadoutConfidence = 0.98,
                    hardwareReadoutP95JitterMillis = 0.7,
                    hardwareSynchronized = true,
                    logicalCalibratedSynchronization = true,
                    visualBeaconAvailable = true,
                    visualBeaconConfidence = 0.97,
                    visualBeaconResidualP95Millis = 0.6,
                    peerSynchronizationConfidence = 0.96,
                    peerResidualP95Millis = 0.8,
                    rollingShutterRisk = 0.08,
                ),
            )

        assertTrue(result.referenceGrade)
        assertTrue(result.confidence >= 0.95)
        assertEquals(
            TimingAuthorityFusionEngine.Source.COMPOSITE,
            result.source,
        )
    }

    @Test
    fun weakSoftwareTimingDoesNotMasqueradeAsHardwareAuthority() {
        val result =
            TimingAuthorityFusionEngine().evaluate(
                TimingAuthorityFusionEngine.Input(
                    peerSynchronizationConfidence = 0.82,
                    peerResidualP95Millis = 4.0,
                    peerSynchronizationEvidenceAvailable = true,
                    rollingShutterRisk = 0.20,
                ),
            )

        assertTrue(!result.hardwareGrade)
        assertTrue(!result.referenceGrade)
        assertTrue(
            result.source ==
                TimingAuthorityFusionEngine.Source.PEER_CLOCK,
        )
        assertTrue(result.warnings.isNotEmpty())
    }

    @Test
    fun rollingShutterRiskCapsAuthorityEvenWithStrongClockEvidence() {
        val result =
            TimingAuthorityFusionEngine().evaluate(
                TimingAuthorityFusionEngine.Input(
                    hardwareReadoutAvailable = true,
                    hardwareReadoutConfidence = 0.99,
                    hardwareReadoutP95JitterMillis = 0.5,
                    hardwareSynchronized = true,
                    visualBeaconAvailable = true,
                    visualBeaconConfidence = 0.99,
                    visualBeaconResidualP95Millis = 0.5,
                    peerSynchronizationConfidence = 0.99,
                    peerResidualP95Millis = 0.5,
                    rollingShutterRisk = 0.95,
                ),
            )

        assertTrue(result.confidence < 0.90)
        assertTrue(
            result.warnings.contains(
                "TIMING_AUTHORITY_ROLLING_SHUTTER_RISK_HIGH",
            ),
        )
    }

    @Test
    fun absentTimingEvidenceCannotProduceReferenceAuthority() {
        val result =
            TimingAuthorityFusionEngine().evaluate(
                TimingAuthorityFusionEngine.Input(
                    rollingShutterRisk = 0.0,
                ),
            )

        assertEquals(
            TimingAuthorityFusionEngine.Source.NONE,
            result.source,
        )
        assertTrue(result.confidence < 0.50)
        assertTrue(!result.referenceGrade)
        assertTrue(
            result.warnings.contains(
                "TIMING_AUTHORITY_NO_USABLE_SOURCE",
            ),
        )
    }
}

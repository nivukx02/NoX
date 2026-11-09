package com.nivukx.nox.core.ai

import org.junit.Assert.assertTrue
import org.junit.Test

class ProductionCaptureEvidenceLedgerTest {

    @Test
    fun stableEvidenceProducesStableTrend() {
        val ledger = ProductionCaptureEvidenceLedger()

        repeat(30) { index ->
            val snapshot = ledger.append(
                sample(
                    timestampNanos = 1_000_000_000L +
                        index * 16_666_667L,
                    integrity = 0.92,
                    kinematic = 0.94,
                    surface = 0.91,
                ),
            )

            if (index > 10) {
                assertTrue(
                    snapshot.trend ==
                        ProductionCaptureEvidenceLedger.Trend.STABLE ||
                        snapshot.trend ==
                        ProductionCaptureEvidenceLedger.Trend.IMPROVING,
                )
            }
        }
    }

    @Test
    fun sustainedDegradationBecomesVisibleBeforeHardFailure() {
        val ledger = ProductionCaptureEvidenceLedger()

        repeat(40) { index ->
            val degradation = index * 0.008
            ledger.append(
                sample(
                    timestampNanos = 2_000_000_000L +
                        index * 16_666_667L,
                    integrity =
                        (0.94 - degradation).coerceAtLeast(0.45),
                    kinematic =
                        (0.94 - degradation).coerceAtLeast(0.42),
                    surface =
                        (0.93 - degradation).coerceAtLeast(0.44),
                    outputAllowed = index < 28,
                ),
            )
        }

        val snapshot = ledger.latest()
            ?: error("ledger snapshot missing")

        assertTrue(
            snapshot.trend ==
                ProductionCaptureEvidenceLedger.Trend.DEGRADING ||
                snapshot.trend ==
                ProductionCaptureEvidenceLedger.Trend.CRITICAL,
        )
        assertTrue(
            snapshot.warnings.any {
                it.contains("degrading") ||
                    it.contains("negative-slope")
            },
        )
    }

    @Test
    fun timelineGapReducesContinuityWithoutInventingQuality() {
        val ledger = ProductionCaptureEvidenceLedger()

        repeat(12) { index ->
            ledger.append(
                sample(
                    timestampNanos = 3_000_000_000L +
                        index * 16_666_667L,
                    integrity = 0.90,
                ),
            )
        }

        val snapshot = ledger.append(
            sample(
                timestampNanos = 4_000_000_000L,
                integrity = 0.90,
            ),
        )

        assertTrue(snapshot.continuityScore < 1.0)
        assertTrue(snapshot.currentScore > 0.50)
    }

    @Test
    fun rejectedFaceEyeEvidenceCannotLookHealthyInRollingLedger() {
        val ledger = ProductionCaptureEvidenceLedger()

        repeat(20) { index ->
            ledger.append(
                sample(
                    timestampNanos =
                        6_000_000_000L +
                            index * 16_666_667L,
                    integrity = 0.93,
                    faceEyeAccepted = true,
                ),
            )
        }

        val snapshot =
            ledger.append(
                sample(
                    timestampNanos = 6_400_000_000L,
                    integrity = 0.93,
                    faceEyeAccepted = false,
                ),
            )

        assertTrue(snapshot.currentScore < 0.93)
        assertTrue(
            snapshot.warnings.any {
                it.contains("face-eye")
            },
        )
    }

    @Test
    fun faceEyeReliabilityCollapseBecomesDominantEvidenceTrend() {
        val ledger = ProductionCaptureEvidenceLedger()

        repeat(18) { index ->
            ledger.append(
                sample(
                    timestampNanos =
                        7_000_000_000L +
                            index * 16_666_667L,
                    integrity = 0.94,
                    faceEyeReliabilityTrend = 0.02,
                ),
            )
        }

        repeat(10) { index ->
            ledger.append(
                sample(
                    timestampNanos =
                        7_400_000_000L +
                            index * 16_666_667L,
                    integrity = 0.88,
                    faceEyeReliabilityTrend = -0.48,
                ),
            )
        }

        val snapshot =
            ledger.latest()
                ?: error("ledger snapshot missing")

        assertTrue(
            snapshot.components.any {
                it.name == "face-eye-trend" &&
                    it.direction ==
                        ProductionCaptureEvidenceLedger.Trend.DEGRADING
            },
        )
        assertTrue(
            snapshot.warnings.any {
                it.contains("face-eye") ||
                    it.contains("degrading")
            },
        )
    }

    @Test
    fun persistentFaceEyeCalibrationDriftBecomesRollingBottleneck() {
        val ledger = ProductionCaptureEvidenceLedger()

        repeat(18) { index ->
            ledger.append(
                sample(
                    timestampNanos =
                        8_000_000_000L +
                            index * 16_666_667L,
                    integrity = 0.94,
                    faceEyeCalibrationDriftScore = 0.04,
                    faceEyeCalibrationDriftPersistent = false,
                ),
            )
        }

        repeat(10) { index ->
            ledger.append(
                sample(
                    timestampNanos =
                        8_400_000_000L +
                            index * 16_666_667L,
                    integrity = 0.90,
                    faceEyeCalibrationDriftScore = 0.76,
                    faceEyeCalibrationDriftPersistent = true,
                ),
            )
        }

        val snapshot =
            ledger.latest()
                ?: error("ledger snapshot missing")

        assertTrue(
            snapshot.components.any {
                it.name == "face-eye-calibration" &&
                    it.direction ==
                        ProductionCaptureEvidenceLedger.Trend.DEGRADING
            },
        )
        assertTrue(
            snapshot.warnings.any {
                it.contains("face-eye-calibration-drift-persistent")
            },
        )
    }

    @Test
    fun metricAndEpipolarAuthorityBecomeVisibleInRollingSnapshot() {
        val ledger = ProductionCaptureEvidenceLedger()

        repeat(20) { index ->
            ledger.append(
                sample(
                    timestampNanos =
                        9_000_000_000L +
                            index * 16_666_667L,
                    integrity = 0.96,
                    metricAuthority = 0.91,
                    epipolarAuthority = 0.93,
                    motionInnovationScore = 0.90,
                ),
            )
        }

        val snapshot =
            ledger.append(
                sample(
                    timestampNanos = 9_400_000_000L,
                    integrity = 0.90,
                    metricAuthority = 0.35,
                    epipolarAuthority = 0.40,
                    motionInnovationScore = 0.38,
                ),
            )

        assertEquals(0.35, snapshot.metricAuthorityScore, 1e-9)
        assertEquals(0.40, snapshot.epipolarAuthorityScore, 1e-9)
        assertEquals(0.38, snapshot.motionInnovationScore, 1e-9)
        assertTrue(
            snapshot.components.any {
                it.name == "metric-authority"
            },
        )
        assertTrue(
            snapshot.components.any {
                it.name == "epipolar-authority"
            },
        )
        assertTrue(
            snapshot.warnings.contains(
                "metric-authority-below-trust-floor",
            ),
        )
        assertTrue(
            snapshot.warnings.contains(
                "epipolar-authority-below-trust-floor",
            ),
        )
        assertTrue(
            snapshot.warnings.contains(
                "motion-model-innovation-degraded",
            ),
        )
    }

    @Test
    fun hardFailureEscalatesSeverity() {
        val ledger = ProductionCaptureEvidenceLedger()

        repeat(20) { index ->
            ledger.append(
                sample(
                    timestampNanos = 5_000_000_000L +
                        index * 16_666_667L,
                    integrity = 0.90,
                    hardFailure = false,
                ),
            )
        }

        val snapshot = ledger.append(
            sample(
                timestampNanos = 5_400_000_000L,
                integrity = 0.78,
                outputAllowed = false,
                hardFailure = true,
            ),
        )

        assertTrue(
            snapshot.severity ==
                ProductionCaptureEvidenceLedger.Severity.CRITICAL,
        )
        assertTrue(snapshot.hardFailureRatio > 0.0)
    }

    private fun sample(
        timestampNanos: Long,
        integrity: Double,
        kinematic: Double = 0.90,
        surface: Double = 0.90,
        outputAllowed: Boolean = true,
        hardFailure: Boolean = false,
        faceEyeAccepted: Boolean = true,
        faceEyeReliabilityTrend: Double = 0.0,
        faceEyeCalibrationDriftScore: Double = 0.0,
        faceEyeCalibrationDriftPersistent: Boolean = false,
        metricAuthority: Double = 1.0,
        epipolarAuthority: Double = 1.0,
        motionInnovationScore: Double = 1.0,
    ) =
        ProductionCaptureEvidenceLedger.Sample(
            timestampNanos = timestampNanos,
            integrity = integrity,
            reconstruction = integrity,
            geometry = integrity,
            temporal = integrity,
            synchronization = integrity,
            calibration = integrity,
            bodyTracking = integrity,
            faceEye = integrity,
            faceEyeAccepted = faceEyeAccepted,
            faceEyeReliabilityTrend = faceEyeReliabilityTrend,
            faceEyeCalibrationDriftScore =
                faceEyeCalibrationDriftScore,
            faceEyeCalibrationDriftPersistent =
                faceEyeCalibrationDriftPersistent,
            hand = integrity,
            kinematic = kinematic,
            surface = surface,
            metricAuthority = metricAuthority,
            epipolarAuthority = epipolarAuthority,
            motionInnovationScore = motionInnovationScore,
            outputAllowed = outputAllowed,
            hardFailure = hardFailure,
        )
}

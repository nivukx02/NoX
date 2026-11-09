package com.nivukx.nox.core.sync

/**
 * Composite temporal-authority evaluator for high-end multi-camera capture.
 *
 * The engine intentionally separates timing evidence sources that have different failure modes:
 * hardware readout timing, calibrated logical-camera synchronization, visual timing beacons, and
 * peer-clock fitting. Related signals are corroborated rather than naively summed, preventing
 * correlated evidence from creating artificial confidence.
 */
class TimingAuthorityFusionEngine(
    private val config: Config = Config(),
) {

    enum class Source {
        COMPOSITE,
        HARDWARE_READOUT,
        CALIBRATED_LOGICAL_SYNC,
        VISUAL_BEACON,
        PEER_CLOCK,
        NONE,
    }

    data class Config(
        val minimumHardwareConfidence: Double = 0.90,
        val maximumHardwareP95JitterMillis: Double = 2.0,
        val minimumBeaconConfidence: Double = 0.90,
        val maximumBeaconResidualP95Millis: Double = 2.0,
        val minimumPeerConfidence: Double = 0.85,
        val maximumPeerResidualP95Millis: Double = 4.0,
        val hardwareWeight: Double = 0.42,
        val beaconWeight: Double = 0.32,
        val peerWeight: Double = 0.20,
        val logicalSyncWeight: Double = 0.18,
        val rollingShutterPenaltyMaximum: Double = 0.45,
        val hardwareGradeThreshold: Double = 0.90,
        val referenceGradeThreshold: Double = 0.95,
        val referenceResidualMillis: Double = 1.0,
    ) {
        init {
            listOf(
                minimumHardwareConfidence,
                minimumBeaconConfidence,
                minimumPeerConfidence,
                hardwareWeight,
                beaconWeight,
                peerWeight,
                logicalSyncWeight,
                rollingShutterPenaltyMaximum,
                hardwareGradeThreshold,
                referenceGradeThreshold,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            require(maximumHardwareP95JitterMillis > 0.0)
            require(maximumBeaconResidualP95Millis > 0.0)
            require(maximumPeerResidualP95Millis > 0.0)
            require(referenceResidualMillis > 0.0)
        }
    }

    data class Input(
        val hardwareReadoutAvailable: Boolean = false,
        val hardwareReadoutConfidence: Double = 0.0,
        val hardwareReadoutP95JitterMillis: Double = 0.0,
        val hardwareSynchronized: Boolean = false,
        val logicalCalibratedSynchronization: Boolean = false,
        val visualBeaconAvailable: Boolean = false,
        val visualBeaconConfidence: Double = 0.0,
        val visualBeaconResidualP95Millis: Double = 0.0,
        val peerSynchronizationConfidence: Double = 0.0,
        val peerResidualP95Millis: Double = 0.0,
        val peerSynchronizationEvidenceAvailable: Boolean = false,
        val rollingShutterRisk: Double = 0.0,
    ) {
        init {
            listOf(
                hardwareReadoutConfidence,
                visualBeaconConfidence,
                peerSynchronizationConfidence,
                rollingShutterRisk,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            listOf(
                hardwareReadoutP95JitterMillis,
                visualBeaconResidualP95Millis,
                peerResidualP95Millis,
            ).forEach {
                require(it.isFinite() && it >= 0.0)
            }
        }
    }

    data class Result(
        val source: Source,
        val confidence: Double,
        val residualP95Millis: Double,
        val sourceAgreement: Double,
        val hardwareScore: Double,
        val logicalSyncScore: Double,
        val beaconScore: Double,
        val peerScore: Double,
        val referenceGrade: Boolean,
        val hardwareGrade: Boolean,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    fun evaluate(input: Input): Result {
        val hardwareScore =
            if (!input.hardwareReadoutAvailable) {
                0.0
            } else {
                val jitterScore =
                    (
                        1.0 -
                            input.hardwareReadoutP95JitterMillis /
                                config.maximumHardwareP95JitterMillis
                    ).coerceIn(0.0, 1.0)
                (
                    input.hardwareReadoutConfidence *
                        0.72 +
                        jitterScore *
                        0.28
                ).coerceIn(0.0, 1.0)
            }

        val logicalScore =
            if (input.logicalCalibratedSynchronization) {
                1.0
            } else if (input.hardwareSynchronized) {
                0.86
            } else {
                0.0
            }

        val beaconScore =
            if (!input.visualBeaconAvailable) {
                0.0
            } else {
                val residualScore =
                    (
                        1.0 -
                            input.visualBeaconResidualP95Millis /
                                config.maximumBeaconResidualP95Millis
                    ).coerceIn(0.0, 1.0)
                (
                    input.visualBeaconConfidence *
                        0.76 +
                        residualScore *
                        0.24
                ).coerceIn(0.0, 1.0)
            }

        val peerScore =
            if (
                !input.peerSynchronizationEvidenceAvailable ||
                    input.peerSynchronizationConfidence <= 0.0
            ) {
                0.0
            } else {
                val residualScore =
                    (
                        1.0 -
                            input.peerResidualP95Millis /
                                config.maximumPeerResidualP95Millis
                    ).coerceIn(0.0, 1.0)
                (
                    input.peerSynchronizationConfidence *
                        0.78 +
                        residualScore *
                        0.22
                ).coerceIn(0.0, 1.0)
            }

        val activeScores =
            buildList {
                if (hardwareScore > 0.0) add(hardwareScore)
                if (logicalScore > 0.0) add(logicalScore)
                if (beaconScore > 0.0) add(beaconScore)
                if (peerScore > 0.0) add(peerScore)
            }

        val sourceAgreement =
            if (activeScores.size < 2) {
                1.0
            } else {
                val maximum = activeScores.maxOrNull() ?: 0.0
                val minimum = activeScores.minOrNull() ?: 0.0
                (
                    1.0 -
                        (maximum - minimum)
                ).coerceIn(0.0, 1.0)
            }

        /*
         * Correlated hardware signals are combined as a bounded composite. Visual beacon evidence
         * is treated as the strongest independent corroboration because it derives from the event
         * itself rather than the camera's own timestamp domain.
         */
        val weightedBase =
            (
                hardwareScore * config.hardwareWeight +
                    beaconScore * config.beaconWeight +
                    peerScore * config.peerWeight +
                    logicalScore * config.logicalSyncWeight
            ) /
                (
                    config.hardwareWeight +
                        config.beaconWeight +
                        config.peerWeight +
                        config.logicalSyncWeight
                ).coerceAtLeast(1e-9)

        val corroboration =
            if (
                hardwareScore >= config.minimumHardwareConfidence &&
                    beaconScore >= config.minimumBeaconConfidence
            ) {
                1.08
            } else if (
                hardwareScore >= config.minimumHardwareConfidence ||
                    beaconScore >= config.minimumBeaconConfidence
            ) {
                1.0
            } else {
                0.94
            }

        val rollingPenalty =
            (
                1.0 -
                    input.rollingShutterRisk *
                        config.rollingShutterPenaltyMaximum
            ).coerceIn(0.0, 1.0)

        val confidence =
            (
                weightedBase *
                    corroboration *
                    rollingPenalty *
                    (
                        0.82 +
                            0.18 * sourceAgreement
                    )
            ).coerceIn(0.0, 1.0)

        val activeResiduals =
            buildList {
                if (hardwareScore > 0.0) {
                    add(input.hardwareReadoutP95JitterMillis)
                }
                if (beaconScore > 0.0) {
                    add(input.visualBeaconResidualP95Millis)
                }
                if (peerScore > 0.0) {
                    add(input.peerResidualP95Millis)
                }
            }

        val residualP95 =
            activeResiduals
                .minOrNull()
                ?: maxOf(
                    config.maximumHardwareP95JitterMillis,
                    config.maximumBeaconResidualP95Millis,
                    config.maximumPeerResidualP95Millis,
                )

        val source =
            when {
                hardwareScore > 0.90 &&
                    beaconScore > 0.90 &&
                    peerScore > 0.80 ->
                    Source.COMPOSITE
                hardwareScore >= beaconScore &&
                    hardwareScore >= peerScore &&
                    hardwareScore > 0.0 ->
                    Source.HARDWARE_READOUT
                logicalScore >= beaconScore &&
                    logicalScore >= peerScore &&
                    logicalScore > 0.0 ->
                    Source.CALIBRATED_LOGICAL_SYNC
                beaconScore >= peerScore &&
                    beaconScore > 0.0 ->
                    Source.VISUAL_BEACON
                peerScore > 0.0 ->
                    Source.PEER_CLOCK
                else ->
                    Source.NONE
            }

        val referenceGrade =
            confidence >= config.referenceGradeThreshold &&
                residualP95 <= config.referenceResidualMillis &&
                source != Source.NONE &&
                sourceAgreement >= 0.80

        val hardwareGrade =
            confidence >= config.hardwareGradeThreshold &&
                (
                    (
                        hardwareScore >=
                            config.minimumHardwareConfidence &&
                            (
                                input.hardwareSynchronized ||
                                    input.logicalCalibratedSynchronization
                            )
                    ) ||
                        hardwareScore >=
                            config.minimumHardwareConfidence
                )

        val warnings =
            buildList {
                if (source == Source.NONE) {
                    add("TIMING_AUTHORITY_NO_USABLE_SOURCE")
                }
                if (
                    input.hardwareReadoutAvailable &&
                    hardwareScore < config.minimumHardwareConfidence
                ) {
                    add("TIMING_AUTHORITY_HARDWARE_READOUT_BELOW_TARGET")
                }
                if (
                    input.visualBeaconAvailable &&
                    beaconScore < config.minimumBeaconConfidence
                ) {
                    add("TIMING_AUTHORITY_VISUAL_BEACON_BELOW_TARGET")
                }
                if (
                    peerScore > 0.0 &&
                    peerScore < config.minimumPeerConfidence
                ) {
                    add("TIMING_AUTHORITY_PEER_CLOCK_BELOW_TARGET")
                }
                if (
                    sourceAgreement < 0.70 &&
                    activeScores.size >= 2
                ) {
                    add("TIMING_AUTHORITY_SOURCE_DISAGREEMENT")
                }
                if (
                    input.rollingShutterRisk >
                        config.rollingShutterPenaltyMaximum
                ) {
                    add(
                        "TIMING_AUTHORITY_ROLLING_SHUTTER_RISK_HIGH",
                    )
                }
                if (!referenceGrade) {
                    add(
                        "TIMING_AUTHORITY_REFERENCE_QUALIFICATION_NOT_MET",
                    )
                }
            }.distinct()

        val recommendations =
            buildList {
                if (!input.hardwareReadoutAvailable) {
                    add(
                        "Prefer cameras exposing hardware readout timestamps for high-speed reference capture.",
                    )
                }
                if (!input.visualBeaconAvailable) {
                    add(
                        "Run a visual timing-beacon calibration sequence when heterogeneous device clocks are used.",
                    )
                }
                if (
                    input.rollingShutterRisk >
                        config.rollingShutterPenaltyMaximum
                ) {
                    add(
                        "Increase exposure-speed headroom or use rolling-shutter-aware sub-frame warping before precision fusion.",
                    )
                }
            }.distinct()

        return Result(
            source = source,
            confidence = confidence,
            residualP95Millis = residualP95,
            sourceAgreement = sourceAgreement,
            hardwareScore = hardwareScore,
            logicalSyncScore = logicalScore,
            beaconScore = beaconScore,
            peerScore = peerScore,
            referenceGrade = referenceGrade,
            hardwareGrade = hardwareGrade,
            warnings = warnings,
            recommendations = recommendations,
        )
    }
}

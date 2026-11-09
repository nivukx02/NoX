package com.nivukx.nox.core.reconstruction

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Conservative evidence authority for high-end metric motion capture.
 *
 * This class does not estimate physical accuracy from source-code sophistication. It combines
 * independent geometry, timing, identity, temporal and runtime evidence and keeps empirical
 * reference qualification separate from model-derived readiness.
 */
class PrecisionEvidenceFieldEngine(
    private val config: Config = Config(),
) {
    enum class Band {
        BLOCKED,
        CONDITIONAL,
        OPERATIONAL,
        REFERENCE_CANDIDATE,
        EMPIRICALLY_QUALIFIED,
    }

    enum class Bottleneck {
        NONE,
        CALIBRATION,
        SYNCHRONIZATION,
        GEOMETRY,
        UNCERTAINTY,
        IDENTITY,
        TEMPORAL,
        THERMAL,
        COMPUTE,
        COVERAGE,
        REFERENCE_EVIDENCE,
    }

    data class Config(
        val requestedP95Millimeters: Double = 1.0,
        val requestedP99Millimeters: Double = 2.0,
        val minimumMetricCameraCount: Int = 2,
        val minimumCalibrationConfidence: Double = 0.92,
        val minimumSynchronizationConfidence: Double = 0.92,
        val maximumSynchronizationJitterMillis: Double = 1.5,
        val maximumReprojectionErrorPixels: Double = 2.5,
        val minimumCrossViewEvidence: Double = 0.72,
        val minimumViewDiversity: Double = 0.62,
        val minimumGeometricObservability: Double = 0.72,
        val minimumOperationalScore: Double = 0.72,
        val minimumReferenceCandidateScore: Double = 0.90,
        val minimumEmpiricalScore: Double = 0.94,
        val minimumReferenceSamples: Int = 250,
        val maximumReferenceP95Millimeters: Double = 1.0,
        val maximumReferenceP99Millimeters: Double = 2.0,
        val minimumReferenceAgreement: Double = 0.90,
    ) {
        init {
            require(requestedP95Millimeters > 0.0)
            require(requestedP99Millimeters >= requestedP95Millimeters)
            require(minimumMetricCameraCount >= 1)
            require(minimumCalibrationConfidence in 0.0..1.0)
            require(minimumSynchronizationConfidence in 0.0..1.0)
            require(maximumSynchronizationJitterMillis > 0.0)
            require(maximumReprojectionErrorPixels > 0.0)
            require(minimumCrossViewEvidence in 0.0..1.0)
            require(minimumViewDiversity in 0.0..1.0)
            require(minimumGeometricObservability in 0.0..1.0)
            require(minimumOperationalScore in 0.0..1.0)
            require(minimumReferenceCandidateScore in 0.0..1.0)
            require(minimumEmpiricalScore in 0.0..1.0)
            require(minimumReferenceSamples >= 1)
            require(maximumReferenceP95Millimeters > 0.0)
            require(maximumReferenceP99Millimeters >= maximumReferenceP95Millimeters)
            require(minimumReferenceAgreement in 0.0..1.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val cameraCount: Int,
        val calibrationConfidence: Double,
        val reprojectionP95Pixels: Double,
        val synchronizationConfidence: Double,
        val synchronizationJitterMillis: Double,
        val crossViewEvidence: Double,
        val viewDiversity: Double,
        val geometricObservability: Double,
        val geometricConditionScore: Double,
        val directionalMajor95Meters: Double,
        val directionalDepth95Meters: Double,
        val reconstructionUncertaintyP95Meters: Double,
        val denseSurfaceUncertaintyP95Meters: Double,
        val temporalStability: Double,
        val worldConsistency: Double,
        val identityContinuity: Double,
        val identitySwapRisk: Double,
        val bodyConfidence: Double,
        val faceEyeConfidence: Double,
        val handConfidence: Double,
        val thermalHeadroom: Double,
        val computeHeadroom: Double,
        val predictedObservationRatio: Double,
        val empiricalReferenceAvailable: Boolean = false,
        val empiricalP95Millimeters: Double? = null,
        val empiricalP99Millimeters: Double? = null,
        val empiricalReferenceAgreement: Double = 0.0,
        val empiricalReferenceSamples: Int = 0,
        val matchedReferenceSystem: Boolean = false,
        val referenceUncertaintyMillimeters: Double? = null,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(cameraCount >= 0)
            require(calibrationConfidence in 0.0..1.0)
            require(reprojectionP95Pixels >= 0.0 || reprojectionP95Pixels == Double.POSITIVE_INFINITY)
            require(synchronizationConfidence in 0.0..1.0)
            require(synchronizationJitterMillis >= 0.0 || synchronizationJitterMillis == Double.POSITIVE_INFINITY)
            require(crossViewEvidence in 0.0..1.0)
            require(viewDiversity in 0.0..1.0)
            require(geometricObservability in 0.0..1.0)
            require(geometricConditionScore in 0.0..1.0)
            require(directionalMajor95Meters >= 0.0 || directionalMajor95Meters == Double.POSITIVE_INFINITY)
            require(directionalDepth95Meters >= 0.0 || directionalDepth95Meters == Double.POSITIVE_INFINITY)
            require(reconstructionUncertaintyP95Meters >= 0.0 || reconstructionUncertaintyP95Meters == Double.POSITIVE_INFINITY)
            require(denseSurfaceUncertaintyP95Meters >= 0.0 || denseSurfaceUncertaintyP95Meters == Double.POSITIVE_INFINITY)
            require(temporalStability in 0.0..1.0)
            require(worldConsistency in 0.0..1.0)
            require(identityContinuity in 0.0..1.0)
            require(identitySwapRisk in 0.0..1.0)
            require(bodyConfidence in 0.0..1.0)
            require(faceEyeConfidence in 0.0..1.0)
            require(handConfidence in 0.0..1.0)
            require(thermalHeadroom in 0.0..1.0)
            require(computeHeadroom in 0.0..1.0)
            require(predictedObservationRatio in 0.0..1.0)
            require(empiricalP95Millimeters == null || empiricalP95Millimeters >= 0.0)
            require(empiricalP99Millimeters == null || empiricalP99Millimeters >= 0.0)
            require(empiricalReferenceAgreement in 0.0..1.0)
            require(empiricalReferenceSamples >= 0)
            require(referenceUncertaintyMillimeters == null || referenceUncertaintyMillimeters >= 0.0)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val band: Band,
        val score: Double,
        val operationalScore: Double,
        val modeledP95Millimeters: Double,
        val modeledP99Millimeters: Double,
        val geometricUncertaintyMillimeters: Double,
        val temporalPenaltyMillimeters: Double,
        val timingPenaltyMillimeters: Double,
        val evidenceIndependenceScore: Double,
        val empiricalReady: Boolean,
        val empiricallyQualified: Boolean,
        val bottleneck: Bottleneck,
        val warnings: List<String>,
        val actions: List<String>,
    )

    fun evaluate(input: Input): Result {
        val calibration =
            ratio(
                input.calibrationConfidence,
                config.minimumCalibrationConfidence,
            )

        val synchronization =
            min(
                ratio(
                    input.synchronizationConfidence,
                    config.minimumSynchronizationConfidence,
                ),
                lowerIsBetter(
                    input.synchronizationJitterMillis,
                    config.maximumSynchronizationJitterMillis,
                ),
            )

        val geometry =
            listOf(
                input.crossViewEvidence,
                input.viewDiversity,
                input.geometricObservability,
                input.geometricConditionScore,
            ).minOrNull() ?: 0.0

        val identity =
            min(
                input.identityContinuity,
                1.0 - input.identitySwapRisk,
            )

        val temporal =
            min(
                input.temporalStability,
                input.worldConsistency,
            )

        val modality =
            listOf(
                input.bodyConfidence,
                max(0.45, input.faceEyeConfidence),
                max(0.45, input.handConfidence),
            ).minOrNull() ?: 0.0

        val runtime =
            min(
                input.thermalHeadroom,
                input.computeHeadroom,
            )

        val coverage =
            if (input.cameraCount >= config.minimumMetricCameraCount) {
                min(
                    1.0,
                    0.50 +
                        0.30 * input.crossViewEvidence +
                        0.20 * input.viewDiversity,
                )
            } else {
                0.25
            }

        val independence =
            (
                0.42 * (input.cameraCount / 6.0).coerceIn(0.0, 1.0) +
                    0.34 * input.crossViewEvidence +
                    0.24 * (1.0 - input.predictedObservationRatio)
                ).coerceIn(0.0, 1.0)

        val geometricUncertaintyMeters =
            listOf(
                input.reconstructionUncertaintyP95Meters,
                input.denseSurfaceUncertaintyP95Meters,
                input.directionalMajor95Meters,
                input.directionalDepth95Meters,
            ).filter(Double::isFinite)
                .ifEmpty { listOf(Double.POSITIVE_INFINITY) }
                .maxOrNull()
                ?: Double.POSITIVE_INFINITY

        val geometricUncertaintyMillimeters =
            geometricUncertaintyMeters * 1000.0

        val jitterFactor =
            if (input.synchronizationJitterMillis.isFinite()) {
                (input.synchronizationJitterMillis / 2.0)
                    .coerceIn(0.0, 1.0)
            } else {
                1.0
            }

        val timingPenaltyMillimeters =
            (
                0.15 +
                    1.65 * (1.0 - input.synchronizationConfidence) +
                    0.90 * jitterFactor
                ) * (
                    1.0 +
                        0.85 * (1.0 - input.crossViewEvidence) +
                        0.50 * (1.0 - input.viewDiversity)
                    )

        val temporalPenaltyMillimeters =
            0.15 +
                1.80 * (1.0 - temporal) +
                1.10 * input.predictedObservationRatio

        val modeledP95 =
            (
                geometricUncertaintyMillimeters +
                    timingPenaltyMillimeters +
                    temporalPenaltyMillimeters
                ).coerceAtLeast(0.0)

        val reprojectionPenalty =
            if (input.reprojectionP95Pixels.isFinite()) {
                max(0.0, input.reprojectionP95Pixels) * 0.05
            } else {
                5.0
            }

        val modeledP99 =
            (
                modeledP95 * 1.65 +
                    reprojectionPenalty
                ).coerceAtLeast(modeledP95)

        val operationalScore =
            (
                0.18 * calibration +
                    0.18 * synchronization +
                    0.18 * geometry +
                    0.12 * identity +
                    0.12 * temporal +
                    0.08 * modality +
                    0.08 * coverage +
                    0.06 * runtime
                ).coerceIn(0.0, 1.0)

        val criticalEvidence =
            listOf(
                calibration,
                synchronization,
                geometry,
                identity,
                temporal,
                coverage,
                independence,
                runtime,
            ).sorted().getOrNull(1)
                ?: 0.0

        val score =
            (
                0.58 * operationalScore +
                    0.24 * criticalEvidence +
                    0.18 * independence
                ).coerceIn(0.0, 1.0)

        val empirical =
            evaluateEmpirical(
                input = input,
                modeledP95 = modeledP95,
                modeledP99 = modeledP99,
                score = score,
            )

        val bottleneck =
            findBottleneck(
                input = input,
                modeledP95 = modeledP95,
                modeledP99 = modeledP99,
                calibration = calibration,
                synchronization = synchronization,
                geometry = geometry,
                identity = identity,
                temporal = temporal,
                runtime = runtime,
                empiricalReady = empirical.first,
            )

        val hardBlock =
            input.cameraCount < 1 ||
                calibration < 0.35 ||
                synchronization < 0.35 ||
                geometry < 0.28 ||
                runtime < 0.16 ||
                modeledP95 >
                    config.requestedP95Millimeters * 4.0

        val band =
            when {
                empirical.second ->
                    Band.EMPIRICALLY_QUALIFIED
                hardBlock ->
                    Band.BLOCKED
                score >= config.minimumReferenceCandidateScore &&
                    modeledP95 <= config.requestedP95Millimeters &&
                    modeledP99 <= config.requestedP99Millimeters ->
                    Band.REFERENCE_CANDIDATE
                score >= config.minimumOperationalScore ->
                    Band.OPERATIONAL
                else ->
                    Band.CONDITIONAL
            }

        val warnings =
            buildList {
                if (input.cameraCount < config.minimumMetricCameraCount) {
                    add("metric-geometry-needs-multiple-calibrated-cameras")
                }
                if (calibration < config.minimumCalibrationConfidence) {
                    add("calibration-below-gate")
                }
                if (synchronization < config.minimumSynchronizationConfidence) {
                    add("synchronization-below-gate")
                }
                if (geometry < config.minimumCrossViewEvidence) {
                    add("cross-view-geometry-weak")
                }
                if (input.viewDiversity < config.minimumViewDiversity) {
                    add("view-diversity-weak")
                }
                if (identity < 0.70) {
                    add("identity-continuity-degraded")
                }
                if (temporal < 0.70) {
                    add("temporal-consistency-degraded")
                }
                if (input.predictedObservationRatio > 0.20) {
                    add("prediction-dependence-elevated")
                }
                if (modeledP95 > config.requestedP95Millimeters) {
                    add("modeled-p95-above-requested-target")
                }
                if (modeledP99 > config.requestedP99Millimeters) {
                    add("modeled-p99-above-requested-target")
                }
                if (!empirical.first) {
                    add("empirical-reference-qualification-pending")
                }
                if (
                    input.empiricalReferenceAvailable &&
                    !input.matchedReferenceSystem
                ) {
                    add("reference-is-not-matched-to-independent-system")
                }
            }.distinct()

        val actions =
            buildList {
                when (bottleneck) {
                    Bottleneck.CALIBRATION ->
                        add("recalibrate-intrinsics-extrinsics")
                    Bottleneck.SYNCHRONIZATION ->
                        add("improve-clock-fit-and-exposure-window-alignment")
                    Bottleneck.GEOMETRY,
                    Bottleneck.COVERAGE ->
                        add("increase-baseline-and-view-angle-diversity")
                    Bottleneck.UNCERTAINTY ->
                        add("increase-observation-quality-and-cross-view-support")
                    Bottleneck.IDENTITY ->
                        add("retain-appearance-and-pose-history-for-reassociation")
                    Bottleneck.TEMPORAL ->
                        add("reduce-prediction-reliance-and-increase-temporal-evidence")
                    Bottleneck.THERMAL ->
                        add("shed-optional-work-before-core-tracking")
                    Bottleneck.COMPUTE ->
                        add("preserve-core-motion-and-throttle-dense-detail")
                    Bottleneck.REFERENCE_EVIDENCE ->
                        add("run-matched-reference-metrology-campaign")
                    Bottleneck.NONE ->
                        Unit
                }
            }.distinct()

        return Result(
            timestampNanos = input.timestampNanos,
            band = band,
            score = score,
            operationalScore = operationalScore,
            modeledP95Millimeters = modeledP95,
            modeledP99Millimeters = modeledP99,
            geometricUncertaintyMillimeters = geometricUncertaintyMillimeters,
            temporalPenaltyMillimeters = temporalPenaltyMillimeters,
            timingPenaltyMillimeters = timingPenaltyMillimeters,
            evidenceIndependenceScore = independence,
            empiricalReady = empirical.first,
            empiricallyQualified = empirical.second,
            bottleneck = bottleneck,
            warnings = warnings,
            actions = actions,
        )
    }

    fun reset() = Unit

    private fun evaluateEmpirical(
        input: Input,
        modeledP95: Double,
        modeledP99: Double,
        score: Double,
    ): Pair<Boolean, Boolean> {
        val p95 =
            input.empiricalP95Millimeters
                ?: return false to false
        val p99 =
            input.empiricalP99Millimeters
                ?: return false to false
        val referenceUncertainty =
            input.referenceUncertaintyMillimeters
                ?: return false to false

        val ready =
            input.empiricalReferenceAvailable &&
                input.matchedReferenceSystem &&
                input.empiricalReferenceSamples >=
                    config.minimumReferenceSamples &&
                input.empiricalReferenceAgreement >=
                    config.minimumReferenceAgreement &&
                score >= config.minimumEmpiricalScore &&
                modeledP95.isFinite() &&
                modeledP99.isFinite()

        if (!ready) return false to false

        val adjustedP95 =
            sqrt(
                p95 * p95 +
                    referenceUncertainty * referenceUncertainty,
            )

        val adjustedP99 =
            sqrt(
                p99 * p99 +
                    referenceUncertainty * referenceUncertainty,
            )

        return true to (
            adjustedP95 <= config.maximumReferenceP95Millimeters &&
                adjustedP99 <= config.maximumReferenceP99Millimeters
            )
    }

    private fun ratio(
        value: Double,
        threshold: Double,
    ): Double {
        if (!value.isFinite()) return 0.0
        return (value / threshold).coerceIn(0.0, 1.0)
    }

    private fun lowerIsBetter(
        value: Double,
        threshold: Double,
    ): Double {
        if (!value.isFinite()) return 0.0
        if (value <= threshold) return 1.0
        return (threshold / value).coerceIn(0.0, 1.0)
    }

    private fun findBottleneck(
        input: Input,
        modeledP95: Double,
        modeledP99: Double,
        calibration: Double,
        synchronization: Double,
        geometry: Double,
        identity: Double,
        temporal: Double,
        runtime: Double,
        empiricalReady: Boolean,
    ): Bottleneck {
        val uncertainty =
            if (modeledP95.isFinite()) {
                (
                    1.0 -
                        modeledP95 /
                            (config.maximumModeledUncertaintyForBottleneck())
                    ).coerceIn(0.0, 1.0)
            } else {
                0.0
            }

        val candidates =
            listOf(
                Bottleneck.CALIBRATION to calibration,
                Bottleneck.SYNCHRONIZATION to synchronization,
                Bottleneck.GEOMETRY to geometry,
                Bottleneck.IDENTITY to identity,
                Bottleneck.TEMPORAL to temporal,
                Bottleneck.THERMAL to input.thermalHeadroom,
                Bottleneck.COMPUTE to runtime,
                Bottleneck.UNCERTAINTY to uncertainty,
                Bottleneck.COVERAGE to input.viewDiversity,
            )

        val minimum =
            candidates.minByOrNull { it.second }
                ?: return Bottleneck.NONE

        if (minimum.second < 0.55) return minimum.first

        if (
            !empiricalReady &&
            modeledP95 <= config.requestedP95Millimeters &&
            modeledP99 <= config.requestedP99Millimeters
        ) {
            return Bottleneck.REFERENCE_EVIDENCE
        }

        return Bottleneck.NONE
    }

    private fun Config.maximumModeledUncertaintyForBottleneck(): Double =
        maximum(
            4.0 * maximumModeledP95Millimeters(),
            requestedP95Millimeters * 4.0,
        )

    private fun Config.maximumModeledP95Millimeters(): Double =
        requestedP95Millimeters.coerceAtLeast(0.25)

    private fun maximum(a: Double, b: Double): Double =
        max(a, b)
}

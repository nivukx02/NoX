package com.nivukx.nox.core.reconstruction

import kotlin.math.max
import kotlin.math.min

/**
 * Production contract for dense markerless landmark evidence.
 *
 * The contract is intentionally separate from the detector. It answers a different question:
 * "Is the current dense-landmark evidence strong enough to authorize downstream geometry?"
 *
 * It never converts model confidence into physical millimeter accuracy. The contract only gates
 * evidence quality using observable quantities and explicit operator-selected thresholds.
 */
class DenseLandmarkPrecisionContractEngine(
    config: Config = Config(),
) {
    @Volatile
    private var runtimeConfig: Config = config

    @Synchronized
    fun configure(next: Config) {
        runtimeConfig = next
    }

    fun currentConfig(): Config = runtimeConfig

    data class Config(
        val minimumCoverage: Double = 0.70,
        val minimumVisibleCoverage: Double = 0.58,
        val maximumUncertaintyP95: Double = 0.30,
        val minimumCrossViewAgreement: Double = 0.60,
        val minimumTemporalStability: Double = 0.66,
        val maximumRejectedObservationRatio: Double = 0.10,
        val maximumPredictionBridgeRatio: Double = 0.35,
        val minimumContactCoverageWhenActive: Double = 0.45,
        val minimumEvidenceScore: Double = 0.68,
        val minimumAuthoritativeScore: Double = 0.76,
        val hardUncertaintyP95: Double = 0.80,
        val hardCoverageFloor: Double = 0.20,
        val hardVisibleCoverageFloor: Double = 0.16,
        val hardCrossViewAgreement: Double = 0.18,
        val hardTemporalStability: Double = 0.16,
        val hardRejectedObservationRatio: Double = 0.60,
        val hardPredictionBridgeRatio: Double = 0.92,
        val minimumCrossViewCameras: Int = 2,
        val fullEvidenceSampleCount: Int = 256,
    ) {
        init {
            require(minimumCoverage in 0.0..1.0)
            require(minimumVisibleCoverage in 0.0..1.0)
            require(maximumUncertaintyP95 in 0.0..1.0)
            require(minimumCrossViewAgreement in 0.0..1.0)
            require(minimumTemporalStability in 0.0..1.0)
            require(maximumRejectedObservationRatio in 0.0..1.0)
            require(maximumPredictionBridgeRatio in 0.0..1.0)
            require(minimumContactCoverageWhenActive in 0.0..1.0)
            require(minimumEvidenceScore in 0.0..1.0)
            require(minimumAuthoritativeScore >= minimumEvidenceScore)
            require(minimumAuthoritativeScore <= 1.0)
            require(hardUncertaintyP95 >= maximumUncertaintyP95)
            require(hardCoverageFloor <= minimumCoverage)
            require(hardVisibleCoverageFloor <= minimumVisibleCoverage)
            require(hardCrossViewAgreement <= minimumCrossViewAgreement)
            require(hardTemporalStability <= minimumTemporalStability)
            require(hardRejectedObservationRatio >= maximumRejectedObservationRatio)
            require(hardPredictionBridgeRatio >= maximumPredictionBridgeRatio)
            require(minimumCrossViewCameras >= 1)
            require(fullEvidenceSampleCount >= 4)
        }
    }

    data class Input(
        val coverage: Double,
        val visibleCoverage: Double,
        val uncertaintyP95: Double,
        val crossViewAgreement: Double,
        val crossViewEvidenceAvailable: Boolean,
        val supportingCameraCount: Int,
        val temporalStability: Double,
        val contactCoverage: Double,
        val contactEvidenceActive: Boolean,
        val predictionBridgeRatio: Double,
        val acceptedObservations: Long,
        val rejectedObservations: Long,
        val staleObservations: Long,
        val nonFiniteObservations: Long,
        val hardFailure: Boolean,
    ) {
        init {
            require(coverage in 0.0..1.0)
            require(visibleCoverage in 0.0..1.0)
            require(uncertaintyP95 in 0.0..1.0)
            require(crossViewAgreement in 0.0..1.0)
            require(supportingCameraCount >= 0)
            require(temporalStability in 0.0..1.0)
            require(contactCoverage in 0.0..1.0)
            require(predictionBridgeRatio in 0.0..1.0)
            require(acceptedObservations >= 0L)
            require(rejectedObservations >= 0L)
            require(staleObservations >= 0L)
            require(nonFiniteObservations >= 0L)
        }
    }

    enum class State {
        UNKNOWN,
        READY,
        DEGRADED,
        HOLD,
        BLOCKED,
    }

    enum class FailureClass {
        NONE,
        COVERAGE,
        VISIBILITY,
        UNCERTAINTY,
        CROSS_VIEW,
        TEMPORAL,
        REJECTION,
        PREDICTION,
        CONTACT,
        UPSTREAM_HARD_FAILURE,
        INSUFFICIENT_EVIDENCE,
    }

    data class Result(
        val state: State,
        val score: Double,
        val evidenceScore: Double,
        val authoritative: Boolean,
        val accepted: Boolean,
        val hardFailure: Boolean,
        val failureClass: FailureClass,
        val coverageScore: Double,
        val visibilityScore: Double,
        val uncertaintyScore: Double,
        val crossViewScore: Double,
        val temporalScore: Double,
        val rejectionScore: Double,
        val predictionScore: Double,
        val contactScore: Double,
        val effectiveCameraRedundancy: Double,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val blocked: Boolean
            get() = state == State.BLOCKED
    }

    fun evaluate(input: Input): Result {
        val config = runtimeConfig

        val acceptedViews =
            input.acceptedObservations.toDouble().coerceAtLeast(0.0)
        val rejectedViews =
            input.rejectedObservations.toDouble().coerceAtLeast(0.0)
        val evidenceTotal =
            (acceptedViews + rejectedViews).coerceAtLeast(1.0)

        val rejectedRatio =
            (rejectedViews / evidenceTotal).coerceIn(0.0, 1.0)

        val coverageScore =
            lowerBoundScore(
                measured = input.coverage,
                minimum = runtimeConfig.minimumCoverage,
            )

        val visibilityScore =
            lowerBoundScore(
                measured = input.visibleCoverage,
                minimum = runtimeConfig.minimumVisibleCoverage,
            )

        val uncertaintyScore =
            upperBoundScore(
                measured = input.uncertaintyP95,
                maximum = runtimeConfig.maximumUncertaintyP95,
            )

        val crossViewScore =
            if (!input.crossViewEvidenceAvailable) {
                if (runtimeConfig.minimumCrossViewCameras <= 1) 1.0 else 0.54
            } else {
                lowerBoundScore(
                    measured = input.crossViewAgreement,
                    minimum = runtimeConfig.minimumCrossViewAgreement,
                ) *
                    redundancyScore(
                        supportingCameraCount =
                            input.supportingCameraCount,
                    )
            }

        val temporalScore =
            lowerBoundScore(
                measured = input.temporalStability,
                minimum = runtimeConfig.minimumTemporalStability,
            )

        val rejectionScore =
            upperBoundScore(
                measured = rejectedRatio,
                maximum = runtimeConfig.maximumRejectedObservationRatio,
                invert = true,
            )

        val predictionScore =
            upperBoundScore(
                measured = input.predictionBridgeRatio,
                maximum = runtimeConfig.maximumPredictionBridgeRatio,
                invert = true,
            )

        val contactScore =
            if (!input.contactEvidenceActive) {
                1.0
            } else {
                lowerBoundScore(
                    measured = input.contactCoverage,
                    minimum = runtimeConfig.minimumContactCoverageWhenActive,
                )
            }

        val effectiveCameraRedundancy =
            redundancyScore(
                supportingCameraCount = input.supportingCameraCount,
            )

        val weightedEvidence =
            geometricMean(
                listOf(
                    coverageScore to 0.19,
                    visibilityScore to 0.13,
                    uncertaintyScore to 0.18,
                    crossViewScore to 0.18,
                    temporalScore to 0.13,
                    rejectionScore to 0.08,
                    predictionScore to 0.07,
                    contactScore to 0.04,
                ),
            )

        val evidenceSupport =
            (
                min(1.0, acceptedViews / 32.0) * 0.22 +
                    (1.0 - min(1.0, input.staleObservations / 32.0)) * 0.08 +
                    (1.0 - min(1.0, input.nonFiniteObservations / 8.0)) * 0.08 +
                    effectiveCameraRedundancy * 0.30 +
                    weightedEvidence * 0.32
                ).coerceIn(0.0, 1.0)

        val score =
            (
                weightedEvidence * 0.74 +
                    evidenceSupport * 0.26
                ).coerceIn(0.0, 1.0)

        val hardCoverage =
            input.coverage < runtimeConfig.hardCoverageFloor ||
                input.visibleCoverage < runtimeConfig.hardVisibleCoverageFloor

        val hardUncertainty =
            input.uncertaintyP95 >= runtimeConfig.hardUncertaintyP95

        val hardCrossView =
            input.crossViewEvidenceAvailable &&
                input.crossViewAgreement < runtimeConfig.hardCrossViewAgreement &&
                input.supportingCameraCount >= runtimeConfig.minimumCrossViewCameras

        val hardTemporal =
            input.temporalStability < runtimeConfig.hardTemporalStability

        val hardRejection =
            rejectedRatio >= runtimeConfig.hardRejectedObservationRatio

        val hardPrediction =
            input.predictionBridgeRatio >= runtimeConfig.hardPredictionBridgeRatio

        val hardFailure =
            input.hardFailure ||
                input.nonFiniteObservations > 8L ||
                hardCoverage ||
                hardUncertainty ||
                hardCrossView ||
                hardTemporal ||
                hardRejection ||
                hardPrediction

        val failure =
            dominantFailure(
                input = input,
                rejectedRatio = rejectedRatio,
            )

        val warnings = buildList {
            if (coverageScore < runtimeConfig.minimumCoverage) {
                add("Dense landmark coverage is below the production target.")
            }
            if (visibilityScore < runtimeConfig.minimumVisibleCoverage) {
                add("Visible dense landmark coverage is limiting geometry.")
            }
            if (uncertaintyScore < 1.0) {
                add("Dense landmark uncertainty P95 is elevated.")
            }
            if (
                input.crossViewEvidenceAvailable &&
                crossViewScore <
                    runtimeConfig.minimumCrossViewAgreement
            ) {
                add("Cross-view landmark agreement is limiting triangulation.")
            } else if (!input.crossViewEvidenceAvailable) {
                add("Cross-view dense evidence is unavailable for this frame.")
            }
            if (temporalScore < runtimeConfig.minimumTemporalStability) {
                add("Dense landmark temporal stability is degraded.")
            }
            if (rejectedRatio > runtimeConfig.maximumRejectedObservationRatio) {
                add("Dense landmark rejection ratio is above the configured contract.")
            }
            if (
                input.predictionBridgeRatio >
                    runtimeConfig.maximumPredictionBridgeRatio
            ) {
                add("Temporal prediction is carrying too much dense evidence.")
            }
            if (
                input.contactEvidenceActive &&
                contactScore < 0.70
            ) {
                add("Contact-aware dense evidence is not sufficiently covered.")
            }
            if (input.staleObservations > 0L) {
                add("Stale dense observations were discarded.")
            }
            if (input.nonFiniteObservations > 0L) {
                add("Non-finite dense observations were discarded.")
            }
            if (input.supportingCameraCount < runtimeConfig.minimumCrossViewCameras) {
                add("Camera redundancy is below the dense reconstruction target.")
            }
        }

        val recommendations = buildRecommendations(
            input = input,
            failureClass = failure,
        )

        val sufficientEvidence =
            acceptedViews >= 2.0 &&
                coverageScore > 0.25 &&
                visibilityScore > 0.25 &&
                uncertaintyScore > 0.25

        val evidenceObserved =
            input.supportingCameraCount > 0 ||
                input.coverage > 0.0 ||
                input.visibleCoverage > 0.0 ||
                input.crossViewEvidenceAvailable

        val crossViewBelowMinimum =
            input.crossViewEvidenceAvailable &&
                input.supportingCameraCount >= runtimeConfig.minimumCrossViewCameras &&
                input.crossViewAgreement < runtimeConfig.minimumCrossViewAgreement

        val state = when {
            hardFailure -> State.BLOCKED
            !evidenceObserved -> State.UNKNOWN
            !sufficientEvidence -> State.HOLD
            crossViewBelowMinimum -> State.DEGRADED
            score >= runtimeConfig.minimumAuthoritativeScore -> State.READY
            score >= runtimeConfig.minimumEvidenceScore -> State.DEGRADED
            else -> State.HOLD
        }

        val authoritativeEvidence =
            (runtimeConfig.minimumCrossViewCameras <= 1 ||
                (
                    input.crossViewEvidenceAvailable &&
                        input.crossViewAgreement >=
                            runtimeConfig.minimumCrossViewAgreement &&
                        input.supportingCameraCount >=
                            runtimeConfig.minimumCrossViewCameras
                )) &&
                input.coverage >= runtimeConfig.minimumCoverage &&
                input.visibleCoverage >= runtimeConfig.minimumVisibleCoverage &&
                input.uncertaintyP95 <= runtimeConfig.maximumUncertaintyP95 &&
                input.temporalStability >= runtimeConfig.minimumTemporalStability &&
                rejectedRatio <= runtimeConfig.maximumRejectedObservationRatio &&
                input.predictionBridgeRatio <=
                    runtimeConfig.maximumPredictionBridgeRatio &&
                sufficientEvidence

        return Result(
            state = state,
            score = score,
            evidenceScore = evidenceSupport,
            authoritative =
                state == State.READY &&
                    !hardFailure &&
                    authoritativeEvidence,
            accepted =
                state == State.READY ||
                    state == State.DEGRADED,
            hardFailure = hardFailure,
            failureClass = failure,
            coverageScore = coverageScore,
            visibilityScore = visibilityScore,
            uncertaintyScore = uncertaintyScore,
            crossViewScore = crossViewScore,
            temporalScore = temporalScore,
            rejectionScore = rejectionScore,
            predictionScore = predictionScore,
            contactScore = contactScore,
            effectiveCameraRedundancy = effectiveCameraRedundancy,
            warnings = warnings.distinct(),
            recommendations = recommendations,
        )
    }

    private fun dominantFailure(
        input: Input,
        rejectedRatio: Double,
    ): FailureClass =
        when {
            input.hardFailure -> FailureClass.UPSTREAM_HARD_FAILURE
            input.coverage < runtimeConfig.hardCoverageFloor ->
                FailureClass.COVERAGE
            input.visibleCoverage < runtimeConfig.hardVisibleCoverageFloor ->
                FailureClass.VISIBILITY
            input.uncertaintyP95 >= runtimeConfig.hardUncertaintyP95 ->
                FailureClass.UNCERTAINTY
            input.crossViewEvidenceAvailable &&
                input.crossViewAgreement < runtimeConfig.hardCrossViewAgreement ->
                FailureClass.CROSS_VIEW
            input.temporalStability < runtimeConfig.hardTemporalStability ->
                FailureClass.TEMPORAL
            rejectedRatio >= runtimeConfig.hardRejectedObservationRatio ->
                FailureClass.REJECTION
            input.predictionBridgeRatio >= runtimeConfig.hardPredictionBridgeRatio ->
                FailureClass.PREDICTION
            input.contactEvidenceActive &&
                input.contactCoverage <
                    runtimeConfig.minimumContactCoverageWhenActive ->
                FailureClass.CONTACT
            input.supportingCameraCount < runtimeConfig.minimumCrossViewCameras ->
                FailureClass.INSUFFICIENT_EVIDENCE
            input.coverage < runtimeConfig.minimumCoverage ->
                FailureClass.COVERAGE
            input.visibleCoverage < runtimeConfig.minimumVisibleCoverage ->
                FailureClass.VISIBILITY
            input.uncertaintyP95 > runtimeConfig.maximumUncertaintyP95 ->
                FailureClass.UNCERTAINTY
            input.crossViewEvidenceAvailable &&
                input.crossViewAgreement <
                    runtimeConfig.minimumCrossViewAgreement ->
                FailureClass.CROSS_VIEW
            input.temporalStability < runtimeConfig.minimumTemporalStability ->
                FailureClass.TEMPORAL
            rejectedRatio > runtimeConfig.maximumRejectedObservationRatio ->
                FailureClass.REJECTION
            input.predictionBridgeRatio >
                runtimeConfig.maximumPredictionBridgeRatio ->
                FailureClass.PREDICTION
            else -> FailureClass.NONE
        }

    private fun buildRecommendations(
        input: Input,
        failureClass: FailureClass,
    ): List<String> =
        buildList {
            when (failureClass) {
                FailureClass.COVERAGE,
                FailureClass.VISIBILITY ->
                    add("Improve subject framing and camera coverage before increasing model detail.")

                FailureClass.UNCERTAINTY ->
                    add("Increase parallax, camera quality, exposure quality, and landmark evidence before tightening geometry gates.")

                FailureClass.CROSS_VIEW ->
                    add("Add or reposition cameras so the same landmarks are observed from independent viewpoints.")

                FailureClass.TEMPORAL ->
                    add("Improve synchronization and frame-time stability before relying on temporal interpolation.")

                FailureClass.REJECTION ->
                    add("Inspect camera quality and residuals; excessive rejection means the rig is not producing consistent evidence.")

                FailureClass.PREDICTION ->
                    add("Reduce detector starvation or increase physical visibility instead of extending prediction indefinitely.")

                FailureClass.CONTACT ->
                    add("Increase hand/object visibility and interaction coverage before trusting contact-constrained reconstruction.")

                FailureClass.INSUFFICIENT_EVIDENCE ->
                    add("Increase independent camera coverage before requesting dense reference-grade geometry.")

                FailureClass.UPSTREAM_HARD_FAILURE ->
                    add("Resolve the upstream hard-failure reason before dense reconstruction is authorized.")

                FailureClass.NONE -> Unit
            }

            if (
                input.crossViewEvidenceAvailable &&
                input.supportingCameraCount >= 3 &&
                input.crossViewAgreement >= 0.75
            ) {
                add("Cross-view redundancy is strong enough for aggressive geometric refinement.")
            }
        }

    private fun lowerBoundScore(
        measured: Double,
        minimum: Double,
    ): Double {
        if (!measured.isFinite()) return 0.0
        if (measured >= minimum) return 1.0
        if (minimum <= 0.0) return measured.coerceIn(0.0, 1.0)
        return (measured / minimum).coerceIn(0.0, 1.0)
    }

    private fun upperBoundScore(
        measured: Double,
        maximum: Double,
        invert: Boolean = false,
    ): Double {
        if (!measured.isFinite()) return 0.0
        if (maximum <= 0.0) return if (measured <= 0.0) 1.0 else 0.0
        return if (!invert) {
            if (measured <= maximum) 1.0
            else (maximum / measured.coerceAtLeast(1e-9)).coerceIn(0.0, 1.0)
        } else {
            if (measured <= maximum) 1.0
            else (maximum / measured.coerceAtLeast(1e-9)).coerceIn(0.0, 1.0)
        }
    }

    private fun redundancyScore(
        supportingCameraCount: Int,
    ): Double =
        when {
            supportingCameraCount >= 6 -> 1.0
            supportingCameraCount == 5 -> 0.94
            supportingCameraCount == 4 -> 0.88
            supportingCameraCount == 3 -> 0.76
            supportingCameraCount == 2 -> 0.58
            else -> 0.25
        }

    private fun geometricMean(
        values: List<Pair<Double, Double>>,
    ): Double {
        val weightedLog =
            values.sumOf { (value, weight) ->
                weight * kotlin.math.ln(
                    value.coerceIn(1e-6, 1.0),
                )
            }
        val weightSum =
            values.sumOf { it.second }.coerceAtLeast(1e-6)
        return kotlin.math.exp(weightedLog / weightSum)
    }
}

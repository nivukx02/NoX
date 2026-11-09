package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.max

/**
 * Traceable parity/superiority gate against an active-optical marker reference.
 *
 * The engine intentionally requires the reference family to be explicit. An anonymous "optical"
 * score is not enough to claim superiority over an active infrared-marker system. Conditions must
 * be matched, the reference uncertainty must be materially smaller than the claimed bound, and the
 * comparison must have enough samples and independent sessions.
 *
 * This is a qualification protocol, not a promise of physical accuracy. The only authoritative
 * result is the state returned from measured matched-reference evidence.
 */
class ActiveOpticalReferenceParityEngine(
    private val config: Config = Config(),
) {

    enum class State {
        NOT_READY,
        INCONCLUSIVE,
        REFERENCE_SUPERIOR,
        NOX_WITHIN_REFERENCE,
        NOX_SUPERIOR_EVIDENCE,
    }

    data class Config(
        val minimumSamples: Int = 240,
        val minimumIndependentSessions: Int = 3,
        val maximumReferenceUncertaintyMillimeters: Double = 0.25,
        val maximumNoxMeasurementUncertaintyMillimeters: Double = 0.50,
        val noxTargetP95Millimeters: Double = 1.0,
        val noxTargetP99Millimeters: Double = 2.5,
        val superiorityMarginMillimeters: Double = 0.10,
        val maximumConditionMismatchPenalty: Double = 0.15,
    ) {
        init {
            require(minimumSamples >= 1)
            require(minimumIndependentSessions >= 1)
            require(maximumReferenceUncertaintyMillimeters > 0.0)
            require(maximumNoxMeasurementUncertaintyMillimeters > 0.0)
            require(noxTargetP95Millimeters > 0.0)
            require(noxTargetP99Millimeters >= noxTargetP95Millimeters)
            require(superiorityMarginMillimeters >= 0.0)
            require(maximumConditionMismatchPenalty in 0.0..1.0)
        }
    }

    data class Input(
        val referenceType:
            CapturePrecisionBenchmarkEngine.ReferenceType?,
        val matchedConditions: Boolean,
        val noxP95Millimeters: Double?,
        val noxP99Millimeters: Double?,
        val noxMeasurementUncertaintyMillimeters: Double?,
        val referenceP95Millimeters: Double?,
        val referenceP99Millimeters: Double?,
        val referenceUncertaintyMillimeters: Double?,
        val sampleCount: Int,
        val independentSessions: Int,
        val commonTargetDistanceMeters: Double? = null,
        val commonTargetAngularSpeedDegreesPerSecond: Double? = null,
    ) {
        init {
            require(sampleCount >= 0)
            require(independentSessions >= 0)
            listOfNotNull(
                noxP95Millimeters,
                noxP99Millimeters,
                noxMeasurementUncertaintyMillimeters,
                referenceP95Millimeters,
                referenceP99Millimeters,
                referenceUncertaintyMillimeters,
                commonTargetDistanceMeters,
                commonTargetAngularSpeedDegreesPerSecond,
            ).forEach {
                require(it.isFinite() && it >= 0.0)
            }
        }
    }

    data class Result(
        val state: State,
        val eligible: Boolean,
        val referenceFamilyMatched: Boolean,
        val matchedConditions: Boolean,
        val referenceUncertaintyAdequate: Boolean,
        val sampleCountAdequate: Boolean,
        val independentSessionsAdequate: Boolean,
        val noxMeasurementUncertaintyAdequate: Boolean,
        val noxP95Millimeters: Double?,
        val noxP99Millimeters: Double?,
        val referenceP95Millimeters: Double?,
        val referenceP99Millimeters: Double?,
        val conservativeNoxP95Millimeters: Double?,
        val conservativeNoxP99Millimeters: Double?,
        val superiorityMarginP95Millimeters: Double?,
        val score: Double,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    fun evaluate(input: Input): Result {
        val referenceFamilyMatched =
            input.referenceType ==
                CapturePrecisionBenchmarkEngine.ReferenceType
                    .ACTIVE_OPTICAL_MARKER_REFERENCE

        val referenceUncertaintyAdequate =
            input.referenceUncertaintyMillimeters?.let {
                it <= config.maximumReferenceUncertaintyMillimeters
            } == true

        val sampleCountAdequate =
            input.sampleCount >= config.minimumSamples

        val independentSessionsAdequate =
            input.independentSessions >=
                config.minimumIndependentSessions

        val noxMeasurementUncertaintyAdequate =
            input.noxMeasurementUncertaintyMillimeters?.let {
                it <=
                    config.maximumNoxMeasurementUncertaintyMillimeters
            } == true

        val evidenceComplete =
            referenceFamilyMatched &&
                input.matchedConditions &&
                referenceUncertaintyAdequate &&
                sampleCountAdequate &&
                independentSessionsAdequate &&
                noxMeasurementUncertaintyAdequate &&
                input.noxP95Millimeters != null &&
                input.noxP99Millimeters != null &&
                input.referenceP95Millimeters != null &&
                input.referenceP99Millimeters != null

        val conservativeNoxP95 =
            input.noxP95Millimeters?.let { p95 ->
                p95 +
                    max(
                        0.0,
                        input.noxMeasurementUncertaintyMillimeters ?: 0.0,
                    )
            }

        val conservativeNoxP99 =
            input.noxP99Millimeters?.let { p99 ->
                p99 +
                    max(
                        0.0,
                        (
                            input.noxMeasurementUncertaintyMillimeters
                                ?: 0.0
                        ) * 1.25,
                    )
            }

        val superiorityMarginP95 =
            if (
                conservativeNoxP95 != null &&
                    input.referenceP95Millimeters != null
            ) {
                input.referenceP95Millimeters -
                    conservativeNoxP95
            } else {
                null
            }

        val targetQuality =
            if (
                conservativeNoxP95 != null &&
                    conservativeNoxP99 != null
            ) {
                (
                    inversePenalty(
                        conservativeNoxP95,
                        config.noxTargetP95Millimeters * 2.0,
                    ) * 0.46 +
                        inversePenalty(
                            conservativeNoxP99,
                            config.noxTargetP99Millimeters * 2.0,
                        ) * 0.34
                ).coerceIn(0.0, 1.0)
            } else {
                0.0
            }

        val protocolQuality =
            (
                (if (referenceFamilyMatched) 0.18 else 0.0) +
                    (if (input.matchedConditions) 0.18 else 0.0) +
                    (if (referenceUncertaintyAdequate) 0.16 else 0.0) +
                    (if (sampleCountAdequate) 0.14 else 0.0) +
                    (if (independentSessionsAdequate) 0.14 else 0.0) +
                    (if (noxMeasurementUncertaintyAdequate) 0.12 else 0.0) +
                    targetQuality * 0.08
                ).coerceIn(0.0, 1.0)

        val state =
            when {
                !evidenceComplete ->
                    if (
                        input.referenceType == null &&
                            input.noxP95Millimeters == null
                    ) {
                        State.NOT_READY
                    } else {
                        State.INCONCLUSIVE
                    }
                conservativeNoxP95 == null ||
                    conservativeNoxP99 == null ||
                    input.referenceP95Millimeters == null ||
                    input.referenceP99Millimeters == null ->
                    State.INCONCLUSIVE
                conservativeNoxP95 <
                    input.referenceP95Millimeters -
                        config.superiorityMarginMillimeters &&
                    conservativeNoxP99 <=
                        input.referenceP99Millimeters &&
                    conservativeNoxP95 <=
                        config.noxTargetP95Millimeters &&
                    conservativeNoxP99 <=
                        config.noxTargetP99Millimeters ->
                    State.NOX_SUPERIOR_EVIDENCE
                conservativeNoxP95 <=
                    input.referenceP95Millimeters &&
                    conservativeNoxP99 <=
                        input.referenceP99Millimeters ->
                    State.NOX_WITHIN_REFERENCE
                else ->
                    State.REFERENCE_SUPERIOR
            }

        val warnings =
            buildList {
                if (!referenceFamilyMatched) {
                    add("ACTIVE_OPTICAL_REFERENCE_FAMILY_NOT_CONFIRMED")
                }
                if (!input.matchedConditions) {
                    add("ACTIVE_OPTICAL_REFERENCE_CONDITIONS_NOT_MATCHED")
                }
                if (!referenceUncertaintyAdequate) {
                    add("ACTIVE_OPTICAL_REFERENCE_UNCERTAINTY_TOO_LARGE")
                }
                if (!sampleCountAdequate) {
                    add("ACTIVE_OPTICAL_REFERENCE_SAMPLE_COUNT_TOO_LOW")
                }
                if (!independentSessionsAdequate) {
                    add("ACTIVE_OPTICAL_REFERENCE_INDEPENDENT_SESSION_COUNT_TOO_LOW")
                }
                if (!noxMeasurementUncertaintyAdequate) {
                    add("NOX_MEASUREMENT_UNCERTAINTY_NOT_BOUNDED")
                }
                if (
                    input.noxP95Millimeters != null &&
                        input.noxP95Millimeters >
                            config.noxTargetP95Millimeters
                ) {
                    add("NOX_P95_TARGET_ABOVE_ONE_MILLIMETER")
                }
                if (
                    input.referenceP95Millimeters != null &&
                        input.noxP95Millimeters != null &&
                        input.referenceP95Millimeters <
                            input.noxP95Millimeters
                ) {
                    add("REFERENCE_P95_LOWER_THAN_NOX")
                }
            }.distinct()

        val recommendations =
            buildList {
                if (!referenceFamilyMatched) {
                    add(
                        "Run a matched trial against an explicitly identified active optical marker reference.",
                    )
                }
                if (!input.matchedConditions) {
                    add(
                        "Match subject motion, capture distance, camera layout, frame rate and lighting before comparing errors.",
                    )
                }
                if (!referenceUncertaintyAdequate) {
                    add(
                        "Use a traceable reference measurement whose uncertainty is materially below the target bound.",
                    )
                }
                if (!sampleCountAdequate ||
                    !independentSessionsAdequate
                ) {
                    add(
                        "Repeat the qualification across enough frames and independent sessions to reject a lucky short trial.",
                    )
                }
                if (state == State.REFERENCE_SUPERIOR) {
                    add(
                        "Increase camera geometry quality, synchronization quality or reconstruction fidelity before claiming parity.",
                    )
                }
            }.distinct()

        return Result(
            state = state,
            eligible = evidenceComplete,
            referenceFamilyMatched = referenceFamilyMatched,
            matchedConditions = input.matchedConditions,
            referenceUncertaintyAdequate =
                referenceUncertaintyAdequate,
            sampleCountAdequate = sampleCountAdequate,
            independentSessionsAdequate =
                independentSessionsAdequate,
            noxMeasurementUncertaintyAdequate =
                noxMeasurementUncertaintyAdequate,
            noxP95Millimeters = input.noxP95Millimeters,
            noxP99Millimeters = input.noxP99Millimeters,
            referenceP95Millimeters =
                input.referenceP95Millimeters,
            referenceP99Millimeters =
                input.referenceP99Millimeters,
            conservativeNoxP95Millimeters =
                conservativeNoxP95,
            conservativeNoxP99Millimeters =
                conservativeNoxP99,
            superiorityMarginP95Millimeters =
                superiorityMarginP95,
            score = protocolQuality,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    private fun inversePenalty(
        value: Double,
        limit: Double,
    ): Double =
        (
            1.0 -
                value /
                    max(
                        limit,
                        0.001,
                    )
        ).coerceIn(0.0, 1.0)
}

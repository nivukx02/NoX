package com.nivukx.nox.core.capture

import kotlin.math.abs
import kotlin.math.max

/**
 * Focus-conditioned geometric calibration authority.
 *
 * Mobile lenses can change effective intrinsics as focus distance changes. This engine does not
 * assume a vendor-specific breathing model; it measures the focus-state delta relative to the
 * focus state at which calibration was established and converts that delta into a conservative
 * geometry-conditioning score.
 *
 * Reference-grade capture therefore requires either:
 * - stable manual focus matching the calibration state, or
 * - an empirically established focus-to-intrinsics model outside this engine.
 */
class FocusConditioningEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val warningDiopterDelta: Double = 0.08,
        val hardDiopterDelta: Double = 0.35,
        val minimumFocusConfidence: Double = 0.80,
        val referenceMaximumDiopterDelta: Double = 0.04,
        val manualFocusBonus: Double = 0.06,
    ) {
        init {
            require(warningDiopterDelta > 0.0)
            require(hardDiopterDelta > warningDiopterDelta)
            require(minimumFocusConfidence in 0.0..1.0)
            require(referenceMaximumDiopterDelta > 0.0)
            require(manualFocusBonus in 0.0..1.0)
        }
    }

    data class Input(
        val calibrationFocusDistanceDiopters: Double?,
        val currentFocusDistanceDiopters: Double?,
        val focusConfidence: Double,
        val manualFocus: Boolean,
    ) {
        init {
            listOf(
                calibrationFocusDistanceDiopters,
                currentFocusDistanceDiopters,
            ).forEach { value ->
                require(
                    value == null ||
                        (
                            value.isFinite() &&
                                value >= 0.0
                            ),
                )
            }
            require(
                focusConfidence.isFinite() &&
                    focusConfidence in 0.0..1.0,
            )
        }
    }

    data class Result(
        val conditioningScore: Double,
        val focusDeltaDiopters: Double?,
        val referenceGradeCompatible: Boolean,
        val measuredStateAvailable: Boolean,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    fun evaluate(
        input: Input,
    ): Result {
        val calibrationFocus =
            input.calibrationFocusDistanceDiopters
        val currentFocus =
            input.currentFocusDistanceDiopters

        if (
            calibrationFocus == null ||
                currentFocus == null
        ) {
            return Result(
                conditioningScore =
                    (
                        0.48 *
                            input.focusConfidence
                    ).coerceIn(0.0, 1.0),
                focusDeltaDiopters = null,
                referenceGradeCompatible = false,
                measuredStateAvailable = false,
                warnings =
                    listOf(
                        "FOCUS_CONDITIONING_STATE_UNAVAILABLE",
                    ),
                recommendations =
                    listOf(
                        "Preserve lens focus metadata or use fixed manual focus for metric reference capture.",
                    ),
            )
        }

        val delta =
            abs(
                currentFocus -
                    calibrationFocus,
            )

        val deltaScore =
            when {
                delta <= config.warningDiopterDelta ->
                    1.0
                delta >= config.hardDiopterDelta ->
                    0.0
                else ->
                    (
                        1.0 -
                            (
                                delta -
                                    config.warningDiopterDelta
                            ) /
                            (
                                config.hardDiopterDelta -
                                    config.warningDiopterDelta
                            )
                    ).coerceIn(0.0, 1.0)
            }

        val confidenceScore =
            input.focusConfidence.coerceIn(
                0.0,
                1.0,
            )

        val modeScore =
            if (input.manualFocus) {
                1.0
            } else {
                0.86
            }

        val manualBonus =
            if (input.manualFocus) {
                config.manualFocusBonus
            } else {
                0.0
            }

        val conditioningScore =
            (
                deltaScore * 0.62 +
                    confidenceScore * 0.25 +
                    modeScore * 0.13 +
                    manualBonus
            ).coerceIn(0.0, 1.0)

        val referenceGradeCompatible =
            delta <=
                config.referenceMaximumDiopterDelta &&
                confidenceScore >=
                    config.minimumFocusConfidence

        val warnings =
            buildList {
                if (delta > config.warningDiopterDelta) {
                    add(
                        "FOCUS_CONDITIONING_DRIFT_DETECTED",
                    )
                }
                if (
                    confidenceScore <
                        config.minimumFocusConfidence
                ) {
                    add(
                        "FOCUS_CONDITIONING_CONFIDENCE_LOW",
                    )
                }
                if (
                    delta >=
                        config.hardDiopterDelta
                ) {
                    add(
                        "FOCUS_CONDITIONING_HARD_DRIFT",
                    )
                }
                if (
                    !input.manualFocus &&
                    delta >
                        config.referenceMaximumDiopterDelta
                ) {
                    add(
                        "FOCUS_CONDITIONING_AUTOFOCUS_UNSAFE_FOR_REFERENCE",
                    )
                }
            }.distinct()

        val recommendations =
            buildList {
                if (
                    delta >
                        config.referenceMaximumDiopterDelta
                ) {
                    add(
                        "Restore the calibrated focus distance before reference-grade metric capture.",
                    )
                }
                if (
                    !input.manualFocus
                ) {
                    add(
                        "Lock focus after the reference calibration pass to prevent focus-breathing geometry changes.",
                    )
                }
            }.distinct()

        return Result(
            conditioningScore = conditioningScore,
            focusDeltaDiopters = delta,
            referenceGradeCompatible =
                referenceGradeCompatible,
            measuredStateAvailable = true,
            warnings = warnings,
            recommendations = recommendations,
        )
    }
}

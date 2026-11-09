package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.reconstruction.MetricPrecisionBudgetEngine
import kotlin.math.abs

/**
 * High-end preflight gate for reference-grade metric capture.
 *
 * This is intentionally stricter than ordinary production calibration. A rig can be perfectly
 * usable for normal motion capture and still fail the reference-grade protocol because timing,
 * drift, parallax or metrology uncertainty are not sufficiently controlled.
 */
class ReferenceGradeCalibrationProtocolEngine(
    private val config: Config = Config(),
) {

    enum class Phase {
        IDLE,
        WORKFLOW,
        GEOMETRY,
        TIMING,
        DRIFT,
        METROLOGY,
        THERMAL,
        READY,
        BLOCKED,
    }

    data class Config(
        val minimumRigScore: Double = 0.85,
        val minimumCoverage: Double = 0.88,
        val maximumReprojectionErrorPx: Double = 1.25,
        val maximumClockSkewMilliseconds: Double = 0.75,
        val minimumTemporalAlignmentConfidence: Double = 0.92,
        val minimumTimingAuthorityConfidence: Double = 0.90,
        val maximumTimingAuthorityResidualMilliseconds: Double = 1.50,
        val minimumDriftStability: Double = 0.92,
        val maximumScaleDriftPercentPerMinute: Double = 0.25,
        val maximumTemperatureDeltaCelsius: Double = 3.0,
        val minimumThermalStability: Double = 0.90,
        val minimumObservabilityScore: Double = 0.82,
        val requiredP95Millimeters: Double = 2.0,
    ) {
        init {
            listOf(
                minimumRigScore,
                minimumCoverage,
                minimumTemporalAlignmentConfidence,
                minimumTimingAuthorityConfidence,
                minimumDriftStability,
                minimumThermalStability,
                minimumObservabilityScore,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            require(maximumReprojectionErrorPx > 0.0)
            require(maximumClockSkewMilliseconds > 0.0)
            require(maximumTimingAuthorityResidualMilliseconds > 0.0)
            require(maximumScaleDriftPercentPerMinute >= 0.0)
            require(maximumTemperatureDeltaCelsius >= 0.0)
            require(requiredP95Millimeters in 0.1..2.0)
        }
    }

    data class Input(
        val workflowReady: Boolean,
        val rigScore: Double,
        val coverage: Double,
        val reprojectionErrorPx: Double,
        val clockSkewMilliseconds: Double,
        val temporalAlignmentConfidence: Double,
        val timingAuthorityConfidence: Double,
        val timingAuthorityResidualP95Milliseconds: Double,
        val timingAuthorityReferenceGrade: Boolean,
        val driftStability: Double,
        val recalibrationRequired: Boolean,
        val scaleDriftPercentPerMinute: Double,
        val temperatureDeltaCelsius: Double,
        val thermalStability: Double,
        val observabilityScore: Double,
        val metrology: MetricPrecisionBudgetEngine.Decision,
    ) {
        init {
            listOf(
                rigScore,
                coverage,
                temporalAlignmentConfidence,
                timingAuthorityConfidence,
                thermalStability,
                observabilityScore,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            listOf(
                reprojectionErrorPx,
                clockSkewMilliseconds,
                timingAuthorityResidualP95Milliseconds,
                scaleDriftPercentPerMinute,
                temperatureDeltaCelsius,
            ).forEach {
                require(it.isFinite() && it >= 0.0)
            }
        }
    }

    data class Status(
        val phase: Phase,
        val ready: Boolean,
        val referenceGrade: Boolean,
        val progress: Double,
        val blockingReasons: List<String>,
        val warnings: List<String>,
        val guidance: List<String>,
        val diagnostics: List<String>,
    )

    fun evaluate(input: Input): Status {
        val blockers = mutableListOf<String>()
        val warnings = mutableListOf<String>()
        val guidance = mutableListOf<String>()

        if (!input.workflowReady) {
            blockers += "calibration-workflow-not-ready"
            guidance += "Complete the normal multi-camera calibration workflow before entering reference-grade capture."
        }

        if (input.rigScore < config.minimumRigScore) {
            blockers += "rig-score-below-reference-threshold"
            guidance += "Increase calibrated camera coverage and improve pair geometry."
        }

        if (input.coverage < config.minimumCoverage) {
            blockers += "coverage-below-reference-threshold"
            guidance += "Increase overlap across the active subject volume."
        }

        if (input.reprojectionErrorPx > config.maximumReprojectionErrorPx) {
            blockers += "reprojection-error-above-reference-threshold"
            guidance += "Refine intrinsics, distortion and extrinsics before reference-grade capture."
        }

        if (input.clockSkewMilliseconds > config.maximumClockSkewMilliseconds) {
            blockers += "clock-skew-above-reference-threshold"
            guidance += "Run hardware or sub-frame synchronization refinement."
        }

        if (
            input.temporalAlignmentConfidence <
                config.minimumTemporalAlignmentConfidence
        ) {
            blockers += "temporal-alignment-below-reference-threshold"
            guidance += "Refine inter-camera timing before capturing fast motion."
        }

        if (
            input.timingAuthorityConfidence <
                config.minimumTimingAuthorityConfidence
        ) {
            blockers += "timing-authority-below-reference-threshold"
            guidance +=
                "Use hardware readout timing, calibrated logical synchronization, or an independently measured visual timing beacon."
        }

        if (
            input.timingAuthorityResidualP95Milliseconds >
                config.maximumTimingAuthorityResidualMilliseconds
        ) {
            blockers += "timing-authority-residual-above-reference-threshold"
            guidance +=
                "Reduce temporal residuals before reference-grade metric capture."
        }

        if (!input.timingAuthorityReferenceGrade) {
            warnings += "timing-authority-not-reference-grade"
        }

        if (input.recalibrationRequired) {
            blockers += "calibration-drift-requires-recalibration"
            guidance += "Re-run calibration before accepting a reference-grade session."
        }

        if (input.driftStability < config.minimumDriftStability) {
            blockers += "calibration-drift-stability-low"
            guidance += "Stabilize the rig and verify drift over a rolling observation window."
        }

        if (
            input.scaleDriftPercentPerMinute >
                config.maximumScaleDriftPercentPerMinute
        ) {
            blockers += "metric-scale-drift-above-reference-threshold"
            guidance += "Re-establish metric scale and inspect moving-camera or thermal effects."
        }

        if (input.temperatureDeltaCelsius > config.maximumTemperatureDeltaCelsius) {
            warnings += "temperature-delta-large"
            guidance += "Repeat the reference check after thermal stabilization."
        }

        if (input.thermalStability < config.minimumThermalStability) {
            blockers += "thermal-stability-below-reference-threshold"
            guidance += "Allow the capture rig to reach a stable thermal state."
        }

        if (input.observabilityScore < config.minimumObservabilityScore) {
            blockers += "observability-score-below-reference-threshold"
            guidance += "Improve camera health, angular diversity and sector coverage."
        }

        if (!input.metrology.physicalEvidenceSufficient) {
            blockers += "metrology-evidence-insufficient"
            guidance += "Supply stronger calibration, synchronization, view diversity and observation evidence."
        }

        if (
            input.metrology.modeledP95ErrorMillimeters >
                config.requiredP95Millimeters
        ) {
            blockers += "modeled-p95-above-reference-target"
            guidance += "Increase baseline, focal-length support, parallax, calibration quality or reduce residual error."
        }

        if (input.metrology.modeledP99ErrorMillimeters > 3.5) {
            warnings += "modeled-p99-tail-above-reference-envelope"
        }

        if (input.metrology.contribution.stereoDepthMillimeters > 1.0) {
            warnings += "stereo-depth-dominates-error-budget"
        }

        if (input.metrology.contribution.timingMillimeters > 0.5) {
            warnings += "timing-dominates-error-budget"
        }

        if (input.metrology.contribution.cameraPoseMillimeters > 0.8) {
            warnings += "camera-pose-dominates-error-budget"
        }

        val phase = when {
            blockers.contains("calibration-workflow-not-ready") ->
                Phase.WORKFLOW
            blockers.any {
                it.contains("rig-score") ||
                    it.contains("coverage") ||
                    it.contains("observability")
            } ->
                Phase.GEOMETRY
            blockers.any {
                it.contains("clock-skew") ||
                    it.contains("temporal-alignment")
            } ->
                Phase.TIMING
            blockers.any {
                it.contains("drift") ||
                    it.contains("scale")
            } ->
                Phase.DRIFT
            blockers.any { it.contains("metrology") } ->
                Phase.METROLOGY
            blockers.any { it.contains("thermal") } ->
                Phase.THERMAL
            blockers.isNotEmpty() ->
                Phase.BLOCKED
            else ->
                Phase.READY
        }

        val progress = (
            workflowProgress(input.workflowReady) * 0.15 +
                geometryProgress(input) * 0.22 +
                timingProgress(input) * 0.18 +
                driftProgress(input) * 0.18 +
                metrologyProgress(input) * 0.22 +
                thermalProgress(input) * 0.05
        ).coerceIn(0.0, 1.0)

        val ready =
            blockers.isEmpty() &&
                input.metrology.referenceGradeFeasible &&
                input.metrology.modeledP95ErrorMillimeters <=
                    config.requiredP95Millimeters

        return Status(
            phase = if (ready) Phase.READY else phase,
            ready = ready,
            referenceGrade = ready,
            progress = progress,
            blockingReasons = blockers.distinct(),
            warnings = warnings.distinct(),
            guidance = guidance.distinct(),
            diagnostics = listOf(
                "workflowReady=" + input.workflowReady,
                "rigScore=" + format(input.rigScore),
                "coverage=" + format(input.coverage),
                "reprojectionPx=" + format(input.reprojectionErrorPx),
                "clockSkewMs=" + format(input.clockSkewMilliseconds),
                "temporalAlignment=" +
                    format(input.temporalAlignmentConfidence),
                "timingAuthority=" +
                    format(input.timingAuthorityConfidence),
                "timingAuthorityResidualMs=" +
                    format(
                        input.timingAuthorityResidualP95Milliseconds,
                    ),
                "timingAuthorityReferenceGrade=" +
                    input.timingAuthorityReferenceGrade,
                "driftStability=" + format(input.driftStability),
                "scaleDriftPercentPerMinute=" +
                    format(input.scaleDriftPercentPerMinute),
                "temperatureDeltaC=" +
                    format(input.temperatureDeltaCelsius),
                "thermalStability=" + format(input.thermalStability),
                "observability=" + format(input.observabilityScore),
                "modeledP95Mm=" +
                    format(input.metrology.modeledP95ErrorMillimeters),
                "modeledP99Mm=" +
                    format(input.metrology.modeledP99ErrorMillimeters),
                "metrologyGrade=" + input.metrology.grade.name,
                "phase=" + (if (ready) Phase.READY else phase).name,
            ),
        )
    }

    private fun workflowProgress(ready: Boolean): Double =
        if (ready) 1.0 else 0.0

    private fun geometryProgress(input: Input): Double =
        (
            normalized(input.rigScore, config.minimumRigScore) * 0.55 +
                normalized(input.coverage, config.minimumCoverage) * 0.25 +
                normalized(
                    input.observabilityScore,
                    config.minimumObservabilityScore,
                ) * 0.20
        ).coerceIn(0.0, 1.0)

    private fun timingProgress(input: Input): Double =
        (
            (1.0 -
                input.clockSkewMilliseconds /
                    config.maximumClockSkewMilliseconds) * 0.55 +
                input.temporalAlignmentConfidence * 0.45
        ).coerceIn(0.0, 1.0)

    private fun driftProgress(input: Input): Double =
        (
            input.driftStability * 0.55 +
                (
                    1.0 -
                        input.scaleDriftPercentPerMinute /
                            maxOf(
                                config.maximumScaleDriftPercentPerMinute,
                                1e-6,
                            )
                ) * 0.45
        ).coerceIn(0.0, 1.0)

    private fun metrologyProgress(input: Input): Double =
        (
            (1.0 -
                input.metrology.modeledP95ErrorMillimeters /
                    config.requiredP95Millimeters
            ) * 0.72 +
                (if (input.metrology.physicalEvidenceSufficient) 1.0 else 0.0) * 0.28
        ).coerceIn(0.0, 1.0)

    private fun thermalProgress(input: Input): Double =
        (
            input.thermalStability * 0.75 +
                (
                    1.0 -
                        input.temperatureDeltaCelsius /
                            maxOf(
                                config.maximumTemperatureDeltaCelsius,
                                1e-6,
                            )
                ) * 0.25
        ).coerceIn(0.0, 1.0)

    private fun normalized(value: Double, target: Double): Double =
        (value / target.coerceAtLeast(1e-6)).coerceIn(0.0, 1.0)

    private fun format(value: Double): String =
        "%.4f".format(java.util.Locale.US, value)
}

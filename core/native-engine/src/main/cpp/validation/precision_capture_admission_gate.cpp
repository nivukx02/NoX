#include "validation/precision_capture_admission_gate.h"

#include <algorithm>
#include <cmath>

namespace nox::native::validation {

namespace {

double clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;

    return std::clamp(
        value,
        0.0,
        1.0);
}

double inversePrecision(
    double measured,
    double target) noexcept {
    if (
        !std::isfinite(measured) ||
        !std::isfinite(target) ||
        target <= 0.0) {
        return 0.0;
    }

    return std::clamp(
        std::exp(
            -std::max(
                0.0,
                measured -
                    target) /
                target),
        0.0,
        1.0);
}

} // namespace

PrecisionCaptureAdmissionResult
PrecisionCaptureAdmissionGate::evaluate(
    const PrecisionCaptureAdmissionInput& raw) const noexcept {
    PrecisionCaptureAdmissionInput input =
        raw;

    input.requestedP95Millimeters =
        std::clamp(
            std::isfinite(
                input.requestedP95Millimeters)
                ? input.requestedP95Millimeters
                : 1.0,
            0.25,
            10.0);

    input.measuredP95Millimeters =
        std::max(
            0.0,
            std::isfinite(
                input.measuredP95Millimeters)
                ? input.measuredP95Millimeters
                : 1e9);

    input.measuredMaxMillimeters =
        std::max(
            0.0,
            std::isfinite(
                input.measuredMaxMillimeters)
                ? input.measuredMaxMillimeters
                : 1e9);

    input.calibrationConfidence =
        clamp01(
            input.calibrationConfidence);
    input.synchronizationConfidence =
        clamp01(
            input.synchronizationConfidence);
    input.viewDiversity =
        clamp01(
            input.viewDiversity);
    input.crossViewAgreement =
        clamp01(
            input.crossViewAgreement);
    input.metricScaleConfidence =
        clamp01(
            input.metricScaleConfidence);
    input.temporalStability =
        clamp01(
            input.temporalStability);
    input.thermalHeadroom =
        clamp01(
            input.thermalHeadroom);
    input.memoryHeadroom =
        clamp01(
            input.memoryHeadroom);

    input.reconstructionUncertaintyMeters =
        std::max(
            0.0,
            std::isfinite(
                input.reconstructionUncertaintyMeters)
                ? input.reconstructionUncertaintyMeters
                : 1.0);

    PrecisionCaptureAdmissionResult result;

    const double precisionScore =
        input.metrologyAvailable
            ? inversePrecision(
                input.measuredP95Millimeters,
                input.requestedP95Millimeters)
            : 0.0;

    const double uncertaintyScore =
        std::exp(
            -input.reconstructionUncertaintyMeters /
            (
                input.requestedP95Millimeters /
                1000.0
            ));

    const double geometricScore =
        0.45 *
            input.viewDiversity +
        0.35 *
            input.crossViewAgreement +
        0.20 *
            (
                input.geometryConditioned
                    ? 1.0
                    : 0.0
            );

    const double evidenceScore =
        0.26 *
            input.calibrationConfidence +
        0.20 *
            input.synchronizationConfidence +
        0.16 *
            input.metricScaleConfidence +
        0.18 *
            input.temporalStability +
        0.20 *
            std::clamp(
                geometricScore,
                0.0,
                1.0);

    const double environmentScore =
        std::min(
            input.thermalHeadroom,
            input.memoryHeadroom);

    const double completeScore =
        (
            0.40 *
                precisionScore +
            0.30 *
                std::clamp(
                    uncertaintyScore,
                    0.0,
                    1.0) +
            0.22 *
                clamp01(
                    evidenceScore) +
            0.08 *
                environmentScore
        );

    result.score =
        clamp01(
            completeScore);

    result.precisionMarginMillimeters =
        input.requestedP95Millimeters -
        input.measuredP95Millimeters;

    if (!input.metrologyAvailable) {
        result.blockers.emplace_back(
            "metrology-evidence-unavailable");
    }

    if (!input.metrologyTraceable) {
        result.blockers.emplace_back(
            "metrology-not-traceable");
    }

    if (
        input.calibrationConfidence <
        0.92 ||
        !input.calibrationStable) {
        result.blockers.emplace_back(
            "calibration-not-reference-grade");
    }

    if (
        input.synchronizationConfidence <
        0.92) {
        result.blockers.emplace_back(
            "synchronization-not-reference-grade");
    }

    if (
        input.viewDiversity <
        0.65 ||
        input.crossViewAgreement <
        0.75 ||
        !input.geometryConditioned) {
        result.blockers.emplace_back(
            "multiview-geometry-not-conditioned");
    }

    if (
        input.metricScaleConfidence <
        0.92) {
        result.blockers.emplace_back(
            "metric-scale-not-authoritative");
    }

    if (
        input.reconstructionUncertaintyMeters >
        input.requestedP95Millimeters /
            1000.0) {
        result.blockers.emplace_back(
            "reconstruction-uncertainty-too-high");
    }

    if (!input.rigStable) {
        result.blockers.emplace_back(
            "rig-stability-not-qualified");
    }

    if (
        input.measuredP95Millimeters >
        input.requestedP95Millimeters) {
        result.blockers.emplace_back(
            "measured-p95-exceeds-target");
    }

    if (
        input.measuredMaxMillimeters >
        input.requestedP95Millimeters * 2.0) {
        result.blockers.emplace_back(
            "measured-max-error-exceeds-envelope");
    }

    if (
        input.thermalHeadroom <
        0.30) {
        result.warnings.emplace_back(
            "thermal-headroom-low");
    }

    if (
        input.memoryHeadroom <
        0.30) {
        result.warnings.emplace_back(
            "memory-headroom-low");
    }

    if (
        input.temporalStability <
        0.85) {
        result.warnings.emplace_back(
            "temporal-stability-below-reference-band");
    }

    result.referenceGradeEligible =
        result.blockers.empty();

    result.authoritative =
        result.referenceGradeEligible &&
        result.score >= 0.85 &&
        input.metrologyTraceable;

    result.admitted =
        result.score >= 0.55 &&
        (
            result.blockers.empty() ||
            (
                input.metrologyAvailable &&
                input.metrologyTraceable &&
                input.calibrationConfidence >= 0.92 &&
                input.synchronizationConfidence >= 0.92
            )
        );

    return result;
}

} // namespace nox::native::validation

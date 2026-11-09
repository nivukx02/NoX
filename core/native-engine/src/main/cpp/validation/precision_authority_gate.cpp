#include "validation/precision_authority_gate.h"

#include <algorithm>
#include <cmath>

namespace nox::native::validation {

namespace {

double scoreAtMost(
    double value,
    double threshold) noexcept {
    if (!std::isfinite(value)) return 0.0;
    if (value >= threshold) return 0.0;
    return std::clamp(
        1.0 -
            value /
            std::max(
                threshold,
                1e-9),
        0.0,
        1.0);
}

} // namespace

double PrecisionAuthorityGate::clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

PrecisionAuthorityReport PrecisionAuthorityGate::evaluate(
    const PrecisionAuthorityInput& input) const noexcept {
    PrecisionAuthorityReport report;

    report.measuredMetrologyQualified =
        input.metrology.state ==
            QualificationState::qualified &&
        input.metrology.validSamples > 0U &&
        input.metrology.p95ErrorMeters <=
            0.001 &&
        input.metrology.maxErrorMeters <=
            0.002 &&
        input.metrology.referenceUncertaintyP95Meters <=
            0.0005;

    report.calibrationQualified =
        clamp01(
            input.calibrationConfidence) >=
            0.92;

    report.synchronizationQualified =
        clamp01(
            input.synchronizationConfidence) >=
            0.92;

    report.reconstructionQualified =
        clamp01(
            input.reconstructionConfidence) >=
            0.90 &&
        std::isfinite(
            input.reconstructionUncertaintyP95Meters) &&
        input.reconstructionUncertaintyP95Meters <=
            0.001;

    report.rigQualified =
        clamp01(
            input.rigStability) >=
            0.90 &&
        clamp01(
            input.global4DConsistency) >=
            0.90;

    const double metrologyScore =
        report.measuredMetrologyQualified
            ? clamp01(
                input.metrology.confidence)
            : 0.0;

    const double calibrationScore =
        clamp01(
            input.calibrationConfidence);

    const double synchronizationScore =
        clamp01(
            input.synchronizationConfidence);

    const double reconstructionScore =
        clamp01(
            input.reconstructionConfidence) *
        scoreAtMost(
            std::max(
                0.0,
                input.reconstructionUncertaintyP95Meters),
            0.001);

    const double rigScore =
        clamp01(
            input.rigStability) *
        clamp01(
            input.global4DConsistency);

    report.evidenceScore =
        std::clamp(
            metrologyScore * 0.38 +
            calibrationScore * 0.20 +
            synchronizationScore * 0.18 +
            reconstructionScore * 0.14 +
            rigScore * 0.10,
            0.0,
            1.0);

    report.bottleneckScore =
        std::clamp(
            std::min({
                metrologyScore,
                calibrationScore,
                synchronizationScore,
                reconstructionScore,
                rigScore,
            }),
            0.0,
            1.0);

    if (!input.referenceBenchmarkArmed) {
        report.state =
            PrecisionAuthorityState::evidenceOnly;
        report.reason =
            "reference-benchmark-not-armed";
        return report;
    }

    if (
        !input.runtimeHealthy ||
        !input.thermalHealthy ||
        !input.memoryHealthy) {
        report.state =
            PrecisionAuthorityState::blocked;
        report.reason =
            "runtime-resource-health-failed";
        return report;
    }

    if (!report.measuredMetrologyQualified) {
        report.state =
            PrecisionAuthorityState::evidenceOnly;
        report.reason =
            "measured-metrology-contract-not-qualified";
        return report;
    }

    if (!report.calibrationQualified) {
        report.state =
            PrecisionAuthorityState::blocked;
        report.reason =
            "calibration-confidence-below-precision-contract";
        return report;
    }

    if (!report.synchronizationQualified) {
        report.state =
            PrecisionAuthorityState::blocked;
        report.reason =
            "synchronization-confidence-below-precision-contract";
        return report;
    }

    if (!report.reconstructionQualified) {
        report.state =
            PrecisionAuthorityState::blocked;
        report.reason =
            "reconstruction-uncertainty-below-precision-contract";
        return report;
    }

    if (!report.rigQualified) {
        report.state =
            PrecisionAuthorityState::blocked;
        report.reason =
            "rig-stability-below-precision-contract";
        return report;
    }

    report.state =
        PrecisionAuthorityState::qualified;
    report.reason =
        "measured-submillimeter-contract-qualified";
    return report;
}

} // namespace nox::native::validation

#include "validation/metrology_precision_qualification.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace nox::native::validation {

namespace {
double finiteNonNegative(double value) noexcept {
    return std::isfinite(value) && value >= 0.0 ? value : 0.0;
}

double rss(double a, double b, double c, double d) noexcept {
    return std::sqrt(a * a + b * b + c * c + d * d);
}
}

QualificationReport MetrologyPrecisionQualification::evaluate(
    const QualificationInput& input) const {
    QualificationReport report{};

    const double targetP95 = std::max(0.0, input.targetP95Meters);
    const double targetMax = std::max(targetP95, input.targetMaxMeters);
    const double referenceLimit =
        std::max(0.0, input.maximumReferenceUncertaintyMeters);

    std::vector<double> errors;
    std::vector<double> references;
    errors.reserve(input.samples.size());
    references.reserve(input.samples.size());

    for (const auto& sample : input.samples) {
        if (!sample.valid ||
            !std::isfinite(sample.measuredErrorMeters) ||
            !std::isfinite(sample.referenceUncertaintyMeters) ||
            sample.measuredErrorMeters < 0.0 ||
            sample.referenceUncertaintyMeters < 0.0) {
            continue;
        }
        errors.push_back(sample.measuredErrorMeters);
        references.push_back(sample.referenceUncertaintyMeters);
    }

    report.validSamples = errors.size();
    if (errors.empty()) {
        report.reason = "no-valid-reference-samples";
        return report;
    }

    std::sort(errors.begin(), errors.end());
    std::sort(references.begin(), references.end());

    report.meanErrorMeters =
        std::accumulate(errors.begin(), errors.end(), 0.0) /
        static_cast<double>(errors.size());
    report.p95ErrorMeters = percentile(errors, 0.95);
    report.maxErrorMeters = errors.back();
    report.referenceUncertaintyP95Meters = percentile(references, 0.95);
    report.systematicUncertaintyMeters = rss(
        finiteNonNegative(input.calibrationUncertaintyMeters),
        finiteNonNegative(input.synchronizationUncertaintyMeters),
        finiteNonNegative(input.temporalUncertaintyMeters),
        finiteNonNegative(input.systematicUncertaintyMeters));
    report.combinedP95UncertaintyMeters = rss(
        report.p95ErrorMeters,
        report.referenceUncertaintyP95Meters,
        report.systematicUncertaintyMeters,
        0.0);

    std::size_t outliers = 0U;
    for (const double error : errors) {
        if (error > targetP95) ++outliers;
    }
    report.outlierFraction =
        static_cast<double>(outliers) /
        static_cast<double>(errors.size());

    const bool enoughSamples = errors.size() >= input.minimumSamples;
    const bool enoughSessions =
        input.independentSessions >= input.minimumIndependentSessions;
    const bool referenceKnown =
        report.referenceUncertaintyP95Meters <= referenceLimit;
    const bool conditioningControlled =
        clamp01(input.conditioningScore) >=
        clamp01(input.minimumConditioningScore);
    const bool calibrationControlled =
        finiteNonNegative(input.calibrationUncertaintyMeters) <=
        finiteNonNegative(input.maximumCalibrationUncertaintyMeters);
    const bool synchronizationControlled =
        finiteNonNegative(input.synchronizationUncertaintyMeters) <=
        finiteNonNegative(input.maximumSynchronizationUncertaintyMeters);
    const bool temporalControlled =
        finiteNonNegative(input.temporalUncertaintyMeters) <=
        finiteNonNegative(input.maximumTemporalUncertaintyMeters);
    const bool p95Qualified = report.p95ErrorMeters <= targetP95;
    const bool maxQualified = report.maxErrorMeters <= targetMax;
    const bool outlierQualified =
        report.outlierFraction <= clamp01(input.maximumOutlierFraction);

    if (!enoughSamples || !enoughSessions) {
        report.state = QualificationState::indeterminate;
        report.reason = "insufficient-reference-cohort";
    } else if (!referenceKnown) {
        report.state = QualificationState::indeterminate;
        report.reason = "reference-uncertainty-too-large";
    } else if (!conditioningControlled) {
        report.state = QualificationState::indeterminate;
        report.reason = "geometric-conditioning-below-threshold";
    } else if (!calibrationControlled) {
        report.state = QualificationState::indeterminate;
        report.reason = "calibration-uncertainty-too-large";
    } else if (!synchronizationControlled) {
        report.state = QualificationState::indeterminate;
        report.reason = "synchronization-uncertainty-too-large";
    } else if (!temporalControlled) {
        report.state = QualificationState::indeterminate;
        report.reason = "temporal-uncertainty-too-large";
    } else if (p95Qualified && maxQualified && outlierQualified &&
               report.combinedP95UncertaintyMeters <= targetP95) {
        report.state = QualificationState::qualified;
        report.reason =
            "measured-reference-and-live-uncertainty-contract-satisfied";
    } else {
        report.state = QualificationState::notQualified;
        report.reason = "measured-reference-contract-failed";
    }

    report.sampleConfidence = clamp01(
        static_cast<double>(errors.size()) /
        static_cast<double>(std::max<std::size_t>(input.minimumSamples, 1U)));
    report.referenceConfidence =
        referenceLimit <= 0.0
            ? 0.0
            : clamp01(
                1.0 -
                report.referenceUncertaintyP95Meters / referenceLimit);
    report.conditioningConfidence = clamp01(input.conditioningScore);

    const double residualConfidence =
        report.p95ErrorMeters <= 0.0
            ? 1.0
            : clamp01(targetP95 / report.p95ErrorMeters);
    const double uncertaintyConfidence =
        report.combinedP95UncertaintyMeters <= 0.0
            ? 1.0
            : clamp01(
                targetP95 /
                report.combinedP95UncertaintyMeters);

    report.confidence = clamp01(
        report.sampleConfidence * 0.20 +
        report.referenceConfidence * 0.20 +
        report.conditioningConfidence * 0.15 +
        residualConfidence * 0.20 +
        uncertaintyConfidence * 0.25);
    return report;
}

double MetrologyPrecisionQualification::percentile(
    std::vector<double> values,
    double quantile) noexcept {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());

    const double q = std::clamp(quantile, 0.0, 1.0);
    const double position =
        q * static_cast<double>(values.size() - 1U);
    const auto lower =
        static_cast<std::size_t>(std::floor(position));
    const auto upper =
        static_cast<std::size_t>(std::ceil(position));

    if (lower == upper) return values[lower];

    const double alpha =
        position - static_cast<double>(lower);
    return values[lower] * (1.0 - alpha) +
        values[upper] * alpha;
}

double MetrologyPrecisionQualification::clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

} // namespace nox::native::validation

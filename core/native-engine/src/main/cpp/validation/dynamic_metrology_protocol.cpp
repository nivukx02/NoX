#include "validation/dynamic_metrology_protocol.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace nox::native::validation {

namespace {

double clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

} // namespace

DynamicMetrologyProtocol::DynamicMetrologyProtocol(
    DynamicMetrologyConfig config) noexcept
    : config_(config) {
    config_.referenceLengthMeters =
        std::max(
            config_.referenceLengthMeters,
            1e-6);
    config_.maximumGapSeconds =
        std::clamp(
            config_.maximumGapSeconds,
            0.001,
            1.0);
    config_.targetP95ErrorMillimeters =
        std::max(
            config_.targetP95ErrorMillimeters,
            0.01);
    config_.targetRmseMillimeters =
        std::max(
            config_.targetRmseMillimeters,
            0.01);
    config_.targetMaxErrorMillimeters =
        std::max(
            config_.targetMaxErrorMillimeters,
            config_.targetP95ErrorMillimeters);
    config_.minimumConfidence =
        clamp01(
            config_.minimumConfidence);
}

double DynamicMetrologyProtocol::distance(
    double ax,
    double ay,
    double az,
    double bx,
    double by,
    double bz) noexcept {
    return std::hypot(
        std::hypot(
            ax - bx,
            ay - by),
        az - bz);
}

double DynamicMetrologyProtocol::percentile95(
    std::vector<double> values) noexcept {
    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](double value) {
                return
                    !std::isfinite(value) ||
                    value < 0.0;
            }),
        values.end());

    if (values.empty()) return 0.0;

    std::sort(
        values.begin(),
        values.end());

    const double position =
        0.95 *
        static_cast<double>(
            values.size() - 1U);

    const auto lower =
        static_cast<std::size_t>(
            std::floor(position));

    const auto upper =
        std::min(
            values.size() - 1U,
            lower + 1U);

    const double alpha =
        position -
        static_cast<double>(
            lower);

    return values[lower] +
        (
            values[upper] -
            values[lower]
        ) *
        alpha;
}

double DynamicMetrologyProtocol::slopePerMinute(
    const std::vector<double>& timesMinutes,
    const std::vector<double>& values) noexcept {
    if (
        timesMinutes.size() < 2U ||
        timesMinutes.size() !=
            values.size()) {
        return 0.0;
    }

    const double meanTime =
        std::accumulate(
            timesMinutes.begin(),
            timesMinutes.end(),
            0.0) /
        static_cast<double>(
            timesMinutes.size());

    const double meanValue =
        std::accumulate(
            values.begin(),
            values.end(),
            0.0) /
        static_cast<double>(
            values.size());

    double covariance = 0.0;
    double variance = 0.0;

    for (
        std::size_t index = 0;
        index < timesMinutes.size();
        ++index) {
        const double dt =
            timesMinutes[index] -
            meanTime;
        const double dv =
            values[index] -
            meanValue;
        covariance +=
            dt * dv;
        variance +=
            dt * dt;
    }

    if (
        variance <=
        1e-12) {
        return 0.0;
    }

    return covariance /
        variance;
}

DynamicMetrologyReport
DynamicMetrologyProtocol::evaluate(
    const std::vector<MetrologyEndpointSample>& samples) const {
    DynamicMetrologyReport report;
    report.sampleCount =
        samples.size();

    if (
        samples.empty()) {
        return report;
    }

    std::vector<double> lengths;
    std::vector<double> absoluteErrors;
    std::vector<double> timesMinutes;

    lengths.reserve(
        samples.size());
    absoluteErrors.reserve(
        samples.size());
    timesMinutes.reserve(
        samples.size());

    double sumLength = 0.0;
    double sumSquaredError = 0.0;
    double sumAbsoluteError = 0.0;
    double confidenceSum = 0.0;

    std::uint64_t previousTimestamp = 0ULL;
    std::uint64_t firstAcceptedTimestamp = 0ULL;

    for (const auto& sample :
         samples) {
        if (
            sample.timestampNanos == 0ULL ||
            !std::isfinite(sample.ax) ||
            !std::isfinite(sample.ay) ||
            !std::isfinite(sample.az) ||
            !std::isfinite(sample.bx) ||
            !std::isfinite(sample.by) ||
            !std::isfinite(sample.bz) ||
            !std::isfinite(sample.confidence) ||
            sample.confidence <
                config_.minimumConfidence) {
            ++report.rejectedSamples;
            continue;
        }

        if (
            previousTimestamp != 0ULL &&
            sample.timestampNanos <=
                previousTimestamp) {
            ++report.rejectedSamples;
            continue;
        }

        const double length =
            distance(
                sample.ax,
                sample.ay,
                sample.az,
                sample.bx,
                sample.by,
                sample.bz);

        if (
            !std::isfinite(length) ||
            length <= 1e-6) {
            ++report.rejectedSamples;
            continue;
        }

        if (
            previousTimestamp != 0ULL) {
            const double gap =
                static_cast<double>(
                    sample.timestampNanos -
                    previousTimestamp) /
                1e9;

            if (
                gap >
                config_.maximumGapSeconds) {
                previousTimestamp =
                    sample.timestampNanos;
                ++report.rejectedSamples;
                continue;
            }
        }

        previousTimestamp =
            sample.timestampNanos;

        if (
            firstAcceptedTimestamp == 0ULL) {
            firstAcceptedTimestamp =
                sample.timestampNanos;
        }

        const double error =
            length -
            config_.referenceLengthMeters;

        lengths.push_back(
            length);
        absoluteErrors.push_back(
            std::abs(error));

        timesMinutes.push_back(
            static_cast<double>(
                sample.timestampNanos -
                firstAcceptedTimestamp) /
            60e9);

        sumLength +=
            length;
        sumSquaredError +=
            error * error;
        sumAbsoluteError +=
            std::abs(error);
        confidenceSum +=
            clamp01(
                sample.confidence);

        ++report.acceptedSamples;
    }

    if (
        report.acceptedSamples == 0U) {
        return report;
    }

    const double count =
        static_cast<double>(
            report.acceptedSamples);

    report.meanLengthMeters =
        sumLength /
        count;

    report.truenessMillimeters =
        std::abs(
            report.meanLengthMeters -
            config_.referenceLengthMeters) *
        1000.0;

    report.meanAbsoluteErrorMillimeters =
        sumAbsoluteError /
        count *
        1000.0;

    report.p95ErrorMillimeters =
        percentile95(
            absoluteErrors) *
        1000.0;

    report.maxErrorMillimeters =
        absoluteErrors.empty()
            ? 0.0
            : *std::max_element(
                absoluteErrors.begin(),
                absoluteErrors.end()) *
                1000.0;

    report.rmseMillimeters =
        std::sqrt(
            sumSquaredError /
            count) *
        1000.0;

    std::vector<double> deltaErrors;
    deltaErrors.reserve(
        absoluteErrors.size());

    for (
        std::size_t index = 1U;
        index < absoluteErrors.size();
        ++index) {
        deltaErrors.push_back(
            absoluteErrors[index] -
            absoluteErrors[index - 1U]);
    }

    double jitterSquared = 0.0;

    for (const double delta :
         deltaErrors) {
        jitterSquared +=
            delta * delta;
    }

    report.dynamicJitterMillimeters =
        deltaErrors.empty()
            ? 0.0
            : std::sqrt(
                jitterSquared /
                static_cast<double>(
                    deltaErrors.size())) *
                1000.0;

    std::vector<double> signedErrorMillimeters;
    signedErrorMillimeters.reserve(
        lengths.size());

    for (const double length :
         lengths) {
        signedErrorMillimeters.push_back(
            (
                length -
                config_.referenceLengthMeters
            ) *
            1000.0);
    }

    report.driftMillimetersPerMinute =
        slopePerMinute(
            timesMinutes,
            signedErrorMillimeters);

    const double coverage =
        clamp01(
            static_cast<double>(
                report.acceptedSamples) /
            static_cast<double>(
                report.sampleCount));

    const double precisionScore =
        std::exp(
            -report.p95ErrorMillimeters /
            std::max(
                config_.targetP95ErrorMillimeters,
                1e-6));

    const double rmseScore =
        std::exp(
            -report.rmseMillimeters /
            std::max(
                config_.targetRmseMillimeters,
                1e-6));

    const double maxScore =
        std::exp(
            -report.maxErrorMillimeters /
            std::max(
                config_.targetMaxErrorMillimeters,
                1e-6));

    const double confidenceScore =
        clamp01(
            confidenceSum /
            count);

    report.confidence =
        clamp01(
            coverage * 0.20 +
            precisionScore * 0.35 +
            rmseScore * 0.20 +
            maxScore * 0.15 +
            confidenceScore * 0.10);

    report.dynamicCoverageValid =
        report.acceptedSamples >= 10U &&
        coverage >= 0.80;

    report.passesTarget =
        report.dynamicCoverageValid &&
        report.p95ErrorMillimeters <=
            config_.targetP95ErrorMillimeters &&
        report.rmseMillimeters <=
            config_.targetRmseMillimeters &&
        report.maxErrorMillimeters <=
            config_.targetMaxErrorMillimeters;

    return report;
}

} // namespace nox::native::validation

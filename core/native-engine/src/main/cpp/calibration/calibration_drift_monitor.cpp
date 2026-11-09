#include "calibration/calibration_drift_monitor.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nox::native::calibration {
namespace {

constexpr double kNanosPerMinute = 60'000'000'000.0;

} // namespace

CalibrationDriftMonitor::CalibrationDriftMonitor() noexcept
    : CalibrationDriftMonitor(Config{}) {}

CalibrationDriftMonitor::CalibrationDriftMonitor(
    Config config) noexcept
    : config_(config) {
    config_.windowNanos = std::clamp<std::uint64_t>(
        config_.windowNanos,
        5'000'000'000ull,
        300'000'000'000ull);
    config_.minimumSamples = std::clamp<std::uint64_t>(
        config_.minimumSamples,
        3ull,
        128ull);
    config_.maximumResidualSlopePxPerMinute =
        std::max(1e-6, config_.maximumResidualSlopePxPerMinute);
    config_.maximumTemperatureResidualSlopePxPerCelsius =
        std::max(
            1e-9,
            config_.maximumTemperatureResidualSlopePxPerCelsius);
    config_.maximumMetricScaleDriftPerMinute =
        std::max(1e-9, config_.maximumMetricScaleDriftPerMinute);
    config_.maximumTemperatureDeltaCelsius =
        std::max(0.5, config_.maximumTemperatureDeltaCelsius);
    config_.residualInstabilityFloorPxPerMinute =
        std::max(1e-6, config_.residualInstabilityFloorPxPerMinute);
    config_.minimumAuthoritativeConfidence =
        std::clamp(config_.minimumAuthoritativeConfidence, 0.0, 1.0);
    config_.recommendationScore =
        std::clamp(config_.recommendationScore, 0.0, 1.0);
    config_.requiredScore =
        std::clamp(
            std::min(config_.requiredScore, config_.recommendationScore),
            0.0,
            1.0);
}

void CalibrationDriftMonitor::reset() noexcept {
    samples_.clear();
}

bool CalibrationDriftMonitor::valid(
    const CalibrationDriftSample& sample) noexcept {
    return sample.timestampNanos != 0 &&
        std::isfinite(sample.reprojectionRmsPx) &&
        sample.reprojectionRmsPx >= 0.0 &&
        std::isfinite(sample.confidence) &&
        sample.confidence >= 0.0 &&
        sample.confidence <= 1.0 &&
        std::isfinite(sample.imageCoverage) &&
        sample.imageCoverage >= 0.0 &&
        sample.imageCoverage <= 1.0 &&
        std::isfinite(sample.depthCoverage) &&
        sample.depthCoverage >= 0.0 &&
        sample.depthCoverage <= 1.0 &&
        std::isfinite(sample.metricScale) &&
        sample.metricScale > 1e-9 &&
        std::isfinite(sample.metricScaleConfidence) &&
        sample.metricScaleConfidence >= 0.0 &&
        sample.metricScaleConfidence <= 1.0 &&
        std::isfinite(sample.temperatureCelsius);
}

void CalibrationDriftMonitor::prune(
    std::uint64_t newestTimestamp) noexcept {
    while (!samples_.empty()) {
        const auto oldest = samples_.front().timestampNanos;
        if (newestTimestamp <= oldest ||
            newestTimestamp - oldest <= config_.windowNanos) {
            break;
        }
        samples_.pop_front();
    }
}

double CalibrationDriftMonitor::mean(
    const std::vector<double>& values) noexcept {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (const double value : values) sum += value;
    return sum / static_cast<double>(values.size());
}

double CalibrationDriftMonitor::percentile(
    std::vector<double> values,
    double p) noexcept {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position =
        std::clamp(p, 0.0, 1.0) *
        static_cast<double>(values.size() - 1U);
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

double temperatureSlopeAndCorrelation(
    const std::vector<double>& timesMinutes,
    const std::vector<double>& temperatures,
    const std::vector<double>& residuals,
    double* correlation) noexcept {
    if (
        timesMinutes.size() < 3U ||
        temperatures.size() != timesMinutes.size() ||
        residuals.size() != timesMinutes.size()
    ) {
        if (correlation != nullptr) *correlation = 0.0;
        return 0.0;
    }

    auto vectorMean =
        [](const std::vector<double>& values) noexcept {
            if (values.empty()) return 0.0;
            double sum = 0.0;
            for (const double value : values) {
                sum += value;
            }
            return sum /
                static_cast<double>(values.size());
        };

    const double meanTime = vectorMean(timesMinutes);
    const double meanTemperature =
        vectorMean(temperatures);
    const double meanResidual =
        vectorMean(residuals);

    double sTT = 0.0;
    double sTZ = 0.0;
    double sZZ = 0.0;
    double sTY = 0.0;
    double sZY = 0.0;
    double sYY = 0.0;

    for (std::size_t index = 0; index < timesMinutes.size(); ++index) {
        const double t =
            timesMinutes[index] - meanTime;
        const double z =
            temperatures[index] - meanTemperature;
        const double y =
            residuals[index] - meanResidual;

        sTT += t * t;
        sTZ += t * z;
        sZZ += z * z;
        sTY += t * y;
        sZY += z * y;
        sYY += y * y;
    }

    constexpr double kRidge = 1e-8;
    const double determinant =
        (sTT + kRidge) * (sZZ + kRidge) -
        sTZ * sTZ;

    if (determinant <= 1e-12) {
        if (correlation != nullptr) *correlation = 0.0;
        return 0.0;
    }

    const double temperatureSlope =
        (
            sZY * (sTT + kRidge) -
            sTY * sTZ
        ) /
        determinant;

    if (correlation != nullptr) {
        const double denominator =
            std::sqrt(
                std::max(0.0, sZZ * sYY));
        *correlation =
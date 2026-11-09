#pragma once

#include <cstdint>
#include <deque>
#include <vector>

namespace nox::native::calibration {

struct CalibrationDriftSample {
    std::uint64_t timestampNanos{0};
    double reprojectionRmsPx{0.0};
    double confidence{0.0};
    double imageCoverage{0.0};
    double depthCoverage{0.0};
    double metricScale{1.0};
    double metricScaleConfidence{0.0};
    double temperatureCelsius{25.0};
};

struct CalibrationDriftResult {
    bool authoritative{false};
    bool stable{false};
    bool recalibrationRecommended{false};
    bool recalibrationRequired{false};
    std::uint64_t sampleCount{0};
    double residualSlopePxPerMinute{0.0};
    double temperatureResidualSlopePxPerCelsius{0.0};
    double residualTemperatureCorrelation{0.0};
    double residualP95Px{0.0};
    double metricScaleDriftPerMinute{0.0};
    double temperatureDeltaCelsius{0.0};
    double confidenceTrendPerMinute{0.0};
    double stabilityScore{0.0};
};

class CalibrationDriftMonitor final {
public:
    struct Config {
        std::uint64_t windowNanos{30'000'000'000ull};
        std::uint64_t minimumSamples{6};
        double maximumResidualSlopePxPerMinute{0.25};
        double maximumTemperatureResidualSlopePxPerCelsius{0.08};
        double maximumMetricScaleDriftPerMinute{0.005};
        double maximumTemperatureDeltaCelsius{8.0};
        double residualInstabilityFloorPxPerMinute{0.10};
        double minimumAuthoritativeConfidence{0.55};
        double recommendationScore{0.78};
        double requiredScore{0.48};
    };

    CalibrationDriftMonitor() noexcept;
    explicit CalibrationDriftMonitor(
        Config config) noexcept;

    void reset() noexcept;

    void observe(
        const CalibrationDriftSample& sample);

    [[nodiscard]] CalibrationDriftResult evaluate() const;

    [[nodiscard]] std::size_t size() const noexcept {
        return samples_.size();
    }

private:
    Config config_{};
    std::deque<CalibrationDriftSample> samples_{};

    void prune(
        std::uint64_t newestTimestamp) noexcept;

    static bool valid(
        const CalibrationDriftSample&) noexcept;

    static double slopePerMinute(
        const std::vector<double>& timesMinutes,
        const std::vector<double>& values) noexcept;

    static double percentile(
        std::vector<double> values,
        double p) noexcept;

    static double mean(
        const std::vector<double>& values) noexcept;
};

} // namespace nox::native::calibration

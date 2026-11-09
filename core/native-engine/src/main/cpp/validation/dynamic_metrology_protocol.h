#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nox::native::validation {

struct MetrologyEndpointSample {
    std::uint64_t timestampNanos{0};
    double ax{0.0};
    double ay{0.0};
    double az{0.0};
    double bx{0.0};
    double by{0.0};
    double bz{0.0};
    double confidence{1.0};
};

struct DynamicMetrologyConfig {
    double referenceLengthMeters{0.32088};
    double maximumGapSeconds{0.050};
    double targetP95ErrorMillimeters{1.0};
    double targetRmseMillimeters{0.75};
    double targetMaxErrorMillimeters{2.0};
    double minimumConfidence{0.50};
};

struct DynamicMetrologyReport {
    std::size_t sampleCount{0};
    std::size_t acceptedSamples{0};
    std::size_t rejectedSamples{0};
    double meanLengthMeters{0.0};
    double truenessMillimeters{0.0};
    double meanAbsoluteErrorMillimeters{0.0};
    double p95ErrorMillimeters{0.0};
    double maxErrorMillimeters{0.0};
    double rmseMillimeters{0.0};
    double dynamicJitterMillimeters{0.0};
    double driftMillimetersPerMinute{0.0};
    double confidence{0.0};
    bool dynamicCoverageValid{false};
    bool passesTarget{false};
};

class DynamicMetrologyProtocol final {
public:
    explicit DynamicMetrologyProtocol(
        DynamicMetrologyConfig config = {}) noexcept;

    [[nodiscard]] DynamicMetrologyReport evaluate(
        const std::vector<MetrologyEndpointSample>& samples) const;

private:
    DynamicMetrologyConfig config_{};

    static double distance(
        double ax,
        double ay,
        double az,
        double bx,
        double by,
        double bz) noexcept;

    static double percentile95(
        std::vector<double>) noexcept;

    static double slopePerMinute(
        const std::vector<double>&,
        const std::vector<double>&) noexcept;
};

} // namespace nox::native::validation

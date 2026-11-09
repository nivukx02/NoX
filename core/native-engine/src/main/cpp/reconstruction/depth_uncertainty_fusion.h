#pragma once

#include <cstdint>

#include <cstddef>
#include <vector>

namespace nox::native::reconstruction {

struct DepthEstimate {
    double depth{0.0};
    double variance{1.0};
    double confidence{0.0};
    double biasMeters{0.0};
    double temporalConsistency{1.0};
    double geometricConsistency{1.0};
    double sensorCorrelation{0.0};
};

struct DepthFusionResult {
    double depth{0.0};
    double variance{1.0};
    double confidence{0.0};
    double p95ResidualMeters{0.0};
    double consensus{0.0};
    double effectiveSampleCount{0.0};
    double correlationPenalty{0.0};
    std::size_t acceptedSamples{0};
    std::size_t rejectedSamples{0};
    bool valid{false};
};

class DepthUncertaintyFusion final {
public:
    [[nodiscard]] DepthEstimate fuse(
        const DepthEstimate&,
        const DepthEstimate&) const noexcept;

    [[nodiscard]] DepthFusionResult fuseRobust(
        const std::vector<DepthEstimate>&,
        double huberThresholdSigma = 2.5,
        double maximumDepthResidualMeters = 0.20) const;

private:
    static double clamp01(double value) noexcept;
    static double median(
        std::vector<double> values) noexcept;
    static double percentile95(
        std::vector<double> values) noexcept;
};

} // namespace nox::native::reconstruction

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nox::native::sync {

struct MotionTemporalSample {
    std::uint64_t timestampNanos{0};
    double motionEnergy{0.0};
    double confidence{1.0};
};

struct MotionTemporalAlignmentResult {
    double offsetNanos{0.0};
    double offsetSamples{0.0};
    double coarseOffsetSamples{0.0};
    double confidence{0.0};
    double peakCorrelation{0.0};
    double sideLobeCorrelation{0.0};
    double sharpness{0.0};
    double supportRatio{0.0};
    std::size_t evaluatedLags{0};
    bool valid{false};
};

struct MotionTemporalAlignmentOptions {
    std::size_t minimumSamples{12U};
    std::size_t maximumLags{2048U};
    double minimumCorrelation{0.45};
    double minimumConfidence{0.60};
    double minimumPeakSeparation{0.07};
    bool differentiateSignal{true};
};

class MotionTemporalAlignment final {
public:
    [[nodiscard]] MotionTemporalAlignmentResult estimate(
        const std::vector<MotionTemporalSample>& reference,
        const std::vector<MotionTemporalSample>& remote,
        double sampleIntervalNanos,
        const MotionTemporalAlignmentOptions& options = {}) const;

private:
    static double clamp01(double) noexcept;
    static double correlationAtLag(
        const std::vector<double>&,
        const std::vector<double>&,
        int lag,
        double& support) noexcept;
    static double parabolicOffset(
        double left,
        double center,
        double right) noexcept;
};

} // namespace nox::native::sync

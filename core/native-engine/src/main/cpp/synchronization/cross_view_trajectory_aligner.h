#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nox::native::sync {

struct TrajectorySample {
    std::uint64_t timestampNanos{0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double confidence{1.0};
};

struct TrajectoryAlignmentOptions {
    double maximumOffsetSeconds{0.250};
    double sampleIntervalSeconds{1.0 / 120.0};
    double minimumCorrelation{0.45};
    double velocitySmoothing{0.18};
    std::size_t minimumSamples{16};
};

struct TrajectoryAlignmentResult {
    double offsetSeconds{0.0};
    double offsetSamples{0.0};
    double correlation{0.0};
    double confidence{0.0};
    double curvature{0.0};
    double sideLobeCorrelation{0.0};
    double peakSeparation{0.0};
    double offsetUncertaintySeconds{0.0};
    double overlapRatio{0.0};
    std::size_t comparedSamples{0};
    bool valid{false};
};

class CrossViewTrajectoryAligner final {
public:
    explicit CrossViewTrajectoryAligner(
        TrajectoryAlignmentOptions options = {}) noexcept;

    [[nodiscard]] TrajectoryAlignmentResult align(
        const std::vector<TrajectorySample>& reference,
        const std::vector<TrajectorySample>& remote) const;

private:
    TrajectoryAlignmentOptions options_{};

    static double norm(
        double x,
        double y,
        double z) noexcept;

    static double clamp01(
        double value) noexcept;

    static std::vector<double> motionEnergy(
        const std::vector<TrajectorySample>&,
        double smoothing);

    static double correlationAt(
        const std::vector<double>&,
        const std::vector<double>&,
        int lag);

    static double subSampleOffset(
        double left,
        double center,
        double right) noexcept;
};

} // namespace nox::native::sync

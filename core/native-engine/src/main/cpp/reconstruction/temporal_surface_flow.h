#pragma once
#include <cstdint>
#include <vector>

namespace nox::native::reconstruction {

struct SurfaceSample {
    std::uint32_t id{0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double confidence{0.0};
};

struct SurfaceFlowSample {
    std::uint32_t sourceId{0};
    std::uint32_t targetId{0};
    double dx{0.0};
    double dy{0.0};
    double dz{0.0};
    double speedMetersPerSecond{0.0};
    double confidence{0.0};
    double residualMeters{0.0};
    bool matched{false};
};

struct SurfaceFlowSummary {
    double temporalStability{0.0};
    double meanSpeedMetersPerSecond{0.0};
    double p95ResidualMeters{0.0};
    double correspondenceCoverage{0.0};
    double confidence{0.0};
    std::uint32_t previousCount{0};
    std::uint32_t currentCount{0};
    std::uint32_t matchedCount{0};
    std::uint32_t rejectedCount{0};
    std::uint32_t ambiguousMatchCount{0};
    double mutualConsistency{0.0};
    double correspondenceUniqueness{0.0};
    std::vector<SurfaceFlowSample> samples;
};

class TemporalSurfaceFlow final {
public:
    struct Config {
        double maximumMatchDistanceMeters{0.075};
        double maximumSpeedMetersPerSecond{20.0};
        double confidenceFloor{0.15};
        bool requireMutualBestMatch{true};
        double mutualDistanceSlack{1.35};
        double minimumUniquenessRatio{0.10};
        std::uint32_t maximumSamples{4096};
    };

    TemporalSurfaceFlow() noexcept;
    explicit TemporalSurfaceFlow(Config config) noexcept;

    [[nodiscard]] SurfaceFlowSummary estimate(
        const std::vector<SurfaceSample>& previous,
        const std::vector<SurfaceSample>& current,
        double deltaSeconds) const noexcept;

private:
    Config config_{};
    [[nodiscard]] double distanceSquared(
        const SurfaceSample& a,
        const SurfaceSample& b) const noexcept;
    [[nodiscard]] double clamp01(double value) const noexcept;
};

} // namespace nox::native::reconstruction

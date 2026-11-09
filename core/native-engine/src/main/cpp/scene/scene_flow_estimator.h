#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nox::native::scene {

struct Flow {
    double dx{0.0};
    double dy{0.0};
    double dz{0.0};
    double confidence{0.0};
    double speedMetersPerSecond{0.0};
    double accelerationMetersPerSecond2{0.0};
    double uncertaintyMeters{1.0};
    double directionalConsistency{1.0};
    double innovationMeters{0.0};
    bool outlier{false};
};

struct FlowSample {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    std::uint64_t timestampNanos{0};
    double confidence{1.0};
};

struct SceneFlowOptions {
    double maximumSpeedMetersPerSecond{12.0};
    double maximumAccelerationMetersPerSecond2{80.0};
    double maximumInnovationMeters{0.15};
    double uncertaintyFloorMeters{0.0005};
    double uncertaintyCeilingMeters{0.25};
    double minimumConfidence{0.05};
    double outlierRejectionStrength{0.90};
};

struct SceneFlowSequenceResult {
    std::vector<Flow> samples;
    double meanSpeedMetersPerSecond{0.0};
    double p95SpeedMetersPerSecond{0.0};
    double p95AccelerationMetersPerSecond2{0.0};
    double p95InnovationMeters{0.0};
    double meanUncertaintyMeters{0.0};
    double temporalStability{0.0};
    double confidence{0.0};
    std::size_t rejectedSamples{0};
};

class SceneFlowEstimator final {
public:
    [[nodiscard]] Flow estimate(
        double px,
        double py,
        double pz,
        double cx,
        double cy,
        double cz,
        double dt) const noexcept;

    [[nodiscard]] Flow estimateHighEnd(
        const FlowSample& previous,
        const FlowSample& current,
        const Flow* priorVelocity = nullptr,
        const SceneFlowOptions& options = {}) const noexcept;

    [[nodiscard]] SceneFlowSequenceResult estimateSequence(
        const std::vector<FlowSample>& samples,
        const SceneFlowOptions& options = {}) const;
};

} // namespace nox::native::scene

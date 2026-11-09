#pragma once
#include <cstdint>
#include <optional>
#include <vector>

namespace nox::native::hands {

struct Vec3 {
    double x{0}, y{0}, z{0};
};

struct Joint {
    std::uint32_t id{0};
    Vec3 position{};
    double confidence{0};
};

struct HandTemporalOptions {
    double maximumJointVelocityMetersPerSecond{8.0};
    double maximumInnovationMeters{0.12};
    double innovationScaleMeters{0.02};
    double minimumTemporalConfidence{0.35};
    double maximumDtSeconds{0.25};
};

struct HandState {
    Vec3 wrist{};
    Vec3 palm{};
    Vec3 palmNormal{0,0,1};
    std::vector<Joint> joints;
    double handScale{0};
    double metricScaleMetersPerUnit{0};
    double wristCouplingError{0};
    double confidence{0};
    double occlusionRatio{1};
    double kinematicConsistency{0};
    double p95LengthResidual{0};
    double planarityConfidence{0};
    double metricConfidence{0};
    double wristCouplingConfidence{0};
    double temporalConfidence{0};
    double temporalConsistency{0};
    double meanInnovationMeters{0};
    double p95InnovationMeters{0};
    double maximumInnovationMeters{0};
    double estimatedJointVelocityMetersPerSecond{0};
    bool temporalRejected{false};
    bool metric{false};
    bool accepted{false};
};

class HandGeometryFusion final {
public:
    HandState solve(
        const std::vector<Joint>& joints,
        const Vec3& wrist,
        const Vec3& bodyWrist,
        double bodyCouplingConfidence) const;

    HandState solve(
        const std::vector<Joint>& joints,
        const Vec3& wrist,
        const Vec3& bodyWrist,
        double bodyCouplingConfidence,
        std::optional<double> metricScaleMetersPerUnit) const;

    HandState solveTemporal(
        const std::vector<Joint>& joints,
        const Vec3& wrist,
        const Vec3& bodyWrist,
        double bodyCouplingConfidence,
        const HandState& previous,
        double dtSeconds,
        const HandTemporalOptions& options = {},
        std::optional<double> metricScaleMetersPerUnit = std::nullopt) const;
};

} // namespace nox::native::hands

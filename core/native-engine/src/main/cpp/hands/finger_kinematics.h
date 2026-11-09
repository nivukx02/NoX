#pragma once
#include <cstdint>
#include <vector>
#include "hands/hand_geometry_fusion.h"
namespace nox::native::hands{
struct FingerJointRotation{
    std::uint32_t id{0};
    double bendRadians{0};
    double uncertainty{1};
    double lengthResidualRatio{0.0};
    double angularVelocityRadiansPerSecond{0.0};
    double temporalConfidence{0.0};
    bool withinLimits{false};
};
struct FingerChain{
    std::vector<std::uint32_t> jointIndices;
    double minBendRadians{0.0};
    double maxBendRadians{2.8};
};
class FingerKinematics final{
public:
    std::vector<FingerJointRotation> solve(
        const std::vector<Joint>& joints,
        const std::vector<double>& preferredLengths) const;
    std::vector<FingerJointRotation> solveChains(
        const std::vector<Joint>& joints,
        const std::vector<FingerChain>& chains) const;

    std::vector<FingerJointRotation> solveTemporal(
        const std::vector<Joint>& joints,
        const std::vector<FingerJointRotation>& previous,
        double dtSeconds,
        const std::vector<FingerChain>& chains = {}) const;
};
}
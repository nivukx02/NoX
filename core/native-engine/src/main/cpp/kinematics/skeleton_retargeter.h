#pragma once
#include <cstdint>
#include <vector>
namespace nox::native::kinematics {
struct JointPair { std::uint32_t source{0}, target{0}; double lengthScale{1}; };
struct RetargetResult {
    std::vector<double> weights;
    double meanError{0};
    double confidence{0};
    double globalScale{1.0};
    double mappingCoverage{0.0};
    bool valid{false};
};
class SkeletonRetargeter final { public: RetargetResult solve(const std::vector<double>& sourceLengths,const std::vector<double>& targetLengths,const std::vector<JointPair>& mapping) const; };
}
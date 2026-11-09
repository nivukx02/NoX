#pragma once
#include <cstdint>
#include <vector>
#include "optimization/objective_terms.h"
#include "kinematics/joint_limit_projector.h"
#include "kinematics/contact_constraints.h"
namespace nox::native::optimization {
struct SolveResult {
    kinematics::Pose pose;
    double objective{0};
    std::uint32_t iterations{0};
    bool converged{false};
    ObjectiveBreakdown residuals;
};
class RobustSolver final {
public:
    SolveResult solve(const kinematics::Pose& initial,
                      const kinematics::Pose& observed,
                      const std::vector<kinematics::Bone>& bones,
                      std::uint32_t maxIterations=24,
                      double tolerance=1e-5,
                      const kinematics::JointLimitSet* limits=nullptr,
                      const kinematics::ContactConstraintSet* contacts=nullptr) const;
};
}
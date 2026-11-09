#include "optimization/robust_solver.h"
#include "kinematics/contact_constraints.h"
#include <algorithm>
#include <cmath>

namespace nox::native::optimization {
SolveResult RobustSolver::solve(
    const kinematics::Pose& initial,
    const kinematics::Pose& observed,
    const std::vector<kinematics::Bone>& bones,
    std::uint32_t maxIterations,
    double tolerance,
    const kinematics::JointLimitSet* limits,
    const kinematics::ContactConstraintSet* contacts) const {
    SolveResult output;
    output.pose = initial;
    double previous = 1e300;

    for (std::uint32_t iteration = 0; iteration < maxIterations; ++iteration) {
        const auto residuals =
            ObjectiveTerms{}.evaluate(output.pose, observed, bones);
        output.residuals = residuals;
        output.objective = residuals.total;
        output.iterations = iteration + 1;

        if (residuals.total <= tolerance) {
            output.converged = true;
            return output;
        }
        if (residuals.total > previous + 1e-9) break;
        previous = residuals.total;

        const double alpha =
            0.35 / (1.0 + 0.05 * static_cast<double>(iteration));
        for (std::size_t i = 0;
             i < output.pose.joints.size() && i < observed.joints.size();
             ++i) {
            output.pose.joints[i].x +=
                (observed.joints[i].x - output.pose.joints[i].x) * alpha;
            output.pose.joints[i].y +=
                (observed.joints[i].y - output.pose.joints[i].y) * alpha;
            output.pose.joints[i].z +=
                (observed.joints[i].z - output.pose.joints[i].z) * alpha;
        }

        for (std::size_t i = 0;
             i < output.pose.joints.size() && i < bones.size();
             ++i) {
            const int parent = bones[i].parent;
            if (parent < 0 ||
                static_cast<std::size_t>(parent) >= output.pose.joints.size()) {
                continue;
            }
            auto& parentJoint = output.pose.joints[parent];
            auto& joint = output.pose.joints[i];
            const double dx = joint.x - parentJoint.x;
            const double dy = joint.y - parentJoint.y;
            const double dz = joint.z - parentJoint.z;
            const double length = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (length < 1e-9) continue;
            const double correction =
                (bones[i].length - length) / length * 0.25;
            joint.x += dx * correction;
            joint.y += dy * correction;
            joint.z += dz * correction;
        }

        if (limits != nullptr) {
            kinematics::JointLimitProjector{}.project(
                output.pose,
                bones,
                *limits);
        }
        if (contacts != nullptr) {
            kinematics::ContactConstraints::project(
                output.pose,
                *contacts);
        }
    }
    return output;
}
}

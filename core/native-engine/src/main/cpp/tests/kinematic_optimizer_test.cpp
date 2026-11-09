#include <cassert>
#include "kinematics/skeleton_model.h"
#include "optimization/robust_solver.h"

int main() {
    using namespace nox::native::kinematics;
    using namespace nox::native::optimization;
    const std::vector<Bone> bones={{-1,0.0},{0,1.0},{1,1.0}};
    const std::vector<Vec3> directions={{0,1,0},{1,0,0},{1,0,0}};
    const auto observed=SkeletonModel{}.forwardKinematics({0,0,0},directions,bones);
    auto initial=observed;
    initial.joints[2].x+=0.2;

    const JointLimitSet limits{
        {{2,{0,1,0},0.0,1.6,0.02}},
    };
    const ContactConstraintSet contacts{{{2,0.0,1.0,0.02,true}}};
    const auto result=RobustSolver{}.solve(
        initial,
        observed,
        bones,
        24,
        1e-5,
        &limits,
        &contacts);

    assert(result.iterations>0);
    assert(result.objective<0.05);
    assert(result.converged||result.objective<0.02);
    assert(result.residuals.bone<0.05);

    Pose contactPose=observed;
    contactPose.joints[2].y=-0.04;
    ContactConstraints::project(contactPose,contacts);
    assert(contactPose.joints[2].y>=-1e-9);
    return 0;
}

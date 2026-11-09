#pragma once
#include <vector>
#include "kinematics/skeleton_model.h"

namespace nox::native::kinematics {

struct JointLimit {
    int joint{-1};
    Vec3 referenceAxis{0.0,1.0,0.0};
    double minAngleRadians{0.0};
    double maxAngleRadians{3.141592653589793};
    double toleranceRadians{0.02};
};

struct JointLimitSet {
    std::vector<JointLimit> limits;
};

class JointLimitProjector final {
public:
    void project(
        Pose& pose,
        const std::vector<Bone>& bones,
        const JointLimitSet& limits) const noexcept;
};

} // namespace nox::native::kinematics

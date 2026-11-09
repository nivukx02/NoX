#pragma once
#include "kinematics/skeleton_model.h"
#include <vector>
namespace nox::native::kinematics {
struct ContactConstraint {
    int joint{-1};
    double floorHeight{0.0};
    double stiffness{1.0};
    double tolerance{0.015};
    bool projectToPlane{true};
    bool hasAnchor{false};
    SkeletonModel::Vec3 anchor{};
    double anchorStiffness{0.0};
    double anchorTolerance{0.02};
};

struct ContactConstraintSet {
    std::vector<ContactConstraint> contacts;
};
class ContactConstraints final {
public:
    static double penalty(const Pose&,const ContactConstraint&) noexcept;
    static void project(Pose&,const ContactConstraintSet&) noexcept;
};
}
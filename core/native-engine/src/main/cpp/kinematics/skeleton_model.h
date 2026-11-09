#pragma once
#include <vector>
namespace nox::native::kinematics {
struct Vec3 { double x{0}, y{0}, z{0}; };
struct Bone { int parent{-1}; double length{1}; };
struct Pose { std::vector<Vec3> joints; };
class SkeletonModel final {
public:
    using Vec3 = nox::native::kinematics::Vec3;
    Pose forwardKinematics(const Vec3& root, const std::vector<Vec3>& directions,
                           const std::vector<Bone>& bones) const;
};
}
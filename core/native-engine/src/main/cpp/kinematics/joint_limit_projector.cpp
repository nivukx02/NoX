#include "kinematics/joint_limit_projector.h"

#include <algorithm>
#include <cmath>

namespace nox::native::kinematics {
namespace {
double dot(Vec3 a,Vec3 b) noexcept {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
double norm(Vec3 v) noexcept {
    return std::sqrt(dot(v,v));
}
Vec3 normalize(Vec3 v) noexcept {
    const double n=norm(v);
    if(n<1e-9)return {0.0,1.0,0.0};
    return {v.x/n,v.y/n,v.z/n};
}
Vec3 cross(Vec3 a,Vec3 b) noexcept {
    return {
        a.y*b.z-a.z*b.y,
        a.z*b.x-a.x*b.z,
        a.x*b.y-a.y*b.x,
    };
}
Vec3 add(Vec3 a,Vec3 b) noexcept {
    return {a.x+b.x,a.y+b.y,a.z+b.z};
}
Vec3 scale(Vec3 v,double s) noexcept {
    return {v.x*s,v.y*s,v.z*s};
}
Vec3 blendDirection(Vec3 from,Vec3 to,double alpha) noexcept {
    return normalize(add(
        scale(normalize(from),1.0-alpha),
        scale(normalize(to),alpha)));
}
}

void JointLimitProjector::project(
    Pose& pose,
    const std::vector<Bone>& bones,
    const JointLimitSet& limits) const noexcept {
    for(const auto& limit:limits.limits){
        if(limit.joint<0||
           static_cast<std::size_t>(limit.joint)>=pose.joints.size()||
           static_cast<std::size_t>(limit.joint)>=bones.size())continue;

        const int parent=bones[limit.joint].parent;
        if(parent<0||
           static_cast<std::size_t>(parent)>=pose.joints.size())continue;

        const Vec3 delta{
            pose.joints[limit.joint].x-pose.joints[parent].x,
            pose.joints[limit.joint].y-pose.joints[parent].y,
            pose.joints[limit.joint].z-pose.joints[parent].z,
        };
        const double length=norm(delta);
        if(length<1e-9)continue;

        const Vec3 direction=normalize(delta);
        const Vec3 axis=normalize(limit.referenceAxis);
        const double angle=std::acos(
            std::clamp(dot(direction,axis),-1.0,1.0));

        double targetAngle=angle;
        if(angle<limit.minAngleRadians)targetAngle=limit.minAngleRadians;
        if(angle>limit.maxAngleRadians)targetAngle=limit.maxAngleRadians;
        if(std::abs(targetAngle-angle)<=limit.toleranceRadians)continue;

        Vec3 tangent=cross(axis,direction);
        if(norm(tangent)<1e-9){
            tangent=normalize(cross(
                axis,
                std::abs(axis.x)<0.9
                    ? Vec3{1,0,0}
                    : Vec3{0,1,0}));
        } else {
            tangent=normalize(tangent);
        }

        const Vec3 targetDirection=normalize(add(
            scale(axis,std::cos(targetAngle)),
            scale(tangent,std::sin(targetAngle))));
        const double blend=std::clamp(
            std::abs(targetAngle-angle)/
                std::max(limit.maxAngleRadians,0.1),
            0.0,
            1.0);
        const Vec3 projected=blendDirection(
            direction,
            targetDirection,
            std::max(0.35,blend));
        pose.joints[limit.joint]=add(
            pose.joints[parent],
            scale(projected,bones[limit.joint].length));
    }
}

} // namespace nox::native::kinematics

#include <algorithm>
#include <cassert>
#include <cmath>
#include "kinematics/joint_limit_projector.h"

int main(){
    using namespace nox::native::kinematics;
    const std::vector<Bone> bones={{-1,0.0},{0,1.0},{1,1.0}};
    Pose pose;
    pose.joints={{0,0,0},{0,1,0},{1,1,0}};

    JointLimitSet limits{
        {{2,{0,1,0},0.0,0.35,0.01}},
    };
    JointLimitProjector{}.project(pose,bones,limits);

    const Vec3 delta{
        pose.joints[2].x-pose.joints[1].x,
        pose.joints[2].y-pose.joints[1].y,
        pose.joints[2].z-pose.joints[1].z,
    };
    const double length=std::sqrt(
        delta.x*delta.x+delta.y*delta.y+delta.z*delta.z);
    const double angle=std::acos(
        std::clamp(delta.y/length,-1.0,1.0));
    assert(angle<0.36);
    assert(length>0.95&&length<1.05);
    return 0;
}

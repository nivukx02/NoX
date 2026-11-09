#include "kinematics/contact_constraints.h"
#include <algorithm>
#include <cmath>

namespace nox::native::kinematics {

double ContactConstraints::penalty(
    const Pose& pose,
    const ContactConstraint& constraint) noexcept {
    if(constraint.joint<0||
       static_cast<std::size_t>(constraint.joint)>=pose.joints.size()){
        return 0.0;
    }
    const double distance=
        pose.joints[constraint.joint].y-constraint.floorHeight;
    double penalty=constraint.stiffness*distance*distance;
    if(constraint.hasAnchor){
        const auto& joint=pose.joints[constraint.joint];
        const double dx=joint.x-constraint.anchor.x;
        const double dy=joint.y-constraint.anchor.y;
        const double dz=joint.z-constraint.anchor.z;
        const double anchorDistanceSquared=dx*dx+dy*dy+dz*dz;
        penalty+=constraint.anchorStiffness*anchorDistanceSquared;
    }
    return penalty;
}

void ContactConstraints::project(
    Pose& pose,
    const ContactConstraintSet& set) noexcept {
    for(const auto& constraint:set.contacts){
        if(!constraint.projectToPlane||
           constraint.joint<0||
           static_cast<std::size_t>(constraint.joint)>=pose.joints.size()){
            continue;
        }
        auto& joint=pose.joints[constraint.joint];
        const double distance=joint.y-constraint.floorHeight;
        if(distance<0.0){
            joint.y=constraint.floorHeight;
        }else if(distance<=constraint.tolerance){
            const double alpha=std::clamp(
                constraint.stiffness*0.5,0.0,1.0);
            joint.y=
                joint.y*(1.0-alpha)+
                constraint.floorHeight*alpha;
        }

        if(constraint.hasAnchor){
            const double dx=constraint.anchor.x-joint.x;
            const double dy=constraint.anchor.y-joint.y;
            const double dz=constraint.anchor.z-joint.z;
            const double anchorDistance=std::sqrt(
                dx*dx+dy*dy+dz*dz);
            if(anchorDistance>constraint.anchorTolerance){
                const double alpha=std::clamp(
                    constraint.anchorStiffness*0.5,0.0,1.0);
                joint.x+=dx*alpha;
                joint.y+=dy*alpha;
                joint.z+=dz*alpha;
            }else if(anchorDistance>0.0){
                const double alpha=std::clamp(
                    constraint.anchorStiffness,0.0,1.0);
                joint.x+=dx*alpha;
                joint.y+=dy*alpha;
                joint.z+=dz*alpha;
            }
        }
    }
}

} // namespace nox::native::kinematics

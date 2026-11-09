#include "kinematics/foot_lock_solver.h"

#include <algorithm>
#include <cmath>

namespace nox::native::kinematics {
namespace {
double distance(
    FootLockState::Vec3 a,
    FootLockState::Vec3 b) noexcept {
    const double dx=a.x-b.x;
    const double dy=a.y-b.y;
    const double dz=a.z-b.z;
    return std::sqrt(dx*dx+dy*dy+dz*dz);
}
}

FootLockState FootLockSolver::solve(
    double leftSpeed,
    double rightSpeed,
    double leftHeight,
    double rightHeight,
    double speedThreshold,
    double heightThreshold) const noexcept {
    FootLockState output;
    output.leftLocked=
        leftSpeed<speedThreshold&&
        std::abs(leftHeight)<heightThreshold;
    output.rightLocked=
        rightSpeed<speedThreshold&&
        std::abs(rightHeight)<heightThreshold;
    output.leftError=std::abs(leftHeight);
    output.rightError=std::abs(rightHeight);
    return output;
}

FootLockState FootLockSolver::updateAnchored(
    const FootLockState& previous,
    const FootLockState::Vec3& leftPosition,
    const FootLockState::Vec3& rightPosition,
    double leftSpeed,
    double rightSpeed,
    double leftHeight,
    double rightHeight,
    double speedThreshold,
    double heightThreshold,
    double releaseSpeedThreshold,
    double releaseHeightThreshold) const noexcept {
    FootLockState output=previous;
    output.leftError=std::abs(leftHeight);
    output.rightError=std::abs(rightHeight);

    const bool leftEligible=
        leftSpeed<speedThreshold&&
        std::abs(leftHeight)<heightThreshold;
    const bool rightEligible=
        rightSpeed<speedThreshold&&
        std::abs(rightHeight)<heightThreshold;

    if((!previous.leftLocked||!previous.leftHasAnchor)&&leftEligible){
        output.leftLocked=true;
        output.leftHasAnchor=true;
        output.leftAnchor=leftPosition;
    }else if(previous.leftLocked&&
             (leftSpeed>releaseSpeedThreshold||
              std::abs(leftHeight)>releaseHeightThreshold)){
        output.leftLocked=false;
        output.leftHasAnchor=false;
    }

    if((!previous.rightLocked||!previous.rightHasAnchor)&&rightEligible){
        output.rightLocked=true;
        output.rightHasAnchor=true;
        output.rightAnchor=rightPosition;
    }else if(previous.rightLocked&&
             (rightSpeed>releaseSpeedThreshold||
              std::abs(rightHeight)>releaseHeightThreshold)){
        output.rightLocked=false;
        output.rightHasAnchor=false;
    }

    if(output.leftLocked&&output.leftHasAnchor){
        output.leftAnchorDrift=
            distance(leftPosition,output.leftAnchor);
        output.leftCorrection={
            output.leftAnchor.x-leftPosition.x,
            output.leftAnchor.y-leftPosition.y,
            output.leftAnchor.z-leftPosition.z,
        };
    }else{
        output.leftAnchorDrift=0.0;
        output.leftCorrection={};
    }

    if(output.rightLocked&&output.rightHasAnchor){
        output.rightAnchorDrift=
            distance(rightPosition,output.rightAnchor);
        output.rightCorrection={
            output.rightAnchor.x-rightPosition.x,
            output.rightAnchor.y-rightPosition.y,
            output.rightAnchor.z-rightPosition.z,
        };
    }else{
        output.rightAnchorDrift=0.0;
        output.rightCorrection={};
    }

    return output;
}

} // namespace nox::native::kinematics

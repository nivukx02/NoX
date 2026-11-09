#include <cassert>
#include "tracking/body_tracker.h"
#include "tracking/camera_motion_compensator.h"

int main(){
    using namespace nox::native;
    const std::vector<tracking::JointObservation> observations={
        {1,{1,0,0},1.0},
        {2,{2,0,0},1.0},
    };
    const math::SE3d cameraToWorld={
        math::Quaterniond::fromRotationVector({0,0,0.5}),
        {10,0,0},
    };
    const auto world=tracking::CameraMotionCompensator{}.toWorld(
        observations,cameraToWorld);
    assert(world.size()==2);
    assert(world.front().position.x>10.0);

    const auto compensated =
        tracking::CameraMotionCompensator{}.toWorldHighEnd(
            observations,
            tracking::CameraMotionContext{
                .cameraToWorld = cameraToWorld,
                .linearVelocityWorld = {1.0,0.0,0.0},
                .angularVelocityWorld = {0.0,0.0,0.8},
                .linearAccelerationWorld = {2.0,0.0,0.0},
                .angularAccelerationWorld = {0.0,0.0,0.4},
                .captureOffsetSeconds = 0.010,
                .readoutDurationSeconds = 0.004,
                .normalizedSensorRow = 0.75,
                .positionUncertaintyMeters = 0.001,
                .angularUncertaintyRadians = 0.002,
                .motionConfidence = 0.98,
                .motionBlurRisk = 0.20,
            });

    assert(compensated.observations.size()==2);
    assert(compensated.compensated);
    assert(compensated.meanTranslationCorrectionMeters>0.0);
    assert(compensated.maximumTranslationCorrectionMeters >=
           compensated.meanTranslationCorrectionMeters);
    assert(compensated.angularCorrectionRadians>0.0);
    assert(compensated.meanMotionBlurRisk == 0.20);
    assert(compensated.confidence>0.70);

    tracking::BodyTracker tracker;
    const auto state=tracker.updateCameraCompensated(
        observations,
        1'000'000,
        cameraToWorld);
    assert(state.joints.size()==2);
    assert(state.joints.front().position.x>10.0);
    return 0;
}

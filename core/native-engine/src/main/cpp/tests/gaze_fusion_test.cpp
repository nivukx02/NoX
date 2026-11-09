#include <cassert>
#include <cmath>
#include <limits>
#include "eyes/gaze_fusion.h"

int main(){
    using namespace nox::native::eyes;

    EyeObservation leftObservation{
        {-0.03,0,1},
        {0.0,0,2},
        .95,
    };
    EyeObservation rightObservation{
        {0.03,0,1},
        {0.0,0,2},
        .95,
    };

    const auto left=EyeGeometrySolver{}.solve(
        leftObservation,{0,0,1});
    const auto right=EyeGeometrySolver{}.solve(
        rightObservation,{0,0,1});

    const auto forward=GazeFusion{}.update(
        left,right,{0,0,1},{0,1,0});
    assert(left.valid&&right.valid);
    assert(forward.consistent);
    assert(forward.binocular_confidence>.7);
    assert(forward.world_relative.z>0.95);
    assert(forward.angular_disagreement_degrees < 2.0);
    assert(forward.accepted);
    assert(forward.eye_uncertainty < 0.20);
    assert(forward.interpupillary_distance_meters > 0.05);
    assert(forward.ipd_quality > 0.70);
    assert(forward.failure_flags == GAZE_FAILURE_NONE);
    assert(forward.uncertainty_confidence > 0.75);
    assert(forward.fixation_uncertainty_meters >= 0.0);
    if (forward.strict_submillimeter_qualified) {
        assert(forward.fixation_uncertainty_meters <= 0.001);
    }
    assert(forward.parallax_quality >= 0.0);
    assert(forward.parallax_quality <= 1.0);

    GazeFusion fusion;
    const auto rotatedState =
        fusion.update(
            left,
            right,
            Vec3{1,0,0},
            Vec3{0,1,0});
    assert(rotatedState.world_relative.x>0.95);
    assert(std::abs(rotatedState.world_relative.z)<0.1);

    auto bad=EyeGeometrySolver{}.solve(
        {rightObservation.center,{-1,0,1},.9},
        {0,0,1});
    const auto divergent=GazeFusion{}.update(
        left,bad,{0,0,1},{0,1,0});
    assert(divergent.binocular_confidence<forward.binocular_confidence);
    assert(divergent.angular_disagreement_degrees > 20.0);
    assert(!divergent.accepted);
    assert(
        (divergent.failure_flags &
            GAZE_FAILURE_ANGULAR_DISAGREEMENT) != 0u);
    assert(
        (divergent.failure_flags &
            GAZE_FAILURE_LOW_BINOCULAR_CONFIDENCE) != 0u);

    const auto temporalBase =
        GazeFusion{}.updateTemporal(
            left,
            right,
            {0,0,1},
            {0,1,0},
            forward,
            1.0 / 120.0);
    assert(temporalBase.accepted);
    assert(
        temporalBase.temporal_residual_degrees <
        1.0);
    assert(
        temporalBase.angular_velocity_degrees_per_second <
        120.0);
    assert(
        temporalBase.temporal_confidence >
        0.70);

    auto fastEyeLeft = left;
    auto fastEyeRight = right;
    fastEyeLeft.gaze = {1.0, 0.0, 1.0};
    fastEyeRight.gaze = {1.0, 0.0, 1.0};

    const auto temporalJump =
        GazeFusion{}.updateTemporal(
            fastEyeLeft,
            fastEyeRight,
            {0,0,1},
            {0,1,0},
            forward,
            1.0 / 120.0);
    assert(
        temporalJump.angular_velocity_degrees_per_second >
        120.0);
    assert(!temporalJump.accepted);

    EyeGeometry nonFinite = right;
    nonFinite.gaze.x = std::numeric_limits<double>::quiet_NaN();
    const auto rejected =
        GazeFusion{}.update(
            left,
            nonFinite,
            {0,0,1},
            {0,1,0});
    assert(!rejected.accepted);
    assert(!rejected.consistent);
    assert(
        (rejected.failure_flags &
            GAZE_FAILURE_INVALID_EYE) != 0u);
    return 0;
}

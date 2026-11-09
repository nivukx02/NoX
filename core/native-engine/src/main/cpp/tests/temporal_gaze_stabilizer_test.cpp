#include <cassert>
#include <cmath>

#include "eyes/eye_geometry.h"
#include "eyes/temporal_gaze_stabilizer.h"

int main() {
    using namespace nox::native::eyes;

    EyeObservation leftObservation{
        {-0.03,0,1},
        {0.0,0,2},
        .98,
    };
    EyeObservation rightObservation{
        {0.03,0,1},
        {0.0,0,2},
        .98,
    };

    const auto left=
        EyeGeometrySolver{}.solve(
            leftObservation,
            {0,0,1});
    const auto right=
        EyeGeometrySolver{}.solve(
            rightObservation,
            {0,0,1});

    TemporalGazeStabilizer stabilizer;

    const auto first=
        stabilizer.update(
            1'000'000'000ull,
            left,
            right,
            {0,0,1},
            {0,1,0});

    const auto second=
        stabilizer.update(
            1'008'333'333ull,
            left,
            right,
            {0.03,0,0.99955},
            {0,1,0});

    assert(first.accepted);
    assert(second.accepted);
    assert(second.angular_velocity_degrees_per_second < 500.0);
    assert(second.temporal_residual_degrees < 5.0);
    assert(second.temporal_confidence > 0.65);

    const auto fast=
        stabilizer.update(
            1'016'666'666ull,
            left,
            right,
            {0.8,0,0.6},
            {0,1,0});

    assert(
        fast.angular_velocity_degrees_per_second >
        second.angular_velocity_degrees_per_second);
    assert(std::isfinite(
        fast.world_relative.x));
    assert(std::isfinite(
        fast.world_relative.y));
    assert(std::isfinite(
        fast.world_relative.z));

    stabilizer.reset();
    const auto reset=
        stabilizer.update(
            2'000'000'000ull,
            left,
            right,
            {0,0,1},
            {0,1,0});
    assert(reset.temporal_residual_degrees==0.0);

    return 0;
}

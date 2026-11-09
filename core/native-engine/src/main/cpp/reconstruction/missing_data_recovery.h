#pragma once
#include "tracking/occlusion_state.h"
namespace nox::native::recovery{
struct Vec3{double x{0},y{0},z{0};};
struct Track{
    Vec3 position{},velocity{};
    double confidence{0};
    double uncertainty{1};
    Vec3 acceleration{};
    Vec3 jerk{};
};
struct Result{Track track{};occlusion::Observation observation{};double gapSeconds{0};};
class MissingDataRecovery final{
public:
    Result predict(const Track&,double gapSeconds,double maxGapSeconds,double maxSpeedMetersPerSecond=12.0,
        double maxAccelerationMetersPerSecond2=60.0,
        double maxJerkMetersPerSecond3=240.0)const noexcept;
};
}
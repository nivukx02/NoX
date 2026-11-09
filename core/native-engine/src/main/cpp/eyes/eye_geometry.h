#pragma once
#include <cstdint>
namespace nox::native::eyes{
struct Vec3{double x{0},y{0},z{0};};
struct EyeObservation{
    Vec3 center{},
    iris{};
    double confidence{0};
    double irisDiameterMeters{0};
};

struct EyeGeometry{
    Vec3 center{},
    iris{},
    gaze{0,0,1};
    double confidence{0};
    bool valid{false};
    double gazeUncertaintyDegrees{180.0};
    double centerUncertaintyMeters{0.0};
    double irisQuality{0.0};
    double headAlignment{0.0};
};
class EyeGeometrySolver final{public:EyeGeometry solve(const EyeObservation& eye,const Vec3& headForward) const noexcept;};
}
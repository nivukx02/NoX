#pragma once
#include "eyes/eye_geometry.h"

namespace nox::native::eyes {

struct VergenceResult {
    Vec3 point{0,0,0};
    double distance{0};
    double raySeparation{0};
    double convergenceAngleDegrees{0};
    double parallaxDegrees{0};
    double depthAsymmetry{0};
    double angularResidualDegrees{0};
    double rayAlignmentConfidence{0};
    double conditioning{0};
    double uncertaintyMeters{1.0};
    double confidence{0};
    bool valid{false};
};

class BinocularVergenceSolver final {
public:
    VergenceResult solve(
        const EyeGeometry& left,
        const EyeGeometry& right) const noexcept;
};

} // namespace nox::native::eyes

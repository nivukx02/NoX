#include "capture/rolling_shutter_compensator.h"
namespace nox::native::capture {
Point2 RollingShutterCompensator::correct(
    Point2 p, double rowNormalized, double angularVelocity, double readoutSeconds) const noexcept {
    const double dt = (rowNormalized - 0.5) * readoutSeconds;
    const double scale = 1.0 - angularVelocity * dt;
    return {p.x * scale, p.y * scale};
}
}

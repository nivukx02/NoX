#include "scene/normal_estimator.h"
#include <cmath>

namespace nox::native::scene {
Vec3 NormalEstimator::estimate(
    const Point& a, const Point& b, const Point& c) noexcept {
    const double ux = b.p.x - a.p.x;
    const double uy = b.p.y - a.p.y;
    const double uz = b.p.z - a.p.z;
    const double vx = c.p.x - a.p.x;
    const double vy = c.p.y - a.p.y;
    const double vz = c.p.z - a.p.z;
    Vec3 normal{
        uy * vz - uz * vy,
        uz * vx - ux * vz,
        ux * vy - uy * vx,
    };
    const double magnitude = std::sqrt(
        normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (magnitude < 1e-9) return {0.0, 0.0, 1.0};
    return {
        normal.x / magnitude,
        normal.y / magnitude,
        normal.z / magnitude,
    };
}
}

#include "face/surface_visibility.h"
#include <algorithm>
#include <cmath>

namespace nox::native::face {
std::vector<SurfaceVertex> SurfaceVisibility::evaluate(
    const std::vector<SurfaceVertex>& input,
    double vx,
    double vy,
    double vz) const {
    auto output = input;
    const double norm = std::sqrt(vx * vx + vy * vy + vz * vz);
    if (norm < 1e-9) return output;
    vx /= norm;
    vy /= norm;
    vz /= norm;

    for (auto& vertex : output) {
        const double dot = vertex.nx * vx + vertex.ny * vy + vertex.nz * vz;
        vertex.visible = dot > 0.0;
        vertex.confidence *= vertex.visible
            ? std::clamp(dot, 0.0, 1.0)
            : 0.0;
    }
    return output;
}
}

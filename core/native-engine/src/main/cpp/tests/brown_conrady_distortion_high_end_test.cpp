#include <cassert>
#include <cmath>
#include <limits>

#include "calibration/brown_conrady_distortion.h"

int main() {
    using nox::native::calibration::BrownConradyCoefficients;
    using nox::native::calibration::BrownConradyDistortion;

    const BrownConradyCoefficients coefficients{
        .k1 = -0.24,
        .k2 = 0.08,
        .p1 = 0.0015,
        .p2 = -0.0010,
        .k3 = -0.015,
    };

    const double points[][2]{
        {0.00, 0.00},
        {0.15, -0.12},
        {0.42, 0.31},
        {-0.60, 0.48},
        {0.82, -0.67},
    };

    for (const auto& point : points) {
        const auto distorted =
            BrownConradyDistortion::distort(
                point[0],
                point[1],
                coefficients);

        const auto recovered =
            BrownConradyDistortion::undistort(
                distorted.x,
                distorted.y,
                coefficients,
                12u);

        assert(std::isfinite(recovered.x));
        assert(std::isfinite(recovered.y));

        const double error =
            std::hypot(
                recovered.x - point[0],
                recovered.y - point[1]);

        assert(error < 1e-8);
    }

    const auto invalid =
        BrownConradyDistortion::undistort(
            std::numeric_limits<double>::quiet_NaN(),
            0.25,
            coefficients,
            12u);

    assert(std::isfinite(invalid.x));
    assert(std::isfinite(invalid.y));

    return 0;
}

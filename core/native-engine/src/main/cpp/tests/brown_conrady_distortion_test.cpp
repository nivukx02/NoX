#include <cassert>
#include <cmath>

#include "calibration/brown_conrady_distortion.h"

int main() {
    using nox::native::calibration::BrownConradyCoefficients;
    using nox::native::calibration::BrownConradyDistortion;

    BrownConradyCoefficients coefficients{};
    coefficients.k1 = 0.31;
    coefficients.k2 = -0.19;
    coefficients.k3 = 0.08;
    coefficients.p1 = 0.012;
    coefficients.p2 = -0.009;

    const double x = 0.62;
    const double y = -0.47;

    const auto distorted =
        BrownConradyDistortion::distort(
            x,
            y,
            coefficients);

    const auto recovered =
        BrownConradyDistortion::undistort(
            distorted.x,
            distorted.y,
            coefficients,
            16u);

    assert(std::isfinite(recovered.x));
    assert(std::isfinite(recovered.y));
    assert(std::abs(recovered.x - x) < 1e-8);
    assert(std::abs(recovered.y - y) < 1e-8);

    BrownConradyCoefficients benign{};
    const auto identity =
        BrownConradyDistortion::undistort(
            0.2,
            -0.3,
            benign,
            4u);

    assert(std::abs(identity.x - 0.2) < 1e-12);
    assert(std::abs(identity.y + 0.3) < 1e-12);

    return 0;
}

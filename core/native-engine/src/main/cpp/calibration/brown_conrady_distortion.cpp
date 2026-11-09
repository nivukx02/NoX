#include "calibration/brown_conrady_distortion.h"

#include <algorithm>
#include <cmath>

namespace nox::native::calibration {

DistortedPixel BrownConradyDistortion::distort(
    double x,
    double y,
    const BrownConradyCoefficients& c) noexcept {
    const double r2 = x * x + y * y;
    const double radial = 1.0 + r2 * (c.k1 + r2 * (c.k2 + r2 * c.k3));
    const double x2 = x * x;
    const double y2 = y * y;
    const double xy = x * y;
    return {
        x * radial + 2.0 * c.p1 * xy + c.p2 * (r2 + 2.0 * x2),
        y * radial + c.p1 * (r2 + 2.0 * y2) + 2.0 * c.p2 * xy,
    };
}

DistortedPixel BrownConradyDistortion::undistort(
    double distorted_x,
    double distorted_y,
    const BrownConradyCoefficients& c,
    std::uint32_t iterations) noexcept {
    if (
        !std::isfinite(distorted_x) ||
        !std::isfinite(distorted_y)
    ) {
        return {
            0.0,
            0.0,
        };
    }

    double x = distorted_x;
    double y = distorted_y;
    const auto count =
        std::max<std::uint32_t>(
            iterations,
            1u);

    for (
        std::uint32_t i = 0;
        i < count;
        ++i
    ) {
        const auto predicted =
            distort(
                x,
                y,
                c);

        const double residualX =
            distorted_x -
            predicted.x;
        const double residualY =
            distorted_y -
            predicted.y;

        if (
            !std::isfinite(residualX) ||
            !std::isfinite(residualY)
        ) {
            return {
                0.0,
                0.0,
            };
        }

        const double r2 =
            x * x +
            y * y;
        const double radial =
            1.0 +
            r2 *
                (
                    c.k1 +
                    r2 *
                        (
                            c.k2 +
                            r2 * c.k3
                        )
                );
        const double radialSlope =
            c.k1 +
            2.0 * c.k2 * r2 +
            3.0 * c.k3 * r2 * r2;
        const double dRadialDx =
            2.0 * x *
            radialSlope;
        const double dRadialDy =
            2.0 * y *
            radialSlope;

        const double dTxDx =
            2.0 * c.p1 * y +
            6.0 * c.p2 * x;
        const double dTxDy =
            2.0 * c.p1 * x +
            2.0 * c.p2 * y;
        const double dTyDx =
            2.0 * c.p1 * x +
            2.0 * c.p2 * y;
        const double dTyDy =
            6.0 * c.p1 * y +
            2.0 * c.p2 * x;

        const double j00 =
            radial +
            x * dRadialDx +
            dTxDx;
        const double j01 =
            x * dRadialDy +
            dTxDy;
        const double j10 =
            y * dRadialDx +
            dTyDx;
        const double j11 =
            radial +
            y * dRadialDy +
            dTyDy;

        const double determinant =
            j00 * j11 -
            j01 * j10;

        double stepX = residualX;
        double stepY = residualY;

        if (
            std::isfinite(determinant) &&
            std::abs(determinant) > 1e-12
        ) {
            stepX =
                (
                    j11 * residualX -
                    j01 * residualY
                ) /
                determinant;
            stepY =
                (
                    -j10 * residualX +
                    j00 * residualY
                ) /
                determinant;
        }

        if (
            !std::isfinite(stepX) ||
            !std::isfinite(stepY)
        ) {
            stepX = residualX;
            stepY = residualY;
        }

        const double step =
            std::hypot(
                stepX,
                stepY);

        const double damping =
            step <= 0.20
                ? 1.0
                : 0.20 /
                    std::max(
                        step,
                        1e-12);

        x += stepX * damping;
        y += stepY * damping;

        if (
            !std::isfinite(x) ||
            !std::isfinite(y)
        ) {
            return {
                0.0,
                0.0,
            };
        }

        if (
            std::hypot(
                residualX,
                residualY) <
                1e-12
        ) {
            break;
        }
    }

    const auto check =
        distort(
            x,
            y,
            c);

    if (
        !std::isfinite(check.x) ||
        !std::isfinite(check.y)
    ) {
        return {
            0.0,
            0.0,
        };
    }

    return {
        x,
        y,
    };
}

} // namespace nox::native::calibration

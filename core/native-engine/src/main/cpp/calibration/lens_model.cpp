#include "calibration/lens_model.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nox::native::calibration {


double LensModel::radialScale(
    double r2,
    const LensParameters& p) const noexcept {
    if (
        !std::isfinite(r2) ||
        r2 < 0.0 ||
        !std::isfinite(p.k1) ||
        !std::isfinite(p.k2) ||
        !std::isfinite(p.k3)) {
        return 1.0;
    }

    return
        1.0 +
        r2 *
            (
                p.k1 +
                r2 *
                    (
                        p.k2 +
                        r2 *
                            p.k3
                    )
            );
}

LensJacobian LensModel::jacobian(
    double x,
    double y,
    const LensParameters& p) noexcept {
    LensJacobian output;

    if (
        !std::isfinite(x) ||
        !std::isfinite(y)) {
        output.determinant = 0.0;
        output.conditioning =
            std::numeric_limits<double>::infinity();
        return output;
    }

    const double r2 =
        x * x +
        y * y;

    const double radial =
        1.0 +
        r2 *
            (
                p.k1 +
                r2 *
                    (
                        p.k2 +
                        r2 *
                            p.k3
                    )
            );

    const double radialDerivative =
        p.k1 +
        2.0 * p.k2 * r2 +
        3.0 * p.k3 * r2 * r2;

    const double dRadialDx =
        2.0 *
        x *
        radialDerivative;
    const double dRadialDy =
        2.0 *
        y *
        radialDerivative;

    const double dTxDx =
        2.0 * p.p1 * y +
        6.0 * p.p2 * x;
    const double dTxDy =
        2.0 * p.p1 * x +
        2.0 * p.p2 * y;
    const double dTyDx =
        2.0 * p.p1 * x +
        2.0 * p.p2 * y;
    const double dTyDy =
        6.0 * p.p1 * y +
        2.0 * p.p2 * x;

    output.dxx =
        radial +
        x * dRadialDx +
        dTxDx;
    output.dxy =
        x * dRadialDy +
        dTxDy;
    output.dyx =
        y * dRadialDx +
        dTyDx;
    output.dyy =
        radial +
        y * dRadialDy +
        dTyDy;

    output.determinant =
        output.dxx *
            output.dyy -
        output.dxy *
            output.dyx;

    const double trace =
        output.dxx +
        output.dyy;

    const double frobenius =
        std::sqrt(
            output.dxx * output.dxx +
            output.dxy * output.dxy +
            output.dyx * output.dyx +
            output.dyy * output.dyy);

    output.conditioning =
        std::isfinite(
            output.determinant)
            ? std::max(
                1.0,
                frobenius /
                    std::max(
                        std::abs(
                            output.determinant),
                        1e-12))
            : std::numeric_limits<double>::infinity();

    (void)trace;
    return output;
}

LensProjectionResult LensModel::distortNormalized(
    double x,
    double y,
    const LensParameters& p) const noexcept {
    LensProjectionResult output;

    if (
        !std::isfinite(x) ||
        !std::isfinite(y) ||
        !isPlausible(p)) {
        return output;
    }

    const double r2 =
        x * x +
        y * y;

    const double radial =
        radialScale(
            r2,
            p);

    const double xTangential =
        2.0 *
            p.p1 *
            x *
            y +
        p.p2 *
            (
                r2 +
                2.0 *
                    x *
                    x
            );

    const double yTangential =
        p.p1 *
            (
                r2 +
                2.0 *
                    y *
                    y
            ) +
        2.0 *
            p.p2 *
            x *
            y;

    output.normalizedX =
        x *
            radial +
        xTangential;
    output.normalizedY =
        y *
            radial +
        yTangential;

    output.x =
        p.fx *
            output.normalizedX +
        p.skew *
            output.normalizedY +
        p.cx;
    output.y =
        p.fy *
            output.normalizedY +
        p.cy;

    output.radial =
        radial;
    output.jacobian =
        jacobian(
            x,
            y,
            p);

    output.valid =
        std::isfinite(output.x) &&
        std::isfinite(output.y) &&
        std::isfinite(
            output.jacobian.determinant) &&
        std::abs(
            output.jacobian.determinant) >
            1e-8;

    return output;
}

LensProjectionResult LensModel::project(
    double x,
    double y,
    double z,
    const LensParameters& p) const noexcept {
    LensProjectionResult output;

    if (
        !std::isfinite(x) ||
        !std::isfinite(y) ||
        !std::isfinite(z) ||
        z <= 1e-8) {
        return output;
    }

    const double normalizedX =
        x / z;
    const double normalizedY =
        y / z;

    return distortNormalized(
        normalizedX,
        normalizedY,
        p);
}

bool LensModel::undistortNormalized(
    double xd,
    double yd,
    const LensParameters& p,
    double& x,
    double& y,
    std::uint32_t iterations) const noexcept {
    if (
        !std::isfinite(xd) ||
        !std::isfinite(yd) ||
        !isPlausible(p)) {
        return false;
    }

    x = xd;
    y = yd;

    const std::uint32_t count =
        std::max(
            std::uint32_t{1},
            iterations);

    for (
        std::uint32_t iteration = 0;
        iteration < count;
        ++iteration) {
        const auto model =
            distortNormalized(
                x,
                y,
                p);

        if (!model.valid) {
            return false;
        }

        const double residualX =
            xd -
            model.normalizedX;
        const double residualY =
            yd -
            model.normalizedY;

        const double determinant =
            model.jacobian.determinant;

        if (
            !std::isfinite(determinant) ||
            std::abs(determinant) <
                1e-10) {
            return false;
        }

        const double dx =
            (
                model.jacobian.dyy *
                    residualX -
                model.jacobian.dxy *
                    residualY
            ) /
            determinant;

        const double dy =
            (
                -model.jacobian.dyx *
                    residualX +
                model.jacobian.dxx *
                    residualY
            ) /
            determinant;

        const double step =
            std::hypot(
                dx,
                dy);

        const double damping =
            step <= 0.10
                ? 1.0
                : 0.10 /
                    step;

        x +=
            dx *
            damping;
        y +=
            dy *
            damping;

        if (
            !std::isfinite(x) ||
            !std::isfinite(y)) {
            return false;
        }

        if (
            std::hypot(
                residualX,
                residualY) <
            1e-11) {
            break;
        }
    }

    const auto check =
        distortNormalized(
            x,
            y,
            p);

    return
        check.valid &&
        std::hypot(
            check.normalizedX - xd,
            check.normalizedY - yd) <
            1e-7;
}

bool LensModel::isPlausible(
    const LensParameters& p) const noexcept {
    if (
        !std::isfinite(p.fx) ||
        !std::isfinite(p.fy) ||
        !std::isfinite(p.cx) ||
        !std::isfinite(p.cy) ||
        !std::isfinite(p.skew) ||
        p.fx <= 0.0 ||
        p.fy <= 0.0) {
        return false;
    }

    if (
        p.width > 0U &&
        p.height > 0U &&
        (
            p.cx <
                -static_cast<double>(p.width) ||
            p.cx >
                static_cast<double>(p.width) * 2.0 ||
            p.cy <
                -static_cast<double>(p.height) ||
            p.cy >
                static_cast<double>(p.height) * 2.0
        )) {
        return false;
    }

    const double distortionMagnitude =
        std::abs(p.k1) +
        std::abs(p.k2) +
        std::abs(p.k3) +
        std::abs(p.p1) +
        std::abs(p.p2);

    return
        std::isfinite(
            distortionMagnitude) &&
        distortionMagnitude <
            10.0;
}

} // namespace nox::native::calibration

#include "calibration/calibration_solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace nox::native::calibration {

namespace {

struct Matrix3 {
    double a[3][3]{};
};

struct Vector3 {
    double v[3]{};
};

double huberWeight(double residual, double delta) {
    const double a = std::abs(residual);
    if (a <= delta || a < 1e-12) return 1.0;
    return delta / a;
}

double huberLoss(double residual, double delta) {
    const double a = std::abs(residual);
    if (a <= delta) {
        return 0.5 * a * a;
    }
    return delta * (a - 0.5 * delta);
}

struct IntrinsicRefinementState {
    double logFocalX{0.0};
    double logFocalY{0.0};
    double principalX{0.0};
    double principalY{0.0};
    double k1{0.0};
    double k2{0.0};
    double p1{0.0};
    double p2{0.0};
    double k3{0.0};
};

struct Matrix9 {
    double a[9][9]{};
};

struct Vector9 {
    double v[9]{};
};

struct CoverageMetrics {
    double image{0.0};
    double depth{0.0};
};

CoverageMetrics coverageFor(
    const CameraSeed& seed);

bool project(
    const CameraIntrinsics& intrinsics,
    const RigidPose& pose,
    const Vec3& world,
    DistortedPixel& pixel);

CoverageMetrics coverageFor(
    const CameraSeed& seed);

bool solve9x9(
    const Matrix9& h,
    const Vector9& b,
    Vector9& x) {
    double m[9][10]{};
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            m[row][col] = h.a[row][col];
        }
        m[row][9] = b.v[row];
    }

    for (int col = 0; col < 9; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 9; ++row) {
            if (std::abs(m[row][col]) > std::abs(m[pivot][col])) {
                pivot = row;
            }
        }

        if (!std::isfinite(m[pivot][col]) ||
            std::abs(m[pivot][col]) < 1e-12) {
            return false;
        }

        if (pivot != col) {
            for (int k = col; k <= 9; ++k) {
                std::swap(m[pivot][k], m[col][k]);
            }
        }

        const double pivotValue = m[col][col];
        for (int k = col; k <= 9; ++k) {
            m[col][k] /= pivotValue;
        }

        for (int row = 0; row < 9; ++row) {
            if (row == col) continue;
            const double factor = m[row][col];
            if (std::abs(factor) < 1e-18) continue;
            for (int k = col; k <= 9; ++k) {
                m[row][k] -= factor * m[col][k];
            }
        }
    }

    for (int row = 0; row < 9; ++row) {
        x.v[row] = m[row][9];
        if (!std::isfinite(x.v[row])) return false;
    }
    return true;
}

CameraIntrinsics applyIntrinsicState(
    const CameraIntrinsics& seed,
    const IntrinsicRefinementState& state) {
    CameraIntrinsics out = seed;

    const double focalScaleX =
        std::exp(std::clamp(state.logFocalX, -0.6931471805599453, 0.6931471805599453));
    const double focalScaleY =
        std::exp(std::clamp(state.logFocalY, -0.6931471805599453, 0.6931471805599453));

    out.fx =
        seed.fx *
        focalScaleX;
    out.fy =
        seed.fy *
        focalScaleY;

    const double width =
        static_cast<double>(std::max<std::uint32_t>(seed.width, 1u));
    const double height =
        static_cast<double>(std::max<std::uint32_t>(seed.height, 1u));

    out.cx =
        seed.cx +
        state.principalX * width;
    out.cy =
        seed.cy +
        state.principalY * height;

    out.distortion.k1 =
        std::clamp(
            seed.distortion.k1 + state.k1,
            -1.0,
            1.0);
    out.distortion.k2 =
        std::clamp(
            seed.distortion.k2 + state.k2,
            -1.0,
            1.0);
    out.distortion.p1 =
        std::clamp(
            seed.distortion.p1 + state.p1,
            -0.2,
            0.2);
    out.distortion.p2 =
        std::clamp(
            seed.distortion.p2 + state.p2,
            -0.2,
            0.2);
    out.distortion.k3 =
        std::clamp(
            seed.distortion.k3 + state.k3,
            -1.0,
            1.0);
    return out;
}

double refinementCost(
    const CameraSeed& seed,
    const RigidPose& pose,
    const CameraIntrinsics& intrinsics,
    double huberDelta,
    const IntrinsicRefinementState& state,
    double focalPriorSigma,
    double principalPointPriorSigma,
    double distortionPriorSigma,
    bool refineDistortion) {
    double cost = 0.0;

    for (const auto& observation : seed.observations) {
        DistortedPixel predicted;
        if (!project(
                intrinsics,
                pose,
                observation.world_point,
                predicted)) {
            return std::numeric_limits<double>::infinity();
        }

        if (!std::isfinite(predicted.x) ||
            !std::isfinite(predicted.y)) {
            return std::numeric_limits<double>::infinity();
        }

        const double rx =
            predicted.x - observation.pixel.x;
        const double ry =
            predicted.y - observation.pixel.y;
        const double normR =
            std::hypot(rx, ry);

        const double weight =
            std::max(0.0, observation.weight);
        if (weight <= 0.0) continue;

        cost +=
            weight *
            huberLoss(normR, huberDelta);
    }

    const double focalSigma =
        std::max(focalPriorSigma, 1e-6);
    const double principalSigma =
        std::max(principalPointPriorSigma, 1e-6);

    cost +=
        0.5 *
        (state.logFocalX / focalSigma) *
        (state.logFocalX / focalSigma);
    cost +=
        0.5 *
        (state.logFocalY / focalSigma) *
        (state.logFocalY / focalSigma);
    cost +=
        0.5 *
        (state.principalX / principalSigma) *
        (state.principalX / principalSigma);
    cost +=
        0.5 *
        (state.principalY / principalSigma) *
        (state.principalY / principalSigma);

    if (refineDistortion) {
        const double distortionSigma =
            std::max(distortionPriorSigma, 1e-6);
        const double d[5] = {
            state.k1,
            state.k2,
            state.p1,
            state.p2,
            state.k3,
        };
        for (double value : d) {
            cost +=
                0.5 *
                (value / distortionSigma) *
                (value / distortionSigma);
        }
    }

    return std::isfinite(cost)
        ? cost
        : std::numeric_limits<double>::infinity();
}

bool intrinsicJacobian(
    const CameraIntrinsics& baseline,
    const RigidPose& pose,
    const Vec3& world,
    const IntrinsicRefinementState& state,
    bool refineDistortion,
    double jacobian[2][9]) {
    const Vec3 camera =
        pose.transform(world);

    if (
        !std::isfinite(camera.x) ||
        !std::isfinite(camera.y) ||
        !std::isfinite(camera.z) ||
        camera.z <= 1e-8) {
        return false;
    }

    const double x =
        camera.x / camera.z;
    const double y =
        camera.y / camera.z;

    if (!std::isfinite(x) || !std::isfinite(y)) {
        return false;
    }

    /*
     * applyIntrinsicState clamps the state at physical safety bounds. At an active clamp the
     * objective is piecewise-defined, so the closed-form derivative is intentionally bypassed.
     * The existing finite-difference path then captures the actual clamped behavior.
     */
    constexpr double kClampEpsilon = 1e-9;
    if (
        std::abs(std::abs(state.logFocalX) -
            0.6931471805599453) < kClampEpsilon ||
        std::abs(std::abs(state.logFocalY) -
            0.6931471805599453) < kClampEpsilon ||
        std::abs(std::abs(state.principalX) -
            0.25) < kClampEpsilon ||
        std::abs(std::abs(state.principalY) -
            0.25) < kClampEpsilon) {
        return false;
    }

    const double r2 =
        x * x +
        y * y;
    const double r4 =
        r2 * r2;
    const double r6 =
        r4 * r2;

    const auto distortion =
        baseline.distortion;

    if (
        refineDistortion &&
        (
            std::abs(
                std::abs(distortion.k1 + state.k1) -
                    0.5) < kClampEpsilon ||
            std::abs(
                std::abs(distortion.k2 + state.k2) -
                    0.5) < kClampEpsilon ||
            std::abs(
                std::abs(distortion.p1 + state.p1) -
                    0.1) < kClampEpsilon ||
            std::abs(
                std::abs(distortion.p2 + state.p2) -
                    0.1) < kClampEpsilon ||
            std::abs(
                std::abs(distortion.k3 + state.k3) -
                    0.5) < kClampEpsilon
        )
    ) {
        return false;
    }

    const double focalX =
        baseline.fx *
        std::exp(
            std::clamp(
                state.logFocalX,
                -0.6931471805599453,
                0.6931471805599453));
    const double focalY =
        baseline.fy *
        std::exp(
            std::clamp(
                state.logFocalY,
                -0.6931471805599453,
                0.6931471805599453));

    const double radial =
        1.0 +
        (distortion.k1 + state.k1) * r2 +
        (distortion.k2 + state.k2) * r4 +
        (distortion.k3 + state.k3) * r6;

    const double distortedX =
        x * radial +
        2.0 *
            (distortion.p1 + state.p1) *
            x * y +
        (distortion.p2 + state.p2) *
            (r2 + 2.0 * x * x);

    const double distortedY =
        y * radial +
        (distortion.p1 + state.p1) *
            (r2 + 2.0 * y * y) +
        2.0 *
            (distortion.p2 + state.p2) *
            x * y;

    if (
        !std::isfinite(focalX) ||
        !std::isfinite(focalY) ||
        !std::isfinite(distortedX) ||
        !std::isfinite(distortedY)) {
        return false;
    }

    for (int row = 0; row < 2; ++row) {
        for (int parameter = 0; parameter < 9; ++parameter) {
            jacobian[row][parameter] = 0.0;
        }
    }

    jacobian[0][0] =
        focalX * distortedX;
    jacobian[1][0] = 0.0;

    jacobian[0][1] = 0.0;
    jacobian[1][1] =
        focalY * distortedY;

    const double width =
        static_cast<double>(
            std::max<std::uint32_t>(
                baseline.width,
                1u));
    const double height =
        static_cast<double>(
            std::max<std::uint32_t>(
                baseline.height,
                1u));

    jacobian[0][2] = width;
    jacobian[1][2] = 0.0;
    jacobian[0][3] = 0.0;
    jacobian[1][3] = height;

    if (refineDistortion) {
        const double focalScaleX =
            focalX;
        const double focalScaleY =
            focalY;

        jacobian[0][4] =
            focalScaleX *
            x *
            r2;
        jacobian[1][4] =
            focalScaleY *
            y *
            r2;

        jacobian[0][5] =
            focalScaleX *
            x *
            r4;
        jacobian[1][5] =
            focalScaleY *
            y *
            r4;

        jacobian[0][6] =
            focalScaleX *
            2.0 *
            x *
            y;
        jacobian[1][6] =
            focalScaleY *
            (
                r2 +
                    2.0 *
                    y *
                    y
            );

        jacobian[0][7] =
            focalScaleX *
            (
                r2 +
                    2.0 *
                    x *
                    x
            );
        jacobian[1][7] =
            focalScaleY *
            2.0 *
            x *
            y;

        jacobian[0][8] =
            focalScaleX *
            x *
            r6;
        jacobian[1][8] =
            focalScaleY *
            y *
            r6;
    }

    for (int row = 0; row < 2; ++row) {
        for (int parameter = 0; parameter < 9; ++parameter) {
            if (!std::isfinite(jacobian[row][parameter])) {
                return false;
            }
        }
    }

    return true;
}


bool analyticTranslationJacobian(
    const CameraIntrinsics& intrinsics,
    const RigidPose& pose,
    const Vec3& world,
    double jacobian[2][3]) {
    const Vec3 camera =
        pose.transform(world);

    if (
        !std::isfinite(camera.x) ||
        !std::isfinite(camera.y) ||
        !std::isfinite(camera.z) ||
        camera.z <= 1e-8) {
        return false;
    }

    const double x =
        camera.x / camera.z;
    const double y =
        camera.y / camera.z;
    if (!std::isfinite(x) || !std::isfinite(y)) {
        return false;
    }

    const auto& d =
        intrinsics.distortion;

    const double r2 =
        x * x + y * y;
    const double r4 =
        r2 * r2;
    const double r6 =
        r4 * r2;

    const double radial =
        1.0 +
        d.k1 * r2 +
        d.k2 * r4 +
        d.k3 * r6;
    if (!std::isfinite(radial)) {
        return false;
    }

    const double radialSlope =
        d.k1 +
        2.0 * d.k2 * r2 +
        3.0 * d.k3 * r4;

    const double dRadialDx =
        2.0 * x * radialSlope;
    const double dRadialDy =
        2.0 * y * radialSlope;

    const double dTxDx =
        2.0 * d.p1 * y +
        6.0 * d.p2 * x;
    const double dTxDy =
        2.0 * d.p1 * x +
        2.0 * d.p2 * y;
    const double dTyDx =
        2.0 * d.p1 * x +
        2.0 * d.p2 * y;
    const double dTyDy =
        6.0 * d.p1 * y +
        2.0 * d.p2 * x;

    const double duDx =
        radial +
        x * dRadialDx +
        dTxDx;
    const double duDy =
        x * dRadialDy +
        dTxDy;
    const double dvDx =
        y * dRadialDx +
        dTyDx;
    const double dvDy =
        radial +
        y * dRadialDy +
        dTyDy;

    const double invZ =
        1.0 / camera.z;
    const double invZ2 =
        invZ * invZ;

    for (int axis = 0; axis < 3; ++axis) {
        const double dCameraX =
            axis == 0 ? 1.0 : 0.0;
        const double dCameraY =
            axis == 1 ? 1.0 : 0.0;
        const double dCameraZ =
            axis == 2 ? 1.0 : 0.0;

        const double dNormalizedX =
            (
                dCameraX * camera.z -
                camera.x * dCameraZ
            ) * invZ2;

        const double dNormalizedY =
            (
                dCameraY * camera.z -
                camera.y * dCameraZ
            ) * invZ2;

        jacobian[0][axis] =
            intrinsics.fx *
            (
                duDx * dNormalizedX +
                duDy * dNormalizedY
            );

        jacobian[1][axis] =
            intrinsics.fy *
            (
                dvDx * dNormalizedX +
                dvDy * dNormalizedY
            );

        if (
            !std::isfinite(jacobian[0][axis]) ||
            !std::isfinite(jacobian[1][axis])
        ) {
            return false;
        }
    }

    (void)r6;
    return true;
}

bool refineCameraIntrinsics(
    const CameraSeed& seed,
    double huberDelta,
    const CalibrationProblem& problem,
    RigidPose pose,
    CameraIntrinsics& intrinsics,
    double& initialRms,
    double& improvement,
    std::uint32_t& acceptedIterations,
    std::uint32_t& jacobianFallbackCount) {
    if (!problem.refineIntrinsics ||
        seed.observations.size() < 12 ||
        seed.intrinsics.fx <= 1.0 ||
        seed.intrinsics.fy <= 1.0 ||
        seed.intrinsics.width == 0 ||
        seed.intrinsics.height == 0) {
        return false;
    }

    const auto coverage = coverageFor(seed);
    if (coverage.image < 0.12 || coverage.depth < 0.04) {
        return false;
    }

    const CameraIntrinsics baselineIntrinsics =
        seed.intrinsics;

    auto rmsFor =
        [&](const CameraIntrinsics& candidate) {
            double squared = 0.0;
            std::size_t used = 0u;
            for (const auto& observation : seed.observations) {
                DistortedPixel predicted;
                if (!project(
                        candidate,
                        pose,
                        observation.world_point,
                        predicted)) {
                    continue;
                }
                const double error =
                    std::hypot(
                        predicted.x - observation.pixel.x,
                        predicted.y - observation.pixel.y);
                if (!std::isfinite(error)) continue;
                squared += error * error;
                ++used;
            }
            if (used == 0u) {
                return std::numeric_limits<double>::infinity();
            }
            return std::sqrt(
                squared /
                static_cast<double>(used));
        };

    initialRms = rmsFor(baselineIntrinsics);
    if (!std::isfinite(initialRms)) return false;

    IntrinsicRefinementState state{};
    state.k1 = 0.0;
    state.k2 = 0.0;
    state.p1 = 0.0;
    state.p2 = 0.0;
    state.k3 = 0.0;

    const double focalSigma =
        std::clamp(
            problem.focalPriorSigmaRatio,
            0.005,
            0.5);
    const double ppSigma =
        std::clamp(
            problem.principalPointPriorSigmaRatio,
            0.005,
            0.25);
    const double distortionSigma =
        std::clamp(
            problem.distortionPriorSigma,
            0.001,
            0.5);

    double lambda =
        std::clamp(
            problem.refinementDamping,
            1e-8,
            1e6);

    const std::uint32_t maxIterations =
        std::clamp(
            problem.refinementIterations,
            1u,
            64u);

    auto costForState =
        [&](const IntrinsicRefinementState& candidateState) {
            const auto candidateIntrinsics =
                applyIntrinsicState(
                    baselineIntrinsics,
                    candidateState);
            return refinementCost(
                seed,
                pose,
                candidateIntrinsics,
                huberDelta,
                candidateState,
                focalSigma,
                ppSigma,
                distortionSigma,
                problem.refineDistortion);
        };

    double currentCost = costForState(state);
    if (!std::isfinite(currentCost)) return false;

    const double steps[9] = {
        1e-5,
        1e-5,
        1e-5,
        1e-5,
        1e-6,
        1e-6,
        1e-6,
        1e-6,
        1e-6,
    };

    for (std::uint32_t iteration = 0;
         iteration < maxIterations;
         ++iteration) {
        Matrix9 h{};
        Vector9 b{};

        for (const auto& observation : seed.observations) {
            DistortedPixel basePixel;
            const auto baseIntrinsics =
                applyIntrinsicState(
                    baselineIntrinsics,
                    state);
            if (!project(
                    baseIntrinsics,
                    pose,
                    observation.world_point,
                    basePixel)) {
                continue;
            }

            const double rx =
                basePixel.x - observation.pixel.x;
            const double ry =
                basePixel.y - observation.pixel.y;
            const double normR =
                std::hypot(rx, ry);
            const double measurementWeight =
                std::max(0.0, observation.weight) *
                huberWeight(normR, huberDelta);

            if (measurementWeight <= 0.0) continue;

            double jacobian[2][9]{};

            const auto analyticReady =
                intrinsicJacobian(
                    baselineIntrinsics,
                    pose,
                    observation.world_point,
                    state,
                    problem.refineDistortion,
                    jacobian);

            if (!analyticReady) {
                ++jacobianFallbackCount;

                for (int parameter = 0; parameter < 9; ++parameter) {
                    IntrinsicRefinementState plus = state;
                    IntrinsicRefinementState minus = state;

                    double* plusValue = nullptr;
                    double* minusValue = nullptr;
                    switch (parameter) {
                        case 0:
                            plusValue = &plus.logFocalX;
                            minusValue = &minus.logFocalX;
                            break;
                        case 1:
                            plusValue = &plus.logFocalY;
                            minusValue = &minus.logFocalY;
                            break;
                        case 2:
                            plusValue = &plus.principalX;
                            minusValue = &minus.principalX;
                            break;
                        case 3:
                            plusValue = &plus.principalY;
                            minusValue = &minus.principalY;
                            break;
                        case 4:
                            plusValue = &plus.k1;
                            minusValue = &minus.k1;
                            break;
                        case 5:
                            plusValue = &plus.k2;
                            minusValue = &minus.k2;
                            break;
                        case 6:
                            plusValue = &plus.p1;
                            minusValue = &minus.p1;
                            break;
                        case 7:
                            plusValue = &plus.p2;
                            minusValue = &minus.p2;
                            break;
                        case 8:
                            plusValue = &plus.k3;
                            minusValue = &minus.k3;
                            break;
                        default:
                            break;
                    }

                    if (plusValue == nullptr ||
                        minusValue == nullptr) {
                        continue;
                    }

                    if (!problem.refineDistortion &&
                        parameter >= 4) {
                        continue;
                    }

                    *plusValue += steps[parameter];
                    *minusValue -= steps[parameter];

                    const auto plusIntrinsics =
                        applyIntrinsicState(
                            baselineIntrinsics,
                            plus);
                    const auto minusIntrinsics =
                        applyIntrinsicState(
                            baselineIntrinsics,
                            minus);

                    DistortedPixel plusPixel;
                    DistortedPixel minusPixel;

                    if (!project(
                            plusIntrinsics,
                            pose,
                            observation.world_point,
                            plusPixel) ||
                        !project(
                            minusIntrinsics,
                            pose,
                            observation.world_point,
                            minusPixel)) {
                        continue;
                    }

                    jacobian[0][parameter] =
                        (plusPixel.x - minusPixel.x) /
                        (2.0 * steps[parameter]);
                    jacobian[1][parameter] =
                        (plusPixel.y - minusPixel.y) /
                        (2.0 * steps[parameter]);
                }
            }

            for (int row = 0; row < 9; ++row) {
                b.v[row] +=
                    measurementWeight *
                    (
                        jacobian[0][row] * rx +
                        jacobian[1][row] * ry
                    );
                for (int col = 0; col < 9; ++col) {
                    h.a[row][col] +=
                        measurementWeight *
                        (
                            jacobian[0][row] *
                                jacobian[0][col] +
                            jacobian[1][row] *
                                jacobian[1][col]
                        );
                }
            }
        }

        const double priorWeights[9] = {
            1.0 / (focalSigma * focalSigma),
            1.0 / (focalSigma * focalSigma),
            1.0 / (ppSigma * ppSigma),
            1.0 / (ppSigma * ppSigma),
            problem.refineDistortion
                ? 1.0 / (distortionSigma * distortionSigma)
                : 0.0,
            problem.refineDistortion
                ? 1.0 / (distortionSigma * distortionSigma)
                : 0.0,
            problem.refineDistortion
                ? 1.0 / (distortionSigma * distortionSigma)
                : 0.0,
            problem.refineDistortion
                ? 1.0 / (distortionSigma * distortionSigma)
                : 0.0,
            problem.refineDistortion
                ? 1.0 / (distortionSigma * distortionSigma)
                : 0.0,
        };

        const double stateValues[9] = {
            state.logFocalX,
            state.logFocalY,
            state.principalX,
            state.principalY,
            state.k1,
            state.k2,
            state.p1,
            state.p2,
            state.k3,
        };

        for (int parameter = 0; parameter < 9; ++parameter) {
            const double priorWeight =
                priorWeights[parameter];
            if (priorWeight <= 0.0) continue;
            h.a[parameter][parameter] += priorWeight;
            b.v[parameter] +=
                priorWeight *
                stateValues[parameter];
        }

        for (int parameter = 0; parameter < 9; ++parameter) {
            h.a[parameter][parameter] +=
                lambda *
                (1.0 + std::abs(h.a[parameter][parameter]));
        }

        Vector9 delta{};
        Vector9 rhs{};
        for (int i = 0; i < 9; ++i) {
            rhs.v[i] = -b.v[i];
        }

        if (!solve9x9(h, rhs, delta)) {
            lambda *= 10.0;
            continue;
        }

        IntrinsicRefinementState candidate = state;
        candidate.logFocalX += delta.v[0];
        candidate.logFocalY += delta.v[1];
        candidate.principalX += delta.v[2];
        candidate.principalY += delta.v[3];
        if (problem.refineDistortion) {
            candidate.k1 += delta.v[4];
            candidate.k2 += delta.v[5];
            candidate.p1 += delta.v[6];
            candidate.p2 += delta.v[7];
            candidate.k3 += delta.v[8];
        }

        candidate.logFocalX =
            std::clamp(
                candidate.logFocalX,
                -0.6931471805599453,
                0.6931471805599453);
        candidate.logFocalY =
            std::clamp(
                candidate.logFocalY,
                -0.6931471805599453,
                0.6931471805599453);
        candidate.principalX =
            std::clamp(candidate.principalX, -0.25, 0.25);
        candidate.principalY =
            std::clamp(candidate.principalY, -0.25, 0.25);
        candidate.k1 = std::clamp(candidate.k1, -0.5, 0.5);
        candidate.k2 = std::clamp(candidate.k2, -0.5, 0.5);
        candidate.p1 = std::clamp(candidate.p1, -0.1, 0.1);
        candidate.p2 = std::clamp(candidate.p2, -0.1, 0.1);
        candidate.k3 = std::clamp(candidate.k3, -0.5, 0.5);

        const double candidateCost =
            costForState(candidate);

        if (std::isfinite(candidateCost) &&
            candidateCost < currentCost) {
            const auto candidateIntrinsics =
                applyIntrinsicState(
                    baselineIntrinsics,
                    candidate);
            const double candidateRms =
                rmsFor(candidateIntrinsics);

            if (std::isfinite(candidateRms) &&
                candidateRms <=
                    initialRms * 1.05) {
                state = candidate;
                currentCost = candidateCost;
                ++acceptedIterations;
                lambda =
                    std::max(
                        lambda * 0.35,
                        1e-10);

                if (std::abs(
                        candidateRms - initialRms) <
                    1e-6) {
                    break;
                }
            } else {
                lambda *= 4.0;
            }
        } else {
            lambda *= 4.0;
        }

        if (lambda > 1e12) break;

        const double deltaMagnitude =
            std::abs(delta.v[0]) +
            std::abs(delta.v[1]) +
            std::abs(delta.v[2]) +
            std::abs(delta.v[3]) +
            std::abs(delta.v[4]) +
            std::abs(delta.v[5]) +
            std::abs(delta.v[6]) +
            std::abs(delta.v[7]) +
            std::abs(delta.v[8]);

        if (deltaMagnitude < 1e-8) break;
    }

    const auto finalIntrinsics =
        applyIntrinsicState(
            baselineIntrinsics,
            state);
    const double finalRms =
        rmsFor(finalIntrinsics);

    if (std::isfinite(finalRms)) {
        intrinsics = finalIntrinsics;
    }

    improvement =
        initialRms -
        (std::isfinite(finalRms) ? finalRms : initialRms);

    return
        acceptedIterations > 0u &&
        improvement >= 1e-5;
}


bool solve3x3(const Matrix3& h, const Vector3& b, Vector3& x) {
    double matrixScale = 0.0;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            matrixScale = std::max(
                matrixScale,
                std::abs(h.a[row][col]));
        }
    }
    const double pivotFloor =
        std::max(
            1e-14,
            matrixScale * 1e-10);

    double m[3][4] = {
        {h.a[0][0], h.a[0][1], h.a[0][2], b.v[0]},
        {h.a[1][0], h.a[1][1], h.a[1][2], b.v[1]},
        {h.a[2][0], h.a[2][1], h.a[2][2], b.v[2]},
    };
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row) {
            if (std::abs(m[row][col]) > std::abs(m[pivot][col])) pivot = row;
        }
        if (std::abs(m[pivot][col]) < pivotFloor) return false;
        if (pivot != col) std::swap(m[pivot], m[col]);
        for (int row = col + 1; row < 3; ++row) {
            const double f = m[row][col] / m[col][col];
            for (int k = col; k < 4; ++k) m[row][k] -= f * m[col][k];
        }
    }
    for (int row = 2; row >= 0; --row) {
        double value = m[row][3];
        for (int k = row + 1; k < 3; ++k) value -= m[row][k] * x.v[k];
        x.v[row] = value / m[row][row];
    }
    return true;
}

bool project(
    const CameraIntrinsics& intrinsics,
    const RigidPose& pose,
    const Vec3& world,
    DistortedPixel& pixel) {
    const Vec3 camera = pose.transform(world);
    if (!std::isfinite(camera.x) || !std::isfinite(camera.y) ||
        !std::isfinite(camera.z) || camera.z <= 1e-8) return false;
    const double nx = camera.x / camera.z;
    const double ny = camera.y / camera.z;
    const auto distorted = BrownConradyDistortion::distort(nx, ny, intrinsics.distortion);
    pixel.x = intrinsics.fx * distorted.x + intrinsics.cx;
    pixel.y = intrinsics.fy * distorted.y + intrinsics.cy;
    return std::isfinite(pixel.x) && std::isfinite(pixel.y);
}



CoverageMetrics coverageFor(
    const CameraSeed& seed) {
    if (seed.observations.size() < 2 ||
        seed.intrinsics.width == 0 ||
        seed.intrinsics.height == 0) {
        return {};
    }

    double minX=std::numeric_limits<double>::infinity();
    double maxX=-std::numeric_limits<double>::infinity();
    double minY=std::numeric_limits<double>::infinity();
    double maxY=-std::numeric_limits<double>::infinity();
    double minDepth=std::numeric_limits<double>::infinity();
    double maxDepth=-std::numeric_limits<double>::infinity();
    std::size_t usable=0;

    for(const auto& observation:seed.observations){
        if(!std::isfinite(observation.pixel.x) ||
           !std::isfinite(observation.pixel.y)) {
            continue;
        }
        const Vec3 camera =
            seed.world_to_camera.transform(
                observation.world_point);
        if(!std::isfinite(camera.z) ||
           camera.z<=1e-8) {
            continue;
        }

        minX=std::min(minX,observation.pixel.x);
        maxX=std::max(maxX,observation.pixel.x);
        minY=std::min(minY,observation.pixel.y);
        maxY=std::max(maxY,observation.pixel.y);
        minDepth=std::min(minDepth,camera.z);
        maxDepth=std::max(maxDepth,camera.z);
        ++usable;
    }

    if(usable<2 ||
       !std::isfinite(minX) ||
       !std::isfinite(minY) ||
       !std::isfinite(minDepth)) {
        return {};
    }

    const double imageArea =
        std::max(
            0.0,
            std::min(
                static_cast<double>(seed.intrinsics.width),
                maxX) -
            std::max(0.0,minX)) *
        std::max(
            0.0,
            std::min(
                static_cast<double>(seed.intrinsics.height),
                maxY) -
            std::max(0.0,minY));

    const double imageFraction =
        imageArea /
        std::max(
            1.0,
            static_cast<double>(
                seed.intrinsics.width) *
            static_cast<double>(
                seed.intrinsics.height));

    const double depthRange =
        maxDepth-minDepth;
    const double depthCenter =
        0.5*(maxDepth+minDepth);

    const double relativeDepthRange =
        depthRange/
        std::max(
            0.25,
            std::abs(depthCenter));

    return {
        std::clamp(
            std::sqrt(
                std::clamp(
                    imageFraction,
                    0.0,
                    1.0)),
            0.0,
            1.0),
        std::clamp(
            relativeDepthRange/
                0.75,
            0.0,
            1.0),
    };
}

CameraCalibrationResult solveCamera(
    const CameraSeed& seed,
    double huberDelta,
    std::uint32_t maxIterations,
    const CalibrationProblem& problem) {
    CameraCalibrationResult out;
    out.camera_id = seed.camera_id;
    out.intrinsics = seed.intrinsics;
    out.world_to_camera = seed.world_to_camera;

    if (
        seed.observations.size() < 6 ||
        seed.intrinsics.fx <= 0.0 ||
        seed.intrinsics.fy <= 0.0) {
        return out;
    }

    const double epsilon = 1e-5;
    Matrix3 lastH{};
    std::uint32_t translationJacobianFallbackCount = 0U;

    for (
        std::uint32_t iteration = 0;
        iteration < std::max(maxIterations, 1u);
        ++iteration) {
        Matrix3 h{};
        Vector3 b{};
        double weightedError = 0.0;
        double weightTotal = 0.0;

        for (const auto& observation : seed.observations) {
            DistortedPixel predicted;
            if (!project(
                    seed.intrinsics,
                    out.world_to_camera,
                    observation.world_point,
                    predicted)) {
                continue;
            }

            const double rx =
                predicted.x -
                observation.pixel.x;
            const double ry =
                predicted.y -
                observation.pixel.y;
            const double normR =
                std::hypot(rx, ry);
            const double weight =
                std::max(
                    0.0,
                    observation.weight) *
                huberWeight(
                    normR,
                    huberDelta);
            if (weight <= 0.0) continue;

            double j[2][3]{};
            if (!analyticTranslationJacobian(
                    seed.intrinsics,
                    out.world_to_camera,
                    observation.world_point,
                    j)) {
                ++translationJacobianFallbackCount;
                RigidPose perturbed =
                    out.world_to_camera;

                for (int axis = 0; axis < 3; ++axis) {
                    const double delta =
                        epsilon;
                    RigidPose plus =
                        out.world_to_camera;
                    RigidPose minus =
                        out.world_to_camera;

                    if (axis == 0) {
                        plus.translation.x += delta;
                        minus.translation.x -= delta;
                    } else if (axis == 1) {
                        plus.translation.y += delta;
                        minus.translation.y -= delta;
                    } else {
                        plus.translation.z += delta;
                        minus.translation.z -= delta;
                    }

                    DistortedPixel plusPixel;
                    DistortedPixel minusPixel;
                    if (
                        !project(
                            seed.intrinsics,
                            plus,
                            observation.world_point,
                            plusPixel) ||
                        !project(
                            seed.intrinsics,
                            minus,
                            observation.world_point,
                            minusPixel)
                    ) {
                        continue;
                    }

                    j[0][axis] =
                        (
                            plusPixel.x -
                            minusPixel.x
                        ) /
                        (2.0 * delta);
                    j[1][axis] =
                        (
                            plusPixel.y -
                            minusPixel.y
                        ) /
                        (2.0 * delta);
                }
            }

            const double residual[2]{rx, ry};
            for (int row = 0; row < 3; ++row) {
                b.v[row] +=
                    weight *
                    (
                        j[0][row] * residual[0] +
                        j[1][row] * residual[1]
                    );
                for (int col = 0; col < 3; ++col) {
                    h.a[row][col] +=
                        weight *
                        (
                            j[0][row] * j[0][col] +
                            j[1][row] * j[1][col]
                        );
                }
            }

            weightedError +=
                weight *
                (
                    rx * rx +
                    ry * ry
                );
            weightTotal += weight;
        }

        lastH = h;

        Vector3 delta{};
        const Vector3 rhs{{
            -b.v[0],
            -b.v[1],
            -b.v[2],
        }};

        if (!solve3x3(h, rhs, delta)) break;

        out.world_to_camera.translation.x +=
            delta.v[0];
        out.world_to_camera.translation.y +=
            delta.v[1];
        out.world_to_camera.translation.z +=
            delta.v[2];

        if (
            std::abs(delta.v[0]) +
            std::abs(delta.v[1]) +
            std::abs(delta.v[2]) <
            1e-8) {
            break;
        }

        if (weightTotal <= 0.0) break;

        out.reprojection_rms_px =
            std::sqrt(
                weightedError /
                weightTotal);
    }

    double squared = 0.0;
    std::size_t used = 0U;

    for (const auto& observation : seed.observations) {
        DistortedPixel predicted;
        if (!project(
                seed.intrinsics,
                out.world_to_camera,
                observation.world_point,
                predicted)) {
            continue;
        }

        const double error =
            std::hypot(
                predicted.x -
                    observation.pixel.x,
                predicted.y -
                    observation.pixel.y);

        if (!std::isfinite(error)) continue;

        squared += error * error;
        ++used;
    }

    if (used < 6U) return out;

    out.reprojection_rms_px =
        std::sqrt(
            squared /
            static_cast<double>(used));
    out.initial_reprojection_rms_px =
        out.reprojection_rms_px;

    double intrinsicImprovementPx = 0.0;
    std::uint32_t acceptedIntrinsicIterations = 0U;
    std::uint32_t intrinsicJacobianFallbackCount = 0U;

    if (refineCameraIntrinsics(
            seed,
            huberDelta,
            problem,
            out.world_to_camera,
            out.intrinsics,
            out.initial_reprojection_rms_px,
            intrinsicImprovementPx,
            acceptedIntrinsicIterations,
            intrinsicJacobianFallbackCount)) {
        out.intrinsic_refinement_improvement_px =
            intrinsicImprovementPx;
        out.intrinsics_refined = true;
        out.intrinsic_jacobian_fallback_count =
            intrinsicJacobianFallbackCount;
        out.translation_jacobian_fallback_count =
            translationJacobianFallbackCount;

        double refinedSquared = 0.0;
        std::size_t refinedUsed = 0U;

        for (const auto& observation : seed.observations) {
            DistortedPixel predicted;
            if (!project(
                    out.intrinsics,
                    out.world_to_camera,
                    observation.world_point,
                    predicted)) {
                continue;
            }

            const double error =
                std::hypot(
                    predicted.x -
                        observation.pixel.x,
                    predicted.y -
                        observation.pixel.y);

            if (!std::isfinite(error)) continue;

            refinedSquared += error * error;
            ++refinedUsed;
        }

        if (refinedUsed >= 6U) {
            out.reprojection_rms_px =
                std::sqrt(
                    refinedSquared /
                    static_cast<double>(refinedUsed));
        }
    }

    (void)acceptedIntrinsicIterations;

    double minDiagonal =
        std::numeric_limits<double>::infinity();
    double maxDiagonal = 0.0;

    for (int i = 0; i < 3; ++i) {
        minDiagonal =
            std::min(
                minDiagonal,
                lastH.a[i][i]);
        maxDiagonal =
            std::max(
                maxDiagonal,
                lastH.a[i][i]);
    }

    const double diagonalTrace =
        std::max(
            0.0,
            lastH.a[0][0]) +
        std::max(
            0.0,
            lastH.a[1][1]) +
        std::max(
            0.0,
            lastH.a[2][2]);
    out.conditioning =
        std::min(
            diagonalTrace,
            1e12) /
        1e6;
    if(!std::isfinite(out.conditioning)){
        out.conditioning = 0.0;
    }

    const auto coverage =
        coverageFor(seed);

    out.imageCoverage =
        coverage.image;
    out.depthCoverage =
        coverage.depth;
    out.observationCoverage =
        std::clamp(
            static_cast<double>(used) /
                static_cast<double>(
                    std::max<std::size_t>(
                        1U,
                        seed.observations.size())),
            0.0,
            1.0);

    out.confidence =
        std::exp(
            -out.reprojection_rms_px /
            3.0) *
        std::clamp(
            out.conditioning,
            0.0,
            1.0);

    out.valid =
        std::isfinite(out.confidence) &&
        out.confidence >= 0.55 &&
        out.reprojection_rms_px <=
            huberDelta * 2.5;

    return out;
}
} // namespace

CalibrationResult CalibrationSolver::solve(const CalibrationProblem& problem) const {
    CalibrationResult result;
    if (problem.cameras.empty()) return result;

    double weightedSquared = 0.0;
    std::size_t totalObservations = 0;
    double confidenceSum = 0.0;

    for (const auto& camera : problem.cameras) {
        auto solved = solveCamera(camera, problem.huberDeltaPx, problem.maxIterations, problem);
        result.cameras.push_back(solved);
        if (solved.valid) {
            weightedSquared += solved.reprojection_rms_px * solved.reprojection_rms_px *
                static_cast<double>(camera.observations.size());
            totalObservations += camera.observations.size();
            confidenceSum += solved.confidence;
        }
    }

    result.residual_rms_px = totalObservations == 0
        ? std::numeric_limits<double>::infinity()
        : std::sqrt(weightedSquared / static_cast<double>(totalObservations));
    double imageCoverageSum=0.0;
    double depthCoverageSum=0.0;
    double observationCoverageSum=0.0;

    for(const auto& camera:result.cameras){
        imageCoverageSum+=camera.imageCoverage;
        depthCoverageSum+=camera.depthCoverage;
        observationCoverageSum+=camera.observationCoverage;
    }

    const double cameraCount=
        static_cast<double>(
            std::max<std::size_t>(
                result.cameras.size(),
                1));
    result.confidence = result.cameras.empty()
        ? 0.0
        : confidenceSum / cameraCount;
    result.image_coverage=
        imageCoverageSum/cameraCount;
    result.depth_coverage=
        depthCoverageSum/cameraCount;
    result.observation_coverage=
        observationCoverageSum/cameraCount;

    result.valid =
        totalObservations > 0 &&
        result.confidence >= 0.55 &&
        result.image_coverage >= 0.25 &&
        result.depth_coverage >= 0.10;
    return result;
}

} // namespace nox::native::calibration

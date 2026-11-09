#include "triangulation/multiview_bundle_adjuster.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include "math/robust_loss.h"

namespace nox::native::triangulation {
namespace {

constexpr std::size_t CameraDof = 6;
constexpr std::size_t PointDof = 3;

struct Vec2 {
    double x{0.0};
    double y{0.0};
};

struct CameraBlock {
    std::array<double, CameraDof * CameraDof> h{};
    std::array<double, CameraDof> g{};
};

struct PointBlock {
    std::array<double, PointDof * PointDof> h{};
    std::array<double, PointDof> g{};
};

struct CrossBlock {
    std::array<double, CameraDof * PointDof> h{};
};

struct PointLinearSystem {
    PointBlock point{};
    std::vector<std::pair<std::size_t, CrossBlock>> cross;
};

struct CameraLinearSystem {
    std::vector<CameraBlock> blocks;
    std::vector<std::size_t> cameraToBlock;
    std::vector<double> schurMatrix;
    std::vector<double> schurVector;
};

double determinant3(const std::array<double, 9>& a) noexcept {
    return
        a[0] * (a[4] * a[8] - a[5] * a[7]) -
        a[1] * (a[3] * a[8] - a[5] * a[6]) +
        a[2] * (a[3] * a[7] - a[4] * a[6]);
}

bool invert3(
    const std::array<double, 9>& a,
    std::array<double, 9>& inverse) noexcept {
    const double determinant = determinant3(a);
    if (std::abs(determinant) < 1e-12) return false;

    inverse[0] = (a[4] * a[8] - a[5] * a[7]) / determinant;
    inverse[1] = (a[2] * a[7] - a[1] * a[8]) / determinant;
    inverse[2] = (a[1] * a[5] - a[2] * a[4]) / determinant;
    inverse[3] = (a[5] * a[6] - a[3] * a[8]) / determinant;
    inverse[4] = (a[0] * a[8] - a[2] * a[6]) / determinant;
    inverse[5] = (a[3] * a[6] - a[0] * a[5]) / determinant;
    inverse[6] = (a[3] * a[7] - a[4] * a[6]) / determinant;
    inverse[7] = (a[1] * a[6] - a[0] * a[7]) / determinant;
    inverse[8] = (a[0] * a[4] - a[1] * a[3]) / determinant;
    return true;
}

bool solveDense(
    std::vector<double> matrix,
    std::vector<double> rhs,
    std::vector<double>& solution) {
    const std::size_t n = rhs.size();
    if (matrix.size() != n * n) return false;
    solution = rhs;

    for (std::size_t column = 0; column < n; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < n; ++row) {
            if (std::abs(matrix[row * n + column]) >
                std::abs(matrix[pivot * n + column])) {
                pivot = row;
            }
        }
        if (std::abs(matrix[pivot * n + column]) < 1e-12) return false;

        if (pivot != column) {
            for (std::size_t k = column; k < n; ++k) {
                std::swap(
                    matrix[column * n + k],
                    matrix[pivot * n + k]);
            }
            std::swap(solution[column], solution[pivot]);
        }

        for (std::size_t row = column + 1; row < n; ++row) {
            const double factor =
                matrix[row * n + column] /
                matrix[column * n + column];
            if (factor == 0.0) continue;
            for (std::size_t k = column; k < n; ++k) {
                matrix[row * n + k] -= factor * matrix[column * n + k];
            }
            solution[row] -= factor * solution[column];
        }
    }

    for (std::size_t row = n; row-- > 0;) {
        double value = solution[row];
        for (std::size_t k = row + 1; k < n; ++k) {
            value -= matrix[row * n + k] * solution[k];
        }
        solution[row] =
            value / matrix[row * n + row];
    }
    return true;
}

math::SE3d perturbCamera(
    const math::SE3d& pose,
    std::size_t component,
    double epsilon) {
    math::Twist6d delta{};
    if (component < 3) {
        delta.translation.x = component == 0 ? epsilon : 0.0;
        delta.translation.y = component == 1 ? epsilon : 0.0;
        delta.translation.z = component == 2 ? epsilon : 0.0;
    } else {
        const std::size_t axis = component - 3;
        delta.rotation.x = axis == 0 ? epsilon : 0.0;
        delta.rotation.y = axis == 1 ? epsilon : 0.0;
        delta.rotation.z = axis == 2 ? epsilon : 0.0;
    }
    return math::SE3d::exp(delta).compose(pose);
}

bool project(
    const BundleCamera& camera,
    const math::Vec3d& world,
    Vec2& pixel) {
    const auto cameraPoint =
        camera.worldToCamera.transform(world);
    if (
        !std::isfinite(cameraPoint.x) ||
        !std::isfinite(cameraPoint.y) ||
        !std::isfinite(cameraPoint.z) ||
        cameraPoint.z <= 1e-8) {
        return false;
    }
    const double normalizedX = cameraPoint.x / cameraPoint.z;
    const double normalizedY = cameraPoint.y / cameraPoint.z;
    const auto distorted = calibration::BrownConradyDistortion::distort(
        normalizedX,
        normalizedY,
        camera.distortion);
    pixel.x = camera.fx * distorted.x + camera.cx;
    pixel.y = camera.fy * distorted.y + camera.cy;
    return std::isfinite(pixel.x) && std::isfinite(pixel.y);
}


bool analyticProjectionPointJacobian(
    const BundleCamera& camera,
    const math::Vec3d& world,
    std::array<double, PointDof * 2>& jacobian) {
    const auto cameraPoint =
        camera.worldToCamera.transform(world);

    if (
        !std::isfinite(cameraPoint.x) ||
        !std::isfinite(cameraPoint.y) ||
        !std::isfinite(cameraPoint.z) ||
        cameraPoint.z <= 1e-8
    ) {
        return false;
    }

    const double x =
        cameraPoint.x / cameraPoint.z;
    const double y =
        cameraPoint.y / cameraPoint.z;

    if (!std::isfinite(x) || !std::isfinite(y)) {
        return false;
    }

    const auto& d =
        camera.distortion;

    const double r2 =
        x * x +
        y * y;
    const double r4 =
        r2 * r2;
    const double r6 =
        r4 * r2;

    const double radial =
        1.0 +
        d.k1 * r2 +
        d.k2 * r4 +
        d.k3 * r6;

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

    const double dDistortedXDx =
        radial +
        x * dRadialDx +
        dTxDx;
    const double dDistortedXDy =
        x * dRadialDy +
        dTxDy;
    const double dDistortedYDx =
        y * dRadialDx +
        dTyDx;
    const double dDistortedYDy =
        radial +
        y * dRadialDy +
        dTyDy;

    const double invZ2 =
        1.0 /
        (cameraPoint.z * cameraPoint.z);

    const std::array<math::Vec3d, 3> worldBasisCamera{
        camera.worldToCamera.rotation.rotate(
            {1.0, 0.0, 0.0}),
        camera.worldToCamera.rotation.rotate(
            {0.0, 1.0, 0.0}),
        camera.worldToCamera.rotation.rotate(
            {0.0, 0.0, 1.0}),
    };

    for (std::size_t axis = 0; axis < PointDof; ++axis) {
        const auto& derivative =
            worldBasisCamera[axis];

        const double dxn =
            (
                derivative.x * cameraPoint.z -
                cameraPoint.x * derivative.z
            ) * invZ2;

        const double dyn =
            (
                derivative.y * cameraPoint.z -
                cameraPoint.y * derivative.z
            ) * invZ2;

        jacobian[axis] =
            camera.fx *
            (
                dDistortedXDx * dxn +
                dDistortedXDy * dyn
            );

        jacobian[PointDof + axis] =
            camera.fy *
            (
                dDistortedYDx * dxn +
                dDistortedYDy * dyn
            );

        if (
            !std::isfinite(jacobian[axis]) ||
            !std::isfinite(
                jacobian[PointDof + axis])
        ) {
            return false;
        }
    }

    return true;
}

math::Vec3d cross(
    const math::Vec3d& a,
    const math::Vec3d& b) noexcept {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

bool analyticProjectionCameraJacobian(
    const BundleCamera& camera,
    const math::Vec3d& world,
    std::array<double, CameraDof * 2>& jacobian) {
    const auto cameraPoint =
        camera.worldToCamera.transform(world);

    if (
        !std::isfinite(cameraPoint.x) ||
        !std::isfinite(cameraPoint.y) ||
        !std::isfinite(cameraPoint.z) ||
        cameraPoint.z <= 1e-8
    ) {
        return false;
    }

    const double x =
        cameraPoint.x / cameraPoint.z;
    const double y =
        cameraPoint.y / cameraPoint.z;

    if (!std::isfinite(x) || !std::isfinite(y)) {
        return false;
    }

    const auto& d =
        camera.distortion;

    const double r2 =
        x * x +
        y * y;
    const double r4 =
        r2 * r2;
    const double r6 =
        r4 * r2;

    const double radial =
        1.0 +
        d.k1 * r2 +
        d.k2 * r4 +
        d.k3 * r6;

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

    const double dUx =
        radial +
        x * dRadialDx +
        dTxDx;
    const double dUy =
        x * dRadialDy +
        dTxDy;
    const double dVx =
        y * dRadialDx +
        dTyDx;
    const double dVy =
        radial +
        y * dRadialDy +
        dTyDy;

    const double invZ2 =
        1.0 /
        (cameraPoint.z * cameraPoint.z);

    const std::array<math::Vec3d, 3> rotationAxes{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };

    const std::array<math::Vec3d, 6> derivatives{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        cross(rotationAxes[0], cameraPoint),
        cross(rotationAxes[1], cameraPoint),
        cross(rotationAxes[2], cameraPoint),
    };

    for (std::size_t axis = 0; axis < CameraDof; ++axis) {
        math::Vec3d derivative =
            derivatives[axis];

        if (axis >= 3) {
            derivative =
                camera.worldToCamera.rotation.rotate(
                    derivative);
        }

        const double dxn =
            (
                derivative.x * cameraPoint.z -
                cameraPoint.x * derivative.z
            ) * invZ2;

        const double dyn =
            (
                derivative.y * cameraPoint.z -
                cameraPoint.y * derivative.z
            ) * invZ2;

        jacobian[axis] =
            camera.fx *
            (
                dUx * dxn +
                dUy * dyn
            );

        jacobian[CameraDof + axis] =
            camera.fy *
            (
                dVx * dxn +
                dVy * dyn
            );

        if (
            !std::isfinite(jacobian[axis]) ||
            !std::isfinite(
                jacobian[CameraDof + axis])
        ) {
            return false;
        }
    }

    return true;
}

void accumulateNormal(
    std::array<double, CameraDof * CameraDof>& h,
    std::array<double, CameraDof>& g,
    const std::array<double, CameraDof * 2>& jacobian,
    const std::array<double, 2>& residual,
    double weight) {
    for (std::size_t row = 0; row < CameraDof; ++row) {
        const double jx = jacobian[row];
        const double jy = jacobian[CameraDof + row];
        g[row] += weight * (jx * residual[0] + jy * residual[1]);
        for (std::size_t col = 0; col < CameraDof; ++col) {
            const double kx = jacobian[col];
            const double ky = jacobian[CameraDof + col];
            h[row * CameraDof + col] +=
                weight * (jx * kx + jy * ky);
        }
    }
}

void accumulatePointNormal(
    PointBlock& block,
    const std::array<double, PointDof * 2>& jacobian,
    const std::array<double, 2>& residual,
    double weight) {
    for (std::size_t row = 0; row < PointDof; ++row) {
        const double jx = jacobian[row];
        const double jy = jacobian[PointDof + row];
        block.g[row] += weight * (jx * residual[0] + jy * residual[1]);
        for (std::size_t col = 0; col < PointDof; ++col) {
            block.h[row * PointDof + col] +=
                weight * (jx * jacobian[col] +
                          jy * jacobian[PointDof + col]);
        }
    }
}

struct ObservationStats {
    double rms{0.0};
    double meanResidual{0.0};
    std::size_t accepted{0};
    std::size_t rejected{0};
};

ObservationStats observationStats(
    const std::vector<BundleCamera>& cameras,
    const std::vector<BundlePoint>& points,
    const std::vector<BundleObservation>& observations,
    double& robustCost,
    double robustScale,
    double maximumResidualPx) {
    double squared = 0.0;
    double absolute = 0.0;
    double weightedCount = 0.0;
    robustCost = 0.0;
    ObservationStats statistics;
    for (const auto& observation : observations) {
        if (observation.camera >= cameras.size() ||
            observation.point >= points.size()) continue;
        Vec2 pixel;
        if (!project(cameras[observation.camera], points[observation.point].position, pixel)) continue;
        const double dx = pixel.x - observation.u;
        const double dy = pixel.y - observation.v;
        const double residual =
            std::hypot(
                dx,
                dy);
        const double weight =
            std::max(
                0.0,
                observation.weight);

        if (
            weight <= 0.0 ||
            residual >
                maximumResidualPx) {
            ++statistics.rejected;
            continue;
        }

        squared +=
            weight *
            (
                dx * dx +
                dy * dy
            );
        absolute +=
            weight *
            residual;
        weightedCount +=
            weight;
        robustCost +=
            weight *
            math::RobustLoss::adaptiveCost(
                residual,
                robustScale,
                std::clamp(
                    1.0 -
                        residual /
                            std::max(
                                maximumResidualPx,
                                robustScale * 4.0),
                    0.0,
                    1.0));
        ++statistics.accepted;
    }

    if (
        weightedCount <= 0.0 ||
        statistics.accepted == 0U) {
        statistics.rms =
            std::numeric_limits<double>::infinity();
        return statistics;
    }

    statistics.rms =
        std::sqrt(
            squared /
            weightedCount);
    statistics.meanResidual =
        absolute /
        weightedCount;
    return statistics;
}

} // namespace

BundleAdjustmentResult MultiViewBundleAdjuster::solve(
    std::vector<BundleCamera>& cameras,
    std::vector<BundlePoint>& points,
    const std::vector<BundleObservation>& observations,
    const BundleAdjustmentOptions& options) const {
    BundleAdjustmentResult result;
    result.activeCameras = cameras.size();
    result.activePoints = points.size();
    {
        auto initialStats =
            observationStats(
                cameras,
                points,
                observations,
                result.robustCost,
                options.robustScalePx,
                options.maximumResidualPx);
        result.initialRmsPx =
            initialStats.rms;
        result.meanResidualPx =
            initialStats.meanResidual;
        result.acceptedObservations =
            initialStats.accepted;
        result.rejectedObservations =
            initialStats.rejected;
    }

    if (cameras.empty() || points.empty() || observations.size() < 4) {
        result.finalRmsPx = result.initialRmsPx;
        return result;
    }

    std::vector<std::size_t> cameraBlockIndex(cameras.size(), static_cast<std::size_t>(-1));
    std::size_t cameraVariableCount = 0;
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        if (!cameras[i].fixed) {
            cameraBlockIndex[i] = cameraVariableCount++;
        }
    }

    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        std::vector<CameraBlock> cameraBlocks(cameraVariableCount);
        std::vector<PointLinearSystem> pointSystems(points.size());

        for (auto& system : pointSystems) system.cross.reserve(4);

        for (const auto& observation : observations) {
            if (observation.camera >= cameras.size() ||
                observation.point >= points.size()) continue;

            const auto& camera = cameras[observation.camera];
            const auto& point = points[observation.point];
            Vec2 base;
            const auto cameraPoint =
                camera.worldToCamera.transform(
                    point.position);
            if (
                cameraPoint.z <
                    options.minimumDepthMeters ||
                !project(
                    camera,
                    point.position,
                    base)) {
                continue;
            }

            const std::array<double, 2> residual{
                base.x - observation.u,
                base.y - observation.v,
            };
            const double residualNorm =
                std::hypot(residual[0], residual[1]);
            const double adaptiveTrustedFraction =
                std::clamp(
                    1.0 -
                        residualNorm /
                            std::max(
                                options.maximumResidualPx,
                                options.robustScalePx * 4.0),
                    0.0,
                    1.0);

            const double robustWeight =
                std::max(0.0, observation.weight) *
                math::RobustLoss::adaptiveWeight(
                    residualNorm,
                    options.robustScalePx,
                    adaptiveTrustedFraction);
            if (robustWeight <= 0.0) continue;

            const std::size_t pointIndex = observation.point;
            auto& pointSystem = pointSystems[pointIndex];
            std::array<double, PointDof * 2> pointJacobian{};
            if (!analyticProjectionPointJacobian(
                    camera,
                    point.position,
                    pointJacobian)) {
                ++result.pointJacobianFallbackCount;

                for (std::size_t component = 0;
                     component < PointDof;
                     ++component) {
                    auto plusPoint =
                        point.position;
                    auto minusPoint =
                        point.position;

                    if (component == 0) {
                        plusPoint.x +=
                            options.finiteDifferenceStep;
                        minusPoint.x -=
                            options.finiteDifferenceStep;
                    } else if (component == 1) {
                        plusPoint.y +=
                            options.finiteDifferenceStep;
                        minusPoint.y -=
                            options.finiteDifferenceStep;
                    } else {
                        plusPoint.z +=
                            options.finiteDifferenceStep;
                        minusPoint.z -=
                            options.finiteDifferenceStep;
                    }

                    Vec2 plusPixel;
                    Vec2 minusPixel;
                    if (
                        !project(
                            camera,
                            plusPoint,
                            plusPixel) ||
                        !project(
                            camera,
                            minusPoint,
                            minusPixel)
                    ) {
                        continue;
                    }

                    const double denominator =
                        2.0 *
                        options.finiteDifferenceStep;

                    pointJacobian[component] =
                        (
                            plusPixel.x -
                            minusPixel.x
                        ) /
                        denominator;
                    pointJacobian[PointDof + component] =
                        (
                            plusPixel.y -
                            minusPixel.y
                        ) /
                        denominator;
                }
            }
            accumulatePointNormal(
                pointSystem.point,
                pointJacobian,
                residual,
                robustWeight);

            if (cameraBlockIndex[observation.camera] ==
                static_cast<std::size_t>(-1)) {
                continue;
            }

            const std::size_t cameraIndex = cameraBlockIndex[observation.camera];
            auto& cameraBlock = cameraBlocks[cameraIndex];
            std::array<double, CameraDof * 2> cameraJacobian{};
            if (!analyticProjectionCameraJacobian(
                    camera,
                    point.position,
                    cameraJacobian)) {
                ++result.cameraJacobianFallbackCount;

                for (
                    std::size_t component = 0;
                    component < CameraDof;
                    ++component
                ) {
                    const auto plusPose =
                        perturbCamera(
                            camera.worldToCamera,
                            component,
                            options.finiteDifferenceStep);
                    const auto minusPose =
                        perturbCamera(
                            camera.worldToCamera,
                            component,
                            -options.finiteDifferenceStep);

                    auto plusCamera = camera;
                    plusCamera.worldToCamera =
                        plusPose;

                    auto minusCamera = camera;
                    minusCamera.worldToCamera =
                        minusPose;

                    Vec2 plusPixel;
                    Vec2 minusPixel;
                    if (
                        !project(
                            plusCamera,
                            point.position,
                            plusPixel) ||
                        !project(
                            minusCamera,
                            point.position,
                            minusPixel)
                    ) {
                        continue;
                    }

                    const double denominator =
                        2.0 *
                        options.finiteDifferenceStep;

                    cameraJacobian[component] =
                        (
                            plusPixel.x -
                            minusPixel.x
                        ) /
                        denominator;
                    cameraJacobian[CameraDof + component] =
                        (
                            plusPixel.y -
                            minusPixel.y
                        ) /
                        denominator;
                }
            }

            accumulateNormal(
                cameraBlock.h,
                cameraBlock.g,
                cameraJacobian,
                residual,
                robustWeight);

            CrossBlock crossBlock{};
            for (std::size_t row = 0; row < CameraDof; ++row) {
                for (std::size_t col = 0; col < PointDof; ++col) {
                    crossBlock.h[row * PointDof + col] =
                        robustWeight * (
                            cameraJacobian[row] * pointJacobian[col] +
                            cameraJacobian[CameraDof + row] *
                            pointJacobian[PointDof + col]);
                }
            }

            bool found = false;
            for (auto& existing : pointSystem.cross) {
                if (existing.first == cameraIndex) {
                    for (std::size_t i = 0; i < crossBlock.h.size(); ++i) {
                        existing.second.h[i] += crossBlock.h[i];
                    }
                    found = true;
                    break;
                }
            }
            if (!found) pointSystem.cross.push_back({cameraIndex, crossBlock});
        }

        for (auto& block : cameraBlocks) {
            for (std::size_t i = 0; i < CameraDof; ++i) {
                block.h[i * CameraDof + i] += options.damping;
            }
        }
        for (auto& system : pointSystems) {
            for (std::size_t i = 0; i < PointDof; ++i) {
                system.point.h[i * PointDof + i] += options.damping;
            }
        }

        CameraLinearSystem cameraSystem;
        cameraSystem.blocks = cameraBlocks;
        cameraSystem.cameraToBlock = cameraBlockIndex;
        const std::size_t cameraDofCount = cameraVariableCount * CameraDof;
        cameraSystem.schurMatrix.assign(cameraDofCount * cameraDofCount, 0.0);
        cameraSystem.schurVector.assign(cameraDofCount, 0.0);

        for (std::size_t cameraIndex = 0; cameraIndex < cameraVariableCount; ++cameraIndex) {
            for (std::size_t row = 0; row < CameraDof; ++row) {
                cameraSystem.schurVector[
                    cameraIndex * CameraDof + row] =
                    -cameraBlocks[cameraIndex].g[row];
                for (std::size_t col = 0; col < CameraDof; ++col) {
                    cameraSystem.schurMatrix[
                        (cameraIndex * CameraDof + row) * cameraDofCount +
                        cameraIndex * CameraDof + col] =
                        cameraBlocks[cameraIndex].h[row * CameraDof + col];
                }
            }
        }

        std::vector<std::array<double, 9>> pointInverse(points.size());
        std::vector<bool> pointInvertible(points.size(), false);
        for (std::size_t pointIndex = 0; pointIndex < pointSystems.size(); ++pointIndex) {
            std::array<double, 9> inverse{};
            if (!invert3(pointSystems[pointIndex].point.h, inverse)) continue;
            pointInverse[pointIndex] = inverse;
            pointInvertible[pointIndex] = true;

            const auto multiplyInvVector =
                [&](const std::array<double, PointDof>& vector) {
                    std::array<double, PointDof> output{};
                    for (std::size_t row = 0; row < PointDof; ++row) {
                        for (std::size_t col = 0; col < PointDof; ++col) {
                            output[row] += inverse[row * PointDof + col] * vector[col];
                        }
                    }
                    return output;
                };

            const auto pointGradient = multiplyInvVector(pointSystems[pointIndex].point.g);
            for (const auto& left : pointSystems[pointIndex].cross) {
                const std::size_t leftBase = left.first * CameraDof;
                for (std::size_t row = 0; row < CameraDof; ++row) {
                    double contribution = 0.0;
                    for (std::size_t k = 0; k < PointDof; ++k) {
                        contribution += left.second.h[row * PointDof + k] *
                            pointGradient[k];
                    }
                    cameraSystem.schurVector[leftBase + row] += contribution;
                }

                for (const auto& right : pointSystems[pointIndex].cross) {
                    const std::size_t rightBase = right.first * CameraDof;
                    for (std::size_t row = 0; row < CameraDof; ++row) {
                        for (std::size_t col = 0; col < CameraDof; ++col) {
                            double value = 0.0;
                            for (std::size_t k = 0; k < PointDof; ++k) {
                                for (std::size_t l = 0; l < PointDof; ++l) {
                                    value += left.second.h[row * PointDof + k] *
                                        inverse[k * PointDof + l] *
                                        right.second.h[col * PointDof + l];
                                }
                            }
                            cameraSystem.schurMatrix[
                                (leftBase + row) * cameraDofCount +
                                rightBase + col] -= value;
                        }
                    }
                }
            }
        }

        std::vector<double> cameraDelta;
        if (cameraDofCount > 0 &&
            !solveDense(
                cameraSystem.schurMatrix,
                cameraSystem.schurVector,
                cameraDelta)) {
            break;
        }

        double maxTranslationStep = 0.0;
        double maxRotationStep = 0.0;

        for (
            std::size_t cameraIndex = 0;
            cameraIndex < cameras.size();
            ++cameraIndex) {
            const std::size_t blockIndex =
                cameraBlockIndex[cameraIndex];

            if (
                blockIndex ==
                static_cast<std::size_t>(-1)) {
                continue;
            }

            math::Twist6d delta{};
            delta.translation.x =
                cameraDelta[
                    blockIndex *
                        CameraDof +
                    0];
            delta.translation.y =
                cameraDelta[
                    blockIndex *
                        CameraDof +
                    1];
            delta.translation.z =
                cameraDelta[
                    blockIndex *
                        CameraDof +
                    2];
            delta.rotation.x =
                cameraDelta[
                    blockIndex *
                        CameraDof +
                    3];
            delta.rotation.y =
                cameraDelta[
                    blockIndex *
                        CameraDof +
                    4];
            delta.rotation.z =
                cameraDelta[
                    blockIndex *
                        CameraDof +
                    5];

            const double translationMagnitude =
                std::sqrt(
                    delta.translation.x *
                        delta.translation.x +
                    delta.translation.y *
                        delta.translation.y +
                    delta.translation.z *
                        delta.translation.z);

            const double rotationMagnitude =
                std::sqrt(
                    delta.rotation.x *
                        delta.rotation.x +
                    delta.rotation.y *
                        delta.rotation.y +
                    delta.rotation.z *
                        delta.rotation.z);

            const double translationScale =
                translationMagnitude >
                    options.maximumTranslationStepMeters
                    ? options.maximumTranslationStepMeters /
                        std::max(
                            translationMagnitude,
                            1e-12)
                    : 1.0;

            const double rotationScale =
                rotationMagnitude >
                    options.maximumRotationStepRadians
                    ? options.maximumRotationStepRadians /
                        std::max(
                            rotationMagnitude,
                            1e-12)
                    : 1.0;

            delta.translation.x *=
                translationScale;
            delta.translation.y *=
                translationScale;
            delta.translation.z *=
                translationScale;

            delta.rotation.x *=
                rotationScale;
            delta.rotation.y *=
                rotationScale;
            delta.rotation.z *=
                rotationScale;

            maxTranslationStep =
                std::max(
                    maxTranslationStep,
                    translationMagnitude *
                        translationScale);

            maxRotationStep =
                std::max(
                    maxRotationStep,
                    rotationMagnitude *
                        rotationScale);

            cameras[cameraIndex].worldToCamera =
                math::SE3d::exp(
                    delta).compose(
                        cameras[cameraIndex].worldToCamera);
        }

        result.maximumTranslationStepMeters =
            std::max(
                result.maximumTranslationStepMeters,
                maxTranslationStep);
        result.maximumRotationStepRadians =
            std::max(
                result.maximumRotationStepRadians,
                maxRotationStep);

        for (std::size_t pointIndex = 0; pointIndex < pointSystems.size(); ++pointIndex) {
            if (points[pointIndex].fixed || !pointInvertible[pointIndex]) continue;
            const auto& system = pointSystems[pointIndex];
            std::array<double, PointDof> rhs{
                -system.point.g[0],
                -system.point.g[1],
                -system.point.g[2],
            };

            for (const auto& cross : system.cross) {
                const auto cameraIndex = cross.first;
                const auto cameraBlockBase = cameraIndex * CameraDof;
                for (std::size_t row = 0; row < PointDof; ++row) {
                    double value = 0.0;
                    for (std::size_t col = 0; col < CameraDof; ++col) {
                        value += cross.second.h[col * PointDof + row] *
                            cameraDelta[cameraBlockBase + col];
                    }
                    rhs[row] -= value;
                }
            }

            for (std::size_t row = 0; row < PointDof; ++row) {
                double value = 0.0;
                for (std::size_t col = 0; col < PointDof; ++col) {
                    value += pointInverse[pointIndex][row * PointDof + col] * rhs[col];
                }
                if (row == 0) points[pointIndex].position.x += value;
                if (row == 1) points[pointIndex].position.y += value;
                if (row == 2) points[pointIndex].position.z += value;
            }
        }

        double robustCost = 0.0;
        const auto iterationStats =
            observationStats(
                cameras,
                points,
                observations,
                robustCost,
                options.robustScalePx,
                options.maximumResidualPx);

        const double rms =
            iterationStats.rms;

        result.robustCost =
            robustCost;
        result.finalRmsPx =
            rms;
        result.meanResidualPx =
            iterationStats.meanResidual;
        result.acceptedObservations =
            iterationStats.accepted;
        result.rejectedObservations =
            iterationStats.rejected;
        result.iterations =
            iteration + 1U;

        if (
            std::isfinite(rms) &&
            std::abs(
                result.initialRmsPx -
                rms) <
                options.convergencePx) {
            result.converged = true;
            break;
        }
    }

    {
        const auto finalStats =
            observationStats(
                cameras,
                points,
                observations,
                result.robustCost,
                options.robustScalePx,
                options.maximumResidualPx);

        result.finalRmsPx =
            finalStats.rms;
        result.meanResidualPx =
            finalStats.meanResidual;
        result.acceptedObservations =
            finalStats.accepted;
        result.rejectedObservations =
            finalStats.rejected;
    }

    const double residualConfidence =
        std::isfinite(
            result.finalRmsPx)
            ? std::clamp(
                std::exp(
                    -result.finalRmsPx /
                    std::max(
                        options.robustScalePx,
                        1e-6)),
                0.0,
                1.0)
            : 0.0;

    const double acceptanceRatio =
        observations.empty()
            ? 0.0
            : std::clamp(
                static_cast<double>(
                    result.acceptedObservations) /
                static_cast<double>(
                    observations.size()),
                0.0,
                1.0);

    const double depthCondition =
        observations.empty()
            ? 0.0
            : std::clamp(
                1.0 -
                    static_cast<double>(
                        result.rejectedObservations) /
                    static_cast<double>(
                        observations.size()),
                0.0,
                1.0);

    result.conditionScore =
        acceptanceRatio *
        depthCondition;

    result.confidence =
        std::clamp(
            residualConfidence * 0.55 +
            result.conditionScore * 0.45,
            0.0,
            1.0);

    return result;
}

} // namespace nox::native::triangulation

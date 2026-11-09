#include <cassert>
#include <cmath>
#include <vector>

#include "calibration/brown_conrady_distortion.h"
#include "math/se3.h"
#include "triangulation/multiview_bundle_adjuster.h"

using namespace nox::native::triangulation;

static std::pair<double, double> project(
    const BundleCamera& camera,
    const math::Vec3d& point) {
    const auto cameraPoint =
        camera.worldToCamera.transform(point);

    const auto distorted =
        nox::native::calibration::BrownConradyDistortion::distort(
            cameraPoint.x / cameraPoint.z,
            cameraPoint.y / cameraPoint.z,
            camera.distortion);

    return {
        camera.fx * distorted.x + camera.cx,
        camera.fy * distorted.y + camera.cy,
    };
}

int main() {
    BundleCamera camera{};
    camera.fx = 1200.0;
    camera.fy = 1180.0;
    camera.cx = 640.0;
    camera.cy = 360.0;
    camera.distortion = {
        -0.08, 0.012, 0.001, -0.0008, -0.001
    };
    camera.worldToCamera =
        math::SE3d::exp(
            math::Twist6d{
                .translation = {0.0, 0.0, 0.0},
                .rotation = {0.01, -0.02, 0.015},
            });

    BundlePoint seed{};
    seed.position = {0.32, -0.18, 3.8};

    const auto base =
        project(camera, seed.position);

    std::vector<BundleCamera> cameras{camera};
    std::vector<BundlePoint> points{seed};
    std::vector<BundleObservation> observations{
        BundleObservation{
            0,
            0,
            base.first,
            base.second,
            1.0,
        },
    };

    BundleAdjustmentOptions options{};
    options.iterations = 1;

    const auto result =
        MultiViewBundleAdjuster{}.solve(
            cameras,
            points,
            observations,
            options);

    assert(
        result.activeCameras == 1U);
    assert(
        result.activePoints == 1U);
    assert(
        std::isfinite(result.finalRmsPx));
    assert(
        result.pointJacobianFallbackCount == 0U);
    assert(
        result.cameraJacobianFallbackCount == 0U);

    BundleCamera movable = camera;
    movable.fixed = false;
    movable.worldToCamera =
        math::SE3d::exp(
            math::Twist6d{
                .translation = {0.06, -0.02, 0.03},
                .rotation = {0.005, -0.008, 0.004},
            });

    std::vector<BundlePoint> movingPoints{
        BundlePoint{{0.25, -0.10, 3.5}, false},
        BundlePoint{{-0.20, 0.18, 4.2}, false},
        BundlePoint{{0.38, 0.12, 4.8}, false},
        BundlePoint{{-0.15, -0.22, 3.9}, false},
    };

    std::vector<BundleObservation> movingObservations;
    for (std::size_t i = 0; i < movingPoints.size(); ++i) {
        const auto pixel =
            project(camera, movingPoints[i].position);
        movingObservations.push_back(
            BundleObservation{
                0,
                i,
                pixel.first,
                pixel.second,
                1.0,
            });
    }

    MultiViewBundleAdjuster{}.solve(
        std::vector<BundleCamera>{movable},
        movingPoints,
        movingObservations,
        options);

    return 0;
}

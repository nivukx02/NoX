#include <array>
#include <cassert>
#include <cmath>
#include <vector>
#include "triangulation/multiview_bundle_adjuster.h"

int main() {
    using namespace nox::native;

    std::vector<triangulation::BundleCamera> cameras(3);
    for(auto& camera:cameras){
        camera.fx=800.0;
        camera.fy=800.0;
        camera.cx=640.0;
        camera.cy=360.0;
        camera.distortion={};
        camera.fixed=true;
    }

    cameras[0].worldToCamera={
        math::Quaterniond{1,0,0,0},
        {0,0,0},
    };
    cameras[1].worldToCamera={
        math::Quaterniond{1,0,0,0},
        {1.0,0.0,0.0},
    };
    cameras[2].worldToCamera={
        math::Quaterniond{1,0,0,0},
        {0.0,1.0,0.0},
    };

    std::vector<triangulation::BundlePoint> points={
        {{-0.4,-0.2,3.2},false},
        {{0.3,-0.1,3.8},false},
        {{0.1,0.4,4.3},false},
        {{-0.2,0.3,5.0},false},
    };

    const auto projectPoint=[](
        const triangulation::BundleCamera& camera,
        const math::Vec3d& world) {
        const auto p=camera.worldToCamera.transform(world);
        return std::array<double,2>{
            camera.fx*p.x/p.z+camera.cx,
            camera.fy*p.y/p.z+camera.cy,
        };
    };

    std::vector<triangulation::BundleObservation> observations;
    for(std::size_t cameraIndex=0;cameraIndex<cameras.size();++cameraIndex){
        for(std::size_t pointIndex=0;pointIndex<points.size();++pointIndex){
            const auto pixel=projectPoint(
                cameras[cameraIndex],
                points[pointIndex].position);
            observations.push_back({
                cameraIndex,
                pointIndex,
                pixel[0],
                pixel[1],
                1.0,
            });
        }
    }

    points[2].position.z+=0.18;

    triangulation::BundleAdjustmentOptions options;
    options.iterations=25;
    options.damping=1e-2;
    options.robustScalePx=2.0;
    options.maximumTranslationStepMeters=0.08;
    options.maximumRotationStepRadians=0.06;
    options.minimumDepthMeters=0.05;
    options.maximumResidualPx=200.0;

    const auto result=
        triangulation::MultiViewBundleAdjuster{}.solve(
            cameras,
            points,
            observations,
            options);

    assert(result.finalRmsPx<result.initialRmsPx);
    assert(result.finalRmsPx<0.50);
    assert(result.confidence>0.8);
    assert(result.iterations>0);
    assert(result.acceptedObservations >= 10U);
    assert(result.conditionScore > 0.80);
    assert(result.maximumTranslationStepMeters <= 0.08 + 1e-9);
    assert(result.maximumRotationStepRadians <= 0.06 + 1e-9);
    return 0;
}

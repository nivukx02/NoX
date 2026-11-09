#include <cassert>
#include <cmath>
#include <vector>
#include "calibration/robust_bundle_adjustment.h"
#include "calibration/lens_model.h"
#include "geometry3d/imu_extrinsic_calibration.h"
#include "lighting/camera_color_normalizer.h"

int main() {
    std::vector<nox::native::calibration::BundlePoint> points={{0,0,0}};
    const std::vector<nox::native::calibration::BundleObservation> observations={
        {0,{1,0,0},1.0},
    };
    const auto bundle=
        nox::native::calibration::RobustBundleAdjustment{}.solve(points,observations);
    assert(bundle.rms>0.0);
    assert(bundle.valid);
    assert(bundle.acceptedObservations>=1U);
    assert(bundle.confidence>0.0);
    assert(bundle.p95Error>=0.0);
    assert(bundle.maxError>=bundle.p95Error);

    std::vector<nox::native::calibration::BundlePoint> consensusPoints={
        {0.20,0.10,0.00,false},
        {0.80,0.30,0.10,false},
    };
    const auto consensus =
        nox::native::calibration::RobustBundleAdjustment{}.solve(
            consensusPoints,
            {
                {0,{0.0,0.0,0.0},1.0,0},
                {0,{0.01,0.00,0.00},1.0,0},
                {1,{0.80,0.30,0.10},1.0,1},
                {1,{0.82,0.29,0.10},0.8,1},
                {1,{5.0,5.0,5.0},0.2,1},
            },
            25);
    assert(consensus.valid);
    assert(consensus.acceptedObservations>=4U);
    assert(consensus.rejectedObservations>=0U);
    assert(consensus.p95Error<0.05);


    nox::native::calibration::LensParameters lens{};
    assert(nox::native::calibration::LensModel{}.radialScale(0.25,lens)>0.0);

    nox::native::calibration::LensParameters cameraLens{};
    cameraLens.fx = 800.0;
    cameraLens.fy = 810.0;
    cameraLens.cx = 640.0;
    cameraLens.cy = 360.0;
    cameraLens.width = 1280;
    cameraLens.height = 720;
    cameraLens.k1 = 0.08;
    cameraLens.k2 = -0.015;
    cameraLens.k3 = 0.002;
    cameraLens.p1 = 0.001;
    cameraLens.p2 = -0.001;

    nox::native::calibration::LensModel lensModel;
    const auto projected =
        lensModel.project(
            0.23,
            -0.17,
            1.0,
            cameraLens);
    assert(projected.valid);
    assert(std::isfinite(projected.x));
    assert(std::isfinite(projected.y));
    assert(projected.jacobian.determinant > 0.0);

    double recoveredX = 0.0;
    double recoveredY = 0.0;
    const auto normalizedProjected =
        lensModel.distortNormalized(
            0.23,
            -0.17,
            cameraLens);
    assert(normalizedProjected.valid);
    assert(
        lensModel.undistortNormalized(
            normalizedProjected.normalizedX,
            normalizedProjected.normalizedY,
            cameraLens,
            recoveredX,
            recoveredY,
            16U));
    assert(std::abs(recoveredX - 0.23) < 1e-7);
    assert(std::abs(recoveredY + 0.17) < 1e-7);


    const auto extrinsic=
        nox::native::imu::ImuExtrinsicCalibration{}.solve({1,0,0,0},{1,0,0,0});
    assert(extrinsic.valid);

    const auto batchExtrinsic =
        nox::native::imu::ImuExtrinsicCalibration{}.solveBatch({
            {{1,0,0,0},{1,0,0,0},1.0,0.99},
            {{1,0,0,0.01},{1,0,0,0.005},1.0,0.98},
            {{0.9999,0.0003,0.0002,0.005},{1,0.0001,0.0001,0.002},0.95,0.97},
            {{0.98,0.0,0.0,0.19},{1,0,0,0},0.40,0.20},
        });
    assert(batchExtrinsic.acceptedSamples >= 3U);
    assert(batchExtrinsic.sampleConsensus > 0.70);
    assert(batchExtrinsic.valid);
    assert(batchExtrinsic.p95Residual < 0.35);

    const auto color=
        nox::native::lighting::CameraColorNormalizer{}.normalize({0.5,0.5,0.5},{1,1,1});
    assert(color.r==0.5&&color.g==0.5&&color.b==0.5);
    return 0;
}

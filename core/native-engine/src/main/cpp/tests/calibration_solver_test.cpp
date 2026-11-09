#include <cassert>
#include <cmath>
#include <vector>

#include "calibration/brown_conrady_distortion.h"
#include "calibration/calibration_health.h"
#include "calibration/calibration_solver.h"
#include "calibration/rig_graph.h"

using namespace nox::native::calibration;

static DistortedPixel project(
    const CameraIntrinsics& k,
    const RigidPose& pose,
    const Vec3& p) {
    const auto camera = pose.transform(p);
    const auto d = BrownConradyDistortion::distort(
        camera.x / camera.z,
        camera.y / camera.z,
        k.distortion);
    return {k.fx * d.x + k.cx, k.fy * d.y + k.cy};
}

int main() {
    BrownConradyCoefficients coeffs{0.08, -0.015, 0.001, -0.002, 0.004};
    const auto distorted = BrownConradyDistortion::distort(0.23, -0.17, coeffs);
    const auto recovered = BrownConradyDistortion::undistort(
        distorted.x, distorted.y, coeffs, 12u);
    assert(std::abs(recovered.x - 0.23) < 1e-8);
    assert(std::abs(recovered.y + 0.17) < 1e-8);

    RigGraph graph;
    graph.addEdge({"A", "B", {{1.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}});
    graph.addEdge({"B", "C", {{1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}}});
    const auto aToC = graph.resolve("A", "C");
    assert(aToC.has_value());
    assert(std::abs(aToC->translation.x - 1.0) < 1e-9);
    assert(std::abs(aToC->translation.z - 2.0) < 1e-9);

    CameraIntrinsics intrinsics{
        800.0, 800.0, 640.0, 360.0, {}, 1280u, 720u};
    const RigidPose truePose{
        {1.0, 0.0, 0.0, 0.0},
        {0.12, -0.08, 0.18}};
    const std::vector<Vec3> world = {
        {-0.6, -0.4, 3.0}, {0.4, -0.3, 3.2}, {0.7, 0.2, 3.5},
        {-0.3, 0.5, 4.1}, {0.8, 0.6, 4.4}, {-0.9, 0.2, 4.8},
        {0.0, -0.7, 5.1}, {0.3, 0.9, 5.4}
    };

    CameraSeed seed;
    seed.camera_id = "A";
    seed.intrinsics = intrinsics;
    seed.world_to_camera = RigidPose{{1.0,0.0,0.0,0.0}, {0.0,0.0,0.0}};
    for (const auto& p : world) {
        seed.observations.push_back({p, project(intrinsics, truePose, p), 1.0});
    }

    CalibrationProblem problem;
    problem.cameras.push_back(seed);
    problem.huberDeltaPx = 2.0;
    problem.maxIterations = 24;
    problem.refineIntrinsics = true;
    problem.refineDistortion = true;
    problem.refinementIterations = 12;

    const auto solved = CalibrationSolver{}.solve(problem);
    assert(solved.valid);
    assert(solved.cameras.size() == 1);
    assert(solved.image_coverage >= 0.25);
    assert(solved.depth_coverage >= 0.10);
    const auto t = solved.cameras.front().world_to_camera.translation;
    assert(std::abs(t.x - truePose.translation.x) < 1e-3);
    assert(std::abs(t.y - truePose.translation.y) < 1e-3);
    assert(std::abs(t.z - truePose.translation.z) < 1e-3);
    assert(
        solved.cameras.front().reprojection_rms_px <=
        solved.cameras.front().initial_reprojection_rms_px +
            1e-6 ||
        !solved.cameras.front().intrinsics_refined);
    assert(
        solved.cameras.front().intrinsic_jacobian_fallback_count == 0U ||
        !solved.cameras.front().intrinsics_refined);
    assert(
        solved.cameras.front().translation_jacobian_fallback_count == 0U);

    auto narrowProblem=problem;
    narrowProblem.cameras.clear();
    auto narrowSeed=seed;
    narrowSeed.observations.clear();
    for(std::size_t i=0;i<6;++i){
        narrowSeed.observations.push_back(seed.observations[i]);
    }
    narrowProblem.cameras.push_back(narrowSeed);
    const auto narrowSolved=
        CalibrationSolver{}.solve(narrowProblem);
    assert(narrowSolved.image_coverage < solved.image_coverage ||
           narrowSolved.depth_coverage < solved.depth_coverage);

    auto bad = solved;
    bad.residual_rms_px = 12.0;
    const auto health = CalibrationHealth{}.evaluate(bad);
    assert(health.validity == CalibrationValidity::invalid);

    CalibrationResult healthyCoverage;
    healthyCoverage.valid = true;
    healthyCoverage.confidence = 0.96;
    healthyCoverage.residual_rms_px = 0.45;
    healthyCoverage.image_coverage = 0.92;
    healthyCoverage.depth_coverage = 0.88;
    healthyCoverage.observation_coverage = 0.95;
    for (int i = 0; i < 3; ++i) {
        CameraCalibrationResult camera;
        camera.reprojection_rms_px = 0.42 + 0.04 * i;
        camera.confidence = 0.96;
        camera.imageCoverage = 0.90;
        camera.depthCoverage = 0.86;
        healthyCoverage.cameras.push_back(camera);
    }

    const auto strongHealth =
        CalibrationHealth{}.evaluate(
            healthyCoverage,
            1.5,
            8.0);
    assert(strongHealth.calibrated);
    assert(strongHealth.validity == CalibrationValidity::valid);
    assert(strongHealth.calibration_completeness > 0.80);
    assert(strongHealth.geometry_confidence > 0.80);

    healthyCoverage.cameras.front().reprojection_rms_px = 7.0;
    const auto weakCameraHealth =
        CalibrationHealth{}.evaluate(
            healthyCoverage,
            1.5,
            8.0);
    assert(
        weakCameraHealth.validity !=
        CalibrationValidity::valid);

    healthyCoverage.cameras.clear();
    CameraCalibrationResult refinementWeak;
    refinementWeak.reprojection_rms_px = 0.45;
    refinementWeak.confidence = 0.96;
    refinementWeak.imageCoverage = 0.90;
    refinementWeak.depthCoverage = 0.86;
    refinementWeak.observationCoverage = 0.90;
    refinementWeak.intrinsics_refined = true;
    refinementWeak.intrinsic_refinement_improvement_px = 0.30;
    refinementWeak.intrinsic_jacobian_fallback_count = 80U;
    refinementWeak.translation_jacobian_fallback_count = 12U;
    healthyCoverage.cameras.push_back(refinementWeak);
    healthyCoverage.residual_rms_px = 0.45;

    const auto unstableRefinementHealth =
        CalibrationHealth{}.evaluate(
            healthyCoverage,
            1.5,
            8.0);
    assert(
        unstableRefinementHealth.validity !=
        CalibrationValidity::valid);
    assert(
        unstableRefinementHealth.intrinsic_refinement_health <
        0.80);

    return 0;
}

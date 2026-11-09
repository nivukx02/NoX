#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "calibration/brown_conrady_distortion.h"
#include "calibration/rig_graph.h"

namespace nox::native::calibration {

struct CameraIntrinsics {
    double fx{0.0};
    double fy{0.0};
    double cx{0.0};
    double cy{0.0};
    BrownConradyCoefficients distortion{};
    std::uint32_t width{0};
    std::uint32_t height{0};
};

struct CalibrationObservation {
    Vec3 world_point{};
    DistortedPixel pixel{};
    double weight{1.0};
};

struct CameraSeed {
    std::string camera_id;
    CameraIntrinsics intrinsics{};
    RigidPose world_to_camera{};
    std::vector<CalibrationObservation> observations;
};

struct CalibrationProblem {
    std::vector<CameraSeed> cameras;
    double huberDeltaPx{2.0};
    std::uint32_t maxIterations{16};

    // High-end intrinsic/distortion refinement is intentionally conservative.
    // The initial calibration is treated as a prior and the refinement is
    // bounded so noisy marker geometry cannot invent an implausible camera.
    bool refineIntrinsics{true};
    bool refineDistortion{true};
    double focalPriorSigmaRatio{0.10};
    double principalPointPriorSigmaRatio{0.05};
    double distortionPriorSigma{0.05};
    double refinementDamping{1e-3};
    std::uint32_t refinementIterations{12};
};

struct CameraCalibrationResult {
    std::string camera_id;
    CameraIntrinsics intrinsics{};
    RigidPose world_to_camera{};
    double reprojection_rms_px{0.0};
    double conditioning{0.0};
    double confidence{0.0};
    double initial_reprojection_rms_px{0.0};
    double intrinsic_refinement_improvement_px{0.0};
    double imageCoverage{0.0};
    bool intrinsics_refined{false};
    std::uint32_t intrinsic_jacobian_fallback_count{0};
    std::uint32_t translation_jacobian_fallback_count{0};
    double depthCoverage{0.0};
    double observationCoverage{0.0};
    bool valid{false};
};

struct CalibrationResult {
    std::vector<CameraCalibrationResult> cameras;
    double residual_rms_px{0.0};
    double confidence{0.0};
    double image_coverage{0.0};
    double depth_coverage{0.0};
    double observation_coverage{0.0};
    bool valid{false};
};

class CalibrationSolver final {
public:
    CalibrationResult solve(const CalibrationProblem& problem) const;
};

} // namespace nox::native::calibration

#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "calibration/brown_conrady_distortion.h"
#include "math/se3.h"

namespace nox::native::triangulation {

struct BundleCamera {
    math::SE3d worldToCamera{};
    double fx{1.0};
    double fy{1.0};
    double cx{0.0};
    double cy{0.0};
    calibration::BrownConradyCoefficients distortion{};
    bool fixed{false};
};

struct BundlePoint {
    math::Vec3d position{};
    bool fixed{false};
};

struct BundleObservation {
    std::size_t camera{0};
    std::size_t point{0};
    double u{0.0};
    double v{0.0};
    double weight{1.0};
};

struct BundleAdjustmentOptions {
    std::uint32_t iterations{20};
    double damping{1e-3};
    // Retained only as a defensive fallback for degenerate projection/calibration states.
    double finiteDifferenceStep{1e-6};
    double robustScalePx{2.0};
    double convergencePx{1e-5};
    double maximumTranslationStepMeters{0.10};
    double maximumRotationStepRadians{0.08};
    double minimumDepthMeters{0.03};
    double maximumResidualPx{200.0};
    double minimumConditionScore{0.001};
};

struct BundleAdjustmentResult {
    double initialRmsPx{0.0};
    double finalRmsPx{0.0};
    double robustCost{0.0};
    double confidence{0.0};
    double conditionScore{0.0};
    double maximumTranslationStepMeters{0.0};
    double maximumRotationStepRadians{0.0};
    double meanResidualPx{0.0};
    std::uint32_t iterations{0};
    std::size_t activeCameras{0};
    std::size_t activePoints{0};
    std::size_t acceptedObservations{0};
    std::size_t rejectedObservations{0};
    std::size_t pointJacobianFallbackCount{0};
    std::size_t cameraJacobianFallbackCount{0};
    bool converged{false};
};

class MultiViewBundleAdjuster final {
public:
    BundleAdjustmentResult solve(
        std::vector<BundleCamera>& cameras,
        std::vector<BundlePoint>& points,
        const std::vector<BundleObservation>& observations,
        const BundleAdjustmentOptions& options = {}) const;
};

} // namespace nox::native::triangulation

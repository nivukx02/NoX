#include "calibration/calibration_health.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace nox::native::calibration {

CalibrationHealthResult CalibrationHealth::evaluate(
    const CalibrationResult& result,
    double degradedResidualPx,
    double invalidResidualPx) const noexcept {
    CalibrationHealthResult out;

    out.confidence =
        std::clamp(
            result.confidence,
            0.0,
            1.0);

    out.residual_rms_px =
        result.residual_rms_px;
    out.image_coverage =
        std::clamp(
            result.image_coverage,
            0.0,
            1.0);
    out.depth_coverage =
        std::clamp(
            result.depth_coverage,
            0.0,
            1.0);
    out.observation_coverage =
        std::clamp(
            result.observation_coverage,
            0.0,
            1.0);

    if (
        result.cameras.empty() ||
        !std::isfinite(
            result.residual_rms_px) ||
        result.residual_rms_px < 0.0 ||
        result.residual_rms_px >
            invalidResidualPx) {
        out.validity =
            CalibrationValidity::invalid;
        out.calibrated = false;
        return out;
    }

    std::vector<double> cameraResiduals;
    cameraResiduals.reserve(
        result.cameras.size());

    double cameraConfidenceSum = 0.0;
    double cameraCoverageSum = 0.0;
    double cameraDepthCoverageSum = 0.0;
    double intrinsicFallbackWeighted = 0.0;
    double intrinsicFallbackDenominator = 0.0;
    double translationFallbackWeighted = 0.0;
    double translationFallbackDenominator = 0.0;
    double validCameraCount = 0.0;

    for (const auto& camera :
         result.cameras) {
        if (
            !std::isfinite(
                camera.reprojection_rms_px) ||
            !std::isfinite(
                camera.confidence)) {
            continue;
        }

        cameraResiduals.push_back(
            std::max(
                0.0,
                camera.reprojection_rms_px));

        cameraConfidenceSum +=
            std::clamp(
                camera.confidence,
                0.0,
                1.0);

        cameraCoverageSum +=
            std::clamp(
                camera.imageCoverage,
                0.0,
                1.0);

        cameraDepthCoverageSum +=
            std::clamp(
                camera.depthCoverage,
                0.0,
                1.0);

        if (camera.intrinsics_refined) {
            const double intrinsicWork =
                static_cast<double>(
                    camera.intrinsic_jacobian_fallback_count);

            intrinsicFallbackWeighted +=
                std::min(
                    1.0,
                    intrinsicWork / 64.0);
            intrinsicFallbackDenominator +=
                1.0;
        }

        const double translationWork =
            static_cast<double>(
                camera.translation_jacobian_fallback_count);
        translationFallbackWeighted +=
            std::min(
                1.0,
                translationWork / 64.0);
        translationFallbackDenominator +=
            1.0;

        validCameraCount +=
            1.0;
    }

    if (
        cameraResiduals.empty() ||
        validCameraCount <= 0.0) {
        out.validity =
            CalibrationValidity::invalid;
        out.calibrated = false;
        return out;
    }

    std::sort(
        cameraResiduals.begin(),
        cameraResiduals.end());

    const double p95Position =
        0.95 *
        static_cast<double>(
            cameraResiduals.size() - 1U);

    const auto p95Lower =
        static_cast<std::size_t>(
            std::floor(
                p95Position));

    const auto p95Upper =
        std::min(
            cameraResiduals.size() - 1U,
            p95Lower + 1U);

    const double p95Alpha =
        p95Position -
        static_cast<double>(
            p95Lower);

    out.p95_camera_rms_px =
        cameraResiduals[p95Lower] +
        (
            cameraResiduals[p95Upper] -
            cameraResiduals[p95Lower]
        ) *
        p95Alpha;

    const double meanCameraConfidence =
        cameraConfidenceSum /
        validCameraCount;

    const double meanCameraImageCoverage =
        cameraCoverageSum /
        validCameraCount;

    const double meanCameraDepthCoverage =
        cameraDepthCoverageSum /
        validCameraCount;

    const double residualScore =
        std::exp(
            -std::max(
                0.0,
                result.residual_rms_px) /
                std::max(
                    degradedResidualPx,
                    1e-6));

    const double p95CameraScore =
        std::exp(
            -std::max(
                0.0,
                out.p95_camera_rms_px) /
                std::max(
                    degradedResidualPx,
                    1e-6));

    const double imageScore =
        std::clamp(
            (
                out.image_coverage * 0.60 +
                meanCameraImageCoverage * 0.40
            ),
            0.0,
            1.0);

    const double depthScore =
        std::clamp(
            (
                out.depth_coverage * 0.60 +
                meanCameraDepthCoverage * 0.40
            ),
            0.0,
            1.0);

    const double observationScore =
        std::clamp(
            out.observation_coverage,
            0.0,
            1.0);

    out.intrinsic_refinement_fallback_ratio =
        intrinsicFallbackDenominator <= 0.0
            ? 0.0
            : std::clamp(
                intrinsicFallbackWeighted /
                    intrinsicFallbackDenominator,
                0.0,
                1.0);

    out.intrinsic_refinement_health =
        std::clamp(
            1.0 -
                out.intrinsic_refinement_fallback_ratio,
            0.0,
            1.0);

    out.translation_refinement_health =
        std::clamp(
            1.0 -
                (
                    translationFallbackDenominator <= 0.0
                        ? 0.0
                        : translationFallbackWeighted /
                            translationFallbackDenominator
                ),
            0.0,
            1.0);

    out.geometry_confidence =
        std::clamp(
            meanCameraConfidence * 0.42 +
            out.intrinsic_refinement_health * 0.08 +
            out.translation_refinement_health * 0.05 +
            imageScore * 0.20 +
            depthScore * 0.20 +
            observationScore * 0.15,
            0.0,
            1.0);

    out.calibration_completeness =
        std::clamp(
            residualScore * 0.30 +
            p95CameraScore * 0.20 +
            out.geometry_confidence * 0.35 +
            out.confidence * 0.15,
            0.0,
            1.0);

    const bool catastrophic =
        out.geometry_confidence < 0.25 ||
        imageScore < 0.10 ||
        depthScore < 0.05 ||
        meanCameraConfidence < 0.15 ||
        out.p95_camera_rms_px >
            invalidResidualPx * 1.5 ||
        (
            out.intrinsic_refinement_fallback_ratio > 0.75 &&
            out.intrinsic_refinement_health < 0.25
        ) ||
        out.translation_refinement_health < 0.20;

    if (catastrophic) {
        out.validity =
            CalibrationValidity::invalid;
        out.calibrated = false;
        return out;
    }

    const bool degraded =
        result.residual_rms_px >
            degradedResidualPx ||
        out.p95_camera_rms_px >
            degradedResidualPx * 1.5 ||
        out.confidence <
            0.55 ||
        imageScore <
            0.30 ||
        depthScore <
            0.15 ||
        observationScore <
            0.20 ||
        out.geometry_confidence <
            0.55 ||
        out.intrinsic_refinement_health <
            0.55 ||
        out.translation_refinement_health <
            0.55;

    if (degraded) {
        out.validity =
            CalibrationValidity::degraded;
        out.calibrated = false;
        return out;
    }

    out.validity =
        CalibrationValidity::valid;
    out.calibrated = true;
    return out;
}

} // namespace nox::native::calibration

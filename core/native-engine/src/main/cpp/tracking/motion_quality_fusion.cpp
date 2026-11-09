#include "tracking/motion_quality_fusion.h"
#include <algorithm>
#include <cmath>

#include "tracking/observation_quality_model.h"

namespace nox::native::tracking {

MotionQualityResult MotionQualityFusion::evaluate(
    const MotionQualityEvidence& evidence) const noexcept {
    const auto quality = ObservationQualityModel{}.evaluate({
        evidence.detectorConfidence,
        evidence.reprojectionRmsPx <= 0.0
            ? 1.0
            : std::exp(-evidence.reprojectionRmsPx / 3.0),
        evidence.syncConfidence,
        evidence.calibrationConfidence,
        1.0 - evidence.occlusionRatio,
        std::clamp(
            evidence.cameraCoverage * evidence.cameraHealth,
            0.0,
            1.0),
        evidence.rollingShutterPenalty,
        evidence.motionBlurPenalty,
        evidence.depthConditioning,
        evidence.crossViewAgreement,
        evidence.angularDiversity,
        evidence.temporalAgreement,
        evidence.supportingViews,
        evidence.positionUncertaintyMeters,
        evidence.uncertaintyBudgetMeters,
    });

    const bool calibrationAuthoritative =
        std::isfinite(evidence.calibrationConfidence) &&
        evidence.calibrationConfidence >= 0.50;
    const bool uncertaintyAuthoritative =
        evidence.uncertaintyBudgetMeters <= 0.0 ||
        (
            std::isfinite(evidence.positionUncertaintyMeters) &&
            evidence.positionUncertaintyMeters <=
                evidence.uncertaintyBudgetMeters * 3.0
        );

    return {
        quality.finalConfidence,
        quality.finalConfidence,
        quality.uncertainty,
        quality.geometric,
        quality.temporal,
        quality.visibility,
        quality.accepted &&
            calibrationAuthoritative &&
            uncertaintyAuthoritative,
    };
}

} // namespace nox::native::tracking

// CI trigger: validate the current main-head contract without changing runtime behavior.

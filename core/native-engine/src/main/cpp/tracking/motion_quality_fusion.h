#pragma once

#include <cstdint>

namespace nox::native::tracking {

struct MotionQualityEvidence {
    double detectorConfidence{1.0};
    double reprojectionRmsPx{0.0};
    double positionUncertaintyMeters{0.0};
    double syncConfidence{1.0};
    double cameraCoverage{1.0};
    double occlusionRatio{0.0};
    double rollingShutterPenalty{0.0};
    double motionBlurPenalty{0.0};
    double depthConditioning{1.0};
    double crossViewAgreement{1.0};
    double angularDiversity{1.0};
    double temporalAgreement{1.0};
    double calibrationConfidence{1.0};
    double cameraHealth{1.0};
    std::uint32_t supportingViews{1};
    double uncertaintyBudgetMeters{0.08};
};

struct MotionQualityResult {
    double trustScale{0.0};
    double confidenceScore{0.0};
    double uncertaintyScore{1.0};
    double geometricScore{0.0};
    double temporalScore{0.0};
    double visibilityScore{0.0};
    bool accepted{false};
};

class MotionQualityFusion final {
public:
    MotionQualityResult evaluate(
        const MotionQualityEvidence&) const noexcept;
};

} // namespace nox::native::tracking

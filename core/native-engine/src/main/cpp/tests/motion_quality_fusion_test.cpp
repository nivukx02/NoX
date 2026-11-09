#include <cassert>
#include "tracking/motion_quality_fusion.h"

int main() {
    using namespace nox::native::tracking;

    const auto strong =
        MotionQualityFusion{}.evaluate({
            .detectorConfidence = 0.95,
            .reprojectionRmsPx = 0.4,
            .positionUncertaintyMeters = 0.01,
            .syncConfidence = 0.98,
            .cameraCoverage = 0.9,
            .occlusionRatio = 0.05,
            .crossViewAgreement = 0.96,
            .angularDiversity = 0.92,
            .temporalAgreement = 0.98,
            .calibrationConfidence = 0.98,
            .cameraHealth = 0.96,
            .supportingViews = 4,
            .uncertaintyBudgetMeters = 0.08,
        });
    assert(strong.accepted);
    assert(strong.trustScale > 0.7);
    assert(strong.confidenceScore > 0.7);

    const auto weak =
        MotionQualityFusion{}.evaluate({
            .detectorConfidence = 0.35,
            .reprojectionRmsPx = 12.0,
            .positionUncertaintyMeters = 0.75,
            .syncConfidence = 0.3,
            .cameraCoverage = 0.2,
            .occlusionRatio = 0.8,
            .crossViewAgreement = 0.35,
            .angularDiversity = 0.12,
            .temporalAgreement = 0.55,
            .calibrationConfidence = 0.45,
            .cameraHealth = 0.40,
            .supportingViews = 1,
            .uncertaintyBudgetMeters = 0.08,
        });
    assert(!weak.accepted);
    assert(weak.trustScale < strong.trustScale);
    assert(weak.confidenceScore < strong.confidenceScore);

    const auto overconfidentDetectorButBadGeometry =
        MotionQualityFusion{}.evaluate({
            .detectorConfidence = 0.99,
            .reprojectionRmsPx = 0.5,
            .positionUncertaintyMeters = 0.42,
            .syncConfidence = 0.99,
            .cameraCoverage = 1.0,
            .occlusionRatio = 0.02,
            .crossViewAgreement = 0.96,
            .angularDiversity = 0.95,
            .temporalAgreement = 0.99,
            .calibrationConfidence = 0.30,
            .cameraHealth = 0.98,
            .supportingViews = 4,
            .uncertaintyBudgetMeters = 0.08,
        });
    assert(!overconfidentDetectorButBadGeometry.accepted);
    assert(
        overconfidentDetectorButBadGeometry.confidenceScore <
        strong.confidenceScore);

    const auto singleViewUncertain =
        MotionQualityFusion{}.evaluate({
            .detectorConfidence = 0.98,
            .reprojectionRmsPx = 0.6,
            .positionUncertaintyMeters = 0.30,
            .syncConfidence = 0.98,
            .cameraCoverage = 1.0,
            .occlusionRatio = 0.02,
            .crossViewAgreement = 0.45,
            .angularDiversity = 0.20,
            .temporalAgreement = 0.95,
            .calibrationConfidence = 0.98,
            .cameraHealth = 0.98,
            .supportingViews = 1,
            .uncertaintyBudgetMeters = 0.05,
        });
    assert(
        singleViewUncertain.confidenceScore <
        strong.confidenceScore);
    return 0;
}

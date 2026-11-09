#pragma once
#include <cstdint>
namespace nox::native::tracking {

struct ObservationQualityInput {
    double detectorConfidence{0.0};
    double reprojectionConfidence{0.0};
    double synchronizationConfidence{0.0};
    double calibrationConfidence{0.0};
    double occlusionConfidence{0.0};
    double cameraHealth{0.0};
    double rollingShutterPenalty{0.0};
    double motionBlurPenalty{0.0};
    double depthConditioning{0.0};
    double crossViewAgreement{1.0};
    double angularDiversity{1.0};
    double temporalAgreement{1.0};
    std::uint32_t supportingViews{0};
    double uncertaintyMeters{0.0};
    double uncertaintyBudgetMeters{0.0};
};

enum class QualityBottleneck {
    None,
    Detector,
    Geometry,
    Temporal,
    Visibility,
    Camera,
    Uncertainty,
    CrossView,
};

struct ObservationQuality {
    double detector{0.0};
    double geometric{0.0};
    double temporal{0.0};
    double crossView{0.0};
    double visibility{0.0};
    double camera{0.0};
    double finalConfidence{0.0};
    double uncertainty{1.0};
    QualityBottleneck bottleneck{QualityBottleneck::None};
    bool accepted{false};
    bool recoverable{false};
};

class ObservationQualityModel final {
public:
    ObservationQuality evaluate(const ObservationQualityInput&) const noexcept;
};

} // namespace nox::native::tracking

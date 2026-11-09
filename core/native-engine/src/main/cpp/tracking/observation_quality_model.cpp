#include "tracking/observation_quality_model.h"

#include <algorithm>
#include <cmath>

namespace nox::native::tracking {

ObservationQuality ObservationQualityModel::evaluate(
    const ObservationQualityInput& input) const noexcept {
    const auto clamp01=[](double value) {
        return std::clamp(value,0.0,1.0);
    };

    const double detector=clamp01(input.detectorConfidence);
    const double uncertaintyConfidence=
        input.uncertaintyBudgetMeters>0.0 &&
        std::isfinite(input.uncertaintyMeters)
        ? std::exp(
            -std::max(input.uncertaintyMeters,0.0)/
            std::max(input.uncertaintyBudgetMeters,1e-6))
        : 1.0;
    const double crossView=clamp01(
        0.68*input.crossViewAgreement+
        0.32*std::clamp(input.angularDiversity,0.0,1.0));
    const double geometric=clamp01(
        0.52*input.reprojectionConfidence+
        0.28*input.depthConditioning+
        0.20*crossView);
    const double temporal=clamp01(
        input.synchronizationConfidence*
        clamp01(input.temporalAgreement)*
        (1.0-clamp01(input.motionBlurPenalty)));
    const double visibility=clamp01(
        0.75*input.occlusionConfidence+
        0.25*std::min(1.0,
            static_cast<double>(input.supportingViews)/4.0));
    const double camera=clamp01(
        input.cameraHealth*
        (1.0-clamp01(input.rollingShutterPenalty))*
        (1.0-clamp01(input.motionBlurPenalty))*
        input.calibrationConfidence);

    const double weighted=
        std::pow(std::max(detector,1e-6),0.20)*
        std::pow(std::max(geometric,1e-6),0.22)*
        std::pow(std::max(temporal,1e-6),0.18)*
        std::pow(std::max(visibility,1e-6),0.13)*
        std::pow(std::max(camera,1e-6),0.14)*
        std::pow(std::max(crossView,1e-6),0.10)*
        std::pow(std::max(uncertaintyConfidence,1e-6),0.03);

    ObservationQuality output{
        detector,
        geometric,
        temporal,
        crossView,
        visibility,
        camera,
        clamp01(weighted),
        uncertaintyConfidence,
        QualityBottleneck::None,
        false,
        false,
    };

    output.accepted=output.finalConfidence>=0.45 &&
        geometric>=0.35 &&
        temporal>=0.35 &&
        crossView>=0.30;
    output.recoverable=!output.accepted &&
        output.finalConfidence>=0.18;

    double minimum=detector;
    output.bottleneck=QualityBottleneck::Detector;
    if(geometric<minimum){
        minimum=geometric;
        output.bottleneck=QualityBottleneck::Geometry;
    }
    if(temporal<minimum){
        minimum=temporal;
        output.bottleneck=QualityBottleneck::Temporal;
    }
    if(visibility<minimum){
        minimum=visibility;
        output.bottleneck=QualityBottleneck::Visibility;
    }
    if(camera<minimum){
        minimum=camera;
        output.bottleneck=QualityBottleneck::Camera;
    }
    if(crossView<minimum){
        minimum=crossView;
        output.bottleneck=QualityBottleneck::CrossView;
    }
    if(uncertaintyConfidence<minimum &&
       input.uncertaintyBudgetMeters>0.0){
        output.bottleneck=QualityBottleneck::Uncertainty;
    }
    return output;
}

} // namespace nox::native::tracking

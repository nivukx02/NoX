#include <cassert>
#include "tracking/observation_quality_model.h"

int main() {
    using namespace nox::native::tracking;

    ObservationQualityInput good{
        0.95,0.96,0.98,0.97,0.95,0.96,0.02,0.02,0.95,4
    };
    const auto quality=ObservationQualityModel{}.evaluate(good);
    assert(quality.accepted);
    assert(quality.finalConfidence>0.8);
    assert(quality.uncertainty>0.5);

    auto degraded=good;
    degraded.synchronizationConfidence=0.15;
    degraded.detectorConfidence=0.45;
    degraded.reprojectionConfidence=0.30;
    degraded.supportingViews=1;
    const auto degradedResult=ObservationQualityModel{}.evaluate(degraded);
    assert(!degradedResult.accepted);
    assert(degradedResult.recoverable);

    auto crossViewWeak=good;
    crossViewWeak.crossViewAgreement=0.12;
    crossViewWeak.angularDiversity=0.10;
    const auto crossViewResult=ObservationQualityModel{}.evaluate(
        crossViewWeak);
    assert(!crossViewResult.accepted);
    assert(
        crossViewResult.bottleneck==
        QualityBottleneck::CrossView);

    auto invalid=degraded;
    invalid.calibrationConfidence=0.0;
    invalid.cameraHealth=0.0;
    const auto invalidResult=ObservationQualityModel{}.evaluate(invalid);
    assert(!invalidResult.accepted);
    assert(!invalidResult.recoverable);
    return 0;
}

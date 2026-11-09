#include "tracking/observation_quality_model.h"
#include "reconstruction/depth_uncertainty_fusion.h"

#include <cmath>
#include <vector>

int main() {
    using namespace nox::native::tracking;

    ObservationQualityInput input{
        .95,.96,.98,.97,.95,.96,.02,.02,.95,4
    };

    const auto baseline =
        ObservationQualityModel{}.evaluate(input);

    input.uncertaintyMeters = 0.08;
    input.uncertaintyBudgetMeters = 0.02;

    const auto degraded =
        ObservationQualityModel{}.evaluate(input);

    if (!(
        baseline.finalConfidence >
            degraded.finalConfidence &&
        degraded.finalConfidence >= 0.0 &&
        degraded.finalConfidence <= 1.0
    )) {
        return 1;
    }

    using namespace nox::native::reconstruction;

    DepthEstimate lowCorrelationA{};
    lowCorrelationA.depth = 2.00;
    lowCorrelationA.variance = 0.0025;
    lowCorrelationA.confidence = 0.95;
    lowCorrelationA.sensorCorrelation = 0.05;

    DepthEstimate lowCorrelationB = lowCorrelationA;
    lowCorrelationB.depth = 2.01;

    DepthEstimate highCorrelationA = lowCorrelationA;
    highCorrelationA.sensorCorrelation = 0.90;

    DepthEstimate highCorrelationB = lowCorrelationB;
    highCorrelationB.sensorCorrelation = 0.90;

    const auto lowCorrelation =
        DepthUncertaintyFusion{}.fuse(
            lowCorrelationA,
            lowCorrelationB);

    const auto highCorrelation =
        DepthUncertaintyFusion{}.fuse(
            highCorrelationA,
            highCorrelationB);

    if (!std::isfinite(lowCorrelation.variance) ||
        !std::isfinite(highCorrelation.variance) ||
        highCorrelation.variance + 1e-12 <
            lowCorrelation.variance) {
        return 2;
    }

    const auto robust =
        DepthUncertaintyFusion{}.fuseRobust(
            std::vector<DepthEstimate>{
                lowCorrelationA,
                lowCorrelationB,
                highCorrelationA,
                highCorrelationB,
            });

    if (!std::isfinite(robust.variance) ||
        robust.variance <= 0.0 ||
        robust.effectiveSampleCount <= 0.0) {
        return 3;
    }

    return 0;
}

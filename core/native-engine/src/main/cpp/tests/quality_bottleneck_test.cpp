#include "tracking/observation_quality_model.h"

int main() {
    using namespace nox::native::tracking;
    ObservationQualityInput input{
        .9,.9,.9,.9,.9,.9,.0,.0,.9,4
    };
    input.uncertaintyMeters=0.2;
    input.uncertaintyBudgetMeters=0.03;
    const auto quality=ObservationQualityModel{}.evaluate(input);
    return quality.bottleneck==QualityBottleneck::Uncertainty ? 0 : 1;
}

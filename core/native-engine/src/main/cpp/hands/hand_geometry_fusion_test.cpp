#include <cassert>
#include <cmath>
#include "hands/hand_geometry_fusion.h"
#include "hands/finger_kinematics.h"

int main(){
    using namespace nox::native::hands;

    std::vector<Joint> joints={
        {1,{0.01,0.00,0.0500},.95},
        {2,{0.02,0.00,0.0510},.95},
        {3,{0.01,0.02,0.0505},.80},
    };

    const auto state=
        HandGeometryFusion{}.solve(
            joints,
            {0,0,0},
            {0,0,0},
            .9);

    assert(!state.metric);
    assert(state.confidence>.6);
    assert(state.handScale>0.0);
    assert(state.wristCouplingError<1e-9);
    assert(state.palmNormal.z>0.9);

    const auto metric=
        HandGeometryFusion{}.solve(
            joints,
            {0,0,0},
            {0,0,0},
            .9,
            1.0);
    assert(metric.metric);
    assert(metric.metricScaleMetersPerUnit == 1.0);
    assert(metric.handScale > 0.0);

    const auto coupled=
        HandGeometryFusion{}.solve(
            joints,
            {0.2,0,0},
            {0,0,0},
            .9,
            1.0);
    assert(coupled.wristCouplingError>0.19);
    assert(coupled.confidence<metric.confidence);

    const auto rotations=
        FingerKinematics{}.solve(
            joints,
            {.03,.03,.03});
    assert(rotations.size()==3);

    const std::vector<FingerChain> chains={
        {{1,2,3},0.0,2.8},
    };
    const auto chain=
        FingerKinematics{}.solveChains(joints,chains);
    assert(chain.size()==1);
    assert(chain[0].bendRadians<0.2);
    assert(chain[0].withinLimits);
    return 0;
}

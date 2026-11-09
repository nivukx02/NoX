#include <cassert>
#include <cmath>
#include "hands/hand_geometry_fusion.h"
#include "hands/finger_kinematics.h"

int main(){
    using namespace nox::native::hands;
    std::vector<Joint> joints={
        {1,{0,0,.03},.95},
        {2,{.01,0,.05},.95},
        {3,{.02,0,.07},.8},
    };

    const auto state=
        HandGeometryFusion{}.solve(
            joints,
            {0,0,0},
            {0,0,0},
            .9);

    // Geometry alone is not metric-scale provenance.
    assert(!state.metric);
    assert(state.confidence>.6);
    assert(state.handScale>0.0);
    assert(state.wristCouplingError<1e-9);
    assert(state.palmNormal.z>0.9);
    assert(state.kinematicConsistency>0.30);
    assert(state.p95LengthResidual>=0.0);
    assert(state.planarityConfidence>0.0);
    assert(state.wristCouplingConfidence>0.9);

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
    assert(metric.metricConfidence > 0.40);
    assert(metric.accepted);

    const auto coupled=
        HandGeometryFusion{}.solve(
            joints,
            {0.2,0,0},
            {0,0,0},
            .9,
            1.0);
    assert(coupled.wristCouplingError>0.19);
    assert(coupled.confidence<metric.confidence);

    const auto temporalGood =
        HandGeometryFusion{}.solveTemporal(
            joints,
            {0,0,0},
            {0,0,0},
            .9,
            metric,
            1.0 / 120.0,
            HandTemporalOptions{
                .maximumJointVelocityMetersPerSecond = 8.0,
                .maximumInnovationMeters = 0.12,
                .innovationScaleMeters = 0.02,
                .minimumTemporalConfidence = 0.35,
                .maximumDtSeconds = 0.25,
            },
            1.0);

    assert(temporalGood.temporalConfidence > 0.50);
    assert(temporalGood.temporalConsistency > 0.50);
    assert(!temporalGood.temporalRejected);
    assert(temporalGood.p95InnovationMeters >= 0.0);
    assert(temporalGood.estimatedJointVelocityMetersPerSecond >= 0.0);

    auto jumpJoints = joints;
    jumpJoints[0].position.x += 0.25;
    const auto temporalJump =
        HandGeometryFusion{}.solveTemporal(
            jumpJoints,
            {0,0,0},
            {0,0,0},
            .9,
            metric,
            1.0 / 120.0,
            HandTemporalOptions{
                .maximumJointVelocityMetersPerSecond = 4.0,
                .maximumInnovationMeters = 0.05,
                .innovationScaleMeters = 0.01,
                .minimumTemporalConfidence = 0.40,
                .maximumDtSeconds = 0.25,
            },
            1.0);

    assert(temporalJump.temporalRejected);
    assert(!temporalJump.accepted);


    std::vector<Joint> bentJoints={
        {1,{0,0,0},.98},
        {2,{0.03,0,0},.98},
        {3,{0.05,0.02,0},.98},
        {4,{0.07,0.05,0},.98},
    };
    const auto rotations=
        FingerKinematics{}.solve(
            bentJoints,
            {.03,.03,.03,.03});
    assert(rotations.size()==4);
    assert(rotations[2].bendRadians>0.10);
    assert(rotations[2].bendRadians<1.20);
    assert(rotations[2].uncertainty<0.20);


    const std::vector<FingerChain> chains={
        {{1,2,3},0.0,2.8},
    };
    const auto chain=
        FingerKinematics{}.solveChains(joints,chains);
    assert(chain.size()==1);
    assert(chain[0].bendRadians<0.2);
    assert(chain[0].withinLimits);

    auto previousRotations = rotations;
    const std::vector<Joint> movedJoints={
        {1,{0.01,0.00,0.0500},.95},
        {2,{0.02,0.00,0.0510},.95},
        {3,{0.01,0.022,0.0505},.80},
    };

    const auto temporal =
        FingerKinematics{}.solveTemporal(
            movedJoints,
            previousRotations,
            1.0 / 120.0,
            chains);

    assert(temporal.size() == 1U);
    assert(
        temporal.front().angularVelocityRadiansPerSecond >=
        0.0);
    assert(
        temporal.front().temporalConfidence >=
        0.0);
    assert(
        temporal.front().temporalConfidence <=
        1.0);
    return 0;
}

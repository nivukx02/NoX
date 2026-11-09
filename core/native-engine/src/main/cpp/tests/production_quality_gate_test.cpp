#include <cassert>
#include <cmath>

#include "optimization/production_quality_gate.h"

using nox::native::optimization::ProductionQualityGate;
using nox::native::optimization::ProductionQualityInput;
using nox::native::optimization::QualityPolicy;

int main() {
    ProductionQualityGate gate;

    ProductionQualityInput clean{};
    const auto accepted = gate.evaluate(clean);

    assert(accepted.policy == QualityPolicy::Accept);
    assert(accepted.outputAllowed);
    assert(accepted.filteredScore > 0.70);
    assert(accepted.budget.tsdfStride == 1);

    ProductionQualityInput degraded = clean;
    degraded.synchronizationConfidence = 0.28;
    degraded.calibrationConfidence = 0.40;
    degraded.meanUncertaintyMeters = 0.10;
    degraded.uncertaintyLimitMeters = 0.05;
    degraded.runtimeQualityScale = 0.55;

    const auto held = gate.evaluate(degraded);
    assert(
        held.policy == QualityPolicy::Hold ||
        held.policy == QualityPolicy::Degrade);
    assert(held.budget.tsdfStride >= 2);

    ProductionQualityInput faceEyeDegraded = clean;
    faceEyeDegraded.faceEyeConfidence = 0.32;
    faceEyeDegraded.faceEyeFusionAccepted = false;
    faceEyeDegraded.physicsConsistencyScore = 0.94;
    const auto faceEyeRejected = gate.evaluate(faceEyeDegraded);
    assert(faceEyeRejected.policy == QualityPolicy::Reject);
    assert(!faceEyeRejected.outputAllowed);
    assert(faceEyeRejected.hardFailure);
    assert(faceEyeRejected.bottleneck == 
        nox::native::optimization::QualityBottleneck::FaceEye ||
        faceEyeRejected.bottleneck ==
        nox::native::optimization::QualityBottleneck::Physics);

    gate.reset();


    gate.reset();

    ProductionQualityInput gaussianWeak = clean;
    gaussianWeak.gaussian4DEvidenceAvailable = true;
    gaussianWeak.gaussian4DTemporalStability = 0.32;
    gaussianWeak.gaussian4DTrustedRatio = 0.20;
    gaussianWeak.gaussian4DNodeConfidence = 0.42;
    gaussianWeak.gaussian4DBindingConfidence = 0.38;
    gaussianWeak.gaussian4DP95StrainRatio = 0.44;
    gaussianWeak.gaussian4DMaximumStrainRatio = 0.88;

    const auto gaussianRejected =
        gate.evaluate(gaussianWeak);

    assert(
        gaussianRejected.policy == QualityPolicy::Reject ||
        gaussianRejected.policy == QualityPolicy::Degrade);
    assert(
        gaussianRejected.filteredScore <
        accepted.filteredScore);
    assert(
        gaussianRejected.hardFailure ||
        !gaussianRejected.outputAllowed);

    gate.reset();

    ProductionQualityInput gaussianHealthy = clean;
    gaussianHealthy.gaussian4DEvidenceAvailable = true;
    gaussianHealthy.gaussian4DTemporalStability = 0.94;
    gaussianHealthy.gaussian4DTrustedRatio = 0.91;
    gaussianHealthy.gaussian4DNodeConfidence = 0.95;
    gaussianHealthy.gaussian4DBindingConfidence = 0.94;
    gaussianHealthy.gaussian4DP95StrainRatio = 0.08;
    gaussianHealthy.gaussian4DMaximumStrainRatio = 0.16;

    const auto gaussianAccepted =
        gate.evaluate(gaussianHealthy);

    assert(gaussianAccepted.outputAllowed);
    assert(
        gaussianAccepted.filteredScore >
        gaussianRejected.filteredScore);

    gate.reset();

    ProductionQualityInput faceEyeOod = clean;
    faceEyeOod.faceEyeConfidence = 0.94;
    faceEyeOod.faceEyeOutOfDistributionRisk = 0.98;
    const auto ood = gate.evaluate(faceEyeOod);
    assert(ood.policy == QualityPolicy::Reject);
    assert(!ood.outputAllowed);
    assert(ood.hardFailure);

    gate.reset();

    ProductionQualityInput global4DDegraded = clean;
    global4DDegraded.global4DScore = 0.12;
    global4DDegraded.global4DRootMotionConsistency = 0.20;
    global4DDegraded.global4DSceneConsistency = 0.30;
    global4DDegraded.global4DAccepted = false;
    global4DDegraded.global4DPersistentFailure = true;

    const auto global4D = gate.evaluate(global4DDegraded);
    assert(global4D.policy == QualityPolicy::Reject);
    assert(!global4D.outputAllowed);
    assert(global4D.hardFailure);

    ProductionQualityInput faceEyeTrendDegraded = clean;
    faceEyeTrendDegraded.faceEyeConfidence = 0.92;
    faceEyeTrendDegraded.faceEyeReliabilityTrend = -0.42;
    const auto trendDegraded = gate.evaluate(faceEyeTrendDegraded);
    assert(
        trendDegraded.filteredScore <
        accepted.filteredScore);
    assert(
        trendDegraded.budget.eyeTrackingHz >=
        accepted.budget.eyeTrackingHz ||
        trendDegraded.policy != QualityPolicy::Accept);

    gate.reset();

    ProductionQualityInput faceEyeTrendCatastrophic = clean;
    faceEyeTrendCatastrophic.faceEyeReliabilityTrend = -0.90;
    const auto trendRejected = gate.evaluate(faceEyeTrendCatastrophic);
    assert(trendRejected.policy == QualityPolicy::Reject);
    assert(!trendRejected.outputAllowed);
    assert(trendRejected.hardFailure);

    ProductionQualityInput directionalDegraded = clean;
    directionalDegraded.directionalUncertaintyMajor95Meters = 0.34;
    directionalDegraded.directionalUncertaintyDepthMeters = 0.31;
    directionalDegraded.directionalUncertaintyAnisotropy = 8.4;
    directionalDegraded.directionalUncertaintyConfidence = 0.34;
    directionalDegraded.directionalUncertaintyDegradedJointCount = 7.0;

    const auto directional = gate.evaluate(directionalDegraded);
    assert(
        directional.budget.bundleAdjustmentIterations >=
        accepted.budget.bundleAdjustmentIterations);
    assert(
        directional.budget.denseMeshHz <=
        accepted.budget.denseMeshHz);
    assert(
        directional.bottleneck ==
            nox::native::optimization::QualityBottleneck::DirectionalUncertainty ||
        directional.filteredScore < accepted.filteredScore);

    gate.reset();

    ProductionQualityInput global4DComponentRisk = clean;
    global4DComponentRisk.global4DScore = 0.94;
    global4DComponentRisk.global4DAccepted = true;
    global4DComponentRisk.global4DDirectionalUncertaintyConsistency = 0.22;
    global4DComponentRisk.global4DFaceEyeConsistency = 0.25;

    const auto global4DComponentDecision =
        gate.evaluate(global4DComponentRisk);
    assert(
        global4DComponentDecision.filteredScore <
        accepted.filteredScore);

    ProductionQualityInput weakCrossModalMetric = clean;
    weakCrossModalMetric.faceMetricEvidenceAuthority = 0.10;
    const auto metricAuthorityRejected =
        gate.evaluate(weakCrossModalMetric);
    assert(metricAuthorityRejected.policy == QualityPolicy::Reject);
    assert(!metricAuthorityRejected.outputAllowed);
    assert(metricAuthorityRejected.hardFailure);

    gate.reset();

    ProductionQualityInput weakCrossModalEpipolar = clean;
    weakCrossModalEpipolar.handEpipolarConsistency = 0.12;
    const auto epipolarAuthorityRejected =
        gate.evaluate(weakCrossModalEpipolar);
    assert(epipolarAuthorityRejected.policy == QualityPolicy::Reject);
    assert(!epipolarAuthorityRejected.outputAllowed);
    assert(epipolarAuthorityRejected.hardFailure);

    gate.reset();

    ProductionQualityInput physicsDegraded = clean;
    physicsDegraded.physicsConsistencyScore = 0.34;
    physicsDegraded.physicsConstraintAccepted = true;
    const auto physics = gate.evaluate(physicsDegraded);
    assert(
        physics.budget.bundleAdjustmentIterations >=
        accepted.budget.bundleAdjustmentIterations);
    assert(physics.budget.denseMeshHz <= accepted.budget.denseMeshHz);

    gate.reset();

    ProductionQualityInput rigDegraded = clean;
    rigDegraded.globalRigTrust = 0.36;
    const auto rigDecision = gate.evaluate(rigDegraded);
    assert(
        rigDecision.bottleneck ==
            nox::native::optimization::QualityBottleneck::RigCoverage ||
        rigDecision.filteredScore < accepted.filteredScore);

    ProductionQualityInput catastrophic = degraded;
    catastrophic.hardFailure = true;
    catastrophic.identitySwapRisk = 1.0;

    const auto rejected = gate.evaluate(catastrophic);
    assert(rejected.policy == QualityPolicy::Reject);
    assert(!rejected.outputAllowed);
    assert(rejected.hardFailure);

    gate.reset();

    ProductionQualityInput rigStabilityFailure = clean;
    rigStabilityFailure.rigStabilityRequired = true;
    rigStabilityFailure.rigStabilityConfidence = 0.42;
    rigStabilityFailure.rigUnstable = true;
    const auto rigFailure =
        gate.evaluate(rigStabilityFailure);
    assert(rigFailure.policy == QualityPolicy::Reject);
    assert(!rigFailure.outputAllowed);
    assert(rigFailure.hardFailure);

    gate.reset();

    ProductionQualityInput metrologyFailure = clean;
    metrologyFailure.dynamicMetrologyRequired = true;
    metrologyFailure.dynamicMetrologyValid = true;
    metrologyFailure.dynamicMetrologyP95Millimeters = 1.8;
    metrologyFailure.dynamicMetrologyMaxErrorMillimeters = 2.6;
    metrologyFailure.dynamicMetrologyConfidence = 0.88;
    const auto metrologyDecision =
        gate.evaluate(metrologyFailure);
    assert(
        metrologyDecision.policy == QualityPolicy::Hold ||
        metrologyDecision.policy == QualityPolicy::Reject);
    assert(
        metrologyDecision.bottleneck ==
            nox::native::optimization::QualityBottleneck::Metrology ||
        metrologyDecision.filteredScore < accepted.filteredScore);

    gate.reset();

    ProductionQualityInput referenceEligible = clean;
    referenceEligible.rigStabilityConfidence = 0.97;
    referenceEligible.rigUnstable = false;
    referenceEligible.dynamicMetrologyRequired = true;
    referenceEligible.dynamicMetrologyValid = true;
    referenceEligible.dynamicMetrologyP95Millimeters = 0.80;
    referenceEligible.dynamicMetrologyMaxErrorMillimeters = 1.50;
    referenceEligible.dynamicMetrologyConfidence = 0.95;
    referenceEligible.globalRigTrust = 0.96;

    const auto reference =
        gate.evaluate(referenceEligible);

    if (reference.policy == QualityPolicy::Accept) {
        assert(reference.budget.allowReferenceOptimization);
    }

    gate.reset();

    ProductionQualityInput referenceWithoutMetrology =
        referenceEligible;
    referenceWithoutMetrology.dynamicMetrologyValid = false;

    const auto noReference =
        gate.evaluate(referenceWithoutMetrology);
    assert(!noReference.budget.allowReferenceOptimization);


    gate.reset();
    const auto resetDecision = gate.evaluate(clean);
    assert(resetDecision.policy == QualityPolicy::Accept);

    return 0;
}

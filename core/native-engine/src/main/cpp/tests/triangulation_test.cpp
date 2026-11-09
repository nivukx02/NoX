#include <cassert>
#include <cmath>
#include <limits>
#include "triangulation/weighted_triangulator.h"
#include "triangulation/cheirality_gate.h"
#include "triangulation/covariance_propagator.h"

int main() {
    using namespace nox::native::triangulation;
    const std::vector<RayObservation> rays={
        {{0,0,0},{0,0,1},1},
        {{1,0,0},{-1,0,5},1},
        {{0,1,0},{0,-1,5},1},
    };
    const auto result=WeightedTriangulator{}.solve(rays);
    assert(result.valid);
    assert(std::isfinite(result.point.z));
    assert(result.point.z>4.0&&result.point.z<6.0);
    assert(CheiralityGate::allInFront(result.point,rays));
    assert(result.covarianceTrace>0.0);
    assert(std::isfinite(result.covarianceMatrix[0]));
    assert(result.covarianceP95>0.0);
    assert(result.robustInlierRatio>=0.99);
    assert(result.angularDiversity>0.05);
    assert(result.cheiralityConfidence>=0.99);
    assert(result.minimumAngularSeparationRadians > 0.05);
    assert(result.meanAngularSeparationRadians > 0.05);
    assert(result.baselineMeters > 0.5);
    assert(result.depthConditioning > 0.01);
    assert(result.metricScaleConfidence > 0.5);
    assert(result.distinctCameraCount >= 1u);
    assert(result.viewIndependenceScore > 0.0);
    assert(result.cameraIdentityUnknown);

    const auto covariance=
        CovariancePropagator::propagate(result,0.5);
    assert(covariance.trace>0.0);
    assert(covariance.p95Radius>0.0);
    assert(covariance.p99Radius >= covariance.p95Radius);
    assert(covariance.minimumVariance > 0.0);
    assert(covariance.maximumVariance >= covariance.minimumVariance);
    assert(covariance.conditionNumber >= 1.0);
    assert(covariance.anisotropy >= 0.0);
    assert(covariance.anisotropy <= 1.0);
    assert(covariance.positiveDefinite);
    assert(covariance.confidence > 0.0);
    assert(covariance.numericalRepairMagnitude >= 0.0);
    if (covariance.numericallyRepaired) {
        assert(covariance.numericalRepairMagnitude < 1.0);
    }

    const auto weighted=WeightedTriangulator{}.solve({
        {{0,0,0},{0,0,1},1,1.0},
        {{10,0,0},{-10,0,10},1,10.0},
        {{0,10,0},{0,-10,10},1,1.0},
    });
    assert(weighted.valid);

    const auto robust=WeightedTriangulator{}.solve({
        {{0,0,0},{0,0,1},1,1.0},
        {{1,0,0},{-1,0,5},1,1.0},
        {{0,1,0},{0,-1,5},1,1.0},
        {{8,8,0},{0,0,1},0.8,1.0},
    });
    assert(robust.valid);
    assert(robust.rejectedRays>=1u);
    assert(robust.robustInlierRatio<1.0);
    assert(robust.confidence>0.15);
    assert(robust.distinctCameraCount >= 1u);

    const auto behind=WeightedTriangulator{}.solve({
        {{0,0,0},{0,0,1},1,1.0},
        {{1,0,0},{-1,0,-5},1,1.0},
        {{0,1,0},{0,-1,-5},1,1.0},
    });
    assert(!behind.valid);
    assert(behind.cheiralityConfidence<1.0);

    const auto zeroDirection=WeightedTriangulator{}.solve({
        {{0,0,0},{0,0,0},1,1.0},
        {{1,0,0},{-1,0,5},1,1.0},
        {{0,1,0},{0,-1,5},1,1.0},
    });
    assert(!zeroDirection.valid);
    assert(zeroDirection.contributingRays<=2u);

    const auto multiCamera=WeightedTriangulator{}.solve({
        {{0,0,0},{0,0,1},1,1.0,7},
        {{1,0,0},{-1,0,5},1,1.0,8},
        {{0,1,0},{0,-1,5},1,1.0,9},
    });
    assert(multiCamera.cameraIdentityUnknown==false);
    assert(multiCamera.distinctCameraCount>=3u);
    assert(multiCamera.viewIndependenceScore>0.70);
    assert(multiCamera.valid);

    const auto degenerate=WeightedTriangulator{}.solve({
        {{0,0,0},{0,0,1},1,1.0},
        {{0.001,0,0},{0,0,1},1,1.0},
    });
    assert(!degenerate.valid);

    const auto nonFiniteOrigin =
        WeightedTriangulator{}.solve({
            {{std::numeric_limits<double>::quiet_NaN(),0,0},
                {0,0,1},1,1.0},
            {{1,0,0},{-1,0,5},1,1.0},
            {{0,1,0},{0,-1,5},1,1.0},
        });
    assert(!nonFiniteOrigin.valid);

    return 0;
}

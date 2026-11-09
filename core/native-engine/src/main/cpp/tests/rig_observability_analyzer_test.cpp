#include <cassert>
#include "calibration/rig_observability_analyzer.h"

int main(){
    using namespace nox::native::calibration;

    const auto good=RigObservabilityAnalyzer{}.analyze(
        4,
        {
            {"C0","C1",{0.30,0,0},1.0,0.05,0.05},
            {"C1","C2",{0,0.30,0},0.9,0.06,0.06},
            {"C2","C3",{0,0,0.30},0.9,0.06,0.06},
            {"C0","C2",{0.20,0.20,0},0.8,0.08,0.08},
        });
    assert(good.connectedComponents==1);
    assert(good.baselineCoverage>0.5);
    assert(good.angularDiversity>0.3);
    assert(good.minimumBaselineMeters > 0.0);
    assert(good.p10BaselineMeters > 0.0);
    assert(good.p95BaselineMeters >= good.p10BaselineMeters);
    assert(good.leverageDiversity > 0.3);
    assert(good.p95EdgeUncertainty < 0.20);
    assert(good.validEdgeCount == 4U);
    assert(good.observable);
    assert(good.referenceGradeObservable);

    const auto weak=RigObservabilityAnalyzer{}.analyze(
        4,
        {
            {"C0","C1",{0.01,0,0},0.2,1.0,1.0},
            {"C1","C2",{0.01,0,0},0.2,1.0,1.0},
            {"C2","C3",{0.01,0,0},0.2,1.0,1.0},
        });
    assert(weak.baselineCoverage<0.2);
    assert(weak.weakCameraCount > 0U);
    assert(weak.referenceGradeObservable == false);
    assert(!weak.observable);
    return 0;
}

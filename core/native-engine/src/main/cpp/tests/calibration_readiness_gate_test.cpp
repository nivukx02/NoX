#include <cassert>
#include "calibration/calibration_readiness_gate.h"

int main(){
    using namespace nox::native::calibration;

    CalibrationReadinessInput ready{
        {CalibrationValidity::valid,0.92,1.2,true},
        {1.0,0.8,0.9,0.85,1.0,1.0,1,true},
        0.96,
        1.2,
        true,
        0.92,
    };
    const auto result=CalibrationReadinessGate{}.evaluate(ready);
    assert(result.ready);
    assert(result.blockers.empty());
    assert(result.confidence>0.8);

    auto degraded=ready;
    degraded.synchronizationConfidence=0.2;
    const auto bad=CalibrationReadinessGate{}.evaluate(degraded);
    assert(!bad.ready);
    assert(!bad.blockers.empty());
    return 0;
}

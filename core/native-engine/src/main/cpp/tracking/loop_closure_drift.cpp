#include "tracking/loop_closure_drift.h"
#include <algorithm>
#include <cmath>
namespace nox::native::tracking {
DriftCorrection LoopClosureDriftCorrector::solve(const LoopClosureInput&i,double pt,double rt)const noexcept{
    const double p=std::clamp(i.positionError/std::max(pt,1e-9),0.0,1.0);
    const double r=std::clamp(i.rotationError/std::max(rt,1e-9),0.0,1.0);
    const double confidence=std::clamp(i.confidence*(.65*p+.35*r),0.0,1.0);
    return{p*.2,r*.2,confidence,confidence>.45&&(p>.25||r>.25)};
}
}
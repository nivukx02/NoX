#include <cassert>
#include <cmath>
#include "calibration/graph_cycle_validator.h"

int main(){
    using namespace nox::native::calibration;
    const auto result=GraphCycleValidator{}.evaluate(
        {
            {0,0,0,{1,0,0,0}},
            {1,0,0,{1,0,0,0}},
            {1,1,0,{1,0,0,0}},
        },
        {
            {"C0","C1",{1,0,0,{}},1.0,0.02,0.02},
            {"C1","C2",{0,1,0,{}},1.0,0.02,0.02},
            {"C0","C2",{1,1,0,{}},1.0,0.02,0.02},
        });
    assert(result.cyclesChecked==3);
    assert(result.consistent);
    assert(result.confidence>0.8);
    assert(std::isfinite(result.rmsTranslation));

    auto inconsistentEdges=std::vector<GraphEdge>{
        {"C0","C1",{1.2,0,0,{}},1.0,0.02,0.02},
        {"C1","C2",{0,1,0,{}},1.0,0.02,0.02},
        {"C0","C2",{1,1,0,{}},1.0,0.02,0.02},
    };
    const auto bad=GraphCycleValidator{}.evaluate(
        {
            {0,0,0,{1,0,0,0}},
            {1,0,0,{1,0,0,0}},
            {1,1,0,{1,0,0,0}},
        },
        inconsistentEdges);
    assert(!bad.consistent);
    assert(bad.inconsistentEdges>0);
    return 0;
}

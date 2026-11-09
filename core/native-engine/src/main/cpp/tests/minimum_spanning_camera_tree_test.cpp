#include <cassert>
#include "calibration/minimum_spanning_camera_tree.h"

int main(){
    using namespace nox::native::calibration;
    const auto result=MinimumSpanningCameraTree{}.solve(
        4,
        {
            {"C0","C1",{},1.0,0.2,0.2},
            {"C1","C2",{},0.9,0.3,0.3},
            {"C0","C2",{},0.1,0.01,0.01},
            {"C2","C3",{},0.8,0.2,0.2},
            {"C0","C3",{},0.2,0.05,0.05},
        });
    assert(result.connected);
    assert(result.selectedEdgeIndices.size()==3);
    assert(result.selectedEdgeIndices[0]==0);
    return 0;
}

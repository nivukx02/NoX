#include <cassert>
#include "scene/dynamic_motion_separator.h"

int main(){
    using namespace nox::native::scene;

    std::vector<Point> previous={
        {{0,0,0},{0,0,1},0.9,100,false},
        {{1,0,0},{0,0,1},0.9,100,false},
    };
    std::vector<Point> current={
        {{0.01,0,0},{0,0,1},0.9,105,false},
        {{1.10,0,0},{0,0,1},0.9,105,false},
    };

    const auto result=DynamicMotionSeparator{}.classify(current,previous);
    assert(result.evaluated==2);
    assert(result.dynamic==1);
    assert(result.unmatched==0);
    assert(result.matchRatio>0.99);
    assert(result.confidence>0.70);
    assert(current[0].dynamic==false);
    assert(current[1].dynamic==true);

    std::vector<Point> sparse={
        {{10,10,10},{0,0,1},0.9,105,false},
    };
    const auto sparseResult=
        DynamicMotionSeparator{}.classify(
            sparse,
            previous);
    assert(sparseResult.unmatched==1);
    assert(sparseResult.matchRatio==0.0);
    assert(sparseResult.confidence<0.70);
    return 0;
}

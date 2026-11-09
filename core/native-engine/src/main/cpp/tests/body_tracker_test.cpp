#include <cassert>
#include <cmath>
#include <vector>
#include "tracking/body_tracker.h"
using namespace nox::native::tracking;
int main(){
    BodyTracker tracker;
    auto a=tracker.update({{1,{0,0,0},1.0}},1'000'000'000ull);
    assert(a.healthy);
    tracker.update({{1,{0.1,0,0},1.0}},1'100'000'000ull);
    auto c=tracker.update({{1,{0.2,0,0},1.0}},1'200'000'000ull);
    assert(c.joints.size()>0);
    assert(c.joints.front().velocity.x>0.5);
    assert(c.joints.front().velocityMagnitude>0.5);
    assert(c.joints.front().innovationMagnitude>=0.0);
    assert(c.joints.front().jerkMagnitude>=0.0);
    assert(c.dynamic_stability>0.0);
    assert(c.micro_motion_retention>0.0);
    assert(c.p95_innovation>=0.0);

    /*
     * The robust innovation gate must reject an implausible spatial jump while preserving the
     * ordinary high-speed updates exercised above.
     */
    auto spike=tracker.update(
        {{1,{1.25,0,0},1.0}},
        1'216'000'000ull);
    assert(spike.joints.size()>0);
    assert(spike.joints.front().rejected);
    assert(spike.joints.front().predicted);
    assert(spike.joints.front().position.x < 0.45);
    assert(spike.joints.front().innovationMagnitude > 0.5);

    auto p=tracker.predict(1'300'000'000ull);
    assert(p.joints.front().predicted);
    assert(p.joints.front().uncertainty>=c.joints.front().uncertainty);
}

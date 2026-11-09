#include <cassert>
#include <cmath>
#include "geometry3d/pose_timeline.h"

int main() {
    using namespace nox::native;

    geometry3d::PoseTimeline timeline;
    assert(timeline.addSample({
        0,
        {{1,0,0,0},{0,0,0}},
    }));
    assert(timeline.addSample({
        10'000'000,
        {{1,0,0,0},{1,0,0}},
    }));

    const auto center=timeline.sample(5'000'000);
    assert(std::abs(center.translation.x-0.5)<1e-9);

    const auto top=timeline.sampleRow(5'000'000,1.0,10'000'000);
    const auto bottom=timeline.sampleRow(5'000'000,0.0,10'000'000);
    assert(top.translation.x>center.translation.x);
    assert(bottom.translation.x<center.translation.x);

    const auto before=timeline.sample(0);
    assert(std::abs(before.translation.x)<1e-9);
    const auto after=timeline.sample(10'000'000);
    assert(std::abs(after.translation.x-1.0)<1e-9);
    return 0;
}

#include <cassert>
#include <cmath>
#include "capture/rolling_shutter_pose_compensator.h"

int main(){
    using namespace nox::native;
    geometry3d::PoseTimeline timeline;
    timeline.addSample({0,{{1,0,0,0},{0,0,0}}});
    timeline.addSample({10'000'000,{{1,0,0,0},{1,0,0}}});

    const auto row= capture::RollingShutterPoseCompensator{}.compensatePoint(
        {1,0,2},
        5'000'000,
        1.0,
        10'000'000,
        timeline);
    assert(std::abs(row.x-0.5)>0.001);
    assert(std::isfinite(row.x));
    assert(std::isfinite(row.z));

    const auto detailed =
        capture::RollingShutterPoseCompensator{}
            .compensatePointHighEnd(
                {1,0,2},
                5'000'000,
                1.0,
                10'000'000,
                timeline);
    assert(detailed.compensated);
    assert(detailed.rowOffsetSeconds > 0.0);
    assert(detailed.translationCorrectionMeters >= 0.0);
    assert(detailed.rotationCorrectionRadians >= 0.0);
    assert(detailed.confidence >= 0.0);
    assert(detailed.confidence <= 1.0);
    assert(detailed.timelineConfidence >= 0.0);
    assert(detailed.timelineConfidence <= 1.0);
    return 0;
}

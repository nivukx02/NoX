#include <cassert>
#include <cmath>
#include <vector>

#include "scene/scene_point_map.h"

int main() {
    using namespace nox::native::scene;

    ScenePointMap map(0.02, 1024);

    map.integrateEvidence(
        {0.0,0.0,1.0},
        {0,0,1},
        0.95,
        1'000'000'000ULL,
        0.001,
        0.02,
        0.98);

    map.integrateEvidence(
        {0.001,0.0005,1.0002},
        {0,0,1},
        0.92,
        1'016'000'000ULL,
        0.0015,
        0.03,
        0.96);

    map.integrateEvidence(
        {0.10,0.0,1.0},
        {0,0,1},
        0.90,
        1'016'000'000ULL,
        0.002,
        0.85,
        0.80);

    const auto points = map.snapshot();
    assert(points.size() == 2U);
    assert(map.size() == points.size());

    const auto nearest =
        points.front().p.x < points.back().p.x
            ? points.front()
            : points.back();

    assert(nearest.observationCount >= 2U);
    assert(nearest.uncertaintyMeters > 0.0);
    assert(nearest.normalConsistency > 0.0);
    assert(nearest.temporalConsistency > 0.0);

    const auto dynamicIt =
        points.front().dynamicProbability >
                points.back().dynamicProbability
            ? points.front()
            : points.back();

    assert(dynamicIt.dynamicProbability >= 0.20);

    map.reset();
    assert(map.size() == 0U);
    assert(map.snapshot().empty());

    return 0;
}

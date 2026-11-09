#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "scene/static_dynamic_separator.h"

int main() {
    using namespace nox::native::scene;

    const auto makePoint =
        [](double x, double y, double z,
           double confidence,
           std::uint64_t timestamp,
           bool dynamic) {
            Point point{};
            point.p = {x, y, z};
            point.normal = {0.0, 0.0, 1.0};
            point.confidence = confidence;
            point.lastSeen = timestamp;
            point.dynamic = dynamic;
            return point;
        };

    std::vector<Point> previous{
        makePoint(0.00, 0.00, 1.00, 0.95, 1'000'000'000ULL, false),
        makePoint(0.10, 0.00, 1.00, 0.95, 1'000'000'000ULL, false),
        makePoint(0.20, 0.00, 1.00, 0.90, 1'000'000'000ULL, false),
    };

    std::vector<Point> current{
        makePoint(0.001, 0.000, 1.001, 0.95, 1'016'000'000ULL, false),
        makePoint(0.102, 0.001, 1.000, 0.92, 1'016'000'000ULL, false),
        makePoint(0.260, 0.000, 1.000, 0.90, 1'016'000'000ULL, false),
    };

    const auto stats =
        StaticDynamicSeparator::classifyTemporal(
            current,
            previous,
            1'016'000'000ULL,
            1'000'000'000ULL,
            DynamicSeparatorOptions{
                .maximumStaticDisplacementMeters = 0.004,
                .dynamicDisplacementMeters = 0.020,
                .maximumMatchDistanceMeters = 0.10,
                .staticConfidenceFloor = 0.30,
                .dynamicConfidencePenalty = 0.35,
                .temporalHalfLifeSeconds = 0.50,
                .maximumPoints = 1024,
            });

    assert(stats.staticCount >= 2U);
    assert(stats.dynamicCount >= 1U);
    assert(stats.dynamicRatio > 0.20);
    assert(std::isfinite(stats.p95DisplacementMeters));
    assert(stats.confidence > 0.50);
    assert(current[2].dynamic);
    assert(!current[0].dynamic);

    return 0;
}

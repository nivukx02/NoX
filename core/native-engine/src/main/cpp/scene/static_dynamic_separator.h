#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "scene/scene_point_map.h"

namespace nox::native::scene {

struct DynamicSeparatorOptions {
    double maximumStaticDisplacementMeters{0.003};
    double dynamicDisplacementMeters{0.020};
    double maximumMatchDistanceMeters{0.05};
    double staticConfidenceFloor{0.30};
    double dynamicConfidencePenalty{0.35};
    double temporalHalfLifeSeconds{0.50};
    std::size_t maximumPoints{250'000};
};

struct DynamicSeparatorStats {
    std::size_t staticCount{0};
    std::size_t dynamicCount{0};
    std::size_t ambiguousCount{0};
    double dynamicRatio{0.0};
    double meanDisplacementMeters{0.0};
    double p95DisplacementMeters{0.0};
    double confidence{0.0};
};

class StaticDynamicSeparator final {
public:
    static void classify(
        std::vector<Point>&,
        std::uint64_t,
        std::uint64_t);

    [[nodiscard]] static DynamicSeparatorStats classifyTemporal(
        std::vector<Point>& current,
        const std::vector<Point>& previous,
        std::uint64_t timestampNanos,
        std::uint64_t previousTimestampNanos,
        const DynamicSeparatorOptions& options = {});

private:
    static double clamp01(double value) noexcept;
    static double distance(
        const Point&,
        const Point&) noexcept;
    static double percentile95(
        std::vector<double>) noexcept;
};

} // namespace nox::native::scene

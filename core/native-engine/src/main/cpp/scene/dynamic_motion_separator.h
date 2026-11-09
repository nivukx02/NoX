#pragma once
#include <cstdint>
#include <vector>
#include "scene/scene_point_map.h"

namespace nox::native::scene {

struct DynamicMotionOptions {
    double displacementThresholdMeters{0.03};
    double normalChangeRadians{0.35};
    double dynamicConfidenceThreshold{0.55};
    double maximumMatchDistanceMeters{0.50};
    double confidenceFloor{0.10};
    std::uint64_t maxHistoryGapNanos{100'000'000ULL};
};

struct DynamicMotionResult {
    std::size_t evaluated{0};
    std::size_t dynamic{0};
    std::size_t unmatched{0};
    double dynamicRatio{0.0};
    double matchRatio{0.0};
    double confidence{0.0};
};

class DynamicMotionSeparator final {
public:
    DynamicMotionResult classify(
        std::vector<Point>& current,
        const std::vector<Point>& previous,
        const DynamicMotionOptions& options = {}) const;
};

} // namespace nox::native::scene

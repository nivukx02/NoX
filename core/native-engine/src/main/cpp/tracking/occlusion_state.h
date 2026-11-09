#pragma once

#include <vector>

namespace nox::native::occlusion {

enum class State : unsigned char {
    Visible,
    PartiallyOccluded,
    Predicted,
    Unsupported,
};

struct Observation {
    State state{State::Unsupported};
    double confidence{0.0};
    double uncertainty{1.0};
    double crossViewSupport{0.0};
    double contactProbability{0.0};
    double temporalStability{0.0};
    double predictionAgeSeconds{0.0};
};

class OcclusionStateTracker final {
public:
    Observation visible(
        double confidence,
        double uncertainty,
        double crossViewSupport = 1.0,
        double contactProbability = 0.0,
        double temporalStability = 1.0) const noexcept;

    Observation partiallyOccluded(
        double confidence,
        double uncertainty,
        double crossViewSupport = 0.0,
        double contactProbability = 0.0,
        double temporalStability = 0.6) const noexcept;

    Observation predict(
        double confidence,
        double uncertainty,
        double gapSeconds,
        double maxGapSeconds,
        double crossViewSupport = 0.0,
        double contactProbability = 0.0,
        double temporalStability = 0.0) const noexcept;

    Observation fuse(
        const std::vector<Observation>& evidence) const noexcept;
};

} // namespace nox::native::occlusion

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nox::native::tracking {

struct ViewCandidate {
    std::uint32_t cameraId{0};
    double coverage{0.0};
    double confidence{0.0};
    double latency{0.0};
    bool thermallyConstrained{false};
    double observability{0.0};
    double novelty{1.0};
    double healthScore{1.0};
    double occlusionRisk{0.0};
    double depthUncertainty{0.0};
    double synchronizationConfidence{1.0};
    std::array<double, 3> position{0.0, 0.0, 0.0};
    std::array<double, 3> viewDirection{0.0, 0.0, 1.0};
    double temporalStability{1.0};
    double motionBoundaryConfidence{1.0};
    double appearanceConsistency{1.0};
    double depthConfidence{1.0};
    double viewSelectionScore{1.0};
};

struct ViewDecision {
    std::vector<std::uint32_t> selected;
    double combinedScore{0.0};
    double geometricDiversity{0.0};
    double meanTemporalStability{0.0};
    double meanEvidenceConfidence{0.0};
    double minimumPairwiseBaseline{0.0};
    double minimumPairwiseParallaxRadians{0.0};
    double conditioningScore{0.0};
    double informationGainScore{0.0};
    std::uint32_t rejectedForRedundancy{0};
    std::uint32_t rejectedForHealth{0};
};

class AdaptiveViewSelector final {
public:
    struct Config {
        double minimumHealthScore{0.40};
        double minimumCandidateScore{0.02};
        double minimumPairDiversity{0.08};
        double targetBaselineMeters{0.50};
        double targetParallaxRadians{0.35};
        double latencyScaleMilliseconds{50.0};
        double uncertaintyScale{1.0};
        double thermalPenalty{0.20};
        // Candidate admission spans the full distributed camera envelope; maxViews still bounds the expensive selection set.
        std::size_t maximumCandidates{128};
    };

    AdaptiveViewSelector() noexcept;
    explicit AdaptiveViewSelector(Config config) noexcept;

    [[nodiscard]] ViewDecision select(
        const std::vector<ViewCandidate>& candidates,
        std::size_t maxViews) const;

private:
    Config config_{};

    static double clamp01(double value) noexcept;
};

} // namespace nox::native::tracking

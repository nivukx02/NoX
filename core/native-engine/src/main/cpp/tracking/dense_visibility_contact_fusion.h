#pragma once
#include <cstdint>
#include <vector>

namespace nox::native::tracking {

struct LandmarkEvidence {
    std::uint32_t landmarkId{0};
    double confidence{0.0};
    double visibility{0.0};
    double reprojectionResidualPx{0.0};
    bool observed{false};
    bool occluded{false};
    bool contactHint{false};
};

struct DenseVisibilitySummary {
    double visibleCoverage{0.0};
    double observationCoverage{0.0};
    double confidence{0.0};
    double crossViewSupport{0.0};
    double contactCoverage{0.0};
    double reprojectionAgreement{0.0};
    std::uint32_t evaluatedLandmarks{0};
    std::uint32_t visibleLandmarks{0};
    std::uint32_t contactLandmarks{0};
    std::uint32_t rejectedLandmarks{0};
};

class DenseVisibilityContactFusion final {
public:
    struct Config {
        double minimumConfidence{0.20};
        double minimumVisibility{0.35};
        double maximumResidualPx{8.0};
        double crossViewWeight{0.30};
        double contactWeight{0.15};
    };

    DenseVisibilityContactFusion() noexcept;
    explicit DenseVisibilityContactFusion(Config config) noexcept;

    [[nodiscard]] DenseVisibilitySummary fuse(
        const std::vector<std::vector<LandmarkEvidence>>& views) const noexcept;

private:
    Config config_{};
    [[nodiscard]] double clamp01(double value) const noexcept;
    [[nodiscard]] double residualScore(double residualPx) const noexcept;
};

} // namespace nox::native::tracking

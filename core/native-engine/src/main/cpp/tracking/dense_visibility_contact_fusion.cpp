#include "tracking/dense_visibility_contact_fusion.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace nox::native::tracking {

DenseVisibilityContactFusion::DenseVisibilityContactFusion() noexcept
    : DenseVisibilityContactFusion(Config{}) {}

DenseVisibilityContactFusion::DenseVisibilityContactFusion(
    Config config) noexcept : config_(config) {
    config_.minimumConfidence = std::clamp(config_.minimumConfidence, 0.0, 1.0);
    config_.minimumVisibility = std::clamp(config_.minimumVisibility, 0.0, 1.0);
    config_.maximumResidualPx = std::max(config_.maximumResidualPx, 0.001);
    config_.crossViewWeight = std::clamp(config_.crossViewWeight, 0.0, 1.0);
    config_.contactWeight = std::clamp(config_.contactWeight, 0.0, 1.0);
}

DenseVisibilitySummary DenseVisibilityContactFusion::fuse(
    const std::vector<std::vector<LandmarkEvidence>>& views) const noexcept {
    DenseVisibilitySummary summary{};
    if (views.empty()) return summary;

    struct Aggregate {
        std::uint32_t samples{0};
        std::uint32_t visible{0};
        std::uint32_t contact{0};
        double confidence{0.0};
        double residualScoreSum{0.0};
    };

    std::unordered_map<std::uint32_t, Aggregate> aggregates;

    try {
        for (const auto& view : views) {
            for (const auto& evidence : view) {
                auto& aggregate = aggregates[evidence.landmarkId];
                ++aggregate.samples;

                const double confidence = clamp01(evidence.confidence);
                const double visibility = clamp01(evidence.visibility);
                aggregate.confidence += confidence * visibility;
                aggregate.residualScoreSum +=
                    residualScore(evidence.reprojectionResidualPx);

                if (
                    evidence.observed &&
                    !evidence.occluded &&
                    confidence >= config_.minimumConfidence &&
                    visibility >= config_.minimumVisibility) {
                    ++aggregate.visible;
                }

                if (
                    evidence.contactHint &&
                    confidence >= config_.minimumConfidence * 0.75 &&
                    visibility >= config_.minimumVisibility * 0.75) {
                    ++aggregate.contact;
                }
            }
        }
    } catch (...) {
        return {};
    }

    if (aggregates.empty()) return summary;

    summary.evaluatedLandmarks = static_cast<std::uint32_t>(
        std::min<std::size_t>(
            aggregates.size(),
            std::numeric_limits<std::uint32_t>::max()));

    double visibleSum = 0.0;
    double coverageSum = 0.0;
    double confidenceSum = 0.0;
    double crossViewSum = 0.0;
    double contactSum = 0.0;
    double residualSum = 0.0;

    for (const auto& [_, aggregate] : aggregates) {
        const double count = static_cast<double>(aggregate.samples);
        const double visible =
            static_cast<double>(aggregate.visible) / count;
        const double contact =
            static_cast<double>(aggregate.contact) / count;
        const double meanConfidence =
            aggregate.confidence / count;
        const double meanResidualScore =
            aggregate.residualScoreSum / count;
        const double supportingViews =
            std::min(1.0, (count - 1.0) / 2.0);
        const double crossView =
            supportingViews * meanResidualScore;

        visibleSum += visible;
        coverageSum += 1.0;
        confidenceSum += meanConfidence;
        crossViewSum += crossView;
        contactSum += contact;
        residualSum += meanResidualScore;

        if (visible > 0.5) ++summary.visibleLandmarks;
        if (contact > 0.0) ++summary.contactLandmarks;
        if (
            meanConfidence < config_.minimumConfidence ||
            meanResidualScore < 0.20) {
            ++summary.rejectedLandmarks;
        }
    }

    const double count = static_cast<double>(summary.evaluatedLandmarks);
    summary.visibleCoverage = clamp01(visibleSum / count);
    summary.observationCoverage = clamp01(coverageSum / count);
    summary.confidence = clamp01(confidenceSum / count);
    summary.crossViewSupport = clamp01(crossViewSum / count);
    summary.contactCoverage = clamp01(contactSum / count);
    summary.reprojectionAgreement = clamp01(residualSum / count);

    const double blended =
        summary.visibleCoverage * 0.30 +
        summary.observationCoverage * 0.18 +
        summary.confidence * 0.22 +
        summary.crossViewSupport * config_.crossViewWeight +
        summary.contactCoverage * config_.contactWeight +
        summary.reprojectionAgreement * 0.15;

    summary.confidence = clamp01(
        0.65 * summary.confidence + 0.35 * blended);
    return summary;
}

double DenseVisibilityContactFusion::clamp01(double value) const noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

double DenseVisibilityContactFusion::residualScore(double residualPx) const noexcept {
    if (!std::isfinite(residualPx)) return 0.0;
    return std::exp(
        -std::abs(residualPx) / config_.maximumResidualPx);
}

} // namespace nox::native::tracking

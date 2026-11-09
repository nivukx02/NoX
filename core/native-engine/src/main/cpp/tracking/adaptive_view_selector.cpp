#include <numbers>
#include "tracking/adaptive_view_selector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace nox::native::tracking {
namespace {

double clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

double dot(
    const std::array<double, 3>& a,
    const std::array<double, 3>& b) noexcept {
    return a[0] * b[0] +
        a[1] * b[1] +
        a[2] * b[2];
}

double norm(
    const std::array<double, 3>& value) noexcept {
    return std::sqrt(
        std::max(
            0.0,
            dot(value, value)));
}

double angularDiversity(
    const ViewCandidate& a,
    const ViewCandidate& b) noexcept {
    const double na = norm(a.viewDirection);
    const double nb = norm(b.viewDirection);

    if (na <= 1e-9 || nb <= 1e-9) {
        return clamp01(
            0.5 * (a.novelty + b.novelty));
    }

    const double cosine = std::clamp(
        dot(a.viewDirection, b.viewDirection) /
            (na * nb),
        -1.0,
        1.0);

    const double radians =
        std::acos(cosine);

    return clamp01(
        radians /
            std::numbers::pi);
}

double baselineDiversity(
    const ViewCandidate& a,
    const ViewCandidate& b,
    const AdaptiveViewSelector::Config& config) noexcept {
    const std::array<double, 3> delta{
        a.position[0] - b.position[0],
        a.position[1] - b.position[1],
        a.position[2] - b.position[2],
    };

    const double baseline = norm(delta);
    const double target =
        std::max(
            config.targetBaselineMeters,
            0.001);

    return clamp01(
        baseline / target);
}

double intrinsicScore(
    const ViewCandidate& value,
    const AdaptiveViewSelector::Config& config) noexcept {
    const double latencyPenalty =
        clamp01(
            value.latency /
                std::max(
                    config.latencyScaleMilliseconds,
                    1.0));

    const double occlusionTrust =
        1.0 - clamp01(value.occlusionRisk);

    const double depthTrust =
        1.0 - clamp01(value.depthUncertainty);

    const double temporalTrust =
        clamp01(value.temporalStability);

    const double synchronizationTrust =
        clamp01(value.synchronizationConfidence);

    const double physicalEvidence =
        0.30 * clamp01(value.motionBoundaryConfidence) +
        0.25 * clamp01(value.appearanceConsistency) +
        0.25 * clamp01(value.depthConfidence) +
        0.20 * clamp01(value.viewSelectionScore);

    const double evidence =
        0.58 * clamp01(value.confidence) +
        0.42 * physicalEvidence;

    const double score =
        0.15 * clamp01(value.coverage) +
        0.17 * evidence +
        0.18 * clamp01(value.observability) +
        0.11 * clamp01(value.novelty) +
        0.14 * clamp01(value.healthScore) +
        0.08 * occlusionTrust +
        0.07 * depthTrust +
        0.05 * synchronizationTrust +
        0.05 * temporalTrust -
        0.09 * latencyPenalty -
        (value.thermallyConstrained
            ? config.thermalPenalty
            : 0.0);

    return clamp01(score);
}

double pairDiversity(
    const ViewCandidate& a,
    const ViewCandidate& b,
    const AdaptiveViewSelector::Config& config) noexcept {
    const auto positionDelta = std::array<double, 3>{
        a.position[0] - b.position[0],
        a.position[1] - b.position[1],
        a.position[2] - b.position[2],
    };
    const double positionSeparation = norm(positionDelta);
    const double angular =
        angularDiversity(a, b);

    const double baseline =
        baselineDiversity(a, b, config);

    const double rawParallaxRadians =
        std::acos(
            std::clamp(
                dot(
                    [&]() {
                        const double n = norm(a.viewDirection);
                        return n <= 1e-9
                            ? std::array<double, 3>{
                                0.0, 0.0, 1.0}
                            : std::array<double, 3>{
                                a.viewDirection[0] / n,
                                a.viewDirection[1] / n,
                                a.viewDirection[2] / n};
                    }(),
                    [&]() {
                        const double n = norm(b.viewDirection);
                        return n <= 1e-9
                            ? std::array<double, 3>{
                                0.0, 0.0, 1.0}
                            : std::array<double, 3>{
                                b.viewDirection[0] / n,
                                b.viewDirection[1] / n,
                                b.viewDirection[2] / n};
                    }()),
                -1.0,
                1.0));

    const double parallaxTarget =
        std::max(
            config.targetParallaxRadians,
            0.01);
    const double parallax =
        clamp01(
            rawParallaxRadians /
            parallaxTarget);

    if (positionSeparation <= 1e-9 &&
        angular <= 1e-9 &&
        rawParallaxRadians <= 1e-9) {
        return clamp01(
            0.5 * (
                clamp01(a.novelty) +
                clamp01(b.novelty)));
    }

    return clamp01(
        0.34 * angular +
        0.28 * baseline +
        0.38 * parallax);
}

double marginalScore(
    const ViewCandidate& candidate,
    const std::vector<ViewCandidate>& selected,
    const AdaptiveViewSelector::Config& config,
    double& minimumDiversity) noexcept {
    const double base =
        intrinsicScore(candidate, config);

    if (selected.empty()) {
        minimumDiversity = 1.0;
        return base;
    }

    double bestDiversity = 0.0;
    minimumDiversity = 1.0;

    for (const auto& existing : selected) {
        const double diversity =
            pairDiversity(
                candidate,
                existing,
                config);

        bestDiversity =
            std::max(
                bestDiversity,
                diversity);
        minimumDiversity =
            std::min(
                minimumDiversity,
                diversity);
    }

    const double redundancyPenalty =
        0.32 *
        (1.0 - minimumDiversity);

    const double geometryBonus =
        0.26 * bestDiversity +
        0.12 * minimumDiversity;

    return clamp01(
        0.72 * base +
        geometryBonus -
        redundancyPenalty);
}

double setDiversity(
    const std::vector<ViewCandidate>& selected,
    const AdaptiveViewSelector::Config& config,
    double& minimumBaseline) noexcept {
    if (selected.size() < 2U) {
        minimumBaseline =
            selected.empty()
                ? 0.0
                : 1.0;
        return selected.empty() ? 0.0 : 1.0;
    }

    double diversitySum = 0.0;
    double baselineMin =
        std::numeric_limits<double>::infinity();
    std::size_t pairs = 0;

    for (std::size_t i = 0; i < selected.size(); ++i) {
        for (std::size_t j = i + 1; j < selected.size(); ++j) {
            diversitySum +=
                pairDiversity(
                    selected[i],
                    selected[j],
                    config);

            const std::array<double, 3> delta{
                selected[i].position[0] -
                    selected[j].position[0],
                selected[i].position[1] -
                    selected[j].position[1],
                selected[i].position[2] -
                    selected[j].position[2],
            };

            baselineMin =
                std::min(
                    baselineMin,
                    norm(delta));

            ++pairs;
        }
    }

    minimumBaseline =
        std::isfinite(baselineMin)
            ? baselineMin
            : 0.0;

    return pairs == 0
        ? 0.0
        : clamp01(
            diversitySum /
                static_cast<double>(pairs));
}

} // namespace

AdaptiveViewSelector::AdaptiveViewSelector() noexcept
    : AdaptiveViewSelector(Config{}) {}

AdaptiveViewSelector::AdaptiveViewSelector(
    Config config) noexcept
    : config_(config) {
    config_.minimumHealthScore =
        clamp01(config_.minimumHealthScore);
    config_.minimumCandidateScore =
        clamp01(config_.minimumCandidateScore);
    config_.minimumPairDiversity =
        clamp01(config_.minimumPairDiversity);
    config_.targetBaselineMeters =
        std::max(config_.targetBaselineMeters, 0.001);
    config_.targetParallaxRadians =
        std::clamp(
            config_.targetParallaxRadians,
            0.01,
            std::numbers::pi);
    config_.latencyScaleMilliseconds =
        std::max(
            config_.latencyScaleMilliseconds,
            1.0);
    config_.uncertaintyScale =
        std::max(config_.uncertaintyScale, 0.001);
    config_.thermalPenalty =
        clamp01(config_.thermalPenalty);
    config_.maximumCandidates =
        std::max<std::size_t>(
            config_.maximumCandidates,
            2U);
}

double AdaptiveViewSelector::clamp01(
    double value) noexcept {
    return ::nox::native::tracking::clamp01(value);
}

ViewDecision AdaptiveViewSelector::select(
    const std::vector<ViewCandidate>& input,
    std::size_t maxViews) const {
    ViewDecision output;

    if (input.empty() || maxViews == 0U) {
        return output;
    }

    std::vector<ViewCandidate> candidates;
    candidates.reserve(
        std::min(
            input.size(),
            config_.maximumCandidates));

    for (const auto& candidate : input) {
        if (candidates.size() >= config_.maximumCandidates) {
            break;
        }

        if (
            candidate.healthScore <
                config_.minimumHealthScore ||
            intrinsicScore(candidate, config_) <=
                config_.minimumCandidateScore) {
            ++output.rejectedForHealth;
            continue;
        }

        candidates.push_back(candidate);
    }

    if (candidates.empty()) {
        return output;
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [&](const auto& a, const auto& b) {
            const double sa =
                intrinsicScore(a, config_);
            const double sb =
                intrinsicScore(b, config_);

            if (std::abs(sa - sb) > 1e-12) {
                return sa > sb;
            }

            return a.cameraId < b.cameraId;
        });

    std::vector<ViewCandidate> selected;
    selected.reserve(
        std::min(
            maxViews,
            candidates.size()));

    std::vector<bool> used(
        candidates.size(),
        false);

    for (std::size_t slot = 0;
         slot < maxViews;
         ++slot) {
        double bestScore =
            -std::numeric_limits<double>::infinity();
        double bestDiversity = 0.0;
        std::size_t bestIndex =
            candidates.size();

        for (std::size_t index = 0;
             index < candidates.size();
             ++index) {
            if (used[index]) continue;

            double minimumDiversity = 1.0;
            const double score =
                marginalScore(
                    candidates[index],
                    selected,
                    config_,
                    minimumDiversity);

            if (
                !selected.empty() &&
                minimumDiversity <
                    config_.minimumPairDiversity) {
                ++output.rejectedForRedundancy;
                continue;
            }

            if (
                score > bestScore + 1e-12 ||
                (
                    std::abs(score - bestScore) <= 1e-12 &&
                    (
                        bestIndex == candidates.size() ||
                        candidates[index].cameraId <
                            candidates[bestIndex].cameraId
                    )
                )
            ) {
                bestScore = score;
                bestDiversity = minimumDiversity;
                bestIndex = index;
            }
        }

        if (
            bestIndex == candidates.size() ||
            bestScore <= config_.minimumCandidateScore) {
            break;
        }

        used[bestIndex] = true;
        selected.push_back(candidates[bestIndex]);
        output.selected.push_back(
            candidates[bestIndex].cameraId);
        output.combinedScore +=
            bestScore;

        if (selected.size() > 1U) {
            output.geometricDiversity =
                clamp01(
                    output.geometricDiversity * 0.5 +
                    bestDiversity * 0.5);
        }
    }

    double minimumBaseline = 0.0;
    const double setDiversityScore =
        setDiversity(
            selected,
            config_,
            minimumBaseline);

    output.geometricDiversity =
        selected.size() < 2U
            ? (selected.empty() ? 0.0 : 1.0)
            : setDiversityScore;

    output.minimumPairwiseBaseline =
        minimumBaseline;

    double minimumParallax =
        std::numeric_limits<double>::infinity();
    double physicalGain = 0.0;
    double physicalWeight = 0.0;

    for (std::size_t i = 0;
         i < selected.size();
         ++i) {
        const double viewNormI =
            std::max(norm(selected[i].viewDirection), 1e-9);

        physicalGain +=
            intrinsicScore(selected[i], config_) *
            (
                0.35 * clamp01(selected[i].novelty) +
                0.25 * clamp01(selected[i].observability) +
                0.20 * clamp01(selected[i].depthConfidence) +
                0.20 * clamp01(
                    selected[i].synchronizationConfidence)
            );
        physicalWeight += 1.0;

        for (std::size_t j = i + 1;
             j < selected.size();
             ++j) {
            const double viewNormJ =
                std::max(
                    norm(selected[j].viewDirection),
                    1e-9);
            const double cosine =
                std::clamp(
                    dot(
                        {
                            selected[i].viewDirection[0] /
                                viewNormI,
                            selected[i].viewDirection[1] /
                                viewNormI,
                            selected[i].viewDirection[2] /
                                viewNormI,
                        },
                        {
                            selected[j].viewDirection[0] /
                                viewNormJ,
                            selected[j].viewDirection[1] /
                                viewNormJ,
                            selected[j].viewDirection[2] /
                                viewNormJ,
                        }),
                    -1.0,
                    1.0);
            minimumParallax =
                std::min(
                    minimumParallax,
                    std::acos(cosine));
        }
    }

    output.minimumPairwiseParallaxRadians =
        std::isfinite(minimumParallax)
            ? minimumParallax
            : 0.0;

    const double parallaxConditioning =
        selected.size() < 2U
            ? 1.0
            : clamp01(
                output.minimumPairwiseParallaxRadians /
                std::max(
                    config_.targetParallaxRadians,
                    0.01));

    output.conditioningScore =
        clamp01(
            output.geometricDiversity * 0.46 +
            parallaxConditioning * 0.34 +
            clamp01(
                minimumBaseline /
                std::max(
                    config_.targetBaselineMeters,
                    0.001)) * 0.20);

    output.informationGainScore =
        physicalWeight <= 0.0
            ? 0.0
            : clamp01(
                physicalGain /
                physicalWeight) *
                (0.60 + 0.40 * output.conditioningScore);

    if (!selected.empty()) {
        output.combinedScore /=
            static_cast<double>(
                selected.size());

        output.meanTemporalStability =
            selected
                .begin()
                ->temporalStability;

        output.meanEvidenceConfidence =
            selected
                .begin()
                ->confidence;

        for (std::size_t i = 1;
             i < selected.size();
             ++i) {
            output.meanTemporalStability +=
                selected[i].temporalStability;
            output.meanEvidenceConfidence +=
                selected[i].confidence;
        }

        output.meanTemporalStability /=
            static_cast<double>(
                selected.size());
        output.meanEvidenceConfidence /=
            static_cast<double>(
                selected.size());
    }

    return output;
}

} // namespace nox::native::tracking

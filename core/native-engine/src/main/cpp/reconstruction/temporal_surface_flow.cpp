#include "reconstruction/temporal_surface_flow.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace nox::native::reconstruction {

TemporalSurfaceFlow::TemporalSurfaceFlow() noexcept
    : TemporalSurfaceFlow(Config{}) {}

TemporalSurfaceFlow::TemporalSurfaceFlow(Config config) noexcept
    : config_(config) {
    config_.maximumMatchDistanceMeters =
        std::max(config_.maximumMatchDistanceMeters, 0.0001);
    config_.maximumSpeedMetersPerSecond =
        std::max(config_.maximumSpeedMetersPerSecond, 0.01);
    config_.confidenceFloor =
        std::clamp(config_.confidenceFloor, 0.0, 1.0);
    config_.mutualDistanceSlack =
        std::max(config_.mutualDistanceSlack, 1.0);
    config_.minimumUniquenessRatio =
        std::clamp(config_.minimumUniquenessRatio, 0.0, 1.0);
    config_.maximumSamples =
        std::max<std::uint32_t>(config_.maximumSamples, 1u);
}

SurfaceFlowSummary TemporalSurfaceFlow::estimate(
    const std::vector<SurfaceSample>& previous,
    const std::vector<SurfaceSample>& current,
    double deltaSeconds) const noexcept {
    SurfaceFlowSummary summary{};
    summary.previousCount = static_cast<std::uint32_t>(
        std::min<std::size_t>(
            previous.size(),
            std::numeric_limits<std::uint32_t>::max()));
    summary.currentCount = static_cast<std::uint32_t>(
        std::min<std::size_t>(
            current.size(),
            std::numeric_limits<std::uint32_t>::max()));

    if (
        previous.empty() ||
        current.empty() ||
        !std::isfinite(deltaSeconds) ||
        deltaSeconds <= 0.0) {
        return summary;
    }

    const double maxDistanceSq =
        config_.maximumMatchDistanceMeters *
        config_.maximumMatchDistanceMeters;
    const std::size_t currentLimit =
        std::min<std::size_t>(current.size(), config_.maximumSamples);
    const std::size_t previousLimit =
        std::min<std::size_t>(previous.size(), config_.maximumSamples);

    struct Candidate {
        std::size_t sourceIndex{0};
        std::size_t targetIndex{0};
        double distanceSq{0.0};
        double uniquenessRatio{0.0};
        bool mutual{false};
    };

    std::vector<Candidate> candidates;
    candidates.reserve(previousLimit);

    std::vector<std::size_t> targetBestSource(
        currentLimit,
        previousLimit);

    /*
     * First pass: independently identify the best target for every source and the best source for
     * every target. The second pass below uses these reciprocal decisions as an ambiguity guard.
     */
    for (std::size_t previousIndex = 0u;
         previousIndex < previousLimit;
         ++previousIndex) {
        const auto& source = previous[previousIndex];
        if (
            !std::isfinite(source.confidence) ||
            source.confidence < config_.confidenceFloor) {
            continue;
        }

        double bestDistanceSq =
            std::numeric_limits<double>::infinity();
        double secondBestDistanceSq =
            std::numeric_limits<double>::infinity();
        std::size_t bestIndex = currentLimit;

        for (std::size_t currentIndex = 0u;
             currentIndex < currentLimit;
             ++currentIndex) {
            const auto& target = current[currentIndex];
            if (
                !std::isfinite(target.confidence) ||
                target.confidence < config_.confidenceFloor) {
                continue;
            }

            const double distanceSq =
                distanceSquared(source, target);
            if (distanceSq > maxDistanceSq) {
                continue;
            }

            if (distanceSq < bestDistanceSq) {
                secondBestDistanceSq = bestDistanceSq;
                bestDistanceSq = distanceSq;
                bestIndex = currentIndex;
            } else if (distanceSq < secondBestDistanceSq) {
                secondBestDistanceSq = distanceSq;
            }
        }

        if (bestIndex >= currentLimit) {
            continue;
        }

        const double uniquenessRatio =
            !std::isfinite(secondBestDistanceSq)
                ? 1.0
                : (
                    (secondBestDistanceSq - bestDistanceSq) /
                    std::max(secondBestDistanceSq, 1e-12)
                );

        candidates.push_back({
            previousIndex,
            bestIndex,
            bestDistanceSq,
            std::clamp(uniquenessRatio, 0.0, 1.0),
            false,
        });
    }

    for (std::size_t currentIndex = 0u;
         currentIndex < currentLimit;
         ++currentIndex) {
        const auto& target = current[currentIndex];
        if (
            !std::isfinite(target.confidence) ||
            target.confidence < config_.confidenceFloor) {
            continue;
        }

        double bestDistanceSq =
            std::numeric_limits<double>::infinity();
        std::size_t bestIndex = previousLimit;

        for (std::size_t previousIndex = 0u;
             previousIndex < previousLimit;
             ++previousIndex) {
            const auto& source = previous[previousIndex];
            if (
                !std::isfinite(source.confidence) ||
                source.confidence < config_.confidenceFloor) {
                continue;
            }

            const double distanceSq =
                distanceSquared(source, target);
            if (
                distanceSq <= maxDistanceSq &&
                distanceSq < bestDistanceSq) {
                bestDistanceSq = distanceSq;
                bestIndex = previousIndex;
            }
        }

        targetBestSource[currentIndex] =
            bestIndex;
    }

    for (auto& candidate : candidates) {
        candidate.mutual =
            candidate.targetIndex < targetBestSource.size() &&
            targetBestSource[candidate.targetIndex] ==
                candidate.sourceIndex;

        if (
            !candidate.mutual &&
            config_.requireMutualBestMatch
        ) {
            candidate.distanceSq *=
                config_.mutualDistanceSlack;
        }
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            if (a.mutual != b.mutual) {
                return a.mutual > b.mutual;
            }
            if (a.uniquenessRatio != b.uniquenessRatio) {
                return a.uniquenessRatio > b.uniquenessRatio;
            }
            if (a.distanceSq != b.distanceSq) {
                return a.distanceSq < b.distanceSq;
            }
            if (a.sourceIndex != b.sourceIndex) {
                return a.sourceIndex < b.sourceIndex;
            }
            return a.targetIndex < b.targetIndex;
        });

    std::vector<bool> usedCurrent(
        currentLimit,
        false);

    std::vector<double> residuals;
    residuals.reserve(std::min(previousLimit, currentLimit));
    summary.samples.reserve(std::min(previousLimit, currentLimit));

    double totalSpeed = 0.0;
    double totalConfidence = 0.0;

    try {
        for (const auto& candidate : candidates) {
            if (
                candidate.sourceIndex >= previousLimit ||
                candidate.targetIndex >= currentLimit ||
                usedCurrent[candidate.targetIndex]
            ) {
                continue;
            }

            const auto& source =
                previous[candidate.sourceIndex];
            const auto& target =
                current[candidate.targetIndex];

            if (
                !std::isfinite(source.confidence) ||
                !std::isfinite(target.confidence) ||
                source.confidence < config_.confidenceFloor ||
                target.confidence < config_.confidenceFloor
            ) {
                ++summary.rejectedCount;
                continue;
            }

            const bool acceptableAmbiguity =
                candidate.mutual ||
                (
                    !config_.requireMutualBestMatch &&
                    candidate.uniquenessRatio >=
                        config_.minimumUniquenessRatio
                );

            if (!acceptableAmbiguity) {
                ++summary.ambiguousMatchCount;
                ++summary.rejectedCount;
                continue;
            }

            const double distanceSq =
                distanceSquared(source, target);

            if (
                !std::isfinite(distanceSq) ||
                distanceSq > maxDistanceSq
            ) {
                ++summary.rejectedCount;
                continue;
            }

            usedCurrent[candidate.targetIndex] = true;

            const double dx =
                target.x - source.x;
            const double dy =
                target.y - source.y;
            const double dz =
                target.z - source.z;
            const double distance =
                std::sqrt(
                    std::max(
                        0.0,
                        dx * dx +
                            dy * dy +
                            dz * dz));
            const double speed =
                distance /
                deltaSeconds;
            const double speedScore =
                std::exp(
                    -speed /
                        config_.maximumSpeedMetersPerSecond);

            const double ambiguityScore =
                candidate.mutual
                    ? 1.0
                    : candidate.uniquenessRatio;

            const double confidence =
                clamp01(
                    std::min(
                        source.confidence,
                        target.confidence) *
                    speedScore *
                    (
                        0.72 +
                            ambiguityScore * 0.28
                    ));

            summary.samples.push_back({
                source.id,
                target.id,
                dx,
                dy,
                dz,
                speed,
                confidence,
                distance,
                true,
            });
            residuals.push_back(distance);
            totalSpeed += speed;
            totalConfidence += confidence;
            if (candidate.mutual) {
                summary.mutualConsistency += 1.0;
            }
            summary.correspondenceUniqueness +=
                ambiguityScore;
            ++summary.matchedCount;
        }
    } catch (...) {
        summary.samples.clear();
        summary.matchedCount = 0u;
        summary.confidence = 0.0;
        return summary;
    }

    if (summary.matchedCount == 0u) return summary;

    std::sort(residuals.begin(), residuals.end());
    const std::size_t p95Index = static_cast<std::size_t>(
        std::floor(
            0.95 * static_cast<double>(residuals.size() - 1u)));
    summary.p95ResidualMeters = residuals[p95Index];
    summary.meanSpeedMetersPerSecond =
        totalSpeed / static_cast<double>(summary.matchedCount);
    summary.correspondenceCoverage = clamp01(
        static_cast<double>(summary.matchedCount) /
        static_cast<double>(std::min(previousLimit, currentLimit)));

    summary.mutualConsistency =
        summary.matchedCount == 0u
            ? 0.0
            : clamp01(
                summary.mutualConsistency /
                static_cast<double>(summary.matchedCount));

    summary.correspondenceUniqueness =
        summary.matchedCount == 0u
            ? 0.0
            : clamp01(
                summary.correspondenceUniqueness /
                static_cast<double>(summary.matchedCount));

    const double residualStability = std::exp(
        -summary.p95ResidualMeters /
        config_.maximumMatchDistanceMeters);
    summary.temporalStability = clamp01(
        residualStability *
        (
            1.0 -
            std::min(
                0.90,
                summary.meanSpeedMetersPerSecond /
                (config_.maximumSpeedMetersPerSecond * 4.0))
        ));

    summary.confidence = clamp01(
        totalConfidence /
            static_cast<double>(summary.matchedCount) * 0.55 +
        summary.temporalStability * 0.25 +
        summary.correspondenceCoverage * 0.20);
    return summary;
}

double TemporalSurfaceFlow::distanceSquared(
    const SurfaceSample& a,
    const SurfaceSample& b) const noexcept {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

double TemporalSurfaceFlow::clamp01(double value) const noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

} // namespace nox::native::reconstruction

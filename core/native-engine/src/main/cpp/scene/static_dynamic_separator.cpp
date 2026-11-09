#include "scene/static_dynamic_separator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace nox::native::scene {

namespace {

struct CellHash {
    std::size_t operator()(
        const std::array<std::int64_t,3>& key) const noexcept {
        std::uint64_t x =
            static_cast<std::uint64_t>(key[0]);
        std::uint64_t y =
            static_cast<std::uint64_t>(key[1]);
        std::uint64_t z =
            static_cast<std::uint64_t>(key[2]);

        x ^= x >> 30U;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27U;

        y ^= y >> 30U;
        y *= 0x94d049bb133111ebULL;
        y ^= y >> 31U;

        z ^= z >> 30U;
        z *= 0x9e3779b97f4a7c15ULL;
        z ^= z >> 27U;

        return static_cast<std::size_t>(
            x ^
            (
                y + 0x9e3779b97f4a7c15ULL +
                (x << 6U) +
                (x >> 2U)
            ) ^
            (
                z + 0xc2b2ae3d27d4eb4fULL +
                (y << 7U) +
                (y >> 3U)
            ));
    }
};

std::array<std::int64_t,3> cellFor(
    const Point& point,
    double cellSize) noexcept {
    const double safeCell =
        std::max(
            cellSize,
            1e-5);

    return {
        static_cast<std::int64_t>(
            std::floor(
                point.p.x / safeCell)),
        static_cast<std::int64_t>(
            std::floor(
                point.p.y / safeCell)),
        static_cast<std::int64_t>(
            std::floor(
                point.p.z / safeCell)),
    };
}

} // namespace

double StaticDynamicSeparator::clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

double StaticDynamicSeparator::distance(
    const Point& a,
    const Point& b) noexcept {
    return std::hypot(
        std::hypot(
            a.p.x - b.p.x,
            a.p.y - b.p.y),
        a.p.z - b.p.z);
}

double StaticDynamicSeparator::percentile95(
    std::vector<double> values) noexcept {
    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](double value) {
                return
                    !std::isfinite(value) ||
                    value < 0.0;
            }),
        values.end());

    if (values.empty()) return 0.0;

    std::sort(
        values.begin(),
        values.end());

    const double position =
        0.95 *
        static_cast<double>(
            values.size() - 1U);

    const auto lower =
        static_cast<std::size_t>(
            std::floor(position));

    const auto upper =
        std::min(
            values.size() - 1U,
            lower + 1U);

    const double alpha =
        position -
        static_cast<double>(
            lower);

    return values[lower] +
        (
            values[upper] -
            values[lower]
        ) *
        alpha;
}

void StaticDynamicSeparator::classify(
    std::vector<Point>& points,
    std::uint64_t now,
    std::uint64_t horizon) {
    for (auto& point :
         points) {
        point.dynamic =
            point.lastSeen +
                horizon <
            now;
    }
}

DynamicSeparatorStats StaticDynamicSeparator::classifyTemporal(
    std::vector<Point>& current,
    const std::vector<Point>& previous,
    std::uint64_t timestampNanos,
    std::uint64_t previousTimestampNanos,
    const DynamicSeparatorOptions& options) {
    DynamicSeparatorStats stats;

    if (
        current.empty() ||
        current.size() >
            options.maximumPoints) {
        return stats;
    }

    if (
        previous.empty() ||
        timestampNanos <=
            previousTimestampNanos) {
        for (auto& point :
             current) {
            point.dynamic =
                point.confidence <
                options.staticConfidenceFloor;
            if (point.dynamic) {
                ++stats.dynamicCount;
            } else {
                ++stats.staticCount;
            }
        }

        stats.dynamicRatio =
            current.empty()
                ? 0.0
                : static_cast<double>(
                    stats.dynamicCount) /
                    static_cast<double>(
                        current.size());
        return stats;
    }

    const double dt =
        std::clamp(
            static_cast<double>(
                timestampNanos -
                previousTimestampNanos) /
                1e9,
            1e-4,
            1.0);

    const double maximumMatch =
        std::max(
            1e-4,
            options.maximumMatchDistanceMeters);

    const double dynamicThreshold =
        std::max(
            options.dynamicDisplacementMeters,
            options.maximumStaticDisplacementMeters * 2.0);

    const double indexCellSize =
        std::max(
            options.maximumMatchDistanceMeters *
                0.75,
            options.maximumStaticDisplacementMeters *
                4.0);

    std::unordered_map<
        std::array<std::int64_t,3>,
        std::vector<std::size_t>,
        CellHash> spatialIndex;
    spatialIndex.reserve(
        previous.size());

    for (
        std::size_t previousIndex = 0;
        previousIndex < previous.size();
        ++previousIndex) {
        spatialIndex[
            cellFor(
                previous[previousIndex],
                indexCellSize)
        ].push_back(
            previousIndex);
    }

    const auto searchRadius =
        std::max<std::int64_t>(
            1,
            static_cast<std::int64_t>(
                std::ceil(
                    maximumMatch /
                    indexCellSize)));

    std::vector<double> displacements;
    displacements.reserve(
        current.size());

    double confidenceSum = 0.0;
    std::size_t matched = 0U;

    for (auto& point :
         current) {
        const auto pointCell =
            cellFor(
                point,
                indexCellSize);

        const Point* nearest =
            nullptr;
        double nearestDistance =
            maximumMatch;

        for (
            std::int64_t dz = -searchRadius;
            dz <= searchRadius;
            ++dz) {
            for (
                std::int64_t dy = -searchRadius;
                dy <= searchRadius;
                ++dy) {
                for (
                    std::int64_t dx = -searchRadius;
                    dx <= searchRadius;
                    ++dx) {
                    const std::array<std::int64_t,3>
                        candidateCell{
                            pointCell[0] + dx,
                            pointCell[1] + dy,
                            pointCell[2] + dz,
                        };

                    const auto iterator =
                        spatialIndex.find(
                            candidateCell);

                    if (
                        iterator ==
                        spatialIndex.end()) {
                        continue;
                    }

                    for (
                        const auto index :
                        iterator->second) {
                        if (
                            index >=
                            previous.size()) {
                            continue;
                        }

                        const auto& candidate =
                            previous[index];

                        const double d =
                            distance(
                                point,
                                candidate);

                        if (
                            d <
                            nearestDistance) {
                            nearestDistance =
                                d;
                            nearest =
                                &candidate;
                        }
                    }
                }
            }
        }

        if (nearest == nullptr) {
            point.dynamic =
                point.confidence <
                options.staticConfidenceFloor;

            stats.ambiguousCount += 1U;
            confidenceSum +=
                point.confidence * 0.50;
            continue;
        }

        ++matched;

        const double staticEvidence =
            clamp01(
                1.0 -
                nearestDistance /
                    std::max(
                        options.maximumStaticDisplacementMeters,
                        1e-6));

        const double temporalEvidence =
            std::exp(
                -dt /
                std::max(
                    options.temporalHalfLifeSeconds,
                    1e-4));

        const double confidence =
            clamp01(
                point.confidence);

        const double motionEvidence =
            clamp01(
                (
                    nearestDistance -
                    options.maximumStaticDisplacementMeters
                ) /
                std::max(
                    dynamicThreshold -
                    options.maximumStaticDisplacementMeters,
                    1e-6));

        const double combinedDynamic =
            clamp01(
                motionEvidence * 0.60 +
                (
                    1.0 -
                    confidence
                ) *
                    options.dynamicConfidencePenalty *
                    0.40 +
                (
                    1.0 -
                    staticEvidence
                ) *
                    0.20);

        const bool clearlyStatic =
            nearestDistance <=
                options.maximumStaticDisplacementMeters &&
            confidence >=
                options.staticConfidenceFloor;

        const bool clearlyDynamic =
            nearestDistance >=
                dynamicThreshold ||
            combinedDynamic >=
                0.75;

        point.dynamic =
            clearlyDynamic ||
            (
                !clearlyStatic &&
                combinedDynamic >=
                    0.45
            );

        if (point.dynamic) {
            ++stats.dynamicCount;
        } else {
            ++stats.staticCount;
        }

        if (
            !clearlyStatic &&
            !clearlyDynamic) {
            ++stats.ambiguousCount;
        }

        displacements.push_back(
            nearestDistance);

        stats.meanDisplacementMeters +=
            nearestDistance *
            temporalEvidence;

        confidenceSum +=
            confidence *
            (
                0.50 +
                0.50 *
                    temporalEvidence
            );
    }

    if (!displacements.empty()) {
        stats.p95DisplacementMeters =
            percentile95(
                displacements);

        stats.meanDisplacementMeters /=
            static_cast<double>(
                matched);
    }

    stats.dynamicRatio =
        current.empty()
            ? 0.0
            : static_cast<double>(
                stats.dynamicCount) /
                static_cast<double>(
                    current.size());

    stats.confidence =
        matched == 0U
            ? 0.0
            : clamp01(
                confidenceSum /
                static_cast<double>(
                    current.size()));

    return stats;
}

} // namespace nox::native::scene

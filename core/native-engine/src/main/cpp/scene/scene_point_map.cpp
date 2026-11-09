#include "scene/scene_point_map.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nox::native::scene {

namespace {

double clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

} // namespace

ScenePointMap::ScenePointMap(
    double cellSizeMeters,
    std::size_t maxPoints)
    : cellSizeMeters_(
        std::max(
            cellSizeMeters,
            1e-4)),
      maxPoints_(
          std::max<std::size_t>(
              maxPoints,
              1024U)) {
    points_.reserve(
        std::min(
            maxPoints_,
            std::size_t{16'384}));
    pointCells_.reserve(
        points_.capacity());
}

std::size_t ScenePointMap::CellHash::operator()(
    const std::array<std::int64_t,3>& key) const noexcept {
    std::uint64_t x =
        static_cast<std::uint64_t>(
            key[0]);
    std::uint64_t y =
        static_cast<std::uint64_t>(
            key[1]);
    std::uint64_t z =
        static_cast<std::uint64_t>(
            key[2]);

    x ^= x >> 30U;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27U;

    y ^= y >> 30U;
    y *= 0xbf58476d1ce4e5b9ULL;
    y ^= y >> 27U;

    z ^= z >> 30U;
    z *= 0xbf58476d1ce4e5b9ULL;
    z ^= z >> 27U;

    return static_cast<std::size_t>(
        x ^
        (y + 0x9e3779b97f4a7c15ULL +
            (x << 6U) +
            (x >> 2U)) ^
        (z + 0xc2b2ae3d27d4eb4fULL +
            (y << 7U) +
            (y >> 3U)));
}

std::array<std::int64_t,3> ScenePointMap::cellFor(
    const Vec3& position) const noexcept {
    const double cell =
        std::max(
            cellSizeMeters_,
            1e-6);

    return {
        static_cast<std::int64_t>(
            std::floor(
                position.x / cell)),
        static_cast<std::int64_t>(
            std::floor(
                position.y / cell)),
        static_cast<std::int64_t>(
            std::floor(
                position.z / cell)),
    };
}

double ScenePointMap::distance(
    const Vec3& a,
    const Vec3& b) noexcept {
    return std::hypot(
        std::hypot(
            a.x - b.x,
            a.y - b.y),
        a.z - b.z);
}

double ScenePointMap::normalizeLength(
    const Vec3& v) noexcept {
    return std::hypot(
        std::hypot(
            v.x,
            v.y),
        v.z);
}

Vec3 ScenePointMap::normalized(
    const Vec3& value) noexcept {
    const double n =
        normalizeLength(
            value);

    if (
        !std::isfinite(n) ||
        n < 1e-12) {
        return {
            0.0,
            0.0,
            1.0,
        };
    }

    return {
        value.x / n,
        value.y / n,
        value.z / n,
    };
}

double ScenePointMap::normalAgreement(
    const Vec3& a,
    const Vec3& b) noexcept {
    const auto na =
        normalized(a);
    const auto nb =
        normalized(b);

    return clamp01(
        0.5 +
            0.5 *
            std::clamp(
                na.x * nb.x +
                na.y * nb.y +
                na.z * nb.z,
                -1.0,
                1.0));
}


std::size_t ScenePointMap::findBestNeighbor(
    const Vec3& position,
    const std::array<std::int64_t,3>& cell,
    double maximumDistance) const noexcept {
    std::size_t best =
        points_.size();

    double bestDistance =
        maximumDistance;

    for (
        std::int64_t dz = -1;
        dz <= 1;
        ++dz) {
        for (
            std::int64_t dy = -1;
            dy <= 1;
            ++dy) {
            for (
                std::int64_t dx = -1;
                dx <= 1;
                ++dx) {
                const std::array<std::int64_t,3>
                    neighborCell{
                        cell[0] + dx,
                        cell[1] + dy,
                        cell[2] + dz,
                    };

                const auto iterator =
                    buckets_.find(
                        neighborCell);

                if (
                    iterator ==
                    buckets_.end()) {
                    continue;
                }

                for (
                    const auto index :
                    iterator->second) {
                    if (
                        index >=
                        points_.size()) {
                        continue;
                    }

                    const double candidate =
                        distance(
                            points_[index].p,
                            position);

                    if (
                        candidate <
                        bestDistance) {
                        bestDistance =
                            candidate;
                        best =
                            index;
                    }
                }
            }
        }
    }

    return best;
}

void ScenePointMap::removeFromBucket(
    std::size_t index) {
    if (
        index >=
        pointCells_.size()) {
        return;
    }

    const auto cell =
        pointCells_[index];

    const auto iterator =
        buckets_.find(cell);

    if (
        iterator ==
        buckets_.end()) {
        return;
    }

    auto& bucket =
        iterator->second;

    bucket.erase(
        std::remove(
            bucket.begin(),
            bucket.end(),
            index),
        bucket.end());

    if (bucket.empty()) {
        buckets_.erase(
            iterator);
    }
}

void ScenePointMap::addToBucket(
    std::size_t index) {
    if (
        index >=
        pointCells_.size()) {
        return;
    }

    buckets_[
        pointCells_[index]
    ].push_back(index);
}

void ScenePointMap::reindexPoint(
    std::size_t index,
    const std::array<std::int64_t,3>& newCell) {
    if (
        index >=
        points_.size() ||
        index >=
        pointCells_.size()) {
        return;
    }

    if (
        pointCells_[index] ==
        newCell) {
        return;
    }

    removeFromBucket(index);
    pointCells_[index] =
        newCell;
    addToBucket(index);
}

void ScenePointMap::integrate(
    const Vec3& position,
    const Vec3& normal,
    double confidence,
    std::uint64_t timestampNanos) {
    integrateEvidence(
        position,
        normal,
        confidence,
        timestampNanos,
        0.02,
        0.0,
        1.0);
}

void ScenePointMap::integrateEvidence(
    const Vec3& position,
    const Vec3& normal,
    double confidence,
    std::uint64_t timestampNanos,
    double uncertaintyMeters,
    double dynamicProbability,
    double temporalConsistency) {
    if (
        !std::isfinite(position.x) ||
        !std::isfinite(position.y) ||
        !std::isfinite(position.z) ||
        !std::isfinite(confidence) ||
        !std::isfinite(uncertaintyMeters) ||
        !std::isfinite(dynamicProbability) ||
        !std::isfinite(temporalConsistency)) {
        return;
    }

    const double measurementConfidence =
        clamp01(confidence);

    if (
        measurementConfidence <=
        0.0) {
        return;
    }

    const auto cell =
        cellFor(position);

    constexpr double kMergeDistance =
        0.020;

    const auto best =
        findBestNeighbor(
            position,
            cell,
            kMergeDistance);

    if (
        best ==
        points_.size()) {
        if (
            points_.size() >=
            maxPoints_) {
            return;
        }

        points_.push_back({
            position,
            normalized(normal),
            measurementConfidence,
            timestampNanos,
            dynamicProbability >= 0.55,
            std::max(
                uncertaintyMeters,
                0.0001),
            clamp01(dynamicProbability),
            1.0,
            clamp01(temporalConsistency),
            1ULL,
        });

        pointCells_.push_back(
            cell);

        addToBucket(
            points_.size() - 1U);
        return;
    }

    auto& point =
        points_[best];

    const double displacement =
        distance(
            point.p,
            position);

    const double normalTrust =
        normalAgreement(
            point.normal,
            normal);

    const double existingWeight =
        std::max(
            0.01,
            point.confidence);

    const double measurementWeight =
        std::max(
            0.01,
            measurementConfidence);

    const double totalWeight =
        existingWeight +
        measurementWeight;

    point.p = {
        (
            point.p.x *
                existingWeight +
            position.x *
                measurementWeight
        ) /
            totalWeight,
        (
            point.p.y *
                existingWeight +
            position.y *
                measurementWeight
        ) /
            totalWeight,
        (
            point.p.z *
                existingWeight +
            position.z *
                measurementWeight
        ) /
            totalWeight,
    };

    point.normal =
        normalized({
            point.normal.x *
                existingWeight *
                normalTrust +
            normal.x *
                measurementWeight,
            point.normal.y *
                existingWeight *
                normalTrust +
            normal.y *
                measurementWeight,
            point.normal.z *
                existingWeight *
                normalTrust +
            normal.z *
                measurementWeight,
        });

    point.confidence =
        clamp01(
            point.confidence * 0.55 +
            measurementConfidence * 0.45);

    point.uncertaintyMeters =
        std::clamp(
            (
                point.uncertaintyMeters *
                    existingWeight +
                std::max(
                    uncertaintyMeters,
                    0.0001) *
                    measurementWeight
            ) /
                totalWeight,
            0.0001,
            10.0);

    const double dynamicEvidence =
        clamp01(
            dynamicProbability);

    const double temporalEvidence =
        clamp01(
            temporalConsistency);

    const double motionEvidence =
        clamp01(
            displacement /
            0.020);

    point.dynamicProbability =
        clamp01(
            point.dynamicProbability *
                0.60 +
            dynamicEvidence *
                0.25 +
            motionEvidence *
                0.15);

    point.temporalConsistency =
        clamp01(
            point.temporalConsistency *
                0.60 +
            temporalEvidence *
                0.40);

    point.normalConsistency =
        clamp01(
            point.normalConsistency *
                0.55 +
            normalTrust *
                0.45);

    point.dynamic =
        point.dynamicProbability >=
            0.55 ||
        (
            displacement >
                0.020 &&
            point.confidence <
                0.45
        );

    point.lastSeen =
        std::max(
            point.lastSeen,
            timestampNanos);

    point.observationCount =
        std::min<std::uint64_t>(
            std::uint64_t{1'000'000'000ULL},
            point.observationCount + std::uint64_t{1ULL});

    reindexPoint(
        best,
        cell);
}

std::vector<Point> ScenePointMap::snapshot() const {
    return points_;
}

void ScenePointMap::reset() noexcept {
    points_.clear();
    pointCells_.clear();
    buckets_.clear();
}

} // namespace nox::native::scene

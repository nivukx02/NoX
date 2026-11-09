#include "scene/mesh_temporal_coherence.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace nox::native::scene {

namespace {

double clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

double distance(
    const Vertex& a,
    const Vertex& b) noexcept {
    return std::hypot(
        std::hypot(a.x - b.x, a.y - b.y),
        a.z - b.z);
}

double percentile95(
    std::vector<double> values) {
    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](double value) {
                return !std::isfinite(value) || value < 0.0;
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

struct CellKey {
    std::int64_t x{0};
    std::int64_t y{0};
    std::int64_t z{0};

    bool operator==(const CellKey& other) const noexcept {
        return
            x == other.x &&
            y == other.y &&
            z == other.z;
    }
};

struct CellKeyHash {
    std::size_t operator()(
        const CellKey& key) const noexcept {
        const auto mix =
            [](std::uint64_t value) {
                value ^= value >> 30U;
                value *=
                    0xbf58476d1ce4e5b9ULL;
                value ^= value >> 27U;
                value *=
                    0x94d049bb133111ebULL;
                value ^= value >> 31U;
                return value;
            };

        const auto hx =
            mix(
                static_cast<std::uint64_t>(
                    key.x));
        const auto hy =
            mix(
                static_cast<std::uint64_t>(
                    key.y));
        const auto hz =
            mix(
                static_cast<std::uint64_t>(
                    key.z));

        return static_cast<std::size_t>(
            hx ^
            (
                hy + 0x9e3779b97f4a7c15ULL +
                (hx << 6U) +
                (hx >> 2U)
            ) ^
            (
                hz + 0xc2b2ae3d27d4eb4fULL +
                (hy << 7U) +
                (hy >> 3U)
            ));
    }
};

CellKey cellOf(
    const Vertex& vertex,
    double cellSize) noexcept {
    const double safeCell =
        std::max(
            cellSize,
            1e-6);
    return {
        static_cast<std::int64_t>(
            std::floor(
                vertex.x / safeCell)),
        static_cast<std::int64_t>(
            std::floor(
                vertex.y / safeCell)),
        static_cast<std::int64_t>(
            std::floor(
                vertex.z / safeCell)),
    };
}

} // namespace

std::vector<Vertex> MeshTemporalCoherence::stabilize(
    const std::vector<Vertex>& previous,
    const std::vector<Vertex>& current,
    double alpha) const {
    if (
        !previous.empty() &&
        previous.size() == current.size()) {
        const double blend = clamp01(alpha);
        std::vector<Vertex> legacy = current;
        for (
            std::size_t index = 0;
            index < current.size();
            ++index) {
            legacy[index].x =
                previous[index].x +
                (current[index].x - previous[index].x) *
                    (1.0 - blend);
            legacy[index].y =
                previous[index].y +
                (current[index].y - previous[index].y) *
                    (1.0 - blend);
            legacy[index].z =
                previous[index].z +
                (current[index].z - previous[index].z) *
                    (1.0 - blend);
        }
        return legacy;
    }

    MeshTemporalOptions options;
    options.blendAlpha =
        alpha;
    options.maxDisplacementMeters =
        1e12;
    options.maximumMatchDistanceMeters =
        1e12;
    options.confidenceFloor =
        0.0;
    options.innovationScaleMeters =
        1.0;

    const auto result =
        stabilizeHighEnd(
            previous,
            current,
            options);

    if (
        result.vertices.empty() &&
        !current.empty()) {
        return current;
    }

    return result.vertices;
}

MeshTemporalResult MeshTemporalCoherence::stabilizeHighEnd(
    const std::vector<Vertex>& previous,
    const std::vector<Vertex>& current,
    const MeshTemporalOptions& options) const {
    MeshTemporalResult result;
    result.vertices = current;

    if (
        current.empty() ||
        previous.empty() ||
        current.size() > options.maximumVertices ||
        previous.size() > options.maximumVertices) {
        return result;
    }

    const double cellSize =
        std::max(
            options.spatialCellSizeMeters,
            1e-5);
    const double maxMatchDistance =
        std::max(
            options.maximumMatchDistanceMeters,
            1e-5);
    const double blendAlpha =
        clamp01(
            options.blendAlpha);
    const double confidenceDamping =
        std::clamp(
            options.confidenceDamping,
            0.10,
            1.0);
    const double innovationScale =
        std::max(
            options.innovationScaleMeters,
            1e-5);

    std::unordered_map<
        CellKey,
        std::vector<std::size_t>,
        CellKeyHash> spatialIndex;
    spatialIndex.reserve(
        previous.size());

    for (std::size_t index = 0;
         index < previous.size();
         ++index) {
        spatialIndex[
            cellOf(
                previous[index],
                cellSize)
        ].push_back(index);
    }

    std::vector<double> displacements;
    std::vector<double> innovations;
    displacements.reserve(
        current.size());
    innovations.reserve(
        current.size());

    double confidenceSum = 0.0;
    double stabilitySum = 0.0;
    double innovationSum = 0.0;
    std::size_t correspondences = 0U;

    for (std::size_t currentIndex = 0;
         currentIndex < current.size();
         ++currentIndex) {
        const auto& currentVertex =
            current[currentIndex];
        const CellKey currentCell =
            cellOf(
                currentVertex,
                cellSize);

        std::size_t bestIndex =
            previous.size();
        double bestDistance =
            maxMatchDistance;

        const auto radius =
            static_cast<std::int64_t>(
                std::max(
                    1U,
                    options.searchRadiusCells));

        for (
            std::int64_t dz = -radius;
            dz <= radius;
            ++dz) {
            for (
                std::int64_t dy = -radius;
                dy <= radius;
                ++dy) {
                for (
                    std::int64_t dx = -radius;
                    dx <= radius;
                    ++dx) {
                    const auto iterator =
                        spatialIndex.find(
                            CellKey{
                                currentCell.x + dx,
                                currentCell.y + dy,
                                currentCell.z + dz,
                            });

                    if (
                        iterator ==
                        spatialIndex.end()) {
                        continue;
                    }

                    for (
                        const auto candidateIndex :
                        iterator->second) {
                        const double candidateDistance =
                            distance(
                                currentVertex,
                                previous[candidateIndex]);

                        if (
                            !std::isfinite(
                                candidateDistance) ||
                            candidateDistance >
                                bestDistance) {
                            continue;
                        }

                        if (
                            bestIndex ==
                                previous.size() ||
                            candidateDistance <
                                bestDistance) {
                            bestDistance =
                                candidateDistance;
                            bestIndex =
                                candidateIndex;
                        }
                    }
                }
            }
        }

        const double currentConfidence =
            clamp01(
                currentVertex.confidence);

        if (bestIndex == previous.size()) {
            confidenceSum +=
                currentConfidence *
                0.55;
            stabilitySum +=
                0.20;
            continue;
        }

        const auto& previousVertex =
            previous[bestIndex];

        const double displacement =
            bestDistance;
        const double sourceConfidence =
            clamp01(
                previousVertex.confidence);

        const double matchedConfidence =
            std::sqrt(
                sourceConfidence *
                currentConfidence);

        const double innovation =
            displacement /
            innovationScale;

        const double innovationConfidence =
            clamp01(
                std::exp(
                    -innovation *
                    0.10));

        const double displacementLimitFactor =
            displacement <=
                options.maxDisplacementMeters
                ? 1.0
                : clamp01(
                    options.maxDisplacementMeters /
                    std::max(
                        displacement,
                        1e-8));

        const double temporalWeight =
            clamp01(
                blendAlpha *
                confidenceDamping *
                matchedConfidence *
                innovationConfidence *
                displacementLimitFactor);

        result.vertices[currentIndex].x =
            previousVertex.x +
            (
                currentVertex.x -
                previousVertex.x
            ) *
            (
                1.0 -
                temporalWeight
            );

        result.vertices[currentIndex].y =
            previousVertex.y +
            (
                currentVertex.y -
                previousVertex.y
            ) *
            (
                1.0 -
                temporalWeight
            );

        result.vertices[currentIndex].z =
            previousVertex.z +
            (
                currentVertex.z -
                previousVertex.z
            ) *
            (
                1.0 -
                temporalWeight
            );

        result.vertices[currentIndex].confidence =
            clamp01(
                currentConfidence *
                0.55 +
                sourceConfidence *
                0.45);

        displacements.push_back(
            displacement);
        innovations.push_back(
            innovation);
        confidenceSum +=
            result.vertices[currentIndex]
                .confidence *
            (
                0.55 +
                0.45 *
                    innovationConfidence
            );
        stabilitySum +=
            innovationConfidence;
        innovationSum +=
            innovation;
        ++correspondences;
    }

    result.maxDisplacement =
        displacements.empty()
            ? 0.0
            : *std::max_element(
                displacements.begin(),
                displacements.end());

    result.p95Displacement =
        percentile95(
            displacements);
    result.p95Innovation =
        percentile95(
            innovations);
    result.meanInnovation =
        correspondences == 0U
            ? maxMatchDistance
            : innovationSum /
                static_cast<double>(
                    correspondences);

    result.correspondenceCoverage =
        clamp01(
            static_cast<double>(
                correspondences) /
            static_cast<double>(
                current.size()));

    result.confidence =
        clamp01(
            confidenceSum /
            static_cast<double>(
                std::max(
                    std::size_t{1},
                    correspondences)));

    result.temporalStability =
        clamp01(
            stabilitySum /
            static_cast<double>(
                std::max(
                    std::size_t{1},
                    correspondences)));

    result.accepted =
        result.correspondenceCoverage >=
            0.55 &&
        result.confidence >=
            options.confidenceFloor &&
        result.p95Displacement <=
            std::max(
                options.maxDisplacementMeters,
                1e-6) &&
        result.p95Innovation <=
            6.0 &&
        std::all_of(
            result.vertices.begin(),
            result.vertices.end(),
            [](const Vertex& vertex) {
                return
                    std::isfinite(vertex.x) &&
                    std::isfinite(vertex.y) &&
                    std::isfinite(vertex.z) &&
                    std::isfinite(vertex.confidence);
            });

    return result;
}

} // namespace nox::native::scene

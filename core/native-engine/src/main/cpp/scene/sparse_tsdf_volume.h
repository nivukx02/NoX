#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>
#include "math/se3.h"

namespace nox::native::scene {

struct TsdfVoxel {
    float distance{1.0F};
    float weight{0.0F};
    float variance{1.0F};
    float dynamicProbability{0.0F};
    float confidence{0.0F};
    std::uint64_t observationCount{0};
};

struct TsdfMesh {
    std::vector<math::Vec3d> vertices;
    std::vector<std::array<std::uint32_t,3>> triangles;
    std::vector<math::Vec3d> normals;
    std::vector<double> vertexUncertaintyMeters;
    std::vector<double> vertexDynamicProbability;
    double meanConfidence{0.0};
    double meanUncertaintyMeters{0.0};
    double p95UncertaintyMeters{0.0};
    double meanDynamicProbability{0.0};
    double trustedVertexRatio{0.0};
    std::size_t rejectedTriangleCount{0};
};

struct SparseTsdfConfig {
    double voxelSize{0.01};
    double truncationDistance{0.04};
    float maxWeight{64.0F};
    std::int32_t blockResolution{8};
    std::size_t maxResidentVoxels{2'000'000};
    std::size_t maxResidentMemoryBytes{192u * 1024u * 1024u};
    std::size_t estimatedBytesPerVoxel{96};
    double minimumTriangleAreaSqMeters{1e-8};
    double maximumTriangleEdgeMeters{0.25};
    double maximumTriangleUncertaintyMeters{0.20};
};

class SparseTsdfVolume final {
public:
    explicit SparseTsdfVolume(SparseTsdfConfig config = {});

    void reset();
    void releaseMemory() noexcept;

    void integrateSurfacePoint(
        const math::Vec3d& origin,
        const math::Vec3d& direction,
        double surfaceDepth,
        double observationWeight=1.0);

    void integrateSurfacePointHighEnd(
        const math::Vec3d& origin,
        const math::Vec3d& direction,
        double surfaceDepth,
        double observationWeight,
        double uncertaintyMeters,
        double temporalConsistency=1.0,
        double appearanceConsistency=1.0,
        double contactConsistency=1.0,
        double dynamicProbability=0.0);

    [[nodiscard]] TsdfVoxel sample(
        std::int32_t x,
        std::int32_t y,
        std::int32_t z) const noexcept;

    [[nodiscard]] TsdfMesh extractSurface(
        float isoLevel=0.0F,
        std::size_t maxTriangles=200000) const;

    [[nodiscard]] std::size_t activeVoxelCount() const noexcept;
    [[nodiscard]] std::size_t residentMemoryBytes() const noexcept;
    [[nodiscard]] double voxelUtilization() const noexcept;
    [[nodiscard]] std::size_t rejectedVoxelAllocations() const noexcept;
    [[nodiscard]] std::size_t rejectedTriangleCount() const noexcept;

private:
    struct KeyHash {
        std::size_t operator()(
            const std::array<std::int32_t,3>& key) const noexcept;
    };

    SparseTsdfConfig config_;
    std::unordered_map<std::array<std::int32_t,3>,TsdfVoxel,KeyHash> voxels_;
    mutable std::optional<TsdfMesh> last_extracted_mesh_;
    std::size_t rejected_voxel_allocations_{0};

    [[nodiscard]] std::array<std::int32_t,3> worldToVoxel(
        const math::Vec3d&) const noexcept;

    [[nodiscard]] math::Vec3d voxelToWorld(
        std::int32_t,
        std::int32_t,
        std::int32_t) const noexcept;

    void integrateVoxel(
        std::int32_t,
        std::int32_t,
        std::int32_t,
        float,
        float) noexcept;
};

} // namespace nox::native::scene

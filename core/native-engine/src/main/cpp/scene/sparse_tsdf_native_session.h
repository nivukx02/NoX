#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "scene/sparse_tsdf_volume.h"
#include "scene/adaptive_tsdf_detail_controller.h"
#include "scene/temporal_gaussian_surface_state.h"
#include "scene/gaussian_deformation_graph.h"

namespace nox::native::scene {

enum class DepthSampleConvention : std::uint8_t {
    EuclideanRay = 0,
    OpticalAxisZ = 1,
};

class SparseTsdfNativeSession final {
public:
    explicit SparseTsdfNativeSession(SparseTsdfConfig config = {});

    void reset();
    void releaseMemory() noexcept;

    void integrateSurfacePoint(
        const math::Vec3d& origin,
        const math::Vec3d& direction,
        double surfaceDepth,
        double observationWeight=1.0);

    void integrateDepthFrame(
        const float* depth,
        std::size_t width,
        std::size_t height,
        double fx,
        double fy,
        double cx,
        double cy,
        const math::SE3d& cameraToWorld,
        std::size_t stride=2,
        double observationWeight=1.0,
        DepthSampleConvention convention=DepthSampleConvention::EuclideanRay,
        const float* sampleConfidence=nullptr);

    void integrateDepthFrameHighEnd(
        const float* depth,
        std::size_t width,
        std::size_t height,
        double fx,
        double fy,
        double cx,
        double cy,
        const math::SE3d& cameraToWorld,
        std::size_t stride,
        double observationWeight,
        double depthUncertaintyMeters,
        double temporalConsistency,
        double appearanceConsistency,
        double contactConsistency,
        double dynamicProbability,
        DepthSampleConvention convention=DepthSampleConvention::EuclideanRay,
        const float* sampleConfidence=nullptr);

    TsdfDetailDecision integrateDepthFrameAdaptive(
        const float* depth,
        std::size_t width,
        std::size_t height,
        double fx,
        double fy,
        double cx,
        double cy,
        const math::SE3d& cameraToWorld,
        std::size_t stride,
        double observationWeight,
        DepthSampleConvention convention,
        const TsdfDetailEvidence& evidence,
        const float* sampleConfidence=nullptr);

    [[nodiscard]] TsdfMesh extract(
        float isoLevel,
        std::size_t maxTriangles) const;

    [[nodiscard]] bool integrateGaussianObservation(
        const GaussianSurfaceObservation& observation) noexcept;

    [[nodiscard]] std::size_t integrateGaussianObservations(
        const std::vector<GaussianSurfaceObservation>& observations) noexcept;

    [[nodiscard]] std::vector<GaussianSurfacePrimitive>
    gaussianSurfaceSnapshot() const;

    [[nodiscard]] GaussianSurfaceStats
    gaussianSurfaceStats() const noexcept;

    [[nodiscard]] bool rebuildGaussianDeformationGraph();

    [[nodiscard]] bool updateGaussianDeformationNode(
        std::uint32_t nodeId,
        const math::SE3d& transform,
        const Vec3& velocity,
        double confidence,
        double temporalStability) noexcept;

    [[nodiscard]] std::vector<GaussianSurfacePrimitive>
    deformGaussianSurface() const;

    [[nodiscard]] GaussianDeformationStats
    gaussianDeformationStats() const noexcept;

    [[nodiscard]] std::size_t voxelCount() const noexcept;
    [[nodiscard]] std::size_t residentMemoryBytes() const noexcept;
    [[nodiscard]] double voxelUtilization() const noexcept;
    [[nodiscard]] std::size_t rejectedVoxelAllocations() const noexcept;
    [[nodiscard]] std::size_t rejectedTriangleCount() const noexcept;

private:
    SparseTsdfVolume volume_;
    AdaptiveTsdfDetailController detailController_;
    TemporalGaussianSurfaceState gaussianSurface_;
    GaussianDeformationGraph gaussianDeformationGraph_;
};

} // namespace nox::native::scene

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nox::native::scene {

struct Voxel {
    float tsdf{1.0f};
    float weight{0.0f};
    float variance{1.0f};
    float dynamicProbability{0.0f};
    std::uint64_t observationCount{0};
    std::uint64_t lastTimestampNanos{0};
};

struct VoxelObservation {
    float tsdf{1.0f};
    float confidence{1.0f};
    float variance{0.01f};
    float temporalConsistency{1.0f};
    float appearanceConsistency{1.0f};
    float contactConsistency{1.0f};
    std::uint64_t timestampNanos{0};
};

struct VolumetricFusionOptions {
    float minimumConfidence{0.05f};
    float maximumVariance{1.0f};
    float dynamicSuppression{0.75f};
    float temporalDecayHalfLifeSeconds{0.80f};
    float varianceFloor{0.000001f};
    float maximumWeight{65535.0f};
    std::size_t maximumVoxels{2'000'000};
};

struct VolumetricFusionStats {
    std::size_t voxelCount{0};
    std::uint64_t integratedObservations{0};
    std::uint64_t rejectedObservations{0};
    std::uint64_t evictedVoxels{0};
    double meanVariance{0.0};
    double meanDynamicProbability{0.0};
    double confidence{0.0};
    double memoryUtilization{0.0};
};

class VolumetricFusion final {
public:
    void integrate(
        std::int32_t x,
        std::int32_t y,
        std::int32_t z,
        float tsdf,
        float weight) noexcept;

    [[nodiscard]] bool integrateHighEnd(
        std::int32_t x,
        std::int32_t y,
        std::int32_t z,
        const VoxelObservation& observation,
        const VolumetricFusionOptions& options = {}) noexcept;

    [[nodiscard]] Voxel sample(
        std::int32_t x,
        std::int32_t y,
        std::int32_t z) const noexcept;

    [[nodiscard]] VolumetricFusionStats stats(
        const VolumetricFusionOptions& options = {}) const noexcept;

    void decay(
        std::uint64_t timestampNanos,
        const VolumetricFusionOptions& options = {}) noexcept;

    void clear() noexcept;

    [[nodiscard]] std::size_t voxelCount() const noexcept {
        return grid_.size();
    }

private:
    static std::uint64_t key(
        std::int32_t x,
        std::int32_t y,
        std::int32_t z) noexcept;

    static double clamp01(double value) noexcept;

    void evictWeakest(
        const VolumetricFusionOptions& options) noexcept;

    std::unordered_map<std::uint64_t, Voxel> grid_;
    std::uint64_t lastDecayTimestampNanos_{0};
    std::uint64_t integratedObservations_{0};
    std::uint64_t rejectedObservations_{0};
    std::uint64_t evictedVoxels_{0};
};

} // namespace nox::native::scene

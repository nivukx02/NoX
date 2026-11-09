#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <unordered_map>
#include <vector>
#include <shared_mutex>

#include "math/se3.h"
#include "scene/temporal_gaussian_surface_state.h"

namespace nox::native::scene {

struct GaussianDeformationNode {
    std::uint32_t id{0};
    Vec3 canonicalPosition{};
    math::SE3d currentTransform{};
    Vec3 velocity{};
    double confidence{0.0};
    double temporalStability{0.0};
    std::size_t supportCount{0};
};

struct GaussianDeformationBinding {
    std::uint32_t gaussianIndex{0};
    std::vector<std::uint32_t> nodeIds;
    std::vector<double> weights;
    Vec3 canonicalOffset{};
    double confidence{0.0};
};

struct GaussianDeformationOptions {
    double nodeSpacingMeters{0.10};
    std::size_t maximumNodes{16'384};
    std::size_t maximumInfluencesPerGaussian{4};
    double maximumNodeTranslationMeters{0.50};
    double maximumNodeRotationRadians{1.50};
    double maximumEdgeStrainRatio{0.35};
    double minimumBindingConfidence{0.10};
    double temporalSmoothing{0.72};
};

struct GaussianDeformationStats {
    std::size_t nodeCount{0};
    std::size_t bindingCount{0};
    std::size_t rejectedBindings{0};
    double meanNodeConfidence{0.0};
    double meanBindingConfidence{0.0};
    double p95EdgeStrainRatio{0.0};
    double maximumEdgeStrainRatio{0.0};
    double temporalStability{0.0};
    double trustedBindingRatio{0.0};
};

class GaussianDeformationGraph final {
public:
    explicit GaussianDeformationGraph(
        GaussianDeformationOptions options = {});

    void reset() noexcept;

    [[nodiscard]] bool buildCanonicalGraph(
        const std::vector<GaussianSurfacePrimitive>& primitives);

    [[nodiscard]] bool updateNodePose(
        std::uint32_t nodeId,
        const math::SE3d& transform,
        const Vec3& velocity,
        double confidence,
        double temporalStability) noexcept;

    [[nodiscard]] std::vector<GaussianSurfacePrimitive> deform(
        const std::vector<GaussianSurfacePrimitive>& primitives) const;

    [[nodiscard]] GaussianDeformationStats stats() const noexcept;

private:
    GaussianDeformationOptions options_{};
    struct CellHash {
        std::size_t operator()(
            const std::array<std::int64_t,3>&) const noexcept;
    };

    std::vector<GaussianDeformationNode> nodes_{};
    std::vector<GaussianDeformationBinding> bindings_{};
    std::unordered_map<
        std::array<std::int64_t,3>,
        std::vector<std::size_t>,
        CellHash> nodeBuckets_{};
    mutable std::shared_mutex mutex_{};

    void resetUnlocked() noexcept;

    [[nodiscard]] std::array<std::int64_t,3> cellFor(
        const Vec3&) const noexcept;

    void indexNode(
        std::size_t index);

    static double clamp01(double) noexcept;
    static double distance(
        const Vec3&,
        const Vec3&) noexcept;

    std::size_t nearestNode(
        const Vec3& position) const noexcept;

    std::vector<std::size_t> nearestNodes(
        const Vec3& position) const;

    static Vec3 rotate(
        const math::SE3d&,
        const Vec3&) noexcept;
};

} // namespace nox::native::scene

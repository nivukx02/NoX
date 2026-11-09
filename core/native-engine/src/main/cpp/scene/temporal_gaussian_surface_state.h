#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "scene/scene_point_map.h"

namespace nox::native::scene {

struct GaussianSurfaceObservation {
    Vec3 position{};
    Vec3 normal{0.0,0.0,1.0};
    Vec3 velocity{};
    // Measurement variances are supplied as diagonal sensor-space uncertainty. The fused
    // primitive learns cross-axis covariance from multi-view residuals.
    std::array<double,3> variance{
        0.0004,
        0.0004,
        0.0004,
    };
    double confidence{1.0};
    double dynamicProbability{0.0};
    double temporalConsistency{1.0};
    // Visibility confidence prevents invisible or weakly observed Gaussians from receiving
    // full posterior influence; it is supplied by the real surface evidence producer.
    double visibilityConfidence{1.0};
    // Photometric consistency is an observational support signal, not a replacement for geometry.
    double photometricConsistency{1.0};
    double normalConfidence{1.0};
    std::uint64_t timestampNanos{0};
    // Optional higher-order motion evidence. Trailing defaults preserve legacy aggregate initializers.
    Vec3 acceleration{};
    Vec3 jerk{};
};

struct GaussianSurfacePrimitive {
    Vec3 mean{};
    Vec3 normal{0.0,0.0,1.0};
    Vec3 velocity{};
    // Symmetric covariance layout: [xx, yy, zz, xy, xz, yz].
    std::array<double,6> covariance{
        0.0004,
        0.0004,
        0.0004,
        0.0,
        0.0,
        0.0,
    };
    double confidence{0.0};
    double dynamicProbability{0.0};
    double temporalStability{0.0};
    double opacity{0.0};
    std::uint64_t observationCount{0};
    std::uint64_t lastTimestampNanos{0};
    // Bounded higher-order surface motion state used during short temporal occlusions.
    Vec3 acceleration{};
    Vec3 jerk{};
};

struct GaussianSurfaceOptions {
    double cellSizeMeters{0.0125};
    double maximumAssociationMahalanobis{9.0};
    double maximumSpatialResidualMeters{0.08};
    double minimumNormalAssociation{0.20};
    double normalAssociationWeight{0.55};
    double minimumObservationConfidence{0.05};
    double dynamicSuppression{0.55};
    double temporalHalfLifeSeconds{0.80};
    double velocityDecay{0.88};
    double confidenceDecay{0.92};
    double maximumVelocityMetersPerSecond{25.0};
    double maximumAccelerationMetersPerSecond2{120.0};
    double maximumJerkMetersPerSecond3{900.0};
    double accelerationDecay{0.82};
    double jerkDecay{0.68};
    double minimumVarianceMeters2{1e-7};
    double maximumVarianceMeters2{0.25};
    // Preserves model-mismatch / multi-view disagreement in the posterior covariance.
    double innovationVarianceInflation{0.35};
    double crossCovarianceRetention{0.88};
    double maximumCorrelationMagnitude{0.96};
    std::size_t maximumPrimitives{750'000};
    // Capacity eviction activates for primitives whose composite utility is below this threshold.
    // The default is above the minimum utility reachable under the admission floor so capacity
    // pressure can actually evict weakly supported, uncertain observations.
    double minimumEvictionUtility{0.12};
};

struct GaussianSurfaceStats {
    std::size_t primitiveCount{0};
    std::size_t fusedObservations{0};
    std::size_t rejectedObservations{0};
    std::size_t dynamicPrimitives{0};
    double meanConfidence{0.0};
    double meanUncertaintyMeters{0.0};
    double meanDynamicProbability{0.0};
    double temporalStability{0.0};
    double trustedPrimitiveRatio{0.0};
};

class TemporalGaussianSurfaceState final {
public:
    TemporalGaussianSurfaceState() noexcept;
    explicit TemporalGaussianSurfaceState(
        GaussianSurfaceOptions options);

    void reset() noexcept;

    [[nodiscard]] bool integrate(
        const GaussianSurfaceObservation& observation) noexcept;

    [[nodiscard]] bool predict(
        std::uint64_t timestampNanos) noexcept;

    void prune(
        std::uint64_t timestampNanos) noexcept;

    [[nodiscard]] std::vector<GaussianSurfacePrimitive> snapshot() const;

    [[nodiscard]] GaussianSurfaceStats stats() const noexcept;

private:
    struct CellHash {
        std::size_t operator()(
            const std::array<std::int64_t,3>&) const noexcept;
    };

    GaussianSurfaceOptions options_{};
    std::unordered_map<
        std::array<std::int64_t,3>,
        std::vector<std::size_t>,
        CellHash> cells_;
    std::vector<GaussianSurfacePrimitive> primitives_;
    std::vector<std::array<std::int64_t,3>> primitiveCells_;
    std::uint64_t fusedObservationCount_{0};
    std::uint64_t rejectedObservationCount_{0};

    static double clamp01(double) noexcept;
    static double norm(const Vec3&) noexcept;
    static Vec3 normalize(const Vec3&) noexcept;
    static double dot(
        const Vec3&,
        const Vec3&) noexcept;

    [[nodiscard]] std::array<std::int64_t,3> cellFor(
        const Vec3&) const noexcept;

    [[nodiscard]] double mahalanobis(
        const GaussianSurfacePrimitive&,
        const GaussianSurfaceObservation&) const noexcept;

    void stabilizeCovariance(
        std::array<double,6>&) const noexcept;

    [[nodiscard]] std::size_t findCandidate(
        const GaussianSurfaceObservation&) const noexcept;

    void fusePrimitive(
        GaussianSurfacePrimitive&,
        const GaussianSurfaceObservation&) noexcept;

    [[nodiscard]] double primitiveUtility(
        const GaussianSurfacePrimitive&) const noexcept;

    bool evictLowestUtility() noexcept;
};

} // namespace nox::native::scene

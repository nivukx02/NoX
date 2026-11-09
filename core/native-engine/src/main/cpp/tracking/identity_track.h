#pragma once

#include <cstdint>

#include "tracking/joint_track.h"

namespace nox::native::tracking {

struct IdentityEvidence {
    Vec3 predictedCentroid{};
    Vec3 detectedCentroid{};
    Vec3 predictedVelocity{};
    Vec3 detectedVelocity{};
    double scaleRatio{1.0};
    double appearanceSimilarity{1.0};
    double faceSimilarity{1.0};
    double handSimilarity{1.0};
    double contactConsistency{1.0};
    double visibility{1.0};
    double detectorConfidence{1.0};
    double temporalGapSeconds{0.0};
    double occlusionRisk{0.0};
};

struct IdentityAssociationCost {
    double total{0.0};
    double geometry{0.0};
    double velocity{0.0};
    double appearance{0.0};
    double face{0.0};
    double hand{0.0};
    double contact{0.0};
    double visibility{0.0};
    double temporal{0.0};
    double scale{0.0};
    double ambiguityMargin{0.0};
    bool gated{false};
};

class IdentityTrack final {
public:
    explicit IdentityTrack(
        std::uint64_t id = 1)
        : id_(id) {}

    double associationCost(
        const Vec3& centroid,
        const Vec3& predictedCentroid,
        double detectionConfidence) const noexcept;

    IdentityAssociationCost associationCost(
        const IdentityEvidence& evidence) const noexcept;

    [[nodiscard]] std::uint64_t id() const noexcept {
        return id_;
    }

private:
    std::uint64_t id_;
};

} // namespace nox::native::tracking

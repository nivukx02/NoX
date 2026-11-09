#pragma once
#include <cstdint>
#include "tracking/joint_track.h"
namespace nox::native::tracking{struct AssociationCost{
    double total{0};
    double geometry{0};
    double temporal{0};
    double visibility{0};
    double appearance{0};
    double faceConsistency{0};
    double handConsistency{0};
    double contactConsistency{0};
    double motionConsistency{0};
    double scaleConsistency{0};
    double ambiguityMargin{0};
    bool gated{false};
};class GeometricIdentityTrack final {
public:
    explicit GeometricIdentityTrack(
        std::uint64_t id)
        : id_(id) {}

    AssociationCost cost(
        Vec3 predicted,
        Vec3 detected,
        double dt,
        double visibility,
        double detectorConfidence) const noexcept;

    AssociationCost costMultimodal(
        Vec3 predicted,
        Vec3 detected,
        Vec3 predictedVelocity,
        Vec3 detectedVelocity,
        double dt,
        double visibility,
        double detectorConfidence,
        double appearanceSimilarity,
        double faceConsistency,
        double handConsistency,
        double contactConsistency,
        double scaleRatio) const noexcept;

    std::uint64_t id() const noexcept {
        return id_;
    }

private:
    std::uint64_t id_;
};}
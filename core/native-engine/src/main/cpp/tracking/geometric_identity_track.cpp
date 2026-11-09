#include "tracking/geometric_identity_track.h"

#include <algorithm>
#include <cmath>

namespace nox::native::tracking {

namespace {

double clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

double norm(Vec3 value) noexcept {
    return std::hypot(
        std::hypot(
            value.x,
            value.y),
        value.z);
}

double distance(
    Vec3 a,
    Vec3 b) noexcept {
    return norm({
        a.x - b.x,
        a.y - b.y,
        a.z - b.z,
    });
}

double directionConsistency(
    Vec3 a,
    Vec3 b) noexcept {
    const double aNorm =
        norm(a);
    const double bNorm =
        norm(b);
    if (
        aNorm < 1e-8 ||
        bNorm < 1e-8) {
        return 1.0;
    }

    const double cosine =
        (
            a.x * b.x +
            a.y * b.y +
            a.z * b.z
        ) /
        (
            aNorm *
            bNorm
        );

    return clamp01(
        0.5 +
            0.5 *
            std::clamp(
                cosine,
                -1.0,
                1.0));
}

} // namespace

AssociationCost GeometricIdentityTrack::cost(
    Vec3 predicted,
    Vec3 detected,
    double dt,
    double visibility,
    double detectorConfidence) const noexcept {
    return costMultimodal(
        predicted,
        detected,
        {},
        {},
        dt,
        visibility,
        detectorConfidence,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0);
}

AssociationCost GeometricIdentityTrack::costMultimodal(
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
    double scaleRatio) const noexcept {
    AssociationCost result;

    const double geometry =
        std::clamp(
            distance(
                predicted,
                detected) /
                0.75,
            0.0,
            4.0);
    const double safeDt =
        std::max(
            1e-4,
            std::abs(dt));
    const double temporal =
        std::clamp(
            safeDt / 0.2,
            0.0,
            4.0);
    const double visibilityPenalty =
        1.0 -
        clamp01(visibility);
    const double detectorPenalty =
        1.0 -
        clamp01(detectorConfidence);

    result.geometry =
        geometry;
    result.temporal =
        temporal;
    result.visibility =
        visibilityPenalty;
    result.appearance =
        1.0 -
        clamp01(appearanceSimilarity);
    result.faceConsistency =
        1.0 -
        clamp01(faceConsistency);
    result.handConsistency =
        1.0 -
        clamp01(handConsistency);
    result.contactConsistency =
        1.0 -
        clamp01(contactConsistency);
    result.motionConsistency =
        1.0 -
        directionConsistency(
            predictedVelocity,
            detectedVelocity);
    result.scaleConsistency =
        std::abs(
            std::log(
                std::clamp(
                    scaleRatio,
                    0.125,
                    8.0)));

    result.total =
        result.geometry * 0.34 +
        result.motionConsistency * 0.14 +
        result.appearance * 0.12 +
        result.faceConsistency * 0.08 +
        result.handConsistency * 0.06 +
        result.contactConsistency * 0.08 +
        result.visibility * 0.06 +
        result.temporal * 0.04 +
        result.scaleConsistency * 0.04 +
        detectorPenalty * 0.04;

    result.ambiguityMargin =
        clamp01(
            1.0 -
                result.total /
                    2.8);

    const bool multimodalCrossingRisk =
        result.geometry > 0.45 &&
        result.appearance > 0.55 &&
        result.faceConsistency > 0.55 &&
        result.handConsistency > 0.45 &&
        result.contactConsistency > 0.45 &&
        detectorConfidence > 0.0 &&
        result.visibility > 0.0;

    result.gated =
        result.geometry > 3.2 ||
        result.temporal > 3.5 ||
        detectorConfidence < 0.05 ||
        result.total > 0.65 ||
        multimodalCrossingRisk;

    return result;
}

} // namespace nox::native::tracking

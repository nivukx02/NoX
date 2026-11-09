#include "tracking/identity_track.h"

#include <algorithm>
#include <cmath>

namespace nox::native::tracking {

namespace {

double clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

double distance(
    Vec3 a,
    Vec3 b) noexcept {
    return std::hypot(
        std::hypot(
            a.x - b.x,
            a.y - b.y),
        a.z - b.z);
}

double speedDifference(
    Vec3 a,
    Vec3 b) noexcept {
    return distance(a, b);
}

double cosineSimilarity(
    Vec3 a,
    Vec3 b) noexcept {
    const double aNorm =
        std::hypot(
            std::hypot(a.x, a.y),
            a.z);
    const double bNorm =
        std::hypot(
            std::hypot(b.x, b.y),
            b.z);
    if (
        aNorm < 1e-8 ||
        bNorm < 1e-8) {
        return 0.5;
    }

    return clamp01(
        0.5 +
            0.5 *
            std::clamp(
                (
                    a.x * b.x +
                    a.y * b.y +
                    a.z * b.z
                ) /
                (aNorm * bNorm),
                -1.0,
                1.0));
}

} // namespace

double IdentityTrack::associationCost(
    const Vec3& centroid,
    const Vec3& predictedCentroid,
    double detectionConfidence) const noexcept {
    IdentityEvidence evidence;
    evidence.predictedCentroid =
        predictedCentroid;
    evidence.detectedCentroid =
        centroid;
    evidence.detectorConfidence =
        detectionConfidence;

    return associationCost(evidence).total;
}

IdentityAssociationCost IdentityTrack::associationCost(
    const IdentityEvidence& evidence) const noexcept {
    IdentityAssociationCost result;

    const double geometryDistance =
        distance(
            evidence.predictedCentroid,
            evidence.detectedCentroid);

    const double geometry =
        std::clamp(
            geometryDistance / 0.75,
            0.0,
            4.0);

    const double velocityDistance =
        speedDifference(
            evidence.predictedVelocity,
            evidence.detectedVelocity);
    const double velocity =
        std::clamp(
            velocityDistance / 4.0,
            0.0,
            4.0);

    const double appearance =
        1.0 -
        clamp01(
            evidence.appearanceSimilarity);
    const double face =
        1.0 -
        clamp01(
            evidence.faceSimilarity);
    const double hand =
        1.0 -
        clamp01(
            evidence.handSimilarity);
    const double contact =
        1.0 -
        clamp01(
            evidence.contactConsistency);
    const double visibility =
        1.0 -
        clamp01(
            evidence.visibility);
    const double temporal =
        std::clamp(
            std::max(
                0.0,
                evidence.temporalGapSeconds) /
                0.25,
            0.0,
            4.0);
    const double scale =
        std::abs(
            std::log(
                std::max(
                    0.125,
                    std::min(
                        8.0,
                        evidence.scaleRatio))));

    const double detectorTrust =
        clamp01(
            evidence.detectorConfidence);
    const double occlusionRisk =
        clamp01(
            evidence.occlusionRisk);

    result.geometry =
        geometry;
    result.velocity =
        velocity;
    result.appearance =
        appearance;
    result.face =
        face;
    result.hand =
        hand;
    result.contact =
        contact;
    result.visibility =
        visibility;
    result.temporal =
        temporal;
    result.scale =
        scale;

    result.total =
        geometry * 0.30 +
        velocity * 0.14 +
        appearance * 0.12 +
        face * 0.08 +
        hand * 0.06 +
        contact * 0.08 +
        visibility * 0.06 +
        temporal * 0.05 +
        scale * 0.04 +
        occlusionRisk * 0.04 +
        (
            1.0 -
            detectorTrust
        ) * 0.03;

    const double directionAgreement =
        cosineSimilarity(
            evidence.predictedVelocity,
            evidence.detectedVelocity);

    result.ambiguityMargin =
        clamp01(
            (
                directionAgreement * 0.35 +
                detectorTrust * 0.20 +
                clamp01(
                    evidence.visibility) * 0.15 +
                clamp01(
                    evidence.appearanceSimilarity) * 0.15 +
                clamp01(
                    evidence.faceSimilarity) * 0.05 +
                clamp01(
                    evidence.handSimilarity) * 0.05 +
                clamp01(
                    evidence.contactConsistency) * 0.05
            ));

    const bool multimodalCrossingRisk =
        occlusionRisk > 0.75 &&
        geometry > 0.45 &&
        appearance > 0.55 &&
        face > 0.55 &&
        hand > 0.45 &&
        contact > 0.45;

    result.gated =
        geometry > 3.2 ||
        temporal > 3.5 ||
        detectorTrust < 0.05 ||
        result.total > 0.65 ||
        multimodalCrossingRisk;

    return result;
}

} // namespace nox::native::tracking

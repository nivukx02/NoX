#include <cassert>
#include <cmath>

#include "tracking/geometric_identity_track.h"
#include "tracking/identity_track.h"

int main() {
    using namespace nox::native::tracking;

    IdentityTrack identity(7);

    IdentityEvidence strong{
        .predictedCentroid = {0.0, 0.0, 1.0},
        .detectedCentroid = {0.015, 0.004, 1.002},
        .predictedVelocity = {0.4, 0.1, 0.0},
        .detectedVelocity = {0.39, 0.11, 0.01},
        .scaleRatio = 1.01,
        .appearanceSimilarity = 0.95,
        .faceSimilarity = 0.94,
        .handSimilarity = 0.90,
        .contactConsistency = 0.92,
        .visibility = 0.94,
        .detectorConfidence = 0.96,
        .temporalGapSeconds = 1.0 / 120.0,
        .occlusionRisk = 0.05,
    };

    const auto strongCost =
        identity.associationCost(strong);
    assert(strongCost.total < 0.50);
    assert(strongCost.ambiguityMargin > 0.70);
    assert(!strongCost.gated);

    IdentityEvidence crossing = strong;
    crossing.detectedCentroid = {0.42, 0.18, 0.96};
    crossing.detectedVelocity = {-0.9, 0.7, 0.0};
    crossing.appearanceSimilarity = 0.22;
    crossing.faceSimilarity = 0.18;
    crossing.handSimilarity = 0.30;
    crossing.contactConsistency = 0.30;
    crossing.temporalGapSeconds = 0.12;
    crossing.occlusionRisk = 0.88;

    const auto crossingCost =
        identity.associationCost(crossing);
    assert(crossingCost.total > strongCost.total);
    assert(crossingCost.gated);

    GeometricIdentityTrack geometric(11);

    const auto geometricStrong =
        geometric.costMultimodal(
            strong.predictedCentroid,
            strong.detectedCentroid,
            strong.predictedVelocity,
            strong.detectedVelocity,
            strong.temporalGapSeconds,
            strong.visibility,
            strong.detectorConfidence,
            strong.appearanceSimilarity,
            strong.faceSimilarity,
            strong.handSimilarity,
            strong.contactConsistency,
            strong.scaleRatio);

    assert(geometricStrong.total < 0.60);
    assert(geometricStrong.ambiguityMargin > 0.70);
    assert(!geometricStrong.gated);

    const auto geometricCrossing =
        geometric.costMultimodal(
            crossing.predictedCentroid,
            crossing.detectedCentroid,
            crossing.predictedVelocity,
            crossing.detectedVelocity,
            crossing.temporalGapSeconds,
            crossing.visibility,
            crossing.detectorConfidence,
            crossing.appearanceSimilarity,
            crossing.faceSimilarity,
            crossing.handSimilarity,
            crossing.contactConsistency,
            crossing.scaleRatio);

    assert(geometricCrossing.total > geometricStrong.total);
    assert(geometricCrossing.gated);

    const auto legacy =
        identity.associationCost(
            strong.detectedCentroid,
            strong.predictedCentroid,
            strong.detectorConfidence);
    assert(std::isfinite(legacy));
    assert(legacy >= 0.0);

    return 0;
}

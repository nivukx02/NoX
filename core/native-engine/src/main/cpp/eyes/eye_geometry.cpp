#include "eyes/eye_geometry.h"

#include <algorithm>
#include <cmath>

namespace nox::native::eyes {
namespace {

Vec3 subtract(
    Vec3 a,
    Vec3 b) noexcept {
    return {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z,
    };
}

double dot(
    Vec3 a,
    Vec3 b) noexcept {
    return
        a.x * b.x +
        a.y * b.y +
        a.z * b.z;
}

double normValue(
    Vec3 a) noexcept {
    return std::sqrt(
        dot(
            a,
            a));
}

Vec3 normalize(
    Vec3 a) noexcept {
    const double n =
        normValue(a);

    if (
        !std::isfinite(n) ||
        n < 1e-12) {
        return {
            0.0,
            0.0,
            1.0,
        };
    }

    return {
        a.x / n,
        a.y / n,
        a.z / n,
    };
}

bool finiteVec(
    Vec3 value) noexcept {
    return
        std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z);
}

double clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

} // namespace

EyeGeometry EyeGeometrySolver::solve(
    const EyeObservation& observation,
    const Vec3& headForward) const noexcept {
    EyeGeometry result;

    if (
        !finiteVec(
            observation.center) ||
        !finiteVec(
            observation.iris) ||
        !finiteVec(
            headForward) ||
        !std::isfinite(
            observation.confidence)) {
        return result;
    }

    result.center =
        observation.center;
    result.iris =
        observation.iris;

    const Vec3 irisVector =
        subtract(
            observation.iris,
            observation.center);

    const double irisDistance =
        normValue(
            irisVector);

    if (
        !std::isfinite(
            irisDistance) ||
        irisDistance <=
            1e-6) {
        return result;
    }

    result.gaze =
        normalize(
            irisVector);

    result.headAlignment =
        clamp01(
            0.5 +
                0.5 *
                std::clamp(
                    dot(
                        result.gaze,
                        normalize(
                            headForward)),
                    -1.0,
                    1.0));

    result.irisQuality =
        observation.irisDiameterMeters >
                1e-6 &&
            std::isfinite(
                observation.irisDiameterMeters)
            ? std::clamp(
                observation.irisDiameterMeters /
                    0.012,
                0.0,
                1.0)
            : 0.50;

    const double detectorConfidence =
        clamp01(
            observation.confidence);

    result.gazeUncertaintyDegrees =
        std::clamp(
            (
                1.0 -
                detectorConfidence
            ) *
                35.0 +
            (
                1.0 -
                result.headAlignment
            ) *
                20.0 +
            (
                1.0 -
                result.irisQuality
            ) *
                12.0,
            0.25,
            180.0);

    result.centerUncertaintyMeters =
        std::clamp(
            (
                1.0 -
                detectorConfidence
            ) *
                0.030 +
            (
                1.0 -
                result.irisQuality
            ) *
                0.008,
            0.0001,
            0.25);

    result.confidence =
        std::clamp(
            detectorConfidence *
                (
                    0.45 +
                    0.30 *
                        result.headAlignment +
                    0.25 *
                        result.irisQuality
                ),
            0.0,
            1.0);

    result.valid =
        detectorConfidence > 0.05 &&
        irisDistance > 1e-6 &&
        result.headAlignment > 0.15 &&
        std::isfinite(
            result.confidence);

    return result;
}

} // namespace nox::native::eyes

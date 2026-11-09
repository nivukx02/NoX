#include "eyes/binocular_vergence.h"

#include <algorithm>
#include <cmath>

namespace nox::native::eyes {
namespace {

double clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}


double dot(
    Vec3 a,
    Vec3 b) noexcept {
    return
        a.x * b.x +
        a.y * b.y +
        a.z * b.z;
}

double norm(
    Vec3 v) noexcept {
    return std::sqrt(
        dot(
            v,
            v));
}

Vec3 add(
    Vec3 a,
    Vec3 b) noexcept {
    return {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z,
    };
}

Vec3 sub(
    Vec3 a,
    Vec3 b) noexcept {
    return {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z,
    };
}

Vec3 scale(
    Vec3 v,
    double s) noexcept {
    return {
        v.x * s,
        v.y * s,
        v.z * s,
    };
}

bool finiteVec(
    Vec3 v) noexcept {
    return
        std::isfinite(v.x) &&
        std::isfinite(v.y) &&
        std::isfinite(v.z);
}

Vec3 normalize(
    Vec3 v) noexcept {
    if (!finiteVec(v)) {
        return {
            0.0,
            0.0,
            1.0,
        };
    }

    const double n =
        norm(v);

    if (
        !std::isfinite(n) ||
        n < 1e-9) {
        return {
            0.0,
            0.0,
            1.0,
        };
    }

    return scale(
        v,
        1.0 / n);
}

double angleDegrees(
    double cosine) noexcept {
    return
        std::acos(
            std::clamp(
                cosine,
                -1.0,
                1.0)) *
        180.0 /
        3.14159265358979323846;
}

} // namespace

VergenceResult
BinocularVergenceSolver::solve(
    const EyeGeometry& left,
    const EyeGeometry& right) const noexcept {
    VergenceResult result;

    if (
        !left.valid ||
        !right.valid ||
        !finiteVec(left.center) ||
        !finiteVec(right.center) ||
        !finiteVec(left.gaze) ||
        !finiteVec(right.gaze) ||
        !std::isfinite(left.confidence) ||
        !std::isfinite(right.confidence)) {
        return result;
    }

    const Vec3 baseline =
        sub(
            left.center,
            right.center);

    const double baselineMeters =
        norm(
            baseline);

    if (
        baselineMeters <
            1e-4) {
        return result;
    }

    const Vec3 leftDirection =
        normalize(
            left.gaze);
    const Vec3 rightDirection =
        normalize(
            right.gaze);

    const double cosine =
        dot(
            leftDirection,
            rightDirection);

    const double denominator =
        1.0 -
        cosine *
            cosine;

    if (
        denominator <
        1e-10) {
        return result;
    }

    const double conditioning =
        std::sqrt(
            std::clamp(
                denominator,
                0.0,
                1.0));

    const double baselineProjection =
        dot(
            leftDirection,
            baseline);

    const double reverseProjection =
        dot(
            rightDirection,
            baseline);

    const double leftDepth =
        (
            cosine *
                reverseProjection -
            baselineProjection
        ) /
        denominator;

    const double rightDepth =
        (
            reverseProjection -
            cosine *
                baselineProjection
        ) /
        denominator;

    if (
        leftDepth <= 0.0 ||
        rightDepth <= 0.0 ||
        !std::isfinite(
            leftDepth) ||
        !std::isfinite(
            rightDepth)) {
        return result;
    }

    const Vec3 leftPoint =
        add(
            left.center,
            scale(
                leftDirection,
                leftDepth));

    const Vec3 rightPoint =
        add(
            right.center,
            scale(
                rightDirection,
                rightDepth));

    const Vec3 midpoint =
        scale(
            add(
                leftPoint,
                rightPoint),
            0.5);

    const double separation =
        norm(
            sub(
                leftPoint,
                rightPoint));

    const double meanDepth =
        0.5 *
        (
            leftDepth +
            rightDepth
        );

    const double depthAsymmetry =
        meanDepth <= 1e-9
            ? 1.0
            : std::abs(
                leftDepth -
                rightDepth
            ) /
                meanDepth;

    const double convergenceAngle =
        std::acos(
            std::clamp(
                cosine,
                -1.0,
                1.0));

    const double convergenceAngleDegrees =
        angleDegrees(
            cosine);

    const double parallax =
        std::atan2(
            baselineMeters,
            std::max(
                norm(
                    midpoint),
                1e-6));

    const double parallaxDegrees =
        parallax *
        180.0 /
        3.14159265358979323846;

    const double confidenceBase =
        std::min(
            clamp01(
                left.confidence),
            clamp01(
                right.confidence));

    const double uncertainty =
        std::sqrt(
            std::max(
                0.0,
                left.centerUncertaintyMeters *
                    left.centerUncertaintyMeters +
                right.centerUncertaintyMeters *
                    right.centerUncertaintyMeters +
                (
                    left.gazeUncertaintyDegrees +
                    right.gazeUncertaintyDegrees
                ) /
                    180.0 *
                    separation *
                    separation));

    const double separationConfidence =
        std::exp(
            -separation /
            0.15);

    const double angularConfidence =
        std::sqrt(
            std::clamp(
                convergenceAngle /
                    0.03,
                0.0,
                1.0));

    const double conditioningConfidence =
        std::sqrt(
            std::clamp(
                conditioning /
                    0.12,
                0.0,
                1.0));

    const double depthConfidence =
        std::exp(
            -uncertainty /
            0.05);

    const double asymmetryConfidence =
        std::exp(
            -depthAsymmetry /
            0.12);

    const double angularResidualDegrees =
        convergenceAngleDegrees >
            0.0
            ? std::abs(
                convergenceAngleDegrees -
                parallaxDegrees)
            : 180.0;

    const double rayAlignmentConfidence =
        std::exp(
            -angularResidualDegrees /
            25.0);

    result.point =
        midpoint;
    result.distance =
        norm(
            midpoint);
    result.raySeparation =
        separation;
    result.convergenceAngleDegrees =
        convergenceAngleDegrees;
    result.parallaxDegrees =
        parallaxDegrees;
    result.depthAsymmetry =
        depthAsymmetry;
    result.angularResidualDegrees =
        angularResidualDegrees;
    result.rayAlignmentConfidence =
        std::clamp(
            rayAlignmentConfidence,
            0.0,
            1.0);
    result.conditioning =
        conditioning;
    result.uncertaintyMeters =
        std::max(
            1e-6,
            uncertainty);

    result.confidence =
        std::clamp(
            confidenceBase *
            separationConfidence *
            angularConfidence *
            (
                0.55 +
                0.45 *
                    conditioningConfidence
            ) *
            (
                0.55 +
                0.25 *
                    depthConfidence +
                0.20 *
                    asymmetryConfidence
            ) *
            (
                0.75 +
                0.25 *
                    rayAlignmentConfidence
            ),
            0.0,
            1.0);

    const bool strongFixationGeometry =
        separation < 0.20 &&
        parallaxDegrees > 0.01 &&
        result.distance > 0.10 &&
        convergenceAngleDegrees > 0.10 &&
        std::isfinite(result.distance);

    if (strongFixationGeometry) {
        result.confidence =
            std::max(
                result.confidence,
                confidenceBase * 0.55);
    }

    const bool geometricallyValid =
        separation <
            0.20 &&
        conditioning >
            0.001 &&
        parallaxDegrees >
            0.01 &&
        std::isfinite(result.distance) &&
        result.distance > 0.10;

    result.valid =
        geometricallyValid &&
        std::isfinite(result.confidence) &&
        result.confidence >= 0.15;

    (void)convergenceAngle;
    return result;
}

} // namespace nox::native::eyes

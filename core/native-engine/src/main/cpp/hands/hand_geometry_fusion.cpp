#include "hands/hand_geometry_fusion.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>
#include <numeric>

namespace nox::native::hands {
namespace {
Vec3 sub(Vec3 a,Vec3 b) noexcept {
    return {a.x-b.x,a.y-b.y,a.z-b.z};
}
Vec3 add(Vec3 a,Vec3 b) noexcept {
    return {a.x+b.x,a.y+b.y,a.z+b.z};
}
Vec3 scale(Vec3 a,double s) noexcept {
    return {a.x*s,a.y*s,a.z*s};
}
Vec3 cross(Vec3 a,Vec3 b) noexcept {
    return {
        a.y*b.z-a.z*b.y,
        a.z*b.x-a.x*b.z,
        a.x*b.y-a.y*b.x,
    };
}
double dot(Vec3 a,Vec3 b) noexcept {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
double norm(Vec3 a) noexcept {
    return std::sqrt(dot(a,a));
}
Vec3 normalize(Vec3 a) noexcept {
    const double n=norm(a);
    return n<1e-9?Vec3{0,0,1}:scale(a,1.0/n);
}
}

HandState HandGeometryFusion::solve(
    const std::vector<Joint>& joints,
    const Vec3& wrist,
    const Vec3& bodyWrist,
    double bodyCouplingConfidence) const {
    return solve(
        joints,
        wrist,
        bodyWrist,
        bodyCouplingConfidence,
        std::nullopt);
}

HandState HandGeometryFusion::solve(
    const std::vector<Joint>& joints,
    const Vec3& wrist,
    const Vec3& bodyWrist,
    double bodyCouplingConfidence,
    std::optional<double> metricScaleMetersPerUnit) const {
    HandState output;
    output.wrist=wrist;
    output.joints=joints;

    double confidenceSum =
        std::clamp(
            bodyCouplingConfidence,
            0.0,
            1.0);
    std::size_t visible = 0;

    for(const auto& joint:joints){
        confidenceSum+=std::clamp(joint.confidence,0.0,1.0);
        if(joint.confidence>0.1)++visible;
    }

    output.occlusionRatio=
        1.0-
        static_cast<double>(visible)/
        static_cast<double>(std::max<std::size_t>(joints.size(),1));

    output.palm=wrist;

    bool geometryWellConditioned = false;

    if (joints.size() >= 3) {
        Vec3 centroid{};
        for (const auto& joint : joints) {
            centroid =
                add(
                    centroid,
                    joint.position);
        }
        centroid =
            scale(
                centroid,
                1.0 /
                    static_cast<double>(
                        joints.size()));
        output.palm = centroid;

        const Vec3 a =
            sub(
                joints[1].position,
                joints[0].position);
        const Vec3 b =
            sub(
                joints[2].position,
                joints[0].position);
        const Vec3 normal =
            cross(a, b);
        const double areaScale =
            norm(normal);
        output.palmNormal =
            normalize(normal);

        double pairDistanceSum = 0.0;
        std::size_t pairCount = 0;
        std::vector<double> pairDistances;
        pairDistances.reserve(
            joints.size() *
            (joints.size() - 1U) /
            2U);

        for (std::size_t i = 0;
             i < joints.size();
             ++i) {
            for (std::size_t j = i + 1U;
                 j < joints.size();
                 ++j) {
                const double pairDistance =
                    norm(
                        sub(
                            joints[i].position,
                            joints[j].position));
                if (!std::isfinite(pairDistance)) {
                    continue;
                }
                pairDistanceSum +=
                    pairDistance;
                pairDistances.push_back(
                    pairDistance);
                ++pairCount;
            }
        }

        output.handScale =
            pairCount == 0U
                ? 0.0
                : pairDistanceSum /
                    static_cast<double>(
                        pairCount);

        std::sort(
            pairDistances.begin(),
            pairDistances.end());

        double p95PairDistance = 0.0;
        if (!pairDistances.empty()) {
            const double position =
                0.95 *
                static_cast<double>(
                    pairDistances.size() - 1U);
            const auto lower =
                static_cast<std::size_t>(
                    std::floor(position));
            const auto upper =
                std::min(
                    pairDistances.size() - 1U,
                    lower + 1U);
            const double alpha =
                position -
                static_cast<double>(
                    lower);
            p95PairDistance =
                pairDistances[lower] +
                (
                    pairDistances[upper] -
                    pairDistances[lower]
                ) *
                alpha;
        }

        static_cast<void>(p95PairDistance);

        std::vector<double> chainLengths;
        chainLengths.reserve(
            joints.size() - 1U);

        for (std::size_t i = 1U;
             i < joints.size();
             ++i) {
            const double chainLength =
                norm(
                    sub(
                        joints[i].position,
                        joints[i - 1U].position));

            if (
                std::isfinite(
                    chainLength) &&
                chainLength >=
                    1e-5) {
                chainLengths.push_back(
                    chainLength);
            }
        }

        const double localReferenceLength =
            chainLengths.empty()
                ? output.handScale
                : [&chainLengths]() {
                    auto sorted =
                        chainLengths;
                    std::sort(
                        sorted.begin(),
                        sorted.end());
                    return sorted[
                        sorted.size() / 2U];
                }();

        double lengthResidualSum = 0.0;
        std::size_t lengthCount = 0U;
        std::vector<double> lengthResiduals;
        lengthResiduals.reserve(
            chainLengths.size());

        for (const double chainLength :
             chainLengths) {
            if (
                !std::isfinite(
                    localReferenceLength) ||
                localReferenceLength <
                    1e-5) {
                continue;
            }

            const double normalizedResidual =
                std::abs(
                    chainLength -
                    localReferenceLength) /
                localReferenceLength;

            lengthResidualSum +=
                normalizedResidual;
            lengthResiduals.push_back(
                normalizedResidual);
            ++lengthCount;
        }

        double p95LengthResidual = 0.0;
        if (!lengthResiduals.empty()) {
            std::sort(
                lengthResiduals.begin(),
                lengthResiduals.end());
            const double position =
                0.95 *
                static_cast<double>(
                    lengthResiduals.size() - 1U);
            const auto lower =
                static_cast<std::size_t>(
                    std::floor(position));
            const auto upper =
                std::min(
                    lengthResiduals.size() - 1U,
                    lower + 1U);
            const double alpha =
                position -
                static_cast<double>(
                    lower);
            p95LengthResidual =
                lengthResiduals[lower] +
                (
                    lengthResiduals[upper] -
                    lengthResiduals[lower]
                ) *
                alpha;
        }
        output.p95LengthResidual =
            p95LengthResidual;

        const double meanLengthResidual =
            lengthCount == 0U
                ? 1.0
                : lengthResidualSum /
                    static_cast<double>(
                        lengthCount);

        output.kinematicConsistency =
            std::clamp(
                std::exp(
                    -std::min(
                        6.0,
                        meanLengthResidual * 4.0)),
                0.0,
                1.0);

        double planarResidualSum = 0.0;
        std::size_t planarCount = 0U;
        for (const auto& joint : joints) {
            const double residual =
                std::abs(
                    dot(
                        sub(
                            joint.position,
                            output.palm),
                        output.palmNormal));
            if (!std::isfinite(residual)) {
                continue;
            }
            planarResidualSum += residual;
            ++planarCount;
        }
        const double meanPlanarResidual =
            planarCount == 0U
                ? output.handScale
                : planarResidualSum /
                    static_cast<double>(
                        planarCount);

        output.planarityConfidence =
            std::clamp(
                std::exp(
                    -meanPlanarResidual /
                    std::max(
                        output.handScale * 0.20,
                        1e-4)),
                0.0,
                1.0);

        geometryWellConditioned =
            areaScale > 1e-6 &&
            output.handScale > 1e-4 &&
            output.kinematicConsistency > 0.20 &&
            output.planarityConfidence > 0.20;

        if(metricScaleMetersPerUnit.has_value() &&
           std::isfinite(*metricScaleMetersPerUnit) &&
           *metricScaleMetersPerUnit>0.0 &&
           bodyCouplingConfidence>=0.5) {
            output.metric=true;
            output.metricScaleMetersPerUnit=
                *metricScaleMetersPerUnit;
        }
    }

    output.wristCouplingError =
        norm(
            sub(
                wrist,
                bodyWrist));

    output.wristCouplingConfidence =
        std::clamp(
            std::exp(
                -output.wristCouplingError /
                0.08),
            0.0,
            1.0);

    const double visibilityConfidence =
        1.0 -
        output.occlusionRatio;
    const double geometryConfidence =
        geometryWellConditioned
            ? 0.95
            : 0.55;

    if (output.metric) {
        output.metricConfidence =
            std::clamp(
                (
                    output.wristCouplingConfidence * 0.20 +
                    output.kinematicConsistency * 0.30 +
                    output.planarityConfidence * 0.20 +
                    geometryConfidence * 0.30
                ),
                0.0,
                1.0);
    } else {
        output.metricConfidence =
            0.0;
    }

    output.confidence =
        std::clamp(
            (
                confidenceSum /
                static_cast<double>(
                    joints.size() + 1U)
            ) *
            output.wristCouplingConfidence *
            output.kinematicConsistency *
            (
                0.65 +
                0.35 *
                    geometryConfidence
            ) *
            (
                0.70 +
                0.30 *
                    visibilityConfidence
            ),
            0.0,
            1.0);

    output.accepted =
        joints.size() >= 3U &&
        output.confidence >= 0.35 &&
        output.kinematicConsistency >= 0.34 &&
        output.wristCouplingConfidence >= 0.35 &&
        (
            !output.metric ||
            output.metricConfidence >= 0.45
        );

    return output;
}


HandState HandGeometryFusion::solveTemporal(
    const std::vector<Joint>& joints,
    const Vec3& wrist,
    const Vec3& bodyWrist,
    double bodyCouplingConfidence,
    const HandState& previous,
    double dtSeconds,
    const HandTemporalOptions& options,
    std::optional<double> metricScaleMetersPerUnit) const {
    HandState current =
        solve(
            joints,
            wrist,
            bodyWrist,
            bodyCouplingConfidence,
            metricScaleMetersPerUnit);

    if (
        !std::isfinite(dtSeconds) ||
        dtSeconds <= 0.0 ||
        dtSeconds > options.maximumDtSeconds) {
        current.temporalConfidence = 0.0;
        current.temporalConsistency = 0.0;
        current.temporalRejected = true;
        current.accepted = false;
        return current;
    }

    if (
        previous.joints.empty()) {
        current.temporalConfidence = 0.50;
        current.temporalConsistency = 0.50;
        return current;
    }

    const double safeMaximumVelocity =
        std::max(
            options.maximumJointVelocityMetersPerSecond,
            0.01);
    const double safeMaximumInnovation =
        std::max(
            options.maximumInnovationMeters,
            1e-5);
    const double safeInnovationScale =
        std::max(
            options.innovationScaleMeters,
            1e-5);

    std::vector<double> innovations;
    innovations.reserve(
        joints.size());

    double confidenceSum = 0.0;
    double velocitySum = 0.0;
    std::size_t matched = 0U;
    std::size_t rejected = 0U;

    for (
        const auto& joint :
        joints) {
        const auto previousIt =
            std::find_if(
                previous.joints.begin(),
                previous.joints.end(),
                [&joint](const Joint& candidate) {
                    return candidate.id == joint.id;
                });

        if (
            previousIt ==
            previous.joints.end()) {
            continue;
        }

        const double innovation =
            norm(
                sub(
                    joint.position,
                    previousIt->position));

        const double velocity =
            innovation /
            dtSeconds;

        const double innovationConfidence =
            std::exp(
                -innovation /
                safeInnovationScale);

        const double velocityConfidence =
            std::clamp(
                1.0 -
                    std::max(
                        0.0,
                        velocity -
                            safeMaximumVelocity) /
                    safeMaximumVelocity,
                0.0,
                1.0);

        innovations.push_back(
            innovation);

        confidenceSum +=
            std::clamp(
                joint.confidence,
                0.0,
                1.0) *
            innovationConfidence *
            velocityConfidence;

        velocitySum += velocity;
        ++matched;

        if (
            innovation >
                safeMaximumInnovation ||
            velocity >
                safeMaximumVelocity) {
            ++rejected;
        }
    }

    if (
        innovations.empty()) {
        current.temporalConfidence = 0.20;
        current.temporalConsistency = 0.20;
        current.temporalRejected = true;
        current.accepted = false;
        return current;
    }

    std::sort(
        innovations.begin(),
        innovations.end());

    const double position =
        0.95 *
        static_cast<double>(
            innovations.size() - 1U);

    const auto lower =
        static_cast<std::size_t>(
            std::floor(position));

    const auto upper =
        std::min(
            innovations.size() - 1U,
            lower + 1U);

    const double alpha =
        position -
        static_cast<double>(
            lower);

    current.meanInnovationMeters =
        std::accumulate(
            innovations.begin(),
            innovations.end(),
            0.0) /
        static_cast<double>(
            innovations.size());

    current.p95InnovationMeters =
        innovations[lower] +
        (
            innovations[upper] -
            innovations[lower]
        ) *
        alpha;

    current.maximumInnovationMeters =
        innovations.back();

    current.estimatedJointVelocityMetersPerSecond =
        velocitySum /
        static_cast<double>(
            matched);

    const double matchRatio =
        std::clamp(
            static_cast<double>(
                matched) /
            static_cast<double>(
                std::max(
                    joints.size(),
                    std::size_t{1})),
            0.0,
            1.0);

    const double meanEvidence =
        std::clamp(
            confidenceSum /
            static_cast<double>(
                std::max(
                    matched,
                    std::size_t{1})),
            0.0,
            1.0);

    const double innovationEvidence =
        std::exp(
            -current.p95InnovationMeters /
            safeInnovationScale);

    const double rejectionEvidence =
        1.0 -
        std::clamp(
            static_cast<double>(
                rejected) /
            static_cast<double>(
                std::max(
                    matched,
                    std::size_t{1})),
            0.0,
            1.0);

    current.temporalConfidence =
        std::clamp(
            matchRatio * 0.25 +
            meanEvidence * 0.25 +
            innovationEvidence * 0.30 +
            rejectionEvidence * 0.20,
            0.0,
            1.0);

    current.temporalConsistency =
        std::clamp(
            0.55 * innovationEvidence +
            0.25 * rejectionEvidence +
            0.20 * matchRatio,
            0.0,
            1.0);

    current.temporalRejected =
        current.p95InnovationMeters >
            safeMaximumInnovation ||
        current.estimatedJointVelocityMetersPerSecond >
            safeMaximumVelocity ||
        current.temporalConfidence <
            options.minimumTemporalConfidence;

    if (
        current.temporalRejected) {
        current.accepted = false;
        current.confidence *= 0.65;
    }

    return current;
}

} // namespace nox::native::hands

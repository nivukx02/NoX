#include "tracking/camera_motion_compensator.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace nox::native::tracking {

namespace {

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

Vec3 mul(
    Vec3 value,
    double scalar) noexcept {
    return {
        value.x * scalar,
        value.y * scalar,
        value.z * scalar,
    };
}

Vec3 cross(
    Vec3 a,
    Vec3 b) noexcept {
    return {
        a.y * b.z -
            a.z * b.y,
        a.z * b.x -
            a.x * b.z,
        a.x * b.y -
            a.y * b.x,
    };
}

double norm(
    Vec3 value) noexcept {
    return std::hypot(
        std::hypot(
            value.x,
            value.y),
        value.z);
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

std::vector<JointObservation> CameraMotionCompensator::toWorld(
    const std::vector<JointObservation>& observations,
    const math::SE3d& cameraToWorld) const {
    std::vector<JointObservation> output;
    output.reserve(
        observations.size());

    for (
        const auto& observation :
        observations) {
        const auto world =
            cameraToWorld.transform({
                observation.position.x,
                observation.position.y,
                observation.position.z,
            });

        output.push_back({
            observation.id,
            {
                world.x,
                world.y,
                world.z,
            },
            observation.confidence,
        });
    }

    return output;
}

CameraMotionCompensationResult
CameraMotionCompensator::toWorldHighEnd(
    const std::vector<JointObservation>& observations,
    const CameraMotionContext& context) const {
    CameraMotionCompensationResult result;

    result.observations.reserve(
        observations.size());

    const double captureOffset =
        std::isfinite(
            context.captureOffsetSeconds)
            ? context.captureOffsetSeconds
            : 0.0;

    const double readoutOffset =
        (
            std::clamp(
                std::isfinite(
                    context.normalizedSensorRow)
                    ? context.normalizedSensorRow
                    : 0.5,
                0.0,
                1.0) -
            0.5
        ) *
        std::clamp(
            std::isfinite(
                context.readoutDurationSeconds)
                ? context.readoutDurationSeconds
                : 0.0,
            0.0,
            0.25);

    const double dt =
        std::clamp(
            captureOffset +
                readoutOffset,
            -0.25,
            0.25);

    const double uncertainty =
        std::max(
            0.0,
            std::isfinite(
                context.positionUncertaintyMeters)
                ? context.positionUncertaintyMeters
                : 0.0);

    const double angularUncertainty =
        std::max(
            0.0,
            std::isfinite(
                context.angularUncertaintyRadians)
                ? context.angularUncertaintyRadians
                : 0.0);

    const double motionConfidence =
        clamp01(
            context.motionConfidence);

    double correctionSum = 0.0;
    double correctionMax = 0.0;
    double blurRiskSum = 0.0;

    const double motionBlurRisk =
        clamp01(context.motionBlurRisk);

    for (
        const auto& observation :
        observations) {
        const auto world =
            context.cameraToWorld.transform({
                observation.position.x,
                observation.position.y,
                observation.position.z,
            });

        const Vec3 worldPoint{
            world.x,
            world.y,
            world.z,
        };

        const Vec3 angularVelocityCorrection =
            cross(
                context.angularVelocityWorld,
                worldPoint);

        const Vec3 worldVelocity =
            add(
                context.linearVelocityWorld,
                angularVelocityCorrection);

        const Vec3 angularAccelerationCorrection =
            cross(
                context.angularAccelerationWorld,
                worldPoint);

        const Vec3 centripetalCorrection =
            cross(
                context.angularVelocityWorld,
                angularVelocityCorrection);

        const Vec3 worldAcceleration =
            add(
                context.linearAccelerationWorld,
                add(
                    angularAccelerationCorrection,
                    centripetalCorrection));

        const Vec3 motionDelta =
            add(
                mul(
                    worldVelocity,
                    dt),
                mul(
                    worldAcceleration,
                    0.5 *
                        dt *
                        dt));

        const Vec3 compensated =
            sub(
                worldPoint,
                motionDelta);

        const double translationCorrection =
            norm(
                motionDelta);

        const double confidencePenalty =
            clamp01(
                uncertainty * 25.0 +
                angularUncertainty * 2.0 +
                translationCorrection * 2.0 +
                motionBlurRisk * 0.85);

        const double confidence =
            clamp01(
                observation.confidence *
                motionConfidence *
                (
                    1.0 -
                    0.75 *
                        confidencePenalty
                ));

        result.observations.push_back({
            observation.id,
            compensated,
            confidence,
        });

        correctionSum +=
            translationCorrection;
        correctionMax =
            std::max(
                correctionMax,
                translationCorrection);
    }

    result.meanTranslationCorrectionMeters =
        observations.empty()
            ? 0.0
            : correctionSum /
                static_cast<double>(
                    observations.size());

    result.maximumTranslationCorrectionMeters =
        correctionMax;

    result.angularCorrectionRadians =
        norm(
            context.angularVelocityWorld) *
        std::abs(dt);

    result.meanMotionBlurRisk =
        motionBlurRisk;

    result.confidence =
        clamp01(
            observations.empty()
                ? 0.0
                : (
                    std::accumulate(
                        result.observations.begin(),
                        result.observations.end(),
                        0.0,
                        [](double sum,
                           const JointObservation& item) {
                            return sum +
                                clamp01(item.confidence);
                        }) /
                    static_cast<double>(
                        result.observations.size())
                ));

    result.compensated =
        !result.observations.empty() &&
        (
            result.meanTranslationCorrectionMeters >
                1e-8 ||
            result.angularCorrectionRadians >
                1e-8
        );

    return result;
}

} // namespace nox::native::tracking

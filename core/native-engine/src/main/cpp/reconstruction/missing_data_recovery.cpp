#include "reconstruction/missing_data_recovery.h"
#include <algorithm>
#include <cmath>

namespace nox::native::recovery {
Result MissingDataRecovery::predict(
    const Track& input,
    double gap,
    double maxGap,
    double maxSpeed,
    double maxAcceleration,
    double maxJerk) const noexcept {
    Result output;
    output.gapSeconds = std::max(0.0, gap);
    output.track = input;

    const finiteInput =
        std::isfinite(gap) &&
        std::isfinite(maxGap) &&
        std::isfinite(maxSpeed) &&
        std::isfinite(maxAcceleration) &&
        std::isfinite(maxJerk) &&
        std::isfinite(input.position.x) &&
        std::isfinite(input.position.y) &&
        std::isfinite(input.position.z) &&
        std::isfinite(input.velocity.x) &&
        std::isfinite(input.velocity.y) &&
        std::isfinite(input.velocity.z) &&
        std::isfinite(input.acceleration.x) &&
        std::isfinite(input.acceleration.y) &&
        std::isfinite(input.acceleration.z) &&
        std::isfinite(input.jerk.x) &&
        std::isfinite(input.jerk.y) &&
        std::isfinite(input.jerk.z) &&
        std::isfinite(input.confidence) &&
        std::isfinite(input.uncertainty);

    if (!finiteInput || gap > maxGap || maxGap <= 0.0) {
        output.track.confidence = 0.0;
        output.track.uncertainty = 1.0;
        output.observation = {occlusion::State::Unsupported, 0.0, 1.0};
        return output;
    }
    const double speed = std::sqrt(
        input.velocity.x * input.velocity.x +
        input.velocity.y * input.velocity.y +
        input.velocity.z * input.velocity.z);
    const double boundedSpeed =
        std::min(speed, std::max(0.0, maxSpeed));
    const double velocityScale =
        speed > 1e-9
            ? boundedSpeed / speed
            : 0.0;

    const double accelerationMagnitude =
        std::sqrt(
            input.acceleration.x * input.acceleration.x +
            input.acceleration.y * input.acceleration.y +
            input.acceleration.z * input.acceleration.z);
    const double boundedAccelerationMagnitude =
        std::min(
            accelerationMagnitude,
            std::max(0.0, maxAcceleration));
    const double accelerationScale =
        accelerationMagnitude > 1e-9
            ? boundedAccelerationMagnitude /
                accelerationMagnitude
            : 0.0;

    const double jerkMagnitude =
        std::sqrt(
            input.jerk.x * input.jerk.x +
            input.jerk.y * input.jerk.y +
            input.jerk.z * input.jerk.z);
    const double boundedJerkMagnitude =
        std::min(
            jerkMagnitude,
            std::max(0.0, maxJerk));
    const double jerkScale =
        jerkMagnitude > 1e-9
            ? boundedJerkMagnitude /
                jerkMagnitude
            : 0.0;

    const double t =
        std::max(0.0, gap);
    const double t2 = t * t;
    const double t3 = t2 * t;

    output.track.position.x +=
        input.velocity.x * velocityScale * t +
        0.5 *
            input.acceleration.x *
            accelerationScale *
            t2 +
        (1.0 / 6.0) *
            input.jerk.x *
            jerkScale *
            t3;
    output.track.position.y +=
        input.velocity.y * velocityScale * t +
        0.5 *
            input.acceleration.y *
            accelerationScale *
            t2 +
        (1.0 / 6.0) *
            input.jerk.y *
            jerkScale *
            t3;
    output.track.position.z +=
        input.velocity.z * velocityScale * t +
        0.5 *
            input.acceleration.z *
            accelerationScale *
            t2 +
        (1.0 / 6.0) *
            input.jerk.z *
            jerkScale *
            t3;

    const double ratio =
        std::clamp(gap / maxGap, 0.0, 1.0);
    const double dynamicRisk =
        std::clamp(
            0.40 *
                    (
                        boundedAccelerationMagnitude /
                            std::max(
                                1e-9,
                                std::max(
                                    0.0,
                                    maxAcceleration))
                    ) +
                0.20 *
                    (
                        boundedJerkMagnitude /
                            std::max(
                                1e-9,
                                std::max(
                                    0.0,
                                    maxJerk))
                    ),
            0.0,
            1.0);
    output.track.confidence =
        std::clamp(
            input.confidence *
                (1.0 - 0.85 * ratio) *
                (1.0 - 0.25 * dynamicRisk),
            0.0,
            1.0);
    output.track.uncertainty =
        std::clamp(
            input.uncertainty +
                0.9 * ratio +
                0.15 * dynamicRisk,
            0.0,
            1.0);
    output.observation = occlusion::OcclusionStateTracker{}.predict(
        input.confidence, input.uncertainty, gap, maxGap);
    return output;
}
}

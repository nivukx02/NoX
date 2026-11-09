#include "tracking/adaptive_motion_regime.h"

#include <algorithm>
#include <cmath>

namespace nox::native::tracking {

AdaptiveMotionRegimeModel::AdaptiveMotionRegimeModel() noexcept
    : AdaptiveMotionRegimeModel(Config{}) {}

AdaptiveMotionRegimeModel::AdaptiveMotionRegimeModel(
    Config config) noexcept
    : config_(config) {
    config_.staticVelocityPixelsPerSecond =
        std::max(config_.staticVelocityPixelsPerSecond, 0.1);
    config_.microVelocityPixelsPerSecond =
        std::max(
            config_.microVelocityPixelsPerSecond,
            config_.staticVelocityPixelsPerSecond);
    config_.fastVelocityPixelsPerSecond =
        std::max(
            config_.fastVelocityPixelsPerSecond,
            config_.microVelocityPixelsPerSecond);
    config_.extremeVelocityPixelsPerSecond =
        std::max(
            config_.extremeVelocityPixelsPerSecond,
            config_.fastVelocityPixelsPerSecond);
    config_.staticAccelerationPixelsPerSecond2 =
        std::max(
            config_.staticAccelerationPixelsPerSecond2,
            1.0);
    config_.fastAccelerationPixelsPerSecond2 =
        std::max(
            config_.fastAccelerationPixelsPerSecond2,
            config_.staticAccelerationPixelsPerSecond2);
    config_.extremeAccelerationPixelsPerSecond2 =
        std::max(
            config_.extremeAccelerationPixelsPerSecond2,
            config_.fastAccelerationPixelsPerSecond2);
    config_.discontinuityInnovationPixels =
        std::max(config_.discontinuityInnovationPixels, 1.0);
    config_.extremeInnovationPixels =
        std::max(
            config_.extremeInnovationPixels,
            config_.discontinuityInnovationPixels);
    config_.maximumTemporalGapSeconds =
        std::clamp(config_.maximumTemporalGapSeconds, 0.005, 2.0);
    config_.maximumRollingShutterSeconds =
        std::clamp(config_.maximumRollingShutterSeconds, 0.0001, 0.250);
}

AdaptiveMotionRegimeDecision AdaptiveMotionRegimeModel::evaluate(
    const AdaptiveMotionRegimeInput& raw) const noexcept {
    const double velocity = std::max(
        0.0,
        std::isfinite(raw.velocityMagnitudePixelsPerSecond)
            ? raw.velocityMagnitudePixelsPerSecond
            : 0.0);
    const double acceleration = std::max(
        0.0,
        std::isfinite(raw.accelerationMagnitudePixelsPerSecond2)
            ? raw.accelerationMagnitudePixelsPerSecond2
            : 0.0);
    const double jerk = std::max(
        0.0,
        std::isfinite(raw.jerkMagnitudePixelsPerSecond3)
            ? raw.jerkMagnitudePixelsPerSecond3
            : 0.0);
    const double innovation = std::max(
        0.0,
        std::isfinite(raw.innovationPixels)
            ? raw.innovationPixels
            : 0.0);
    const double confidence = clamp01(raw.confidence);
    const double gap = std::max(
        0.0,
        std::isfinite(raw.temporalGapSeconds)
            ? raw.temporalGapSeconds
            : 0.0);
    const double shutter = std::max(
        0.0,
        std::isfinite(raw.rollingShutterSkewSeconds)
            ? raw.rollingShutterSkewSeconds
            : 0.0);
    const double direction = clamp01(raw.directionConsistency);
    const double microRetention = clamp01(raw.microMotionRetention);

    AdaptiveMotionRegimeDecision out{};

    const double fastVelocity = safeRatio(
        velocity,
        config_.fastVelocityPixelsPerSecond);
    const double extremeVelocity = safeRatio(
        velocity,
        config_.extremeVelocityPixelsPerSecond);
    const double fastAcceleration = safeRatio(
        acceleration,
        config_.fastAccelerationPixelsPerSecond2);
    const double extremeAcceleration = safeRatio(
        acceleration,
        config_.extremeAccelerationPixelsPerSecond2);
    const double innovationRatio = safeRatio(
        innovation,
        config_.discontinuityInnovationPixels);

    out.dynamicStress = clamp01(
        fastVelocity * 0.26 +
        fastAcceleration * 0.26 +
        std::sqrt(
            std::min(
                4.0,
                jerk /
                    std::max(
                        config_.fastAccelerationPixelsPerSecond2 * 8.0,
                        1.0))) *
            0.12 +
        innovationRatio * 0.22 +
        (1.0 - direction) * 0.07 +
        microRetention * 0.07);

    if (
        innovation >= config_.extremeInnovationPixels ||
        extremeVelocity >= 1.0 ||
        extremeAcceleration >= 1.0) {
        out.regime = AdaptiveMotionRegime::extreme;
    } else if (
        innovation >= config_.discontinuityInnovationPixels &&
        direction < 0.25 &&
        confidence < 0.65) {
        out.regime = AdaptiveMotionRegime::discontinuous;
    } else if (
        velocity >= config_.fastVelocityPixelsPerSecond ||
        acceleration >= config_.fastAccelerationPixelsPerSecond2) {
        out.regime = AdaptiveMotionRegime::fast;
    } else if (
        velocity >= config_.microVelocityPixelsPerSecond ||
        acceleration >= config_.staticAccelerationPixelsPerSecond2 * 2.0 ||
        microRetention < 0.50) {
        out.regime = AdaptiveMotionRegime::normal;
    } else if (
        velocity > config_.staticVelocityPixelsPerSecond ||
        microRetention < 0.92) {
        out.regime = AdaptiveMotionRegime::microMotion;
    } else {
        out.regime = AdaptiveMotionRegime::staticScene;
    }

    const double gapRatio =
        clamp01(gap / config_.maximumTemporalGapSeconds);
    const double shutterRatio =
        clamp01(shutter / config_.maximumRollingShutterSeconds);

    out.continuityScore = clamp01(
        1.0 -
        gapRatio * 0.55 -
        shutterRatio * 0.25 -
        (1.0 - confidence) * 0.20);

    switch (out.regime) {
        case AdaptiveMotionRegime::staticScene:
            out.predictionTrustScale = 1.05;
            out.processNoiseScale = 0.70;
            out.innovationGateScale = 0.85;
            out.scoreThresholdScale = 1.10;
            out.microMotionPreservation = 1.15;
            out.motionEnergyScale = 0.65;
            break;
        case AdaptiveMotionRegime::microMotion:
            out.predictionTrustScale = 0.98;
            out.processNoiseScale = 0.85;
            out.innovationGateScale = 0.92;
            out.scoreThresholdScale = 1.04;
            out.microMotionPreservation = 1.35;
            out.motionEnergyScale = 0.78;
            break;
        case AdaptiveMotionRegime::normal:
            out.predictionTrustScale = 0.92;
            out.processNoiseScale = 1.00;
            out.innovationGateScale = 1.00;
            out.scoreThresholdScale = 1.00;
            out.microMotionPreservation = 1.05;
            out.motionEnergyScale = 1.00;
            break;
        case AdaptiveMotionRegime::fast:
            out.predictionTrustScale = 0.76;
            out.processNoiseScale = 1.65;
            out.innovationGateScale = 1.35;
            out.scoreThresholdScale = 0.94;
            out.microMotionPreservation = 0.90;
            out.motionEnergyScale = 1.30;
            break;
        case AdaptiveMotionRegime::extreme:
            out.predictionTrustScale = 0.48;
            out.processNoiseScale = 2.75;
            out.innovationGateScale = 1.85;
            out.scoreThresholdScale = 0.78;
            out.microMotionPreservation = 0.72;
            out.motionEnergyScale = 1.65;
            break;
        case AdaptiveMotionRegime::discontinuous:
            out.predictionTrustScale = 0.22;
            out.processNoiseScale = 3.50;
            out.innovationGateScale = 2.20;
            out.scoreThresholdScale = 0.55;
            out.microMotionPreservation = 0.55;
            out.motionEnergyScale = 1.85;
            break;
    }

    out.predictionTrustScale = std::clamp(
        out.predictionTrustScale *
            (0.60 + 0.40 * confidence) *
            out.continuityScore,
        0.0,
        1.15);
    out.processNoiseScale = std::clamp(
        out.processNoiseScale *
            (1.0 + out.dynamicStress * 0.55),
        0.25,
        5.0);
    out.innovationGateScale = std::clamp(
        out.innovationGateScale *
            (1.0 + gapRatio * 0.30 + shutterRatio * 0.20),
        0.50,
        3.0);

    out.shouldReject =
        out.regime == AdaptiveMotionRegime::discontinuous &&
        out.continuityScore < 0.35;
    out.shouldPredict =
        !out.shouldReject &&
        out.predictionTrustScale > 0.12;

    return out;
}

double AdaptiveMotionRegimeModel::clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

double AdaptiveMotionRegimeModel::safeRatio(
    double numerator,
    double denominator) noexcept {
    if (!std::isfinite(numerator) ||
        !std::isfinite(denominator) ||
        denominator <= 0.0) {
        return 0.0;
    }
    return std::clamp(numerator / denominator, 0.0, 4.0);
}

} // namespace nox::native::tracking

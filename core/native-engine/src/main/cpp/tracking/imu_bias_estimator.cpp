#include "tracking/imu_bias_estimator.h"

#include <algorithm>
#include <cmath>

namespace nox::native::imu {

double ImuBiasEstimator::clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

double ImuBiasEstimator::robustWeight(
    double residual,
    double scale) noexcept {
    const double absolute =
        std::abs(
            residual);
    const double threshold =
        std::max(
            1e-6,
            scale * 2.5);

    if (
        absolute <=
        threshold) {
        return 1.0;
    }

    return threshold /
        std::max(
            absolute,
            threshold);
}

BiasEstimate ImuBiasEstimator::update(
    double gx,
    double gy,
    double gz,
    double ax,
    double ay,
    double az,
    double quality) noexcept {
    const double gyroMagnitude =
        std::hypot(
            std::hypot(gx, gy),
            gz);

    const double accelMagnitude =
        std::hypot(
            std::hypot(ax, ay),
            az);

    const double stationary =
        clamp01(
            std::exp(
                -gyroMagnitude *
                8.0) *
            std::exp(
                -std::abs(
                    accelMagnitude -
                    9.80665) *
                2.0));

    return updateStationary(
        gx,
        gy,
        gz,
        ax,
        ay,
        az,
        clamp01(
            quality) *
            stationary,
        1.0 /
            120.0);
}

BiasEstimate ImuBiasEstimator::updateStationary(
    double gx,
    double gy,
    double gz,
    double ax,
    double ay,
    double az,
    double stationaryConfidence,
    double dtSeconds) noexcept {
    const double stationary =
        clamp01(
            stationaryConfidence);

    const double dt =
        std::clamp(
            std::isfinite(dtSeconds)
                ? dtSeconds
                : 0.0,
            1e-4,
            0.25);

    const double gyroInnovation =
        std::hypot(
            std::hypot(
                gx - state_.gx,
                gy - state_.gy),
            gz - state_.gz);

    const double accelInnovation =
        std::hypot(
            std::hypot(
                ax - state_.ax,
                ay - state_.ay),
            az - state_.az);

    const double gyroWeight =
        stationary *
        robustWeight(
            gyroInnovation,
            0.10);

    const double accelWeight =
        stationary *
        robustWeight(
            accelInnovation,
            0.40);

    const double gyroAlpha =
        std::clamp(
            1.0 -
                std::exp(
                    -dt *
                    5.0 *
                    std::max(
                        gyroWeight,
                        0.0)),
            0.0001,
            0.10);

    const double accelAlpha =
        std::clamp(
            1.0 -
                std::exp(
                    -dt *
                    2.5 *
                    std::max(
                        accelWeight,
                        0.0)),
            0.0001,
            0.06);

    const double previousGx = state_.gx;
    const double previousGy = state_.gy;
    const double previousGz = state_.gz;

    state_.gx +=
        (gx - state_.gx) *
        gyroAlpha;
    state_.gy +=
        (gy - state_.gy) *
        gyroAlpha;
    state_.gz +=
        (gz - state_.gz) *
        gyroAlpha;

    state_.ax +=
        (ax - state_.ax) *
        accelAlpha;
    state_.ay +=
        (ay - state_.ay) *
        accelAlpha;
    state_.az +=
        (az - state_.az) *
        accelAlpha;

    const double gyroResidual =
        std::max(
            0.0,
            gyroInnovation);

    const double accelResidual =
        std::max(
            0.0,
            accelInnovation);

    state_.gyroVariance =
        0.96 *
            state_.gyroVariance +
        0.04 *
            gyroResidual *
            gyroResidual;

    state_.accelVariance =
        0.96 *
            state_.accelVariance +
        0.04 *
            accelResidual *
            accelResidual;

    state_.innovationRms =
        std::sqrt(
            std::max(
                0.0,
                0.5 *
                    (
                        state_.gyroVariance +
                        state_.accelVariance
                    )));

    state_.stationaryConfidence =
        clamp01(
            0.75 *
                state_.stationaryConfidence +
            0.25 *
                stationary);

    const double driftMagnitude =
        std::hypot(
            std::hypot(
                state_.gx -
                    previousGx,
                state_.gy -
                    previousGy),
            state_.gz -
                previousGz);

    const double stabilityPenalty =
        clamp01(
            driftMagnitude /
            0.50);

    state_.stability =
        clamp01(
            state_.stability *
                0.94 +
            stationary *
                0.04 +
            (
                1.0 -
                stabilityPenalty
            ) *
                0.02);

    state_.confidence =
        clamp01(
            state_.confidence *
                0.985 +
            stationary *
                0.012 +
            clamp01(
                1.0 -
                    state_.innovationRms /
                    2.0) *
                0.003);

    state_.sampleCount =
        std::min<std::uint64_t>(
            std::uint64_t{1'000'000'000ULL},
            state_.sampleCount +
                1ULL);

    return state_;
}

} // namespace nox::native::imu

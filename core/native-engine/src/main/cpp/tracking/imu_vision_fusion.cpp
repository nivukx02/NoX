#include "tracking/imu_vision_fusion.h"

#include <algorithm>
#include <cmath>

namespace nox::native::imu {

namespace {

double clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

} // namespace

void ImuVisionFusion::reset() noexcept {
    state_ = {};
    bias_ =
        ImuBiasEstimator{};
}

ImuState ImuVisionFusion::propagate(
    double gx,
    double gy,
    double gz,
    double ax,
    double ay,
    double az,
    double dt,
    const ImuPropagationOptions& options) noexcept {
    const double gyroMagnitude =
        std::hypot(
            std::hypot(
                gx,
                gy),
            gz);

    const double accelMagnitude =
        std::hypot(
            std::hypot(
                ax,
                ay),
            az);

    const double stationaryWeight =
        clamp01(
            std::exp(
                -gyroMagnitude *
                8.0) *
            std::exp(
                -std::abs(
                    accelMagnitude -
                    9.80665) *
                2.0));

    const auto bias =
        bias_.update(
            gx,
            gy,
            gz,
            ax,
            ay,
            az,
            stationaryWeight);

    const double correctedGx =
        gx -
        bias.gx;
    const double correctedGy =
        gy -
        bias.gy;
    const double correctedGz =
        gz -
        bias.gz;

    // Apply the estimated accelerometer bias before rotating into world space.
    // The bias estimator already tracks ax/ay/az, but ignoring those estimates
    // here leaks a persistent DC component directly into velocity/displacement.
    const double correctedAx =
        std::isfinite(ax - bias.ax)
            ? ax - bias.ax
            : 0.0;
    const double correctedAy =
        std::isfinite(ay - bias.ay)
            ? ay - bias.ay
            : 0.0;
    const double correctedAz =
        std::isfinite(az - bias.az)
            ? az - bias.az
            : 0.0;

    const double correctedAccelMagnitude =
        std::hypot(
            std::hypot(
                correctedAx,
                correctedAy),
            correctedAz);

    const double safeDt =
        std::clamp(
            std::isfinite(dt) ? dt : 0.0,
            0.0,
            0.100);

    const double midGx =
        hasPreviousSample_
            ? 0.5 * (
                previousGyroX_ +
                correctedGx)
            : correctedGx;
    const double midGy =
        hasPreviousSample_
            ? 0.5 * (
                previousGyroY_ +
                correctedGy)
            : correctedGy;
    const double midGz =
        hasPreviousSample_
            ? 0.5 * (
                previousGyroZ_ +
                correctedGz)
            : correctedGz;

    state_.orientation =
        integrateAngularVelocity(
            state_.orientation,
            midGx,
            midGy,
            midGz,
            safeDt);

    previousGyroX_ = correctedGx;
    previousGyroY_ = correctedGy;
    previousGyroZ_ = correctedGz;
    hasPreviousSample_ = true;

    state_.gravity_norm =
        accelMagnitude;

    state_.gyroResidual =
        std::hypot(
            std::hypot(
                correctedGx,
                correctedGy),
            correctedGz);

    state_.accelResidual =
        std::abs(
            correctedAccelMagnitude -
            options.gravityReference);

    state_.biasConfidence =
        clamp01(
            bias.confidence);

    state_.stationaryConfidence =
        clamp01(
            bias.stationaryConfidence);

    const double gyroQuality =
        std::exp(
            -state_.gyroResidual *
            0.60);

    const double accelQuality =
        std::exp(
            -state_.accelResidual *
            0.45);

    const double inertialQuality =
        clamp01(
            bias.stability *
            0.35 +
            state_.stationaryConfidence *
            0.30 +
            gyroQuality *
            0.20 +
            accelQuality *
            0.15);

    state_.inertialQuality =
        inertialQuality;

    const double qx = state_.orientation.x;
    const double qy = state_.orientation.y;
    const double qz = state_.orientation.z;
    const double qw = state_.orientation.w;

    // Rotate body-frame acceleration into world frame.
    const double tx =
        2.0 * (
            qy * correctedAz -
            qz * correctedAy);
    const double ty =
        2.0 * (
            qz * correctedAx -
            qx * correctedAz);
    const double tz =
        2.0 * (
            qx * correctedAy -
            qy * correctedAx);

    const double worldAx =
        correctedAx +
        qw * tx +
        (qy * tz - qz * ty);
    const double worldAy =
        correctedAy +
        qw * ty +
        (qz * tx - qx * tz);
    const double worldAz =
        correctedAz +
        qw * tz +
        (qx * ty - qy * tx);

    const double linearAx =
        worldAx;
    const double linearAy =
        worldAy;
    const double linearAz =
        worldAz -
        options.gravityReference;

    state_.worldAccelerationX =
        std::isfinite(linearAx)
            ? linearAx
            : 0.0;
    state_.worldAccelerationY =
        std::isfinite(linearAy)
            ? linearAy
            : 0.0;
    state_.worldAccelerationZ =
        std::isfinite(linearAz)
            ? linearAz
            : 0.0;

    if (safeDt > 0.0) {
        const double avgAx =
            hasPreviousSample_
                ? 0.5 * (
                    previousWorldAccelerationX_ +
                    state_.worldAccelerationX)
                : state_.worldAccelerationX;
        const double avgAy =
            hasPreviousSample_
                ? 0.5 * (
                    previousWorldAccelerationY_ +
                    state_.worldAccelerationY)
                : state_.worldAccelerationY;
        const double avgAz =
            hasPreviousSample_
                ? 0.5 * (
                    previousWorldAccelerationZ_ +
                    state_.worldAccelerationZ)
                : state_.worldAccelerationZ;

        state_.velocityX +=
            avgAx * safeDt;
        state_.velocityY +=
            avgAy * safeDt;
        state_.velocityZ +=
            avgAz * safeDt;

        const double speed =
            std::hypot(
                std::hypot(
                    state_.velocityX,
                    state_.velocityY),
                state_.velocityZ);

        if (
            std::isfinite(speed) &&
            speed >
                options.maximumVelocityMetersPerSecond &&
            options.maximumVelocityMetersPerSecond > 0.0) {
            const double scale =
                options.maximumVelocityMetersPerSecond /
                speed;
            state_.velocityX *= scale;
            state_.velocityY *= scale;
            state_.velocityZ *= scale;
        }

        state_.displacementX +=
            state_.velocityX * safeDt;
        state_.displacementY +=
            state_.velocityY * safeDt;
        state_.displacementZ +=
            state_.velocityZ * safeDt;
    }

    if (state_.stationaryConfidence > 0.65) {
        const double speed =
            std::hypot(
                std::hypot(
                    state_.velocityX,
                    state_.velocityY),
                state_.velocityZ);
        if (
            speed <
                options.stationaryVelocityThreshold &&
            options.stationaryVelocityThreshold > 0.0) {
            const double blend =
                std::clamp(
                    options.stationaryVelocityBlend,
                    0.0,
                    0.99);
            state_.velocityX *=
                1.0 - blend;
            state_.velocityY *=
                1.0 - blend;
            state_.velocityZ *=
                1.0 - blend;
        }

        const double stationaryQuality =
            clamp01(
                state_.stationaryConfidence *
                bias.confidence *
                std::exp(
                    -state_.accelResidual *
                    0.60));
        if (stationaryQuality > 0.72) {
            const double correctionGain =
                clamp01(
                    options.stationaryVelocityCorrectionGain) *
                stationaryQuality;
            state_.velocityX *=
                1.0 - correctionGain;
            state_.velocityY *=
                1.0 - correctionGain;
            state_.velocityZ *=
                1.0 - correctionGain;
        }
    }

    const double displacement =
        std::hypot(
            std::hypot(
                state_.displacementX,
                state_.displacementY),
            state_.displacementZ);
    if (
        std::isfinite(displacement) &&
        displacement >
            options.maximumDisplacementMeters &&
        options.maximumDisplacementMeters > 0.0) {
        const double scale =
            options.maximumDisplacementMeters /
            displacement;
        state_.displacementX *= scale;
        state_.displacementY *= scale;
        state_.displacementZ *= scale;
    }

    const double translationConfidence =
        clamp01(
            inertialQuality *
            std::exp(
                -displacement * 0.025) *
            (
                0.80 +
                0.20 * state_.stationaryConfidence
            ));

    state_.velocityConfidence =
        clamp01(
            translationConfidence *
            (0.70 + 0.30 * inertialQuality));

    state_.translationConfidence =
        clamp01(
            state_.translationConfidence *
                options.translationConfidenceDecay +
            translationConfidence *
                (1.0 -
                    options.translationConfidenceDecay));

    previousWorldAccelerationX_ =
        state_.worldAccelerationX;
    previousWorldAccelerationY_ =
        state_.worldAccelerationY;
    previousWorldAccelerationZ_ =
        state_.worldAccelerationZ;

    if (state_.confidence <= 0.0) {
        state_.confidence =
            inertialQuality;
    } else {
        state_.confidence =
            clamp01(
                state_.confidence *
                    0.90 +
                inertialQuality *
                    0.10);
    }

    return state_;
}

Quaternion ImuVisionFusion::integrateAngularVelocity(
    const Quaternion& orientation,
    double gx,
    double gy,
    double gz,
    double dt) noexcept {
    const Quaternion delta =
        fromAngularVelocity(
            gx,
            gy,
            gz,
            dt);

    return normalize(
        multiply(
            orientation,
            delta));
}

ImuState ImuVisionFusion::correct(
    const Quaternion& vision,
    double confidence) noexcept {
    const double correctionConfidence =
        clamp01(
            confidence);

    const double blend =
        0.12 +
        0.23 *
            correctionConfidence *
            state_.inertialQuality;

    state_.orientation =
        normalize({
            state_.orientation.w *
                    (1.0 - blend) +
                vision.w *
                    blend,
            state_.orientation.x *
                    (1.0 - blend) +
                vision.x *
                    blend,
            state_.orientation.y *
                    (1.0 - blend) +
                vision.y *
                    blend,
            state_.orientation.z *
                    (1.0 - blend) +
                vision.z *
                    blend,
        });

    const double correctionQuality =
        clamp01(
            correctionConfidence *
            (
                0.50 +
                0.50 *
                    state_.inertialQuality
            ));

    state_.confidence =
        clamp01(
            state_.confidence *
                0.70 +
            correctionQuality *
                0.30);

    return state_;
}

} // namespace nox::native::imu

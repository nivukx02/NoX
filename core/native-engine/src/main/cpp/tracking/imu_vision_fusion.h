#pragma once
#include "geometry3d/imu_state.h"
#include "tracking/imu_bias_estimator.h"

namespace nox::native::imu {

struct ImuPropagationOptions {
    double gravityReference{9.80665};
    double maximumVelocityMetersPerSecond{6.0};
    double maximumDisplacementMeters{40.0};
    double stationaryVelocityThreshold{0.08};
    double stationaryVelocityBlend{0.82};
    // High-confidence stationary updates get a bounded ZUPT-like velocity pull
    // without assuming that every low-speed frame is actually stationary.
    double stationaryVelocityCorrectionGain{0.62};
    double translationConfidenceDecay{0.998};
};

class ImuVisionFusion final {
public:
    void reset() noexcept;

    ImuState propagate(
        double gx,
        double gy,
        double gz,
        double ax,
        double ay,
        double az,
        double dt,
        const ImuPropagationOptions& options = {}) noexcept;

    ImuState correct(
        const Quaternion&,
        double) noexcept;

    BiasEstimate biases() const noexcept {
        return bias_.current();
    }

private:
    ImuState state_{};
    ImuBiasEstimator bias_{};

    double previousGyroX_{0.0};
    double previousGyroY_{0.0};
    double previousGyroZ_{0.0};
    double previousWorldAccelerationX_{0.0};
    double previousWorldAccelerationY_{0.0};
    double previousWorldAccelerationZ_{0.0};
    bool hasPreviousSample_{false};

    static Quaternion integrateAngularVelocity(
        const Quaternion& orientation,
        double gx,
        double gy,
        double gz,
        double dt) noexcept;
};

} // namespace nox::native::imu
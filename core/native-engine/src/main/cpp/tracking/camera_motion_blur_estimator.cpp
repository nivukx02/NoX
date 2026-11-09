#include "tracking/camera_motion_blur_estimator.h"

#include <algorithm>
#include <cmath>

namespace nox::native::tracking {

namespace {

constexpr double kLowBlurRadians = 0.008726646259971648;   // 0.50 deg
constexpr double kHighBlurRadians = 0.026179938779914945; // 1.50 deg

double clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

} // namespace

CameraMotionBlurEstimate CameraMotionBlurEstimator::estimate(
    std::uint64_t exposureTimeNanos,
    double angularVelocityRadiansPerSecond) const noexcept {
    if (
        exposureTimeNanos == 0ULL ||
        !std::isfinite(angularVelocityRadiansPerSecond)
    ) {
        return {};
    }

    const double exposureSeconds =
        static_cast<double>(exposureTimeNanos) * 1e-9;
    const double blurRadians =
        std::abs(angularVelocityRadiansPerSecond) *
        exposureSeconds;

    const double penalty =
        blurRadians <= kLowBlurRadians
            ? 0.0
            : clamp01(
                (blurRadians - kLowBlurRadians) /
                std::max(
                    kHighBlurRadians - kLowBlurRadians,
                    1e-9));

    const double confidence =
        clamp01(
            1.0 -
            std::abs(
                angularVelocityRadiansPerSecond) /
            20.0);

    return {
        blurRadians,
        penalty,
        confidence,
    };
}

} // namespace nox::native::tracking

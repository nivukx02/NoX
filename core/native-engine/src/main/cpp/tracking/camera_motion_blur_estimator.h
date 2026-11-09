#pragma once

#include <cstdint>

namespace nox::native::tracking {

struct CameraMotionBlurEstimate {
    double estimated_blur_radians{0.0};
    double penalty{0.0};
    double confidence{0.0};
};

class CameraMotionBlurEstimator final {
public:
    [[nodiscard]] CameraMotionBlurEstimate estimate(
        std::uint64_t exposureTimeNanos,
        double angularVelocityRadiansPerSecond) const noexcept;
};

} // namespace nox::native::tracking

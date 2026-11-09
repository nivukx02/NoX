#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "math/se3.h"

namespace nox::native::calibration {

struct RigPoseSample {
    std::uint64_t timestampNanos{0};
    std::size_t cameraId{0};
    math::SE3d worldToCamera{};
    double confidence{1.0};
};

struct RigStabilityConfig {
    double maximumTranslationDriftMeters{0.001};
    double maximumRotationDriftRadians{0.0015};
    double degradedTranslationDriftMeters{0.0004};
    double degradedRotationDriftRadians{0.0007};
    std::size_t minimumSamplesPerCamera{3};
    double minimumConfidence{0.60};
};

struct RigStabilityResult {
    std::size_t cameraCount{0};
    std::size_t validCameraCount{0};
    double maximumTranslationDriftMeters{0.0};
    double p95TranslationDriftMeters{0.0};
    double maximumRotationDriftRadians{0.0};
    double p95RotationDriftRadians{0.0};
    double confidence{0.0};
    bool stable{false};
    bool degraded{false};
    bool unstable{false};
};

class RigStabilityMonitor final {
public:
    explicit RigStabilityMonitor(
        RigStabilityConfig config = {}) noexcept;

    [[nodiscard]] RigStabilityResult evaluate(
        const std::vector<RigPoseSample>& samples) const;

private:
    RigStabilityConfig config_{};

    static double rotationDelta(
        const math::SE3d& reference,
        const math::SE3d& current) noexcept;

    static double translationDelta(
        const math::SE3d& reference,
        const math::SE3d& current) noexcept;

    static double percentile95(
        std::vector<double>) noexcept;
};

} // namespace nox::native::calibration

#pragma once

#include <cstdint>

#include "eyes/gaze_fusion.h"

namespace nox::native::eyes {

struct TemporalGazeConfig {
    double baseSmoothing{0.28};
    double fastMotionSmoothing{0.92};
    double maximumPredictionSeconds{0.08};
    double continuityAngleScaleDegrees{8.0};
};

class TemporalGazeStabilizer final {
public:
    explicit TemporalGazeStabilizer(
        TemporalGazeConfig config = {}) noexcept;

    [[nodiscard]] GazeState update(
        std::uint64_t timestampNanos,
        const EyeGeometry& left,
        const EyeGeometry& right,
        const Vec3& headForward,
        const Vec3& worldUp) noexcept;

    void reset() noexcept;

    [[nodiscard]] double angularVelocityDegreesPerSecond() const noexcept {
        return angularVelocityDegreesPerSecond_;
    }

private:
    TemporalGazeConfig config_{};
    GazeState state_{};
    std::uint64_t timestampNanos_{0};
    double angularVelocityDegreesPerSecond_{0.0};
    Vec3 angularAxis_{0.0,0.0,1.0};
    double predictionConfidence_{0.0};
    bool initialized_{false};

    static double clamp01(double value) noexcept;
    static Vec3 normalize(const Vec3&) noexcept;
    static double dot(const Vec3&, const Vec3&) noexcept;
    static double angleDegrees(const Vec3&, const Vec3&) noexcept;
    static Vec3 blend(
        const Vec3&,
        const Vec3&,
        double) noexcept;
    static Vec3 cross(
        const Vec3&,
        const Vec3&) noexcept;
    static Vec3 rotateAroundAxis(
        const Vec3& direction,
        const Vec3& axis,
        double angleRadians) noexcept;
};

} // namespace nox::native::eyes

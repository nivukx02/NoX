#pragma once
#include <cstdint>
#include "geometry3d/pose_timeline.h"
#include "math/se3.h"

namespace nox::native::capture {

struct RollingShutterPoseResult {
    math::Vec3d correctedPoint{};
    math::Twist6d rowDelta{};
    double normalizedRow{0.5};
    double rowOffsetSeconds{0.0};
    double translationCorrectionMeters{0.0};
    double rotationCorrectionRadians{0.0};
    double confidence{0.0};
    double timelineConfidence{0.0};
    bool compensated{false};
};

class RollingShutterPoseCompensator final {
public:
    [[nodiscard]] math::Vec3d compensatePoint(
        const math::Vec3d& pointInRowCamera,
        std::uint64_t frameTimestampNanos,
        double normalizedRow,
        std::uint64_t readoutTimeNanos,
        const geometry3d::PoseTimeline& timeline) const;

    [[nodiscard]] RollingShutterPoseResult compensatePointHighEnd(
        const math::Vec3d& pointInRowCamera,
        std::uint64_t frameTimestampNanos,
        double normalizedRow,
        std::uint64_t readoutTimeNanos,
        const geometry3d::PoseTimeline& timeline) const;
};

} // namespace nox::native::capture

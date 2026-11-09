#include "capture/rolling_shutter_pose_compensator.h"

#include <algorithm>
#include <cmath>

namespace nox::native::capture {
namespace {

double norm(
    const math::Vec3d& value) noexcept {
    return std::hypot(
        std::hypot(
            value.x,
            value.y),
        value.z);
}

double clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

} // namespace

math::Vec3d
RollingShutterPoseCompensator::compensatePoint(
    const math::Vec3d& pointInRowCamera,
    std::uint64_t frameTimestampNanos,
    double normalizedRow,
    std::uint64_t readoutTimeNanos,
    const geometry3d::PoseTimeline& timeline) const {
    return compensatePointHighEnd(
        pointInRowCamera,
        frameTimestampNanos,
        normalizedRow,
        readoutTimeNanos,
        timeline).correctedPoint;
}

RollingShutterPoseResult
RollingShutterPoseCompensator::compensatePointHighEnd(
    const math::Vec3d& pointInRowCamera,
    std::uint64_t frameTimestampNanos,
    double normalizedRow,
    std::uint64_t readoutTimeNanos,
    const geometry3d::PoseTimeline& timeline) const {
    RollingShutterPoseResult result;

    result.normalizedRow =
        std::clamp(
            std::isfinite(
                normalizedRow)
                ? normalizedRow
                : 0.5,
            0.0,
            1.0);

    result.rowOffsetSeconds =
        (
            result.normalizedRow -
            0.5
        ) *
        static_cast<double>(
            readoutTimeNanos) /
        1e9;

    const auto rowTimeline =
        timeline.sampleRowHighEnd(
            frameTimestampNanos,
            result.normalizedRow,
            readoutTimeNanos);

    const auto centerTimeline =
        timeline.sampleHighEnd(
            frameTimestampNanos);

    const auto& rowPose =
        rowTimeline.pose;

    const auto& centerPose =
        centerTimeline.pose;

    result.timelineConfidence =
        clamp01(
            rowTimeline.confidence *
            0.70 +
            centerTimeline.confidence *
            0.30);

    const auto worldPoint =
        rowPose.transform(
            pointInRowCamera);

    result.correctedPoint =
        centerPose.inverse().transform(
            worldPoint);

    const auto relative =
        centerPose.inverse().compose(
            rowPose);

    result.rowDelta =
        relative.log();

    result.translationCorrectionMeters =
        norm(
            relative.translation);

    result.rotationCorrectionRadians =
        norm(
            result.rowDelta.rotation);

    const double timingMagnitude =
        std::abs(
            result.rowOffsetSeconds);

    const double translationScore =
        std::exp(
            -result.translationCorrectionMeters /
            0.02);

    const double rotationScore =
        std::exp(
            -result.rotationCorrectionRadians /
            0.05);

    const double timingScore =
        std::exp(
            -timingMagnitude /
            0.005);

    result.confidence =
        clamp01(
            (
                translationScore *
                0.30 +
                rotationScore *
                0.30 +
                timingScore *
                0.15 +
                result.timelineConfidence *
                0.25
            ) *
            (
                rowTimeline.clamped ||
                centerTimeline.clamped
                    ? 0.55
                    : 1.0
            ));

    result.compensated =
        timingMagnitude >
            1e-9 ||
        result.translationCorrectionMeters >
            1e-9 ||
        result.rotationCorrectionRadians >
            1e-9;

    return result;
}

} // namespace nox::native::capture

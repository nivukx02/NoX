#include "geometry3d/pose_timeline.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nox::native::geometry3d {

void PoseTimeline::clear() {
    samples_.clear();
}

bool PoseTimeline::addSample(
    const PoseSample& sample) {
    const auto position =
        std::lower_bound(
            samples_.begin(),
            samples_.end(),
            sample.timestampNanos,
            [](const PoseSample& left,
               std::uint64_t timestamp) {
                return
                    left.timestampNanos <
                    timestamp;
            });

    if (
        position !=
            samples_.end() &&
        position->timestampNanos ==
            sample.timestampNanos) {
        position->pose =
            sample.pose;
        return true;
    }

    samples_.insert(
        position,
        sample);
    return true;
}

double PoseTimeline::translationDistance(
    const math::SE3d& a,
    const math::SE3d& b) noexcept {
    const auto delta =
        b.translation -
        a.translation;

    return std::hypot(
        std::hypot(
            delta.x,
            delta.y),
        delta.z);
}

double PoseTimeline::angularDistance(
    const math::SE3d& a,
    const math::SE3d& b) noexcept {
    const auto relative =
        math::Quaterniond::multiply(
            a.rotation.conjugate(),
            b.rotation);

    return
        std::hypot(
            std::hypot(
                relative.x,
                relative.y),
            relative.z) *
        2.0;
}

PoseTimelineResult PoseTimeline::sampleHighEnd(
    std::uint64_t timestampNanos,
    std::uint64_t maxGap) const {
    PoseTimelineResult result;

    if (
        samples_.empty()) {
        result.clamped = true;
        return result;
    }

    const auto interpolate =
        [&](const PoseSample& left,
            const PoseSample& right,
            double alpha,
            bool extrapolated,
            std::uint64_t temporalGap) {
            const double safeAlpha =
                std::isfinite(alpha)
                    ? alpha
                    : 0.0;

            result.pose = {
                math::Quaterniond::slerp(
                    left.pose.rotation,
                    right.pose.rotation,
                    safeAlpha),
                {
                    left.pose.translation.x +
                        (
                            right.pose.translation.x -
                            left.pose.translation.x
                        ) *
                        safeAlpha,
                    left.pose.translation.y +
                        (
                            right.pose.translation.y -
                            left.pose.translation.y
                        ) *
                        safeAlpha,
                    left.pose.translation.z +
                        (
                            right.pose.translation.z -
                            left.pose.translation.z
                        ) *
                        safeAlpha,
                },
            };

            result.temporalGapNanos =
                temporalGap;
            result.interpolationAlpha =
                safeAlpha;
            result.interpolated =
                !extrapolated;
            result.extrapolated =
                extrapolated;

            const std::uint64_t span =
                right.timestampNanos -
                left.timestampNanos;

            if (
                span > 0ULL) {
                const double seconds =
                    static_cast<double>(
                        span) /
                    1e9;

                result.linearSpeedMetersPerSecond =
                    translationDistance(
                        left.pose,
                        right.pose) /
                    seconds;

                result.angularSpeedRadiansPerSecond =
                    angularDistance(
                        left.pose,
                        right.pose) /
                    seconds;
            }

            const double gapSeconds =
                static_cast<double>(
                    temporalGap) /
                1e9;

            const double gapScore =
                std::exp(
                    -gapSeconds /
                    std::max(
                        1e-3,
                        static_cast<double>(
                            maxGap) /
                            1e9));

            const double extrapolationPenalty =
                extrapolated
                    ? 0.65
                    : 1.0;

            result.confidence =
                std::clamp(
                    gapScore *
                        extrapolationPenalty,
                    0.0,
                    1.0);
        };

    if (
        samples_.size() == 1U) {
        result.pose =
            samples_.front().pose;

        result.temporalGapNanos =
            timestampNanos >=
                samples_.front().timestampNanos
                ? timestampNanos -
                    samples_.front().timestampNanos
                : samples_.front().timestampNanos -
                    timestampNanos;

        result.confidence =
            std::exp(
                -static_cast<double>(
                    result.temporalGapNanos) /
                std::max(
                    1.0,
                    static_cast<double>(
                        maxGap)));

        result.clamped =
            result.temporalGapNanos >
            maxGap;

        if (result.clamped) {
            result.confidence *= 0.25;
        }

        return result;
    }

    if (
        timestampNanos <=
        samples_.front().timestampNanos) {
        const auto& left =
            samples_[0];
        const auto& right =
            samples_[1];

        const std::uint64_t delta =
            left.timestampNanos -
            timestampNanos;

        if (
            delta >
            maxGap) {
            result.pose =
                left.pose;
            result.temporalGapNanos =
                delta;
            result.clamped = true;
            result.confidence = 0.15;
            return result;
        }

        const double alpha =
            (
                static_cast<double>(
                    timestampNanos) -
                static_cast<double>(
                    left.timestampNanos)
            ) /
            static_cast<double>(
                right.timestampNanos -
                left.timestampNanos);

        interpolate(
            left,
            right,
            alpha,
            true,
            delta);

        result.confidence *= 0.80;
        return result;
    }

    if (
        timestampNanos >=
        samples_.back().timestampNanos) {
        const auto& left =
            samples_[samples_.size() - 2U];
        const auto& right =
            samples_.back();

        const std::uint64_t delta =
            timestampNanos -
            right.timestampNanos;

        if (
            delta >
            maxGap) {
            result.pose =
                right.pose;
            result.temporalGapNanos =
                delta;
            result.clamped = true;
            result.confidence = 0.15;
            return result;
        }

        const double alpha =
            static_cast<double>(
                timestampNanos -
                left.timestampNanos) /
            static_cast<double>(
                right.timestampNanos -
                left.timestampNanos);

        interpolate(
            left,
            right,
            alpha,
            true,
            delta);

        result.confidence *= 0.80;
        return result;
    }

    const auto right =
        std::lower_bound(
            samples_.begin(),
            samples_.end(),
            timestampNanos,
            [](const PoseSample& left,
               std::uint64_t timestamp) {
                return
                    left.timestampNanos <
                    timestamp;
            });

    if (
        right ==
        samples_.begin() ||
        right ==
        samples_.end()) {
        result.pose =
            right ==
                samples_.end()
                ? samples_.back().pose
                : samples_.front().pose;
        result.clamped = true;
        result.confidence = 0.20;
        return result;
    }

    const auto left =
        right - 1;

    const std::uint64_t span =
        right->timestampNanos -
        left->timestampNanos;

    if (
        span == 0ULL ||
        span >
            maxGap) {
        result.pose =
            left->pose;
        result.temporalGapNanos =
            std::min(
                timestampNanos -
                    left->timestampNanos,
                right->timestampNanos -
                    timestampNanos);
        result.clamped = true;
        result.confidence = 0.20;
        return result;
    }

    const double alpha =
        static_cast<double>(
            timestampNanos -
            left->timestampNanos) /
        static_cast<double>(
            span);

    const std::uint64_t temporalGap =
        std::min(
            timestampNanos -
                left->timestampNanos,
            right->timestampNanos -
                timestampNanos);

    interpolate(
        *left,
        *right,
        alpha,
        false,
        temporalGap);

    return result;
}

math::SE3d PoseTimeline::sample(
    std::uint64_t timestampNanos,
    std::uint64_t maxGap) const {
    return sampleHighEnd(
        timestampNanos,
        maxGap).pose;
}

PoseTimelineResult PoseTimeline::sampleRowHighEnd(
    std::uint64_t frameTimestampNanos,
    double normalizedRow,
    std::uint64_t readoutTimeNanos,
    std::uint64_t maxGap) const {
    const double row =
        std::clamp(
            std::isfinite(
                normalizedRow)
                ? normalizedRow
                : 0.5,
            0.0,
            1.0);

    const double offsetNanos =
        (
            row -
            0.5
        ) *
        static_cast<double>(
            readoutTimeNanos);

    std::int64_t signedTimestamp =
        static_cast<std::int64_t>(
            frameTimestampNanos);

    if (
        offsetNanos >=
        0.0) {
        signedTimestamp +=
            static_cast<std::int64_t>(
                offsetNanos);
    } else {
        signedTimestamp -=
            static_cast<std::int64_t>(
                -offsetNanos);
    }

    if (
        signedTimestamp < 0) {
        signedTimestamp = 0;
    }

    return sampleHighEnd(
        static_cast<std::uint64_t>(
            signedTimestamp),
        maxGap);
}

math::SE3d PoseTimeline::sampleRow(
    std::uint64_t frameTimestampNanos,
    double normalizedRow,
    std::uint64_t readoutTimeNanos,
    std::uint64_t maxGap) const {
    return sampleRowHighEnd(
        frameTimestampNanos,
        normalizedRow,
        readoutTimeNanos,
        maxGap).pose;
}

} // namespace nox::native::geometry3d

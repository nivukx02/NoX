#include "core/observation/motion_observation.h"

#include <algorithm>
#include <cmath>

namespace nox::native {

MotionObservation MotionObservation::metadata(
    std::uint32_t camera_id,
    std::uint64_t frame_id,
    std::uint64_t sensor_timestamp_nanos,
    std::uint64_t monotonic_timestamp_nanos,
    std::uint64_t exposure_duration_nanos,
    std::uint64_t frame_interval_nanos) noexcept {
    MotionObservation observation;
    observation.camera_id =
        camera_id;
    observation.frame_id =
        frame_id;
    observation.sensor_timestamp_nanos =
        sensor_timestamp_nanos;
    observation.monotonic_timestamp_nanos =
        monotonic_timestamp_nanos;
    observation.exposure_duration_nanos =
        exposure_duration_nanos;
    observation.frame_interval_nanos =
        frame_interval_nanos;
    return observation;
}

bool MotionObservation::validate(
    std::uint32_t maximumKeypoints) const noexcept {
    if (
        keypoint_count == 0U ||
        keypoint_count >
        maximumKeypoints ||
        keypoint_count >
        kMaxObservationKeypoints) {
        return false;
    }

    if (
        sensor_timestamp_nanos == 0ULL ||
        monotonic_timestamp_nanos == 0ULL) {
        return false;
    }

    if (
        exposure_duration_nanos >
        1'000'000'000ULL ||
        rolling_shutter_skew_nanos >
        1'000'000'000ULL ||
        frame_interval_nanos >
        1'000'000'000ULL) {
        return false;
    }

    for (
        std::uint32_t index = 0;
        index < keypoint_count;
        ++index) {
        const auto& keypoint =
            keypoints[index];

        if (
            !std::isfinite(
                static_cast<double>(
                    keypoint.x)) ||
            !std::isfinite(
                static_cast<double>(
                    keypoint.y)) ||
            !std::isfinite(
                static_cast<double>(
                    keypoint.confidence)) ||
            keypoint.confidence <
                0.0f ||
            keypoint.confidence >
                1.0f) {
            return false;
        }

        if (
            index > 0U &&
            keypoint_ids[index] ==
                keypoint_ids[index - 1U]) {
            return false;
        }
    }

    return true;
}

bool MotionObservation::sanitize() noexcept {
    if (
        keypoint_count >
        kMaxObservationKeypoints) {
        keypoint_count =
            static_cast<std::uint32_t>(
                kMaxObservationKeypoints);
    }

    bool changed = false;

    for (
        std::uint32_t index = 0;
        index < keypoint_count;
        ++index) {
        auto& keypoint =
            keypoints[index];

        if (!std::isfinite(
                static_cast<double>(
                    keypoint.x))) {
            keypoint.x =
                0.0f;
            changed = true;
        }

        if (!std::isfinite(
                static_cast<double>(
                    keypoint.y))) {
            keypoint.y =
                0.0f;
            changed = true;
        }

        if (!std::isfinite(
                static_cast<double>(
                    keypoint.confidence))) {
            keypoint.confidence =
                0.0f;
            changed = true;
        }

        const float clampedConfidence =
            std::clamp(
                keypoint.confidence,
                0.0f,
                1.0f);

        if (
            clampedConfidence !=
            keypoint.confidence) {
            keypoint.confidence =
                clampedConfidence;
            changed = true;
        }
    }

    if (
        sensor_timestamp_nanos == 0ULL) {
        sensor_timestamp_nanos =
            monotonic_timestamp_nanos;
        changed = true;
    }

    if (
        monotonic_timestamp_nanos == 0ULL) {
        monotonic_timestamp_nanos =
            sensor_timestamp_nanos;
        changed = true;
    }

    return changed;
}

} // namespace nox::native

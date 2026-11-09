#include "capture/capture_gap_tracker.h"

#include <algorithm>

namespace nox::native {

CaptureGapTracker::CaptureGapTracker(std::uint64_t expected_interval_nanos) noexcept
    : expected_interval_nanos_(std::max<std::uint64_t>(expected_interval_nanos, 1u)) {}

GapAssessment CaptureGapTracker::observe(const MotionObservation& observation) noexcept {
    GapAssessment result{false, false, false, 0, epoch_};
    if (!initialized_) {
        initialized_ = true;
        last_frame_id_ = observation.frame_id;
        last_sensor_timestamp_ = observation.sensor_timestamp_nanos;
        ++epoch_;
        result.epoch = epoch_;
        return result;
    }
    if (observation.frame_id == last_frame_id_ ||
        observation.sensor_timestamp_nanos == last_sensor_timestamp_) {
        result.duplicate = true;
        result.epoch = epoch_;
        return result;
    }
    if (observation.frame_id < last_frame_id_ ||
        observation.sensor_timestamp_nanos < last_sensor_timestamp_) {
        result.out_of_order = true;
        ++epoch_;
        result.epoch = epoch_;
        last_frame_id_ = observation.frame_id;
        last_sensor_timestamp_ = observation.sensor_timestamp_nanos;
        return result;
    }
    const auto delta =
        observation.sensor_timestamp_nanos -
            last_sensor_timestamp_;
    const auto gap_threshold =
        expected_interval_nanos_ +
            expected_interval_nanos_ / 2u;
    if (delta > gap_threshold) {
        result.gap = true;
        result.gap_nanos = delta - expected_interval_nanos_;
    }
    last_frame_id_ = observation.frame_id;
    last_sensor_timestamp_ = observation.sensor_timestamp_nanos;
    result.epoch = epoch_;
    return result;
}

void CaptureGapTracker::set_expected_interval_nanos(
    std::uint64_t expected_interval_nanos) noexcept {
    expected_interval_nanos_ =
        std::clamp(
            expected_interval_nanos,
            1'000'000ULL,
            1'000'000'000ULL);
}

void CaptureGapTracker::reset() noexcept {
    initialized_ = false;
    last_frame_id_ = 0;
    last_sensor_timestamp_ = 0;
    ++epoch_;
}

} // namespace nox::native

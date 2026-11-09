#include "capture/native_capture_ingestor.h"

#include <algorithm>

namespace nox::native {

NativeCaptureIngestor::NativeCaptureIngestor(
    std::uint32_t capacity,
    std::uint64_t expected_interval_nanos,
    std::uint32_t camera_id) noexcept
    : ring_(std::max<std::uint32_t>(capacity, 2u)),
      camera_id_(camera_id),
      gap_tracker_(expected_interval_nanos) {}

IngestionResult NativeCaptureIngestor::push(const MotionObservation& observation) noexcept {
    std::scoped_lock lock(mutex_);
    MotionObservation normalized = observation;
    normalized.camera_id = camera_id_;
    const auto timing = timestamp_normalizer_.normalize(
        observation.sensor_timestamp_nanos,
        observation.monotonic_timestamp_nanos);
    normalized.monotonic_timestamp_nanos = timing.monotonic_nanos;

    if (normalized.frame_interval_nanos > 0ULL) {
        gap_tracker_.set_expected_interval_nanos(
            normalized.frame_interval_nanos);
    }

    const auto assessment = gap_tracker_.observe(normalized);
    IngestionResult result{
        false,
        assessment.duplicate,
        assessment.out_of_order,
        assessment.gap,
        false,
        assessment.gap_nanos,
        static_cast<std::uint32_t>(size_),
        assessment.epoch,
    };

    if (assessment.duplicate || assessment.out_of_order) {
        return result;
    }

    if(size_ == ring_.size()) {
        head_ = (head_ + 1) % ring_.size();
        --size_;
        result.overflow = true;
    }

    ring_[tail_] = normalized;
    tail_ = (tail_ + 1) % ring_.size();
    ++size_;
    result.accepted = true;
    result.queue_depth = static_cast<std::uint32_t>(size_);
    return result;
}

bool NativeCaptureIngestor::pop(MotionObservation& observation) noexcept {
    std::scoped_lock lock(mutex_);
    if (size_ == 0) return false;
    observation = ring_[head_];
    head_ = (head_ + 1) % ring_.size();
    --size_;
    return true;
}

void NativeCaptureIngestor::reset() noexcept {
    std::scoped_lock lock(mutex_);
    head_ = 0;
    tail_ = 0;
    size_ = 0;
    gap_tracker_.reset();
    timestamp_normalizer_.reset();
}

std::uint32_t NativeCaptureIngestor::size() const noexcept {
    std::scoped_lock lock(mutex_);
    return static_cast<std::uint32_t>(size_);
}

std::uint32_t NativeCaptureIngestor::capacity() const noexcept {
    return static_cast<std::uint32_t>(ring_.size());
}

} // namespace nox::native

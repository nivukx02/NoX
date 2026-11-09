#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "capture/capture_gap_tracker.h"
#include "capture/timestamp_normalizer.h"
#include "core/observation/motion_observation.h"

namespace nox::native {

struct IngestionResult {
    bool accepted{false};
    bool duplicate{false};
    bool out_of_order{false};
    bool gap{false};
    bool overflow{false};
    std::uint64_t gap_nanos{0};
    std::uint32_t queue_depth{0};
    std::uint64_t epoch{0};
};

class NativeCaptureIngestor final {
public:
    NativeCaptureIngestor(
        std::uint32_t capacity,
        std::uint64_t expected_interval_nanos,
        std::uint32_t camera_id) noexcept;

    IngestionResult push(const MotionObservation& observation) noexcept;
    bool pop(MotionObservation& observation) noexcept;
    void reset() noexcept;
    [[nodiscard]] std::uint32_t size() const noexcept;
    [[nodiscard]] std::uint32_t capacity() const noexcept;

private:
    std::vector<MotionObservation> ring_;
    std::size_t head_{0};
    std::size_t tail_{0};
    std::size_t size_{0};
    std::uint32_t camera_id_{0};
    mutable std::mutex mutex_;
    CaptureGapTracker gap_tracker_;
    TimestampNormalizer timestamp_normalizer_;
};

} // namespace nox::native

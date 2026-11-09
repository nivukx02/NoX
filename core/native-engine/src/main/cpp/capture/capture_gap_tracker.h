#pragma once

#include <cstdint>

#include "core/observation/motion_observation.h"

namespace nox::native {

struct GapAssessment {
    bool duplicate{false};
    bool out_of_order{false};
    bool gap{false};
    std::uint64_t gap_nanos{0};
    std::uint64_t epoch{0};
};

class CaptureGapTracker final {
public:
    explicit CaptureGapTracker(std::uint64_t expected_interval_nanos) noexcept;
    GapAssessment observe(const MotionObservation& observation) noexcept;
    void set_expected_interval_nanos(
        std::uint64_t expected_interval_nanos) noexcept;
    void reset() noexcept;

private:
    std::uint64_t expected_interval_nanos_;
    std::uint64_t epoch_{0};
    std::uint64_t last_frame_id_{0};
    std::uint64_t last_sensor_timestamp_{0};
    bool initialized_{false};
};

} // namespace nox::native

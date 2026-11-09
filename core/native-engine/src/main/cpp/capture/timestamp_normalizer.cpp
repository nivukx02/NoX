#include "capture/timestamp_normalizer.h"

namespace nox::native {

NormalizedTimestamp TimestampNormalizer::normalize(
    std::uint64_t sensor_nanos,
    std::uint64_t monotonic_nanos) noexcept {
    if (!initialized_) {
        initialized_ = true;
        ++epoch_;
        previous_sensor_nanos_ = sensor_nanos;
        previous_monotonic_nanos_ = monotonic_nanos;
        offset_nanos_ = static_cast<std::int64_t>(monotonic_nanos) -
            static_cast<std::int64_t>(sensor_nanos);
    } else {
        const bool sensor_regressed = sensor_nanos < previous_sensor_nanos_;
        const bool monotonic_regressed = monotonic_nanos < previous_monotonic_nanos_;
        if (sensor_regressed || monotonic_regressed) {
            ++epoch_;
            offset_nanos_ = static_cast<std::int64_t>(monotonic_nanos) -
                static_cast<std::int64_t>(sensor_nanos);
        } else {
            const auto observed_offset =
                static_cast<std::int64_t>(monotonic_nanos) -
                static_cast<std::int64_t>(sensor_nanos);
            offset_nanos_ = (offset_nanos_ * 7 + observed_offset) / 8;
        }
        previous_sensor_nanos_ = sensor_nanos;
        previous_monotonic_nanos_ = monotonic_nanos;
    }

    return NormalizedTimestamp{
        monotonic_nanos,
        sensor_nanos,
        offset_nanos_,
        epoch_,
        true,
    };
}

void TimestampNormalizer::reset() noexcept {
    initialized_ = false;
    ++epoch_;
    previous_sensor_nanos_ = 0;
    previous_monotonic_nanos_ = 0;
    offset_nanos_ = 0;
}

} // namespace nox::native

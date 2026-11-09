#pragma once

#include <cstdint>

namespace nox::native {

struct NormalizedTimestamp {
    std::uint64_t monotonic_nanos{0};
    std::uint64_t sensor_nanos{0};
    std::int64_t sensor_to_monotonic_offset_nanos{0};
    std::uint64_t epoch{0};
    bool initialized{false};
};

class TimestampNormalizer final {
public:
    NormalizedTimestamp normalize(
        std::uint64_t sensor_nanos,
        std::uint64_t monotonic_nanos) noexcept;
    void reset() noexcept;

private:
    bool initialized_{false};
    std::uint64_t epoch_{0};
    std::uint64_t previous_sensor_nanos_{0};
    std::uint64_t previous_monotonic_nanos_{0};
    std::int64_t offset_nanos_{0};
};

} // namespace nox::native

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace nox::native {

inline constexpr std::size_t kMaxObservationKeypoints = 128;

struct Keypoint2D {
    float x{0.0f};
    float y{0.0f};
    float confidence{0.0f};
};

struct MotionObservation {
    std::uint32_t camera_id{0};
    std::uint64_t frame_id{0};
    std::uint64_t sensor_timestamp_nanos{0};
    std::uint64_t monotonic_timestamp_nanos{0};
    std::uint64_t exposure_start_nanos{0};
    std::uint64_t exposure_duration_nanos{0};
    std::uint64_t rolling_shutter_skew_nanos{0};
    std::uint64_t frame_interval_nanos{0};
    std::uint64_t ingest_received_monotonic_nanos{0};
    std::uint32_t keypoint_count{0};
    std::array<std::uint32_t, kMaxObservationKeypoints> keypoint_ids{};
    std::array<Keypoint2D, kMaxObservationKeypoints> keypoints{};

    static MotionObservation metadata(
        std::uint32_t camera_id,
        std::uint64_t frame_id,
        std::uint64_t sensor_timestamp_nanos,
        std::uint64_t monotonic_timestamp_nanos,
        std::uint64_t exposure_duration_nanos,
        std::uint64_t frame_interval_nanos) noexcept;

    [[nodiscard]] bool validate(
        std::uint32_t maximumKeypoints = kMaxObservationKeypoints) const noexcept;

    [[nodiscard]] bool sanitize() noexcept;
};

} // namespace nox::native

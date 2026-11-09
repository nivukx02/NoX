#pragma once

#include <cstdint>

#if defined(_WIN32)
#define NOX_EXPORT __declspec(dllexport)
#else
#define NOX_EXPORT __attribute__((visibility("default")))
#endif

namespace nox::native {

inline constexpr std::uint32_t kNativeEngineAbiVersion = 12u;
inline constexpr std::uint32_t kDefaultWorkerCount = 2u;
inline constexpr std::uint32_t kDefaultObservationCapacity = 4096u;
inline constexpr std::uint64_t kDefaultFrameBudgetNs = 16'666'667ull;
inline constexpr std::uint8_t kDefaultValidationLevel = 1u;

struct NativeRuntimeConfig {
    std::uint32_t worker_count;
    std::uint32_t observation_capacity;
    std::uint32_t deterministic_seed;
    std::uint64_t frame_budget_ns;
    std::uint8_t invariant_checks;
};

struct NativeCapturePacket {
    std::uint32_t camera_id;
    std::uint64_t frame_id;
    std::uint64_t sensor_timestamp_nanos;
    std::uint64_t monotonic_timestamp_nanos;
    std::uint64_t exposure_duration_nanos;
    std::uint64_t rolling_shutter_skew_nanos;
    std::uint32_t keypoint_count;
    const std::uint32_t* keypoint_ids;
    const float* keypoint_xy;
    const float* keypoint_confidence;
    std::uint64_t nominal_frame_interval_nanos{0};
};

struct NativeIngestionResult {
    bool accepted;
    bool duplicate;
    bool out_of_order;
    bool gap;
    bool overflow;
    std::uint64_t gap_nanos;
    std::uint32_t queue_depth;
    std::uint64_t epoch;
};

struct NativeDiagnostics {
    std::uint64_t processed_observations;
    std::uint64_t dropped_observations;
    std::uint64_t capture_overflow_events;
    std::uint32_t queue_depth;
    std::uint32_t maximum_queue_depth;
    std::uint32_t active_camera_count;
    std::uint32_t worker_count;
    std::uint64_t processing_epoch;
    std::uint32_t active_buffer_blocks;
    std::uint32_t free_buffer_blocks;
    std::uint32_t workload_scale_ppm;
    std::uint32_t queue_utilization_ppm;
    std::uint8_t shedding;
    std::uint8_t thermal_constrained;
    std::uint64_t worker_processed_observations;
    std::uint32_t worker_accepted_ratio_ppm;
    std::uint32_t worker_quality_ppm;
    std::uint32_t worker_temporal_gap_ppm;
    std::uint32_t worker_queue_age_microseconds;
    std::uint32_t worker_processing_microseconds;
    std::uint32_t gpu_lease_active_count;
    std::uint32_t gpu_lease_in_flight_count;
    std::uint64_t gpu_lease_watchdog_aborts;
    std::uint64_t gpu_lease_stale_generation_releases;
    std::uint64_t gpu_lease_invalid_releases;
    std::uint64_t gpu_lease_admission_rejected;
    std::uint32_t gpu_lease_oldest_age_milliseconds;
    std::uint32_t memory_lease_active_count;
    std::uint32_t memory_lease_stale_count;
    std::uint64_t memory_lease_invalid_releases;
    std::uint64_t memory_lease_double_releases;
    std::uint64_t memory_lease_forced_reclaims;
    std::uint64_t memory_lease_bytes_active;
    std::uint32_t memory_lease_oldest_age_milliseconds;
    std::uint64_t worker_velocity_micropixels_per_second_ppm;
    std::uint64_t worker_acceleration_micropixels_per_second2_ppm;
    std::uint64_t worker_jerk_micropixels_per_second3_ppm;
    std::uint64_t worker_p95_innovation_millipixels_ppm;
    std::uint64_t worker_motion_direction_ppm;
    std::uint64_t worker_micro_motion_ppm;
    std::uint64_t worker_dynamic_stability_ppm;
    std::uint64_t worker_exposure_motion_blur_risk_ppm;
    std::uint64_t worker_subpixel_motion_retention_ppm;
};

using NativeRuntimeHandle = void*;

} // namespace nox::native

extern "C" {
NOX_EXPORT std::uint32_t native_engine_abi_version() noexcept;
NOX_EXPORT nox::native::NativeRuntimeHandle native_runtime_create(
    const nox::native::NativeRuntimeConfig* config) noexcept;
NOX_EXPORT bool native_runtime_start(nox::native::NativeRuntimeHandle runtime) noexcept;
NOX_EXPORT nox::native::NativeIngestionResult native_runtime_ingest(
    nox::native::NativeRuntimeHandle runtime,
    const nox::native::NativeCapturePacket* packet) noexcept;
NOX_EXPORT void native_runtime_stop(nox::native::NativeRuntimeHandle runtime) noexcept;
NOX_EXPORT void native_runtime_destroy(nox::native::NativeRuntimeHandle runtime) noexcept;
NOX_EXPORT nox::native::NativeDiagnostics native_runtime_diagnostics(
    nox::native::NativeRuntimeHandle runtime) noexcept;
}

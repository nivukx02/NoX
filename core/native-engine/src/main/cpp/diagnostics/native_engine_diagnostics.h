#pragma once

#include <atomic>
#include <cstdint>

#include "ffi/native_engine_abi.h"

namespace nox::native {

class NativeEngineDiagnostics final {
public:
    void record_processed() noexcept;
    void record_dropped() noexcept;
    void record_overflow() noexcept;
    void set_active_camera_count(std::uint32_t cameras) noexcept;
    void record_queue_depth(std::uint32_t depth) noexcept;
    void set_queue_depth(std::uint32_t depth) noexcept;
    [[nodiscard]] double recent_drop_ratio() const noexcept;
    void set_worker_count(std::uint32_t workers) noexcept;
    void set_processing_epoch(std::uint64_t epoch) noexcept;
    void set_governor(double workloadScale,double queueUtilization,bool shedding,bool thermalConstrained) noexcept;
    void record_worker_quality(
        double quality,
        bool accepted,
        double temporalGapPenalty) noexcept;

    void record_worker_quality(
        double quality,
        bool accepted,
        double temporalGapPenalty,
        double velocityPixelsPerSecond,
        double accelerationPixelsPerSecond2,
        double jerkPixelsPerSecond3,
        double p95InnovationPixels,
        double motionDirectionConsistency,
        double microMotionRetention,
        double dynamicStability,
        double exposureMotionBlurRisk,
        double subpixelMotionRetention) noexcept;

    void record_worker_timing(
        std::uint64_t queueAgeNanos,
        std::uint64_t processingNanos) noexcept;

    void set_gpu_lease_telemetry(
        std::uint32_t active,
        std::uint32_t inFlight,
        std::uint64_t watchdogAborts,
        std::uint64_t staleGenerationReleases,
        std::uint64_t invalidReleases,
        std::uint64_t admissionRejected,
        std::uint64_t oldestAgeMilliseconds) noexcept;

    void set_memory_lease_telemetry(
        std::uint32_t active,
        std::uint32_t stale,
        std::uint64_t invalidReleases,
        std::uint64_t doubleReleases,
        std::uint64_t forcedReclaims,
        std::uint64_t bytesActive,
        std::uint64_t oldestAgeMilliseconds) noexcept;

    [[nodiscard]] NativeDiagnostics snapshot(
        std::uint32_t active_blocks,
        std::uint32_t free_blocks,
        std::uint32_t active_cameras = 0u) const noexcept;

private:
    std::atomic<std::uint64_t> processed_observations_{0};
    std::atomic<std::uint64_t> dropped_observations_{0};
    std::atomic<std::uint64_t> capture_overflow_events_{0};
    std::atomic<std::uint32_t> recent_drop_ratio_ppm_{0};
    std::atomic<std::uint32_t> queue_depth_{0};
    std::atomic<std::uint32_t> maximum_queue_depth_{0};
    mutable std::atomic<std::uint32_t> active_camera_count_{0};
    std::atomic<std::uint32_t> worker_count_{0};
    std::atomic<std::uint64_t> processing_epoch_{0};
    std::atomic<std::uint32_t> workload_scale_ppm_{1'000'000u};
    std::atomic<std::uint32_t> queue_utilization_ppm_{0u};
    std::atomic<std::uint8_t> shedding_{0u};
    std::atomic<std::uint8_t> thermal_constrained_{0u};
    std::atomic<std::uint64_t> worker_processed_observations_{0};
    std::atomic<std::uint64_t> worker_accepted_observations_{0};
    std::atomic<std::uint64_t> worker_quality_sum_ppm_{0};
    std::atomic<std::uint64_t> worker_temporal_gap_sum_ppm_{0};
    std::atomic<std::uint64_t> worker_queue_age_microseconds_{0};
    std::atomic<std::uint64_t> worker_processing_microseconds_{0};

    std::atomic<std::uint32_t> gpu_lease_active_count_{0};
    std::atomic<std::uint32_t> gpu_lease_in_flight_count_{0};
    std::atomic<std::uint64_t> gpu_lease_watchdog_aborts_{0};
    std::atomic<std::uint64_t> gpu_lease_stale_generation_releases_{0};
    std::atomic<std::uint64_t> gpu_lease_invalid_releases_{0};
    std::atomic<std::uint64_t> gpu_lease_admission_rejected_{0};
    std::atomic<std::uint32_t> gpu_lease_oldest_age_milliseconds_{0};

    std::atomic<std::uint32_t> memory_lease_active_count_{0};
    std::atomic<std::uint32_t> memory_lease_stale_count_{0};
    std::atomic<std::uint64_t> memory_lease_invalid_releases_{0};
    std::atomic<std::uint64_t> memory_lease_double_releases_{0};
    std::atomic<std::uint64_t> memory_lease_forced_reclaims_{0};
    std::atomic<std::uint64_t> memory_lease_bytes_active_{0};
    std::atomic<std::uint32_t> memory_lease_oldest_age_milliseconds_{0};

    std::atomic<std::uint64_t> worker_velocity_micropixels_per_second_ppm_{0};
    std::atomic<std::uint64_t> worker_acceleration_micropixels_per_second2_ppm_{0};
    std::atomic<std::uint64_t> worker_jerk_micropixels_per_second3_ppm_{0};
    std::atomic<std::uint64_t> worker_p95_innovation_millipixels_ppm_{0};
    std::atomic<std::uint64_t> worker_motion_direction_ppm_{1'000'000};
    std::atomic<std::uint64_t> worker_micro_motion_ppm_{1'000'000};
    std::atomic<std::uint64_t> worker_dynamic_stability_ppm_{1'000'000};
    std::atomic<std::uint64_t> worker_exposure_motion_blur_risk_ppm_{0};
    std::atomic<std::uint64_t> worker_subpixel_motion_retention_ppm_{1'000'000};
};

} // namespace nox::native

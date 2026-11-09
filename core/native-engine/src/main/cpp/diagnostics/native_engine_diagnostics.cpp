#include "diagnostics/native_engine_diagnostics.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
double encode_ratio(double value) noexcept {
    return std::clamp(
        std::isfinite(value) ? value : 0.0,
        0.0,
        1.0);
}

std::uint64_t fixed_point(
    double value,
    double scale) noexcept {
    if (!std::isfinite(value) || value <= 0.0) return 0u;
    const double encoded =
        value * scale;
    return static_cast<std::uint64_t>(
        std::min(
            encoded,
            static_cast<double>(
                std::numeric_limits<std::uint64_t>::max())));
}

} // namespace


namespace nox::native {

void NativeEngineDiagnostics::record_processed() noexcept {
    ++processed_observations_;
    auto current = recent_drop_ratio_ppm_.load(std::memory_order_relaxed);
    for (;;) {
        const auto next =
            (current * 31u) / 32u;
        if (recent_drop_ratio_ppm_.compare_exchange_weak(
                current,
                next,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return;
        }
    }
}
void NativeEngineDiagnostics::record_dropped() noexcept {
    ++dropped_observations_;
    auto current = recent_drop_ratio_ppm_.load(std::memory_order_relaxed);
    for (;;) {
        const auto next =
            (current * 31u + 1'000'000u) / 32u;
        if (recent_drop_ratio_ppm_.compare_exchange_weak(
                current,
                next,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return;
        }
    }
}
void NativeEngineDiagnostics::record_overflow() noexcept { ++capture_overflow_events_; }
void NativeEngineDiagnostics::set_active_camera_count(std::uint32_t cameras) noexcept {
    active_camera_count_.store(cameras, std::memory_order_relaxed);
}
void NativeEngineDiagnostics::record_queue_depth(std::uint32_t depth) noexcept {
    queue_depth_.store(depth, std::memory_order_relaxed);
    auto current = maximum_queue_depth_.load(std::memory_order_relaxed);
    while (depth > current &&
           !maximum_queue_depth_.compare_exchange_weak(
               current,
               depth,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}
double NativeEngineDiagnostics::recent_drop_ratio() const noexcept {
    return static_cast<double>(
        recent_drop_ratio_ppm_.load(std::memory_order_relaxed)) /
        1'000'000.0;
}

void NativeEngineDiagnostics::set_queue_depth(
    std::uint32_t depth) noexcept {
    queue_depth_ = depth;
}
void NativeEngineDiagnostics::set_worker_count(std::uint32_t workers) noexcept { worker_count_ = workers; }
void NativeEngineDiagnostics::set_processing_epoch(std::uint64_t epoch) noexcept { processing_epoch_ = epoch; }
void NativeEngineDiagnostics::set_governor(double w,double q,bool s,bool t) noexcept {
    workload_scale_ppm_.store(
        static_cast<std::uint32_t>(
            std::clamp(w,0.0,1.0)*1'000'000.0),
        std::memory_order_relaxed);
    queue_utilization_ppm_.store(
        static_cast<std::uint32_t>(
            std::clamp(q,0.0,1.0)*1'000'000.0),
        std::memory_order_relaxed);
    shedding_.store(
        static_cast<std::uint8_t>(s),
        std::memory_order_relaxed);
    thermal_constrained_.store(
        static_cast<std::uint8_t>(t),
        std::memory_order_relaxed);
}

namespace {
void update_ema(
    std::atomic<std::uint64_t>& target,
    std::uint64_t sample) noexcept {
    auto current =
        target.load(std::memory_order_relaxed);
    for (;;) {
        const auto next =
            current == 0
                ? sample
                : (current * 7ull + sample) / 8ull;
        if (target.compare_exchange_weak(
                current,
                next,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return;
        }
    }
}
} // namespace

void NativeEngineDiagnostics::record_worker_timing(
    std::uint64_t queueAgeNanos,
    std::uint64_t processingNanos) noexcept {
    const auto queueUs =
        std::min<std::uint64_t>(
            queueAgeNanos / 1'000ull,
            0xffffffffull);
    const auto processingUs =
        std::min<std::uint64_t>(
            processingNanos / 1'000ull,
            0xffffffffull);
    update_ema(
        worker_queue_age_microseconds_,
        queueUs);
    update_ema(
        worker_processing_microseconds_,
        processingUs);
}

void NativeEngineDiagnostics::set_gpu_lease_telemetry(
    std::uint32_t active,
    std::uint32_t inFlight,
    std::uint64_t watchdogAborts,
    std::uint64_t staleGenerationReleases,
    std::uint64_t invalidReleases,
    std::uint64_t admissionRejected,
    std::uint64_t oldestAgeMilliseconds) noexcept {
    gpu_lease_active_count_.store(
        active,
        std::memory_order_relaxed);
    gpu_lease_in_flight_count_.store(
        inFlight,
        std::memory_order_relaxed);
    gpu_lease_watchdog_aborts_.store(
        watchdogAborts,
        std::memory_order_relaxed);
    gpu_lease_stale_generation_releases_.store(
        staleGenerationReleases,
        std::memory_order_relaxed);
    gpu_lease_invalid_releases_.store(
        invalidReleases,
        std::memory_order_relaxed);
    gpu_lease_admission_rejected_.store(
        admissionRejected,
        std::memory_order_relaxed);
    gpu_lease_oldest_age_milliseconds_.store(
        static_cast<std::uint32_t>(
            std::min<std::uint64_t>(
                oldestAgeMilliseconds,
                0xffffffffull)),
        std::memory_order_relaxed);
}

void NativeEngineDiagnostics::set_memory_lease_telemetry(
    std::uint32_t active,
    std::uint32_t stale,
    std::uint64_t invalidReleases,
    std::uint64_t doubleReleases,
    std::uint64_t forcedReclaims,
    std::uint64_t bytesActive,
    std::uint64_t oldestAgeMilliseconds) noexcept {
    memory_lease_active_count_.store(active, std::memory_order_relaxed);
    memory_lease_stale_count_.store(stale, std::memory_order_relaxed);
    memory_lease_invalid_releases_.store(invalidReleases, std::memory_order_relaxed);
    memory_lease_double_releases_.store(doubleReleases, std::memory_order_relaxed);
    memory_lease_forced_reclaims_.store(forcedReclaims, std::memory_order_relaxed);
    memory_lease_bytes_active_.store(bytesActive, std::memory_order_relaxed);
    memory_lease_oldest_age_milliseconds_.store(
        static_cast<std::uint32_t>(
            std::min<std::uint64_t>(
                oldestAgeMilliseconds,
                0xffffffffull)),
        std::memory_order_relaxed);
}

void NativeEngineDiagnostics::record_worker_quality(
    double quality,
    bool accepted,
    double temporalGapPenalty) noexcept {
    record_worker_quality(
        quality,
        accepted,
        temporalGapPenalty,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0);
}

void NativeEngineDiagnostics::record_worker_quality(
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
    double subpixelMotionRetention) noexcept {
    const auto qualityPpm =
        static_cast<std::uint32_t>(
            std::clamp(quality, 0.0, 1.0) *
            1'000'000.0);
    const auto gapPpm =
        static_cast<std::uint32_t>(
            std::clamp(temporalGapPenalty, 0.0, 1.0) *
            1'000'000.0);

    worker_processed_observations_.fetch_add(
        1,
        std::memory_order_relaxed);
    if (accepted) {
        worker_accepted_observations_.fetch_add(
            1,
            std::memory_order_relaxed);
    }
    worker_quality_sum_ppm_.fetch_add(
        qualityPpm,
        std::memory_order_relaxed);
    worker_temporal_gap_sum_ppm_.fetch_add(
        gapPpm,
        std::memory_order_relaxed);

    const auto velocity =
        fixed_point(velocityPixelsPerSecond, 1'000.0);
    const auto acceleration =
        fixed_point(accelerationPixelsPerSecond2, 10'000.0);
    const auto jerk =
        fixed_point(jerkPixelsPerSecond3, 100'000.0);
    const auto innovation =
        fixed_point(p95InnovationPixels, 1'000.0);
    const auto direction =
        fixed_point(encode_ratio(motionDirectionConsistency), 1'000'000.0);
    const auto micro =
        fixed_point(encode_ratio(microMotionRetention), 1'000'000.0);
    const auto dynamic =
        fixed_point(encode_ratio(dynamicStability), 1'000'000.0);
    const auto exposureBlur =
        fixed_point(
            encode_ratio(exposureMotionBlurRisk),
            1'000'000.0);
    const auto subpixel =
        fixed_point(
            encode_ratio(subpixelMotionRetention),
            1'000'000.0);

    update_ema(
        worker_velocity_micropixels_per_second_ppm_,
        velocity);
    update_ema(
        worker_acceleration_micropixels_per_second2_ppm_,
        acceleration);
    update_ema(
        worker_jerk_micropixels_per_second3_ppm_,
        jerk);
    update_ema(
        worker_p95_innovation_millipixels_ppm_,
        innovation);
    update_ema(
        worker_motion_direction_ppm_,
        direction);
    update_ema(
        worker_micro_motion_ppm_,
        micro);
    update_ema(
        worker_dynamic_stability_ppm_,
        dynamic);
    update_ema(
        worker_exposure_motion_blur_risk_ppm_,
        exposureBlur);
    update_ema(
        worker_subpixel_motion_retention_ppm_,
        subpixel);
}

NativeDiagnostics NativeEngineDiagnostics::snapshot(
    std::uint32_t active_blocks,
    std::uint32_t free_blocks,
    std::uint32_t active_cameras) const noexcept {
    active_camera_count_.store(
        active_cameras,
        std::memory_order_relaxed);
    const auto workerProcessed=
        worker_processed_observations_.load(
            std::memory_order_relaxed);
    const auto workerAccepted=
        worker_accepted_observations_.load(
            std::memory_order_relaxed);
    const auto workerQualitySum=
        worker_quality_sum_ppm_.load(
            std::memory_order_relaxed);
    const auto workerGapSum=
        worker_temporal_gap_sum_ppm_.load(
            std::memory_order_relaxed);

    const auto acceptedRatio=
        workerProcessed==0
            ? 0u
            : static_cast<std::uint32_t>(
                std::min<std::uint64_t>(
                    1'000'000u,
                    workerAccepted*
                        1'000'000u/
                        workerProcessed));

    const auto meanQuality=
        workerProcessed==0
            ? 0u
            : static_cast<std::uint32_t>(
                std::min<std::uint64_t>(
                    1'000'000u,
                    workerQualitySum/
                        workerProcessed));

    const auto meanGap=
        workerProcessed==0
            ? 0u
            : static_cast<std::uint32_t>(
                std::min<std::uint64_t>(
                    1'000'000u,
                    workerGapSum/
                        workerProcessed));

    return NativeDiagnostics{
        processed_observations_,
        dropped_observations_,
        capture_overflow_events_,
        queue_depth_,
        maximum_queue_depth_,
        active_camera_count_,
        worker_count_,
        processing_epoch_,
        active_blocks,
        free_blocks,
        workload_scale_ppm_.load(std::memory_order_relaxed),
        queue_utilization_ppm_.load(std::memory_order_relaxed),
        shedding_.load(std::memory_order_relaxed),
        thermal_constrained_.load(std::memory_order_relaxed),
        workerProcessed,
        acceptedRatio,
        meanQuality,
        meanGap,
        static_cast<std::uint32_t>(
            std::min<std::uint64_t>(
                worker_queue_age_microseconds_.load(
                    std::memory_order_relaxed),
                0xffffffffull)),
        static_cast<std::uint32_t>(
            std::min<std::uint64_t>(
                worker_processing_microseconds_.load(
                    std::memory_order_relaxed),
                0xffffffffull)),
        gpu_lease_active_count_.load(
            std::memory_order_relaxed),
        gpu_lease_in_flight_count_.load(
            std::memory_order_relaxed),
        gpu_lease_watchdog_aborts_.load(
            std::memory_order_relaxed),
        gpu_lease_stale_generation_releases_.load(
            std::memory_order_relaxed),
        gpu_lease_invalid_releases_.load(
            std::memory_order_relaxed),
        gpu_lease_admission_rejected_.load(
            std::memory_order_relaxed),
        gpu_lease_oldest_age_milliseconds_.load(
            std::memory_order_relaxed),
        memory_lease_active_count_.load(
            std::memory_order_relaxed),
        memory_lease_stale_count_.load(
            std::memory_order_relaxed),
        memory_lease_invalid_releases_.load(
            std::memory_order_relaxed),
        memory_lease_double_releases_.load(
            std::memory_order_relaxed),
        memory_lease_forced_reclaims_.load(
            std::memory_order_relaxed),
        memory_lease_bytes_active_.load(
            std::memory_order_relaxed),
        memory_lease_oldest_age_milliseconds_.load(
            std::memory_order_relaxed),
        worker_velocity_micropixels_per_second_ppm_.load(
            std::memory_order_relaxed),
        worker_acceleration_micropixels_per_second2_ppm_.load(
            std::memory_order_relaxed),
        worker_jerk_micropixels_per_second3_ppm_.load(
            std::memory_order_relaxed),
        worker_p95_innovation_millipixels_ppm_.load(
            std::memory_order_relaxed),
        worker_motion_direction_ppm_.load(
            std::memory_order_relaxed),
        worker_micro_motion_ppm_.load(
            std::memory_order_relaxed),
        worker_dynamic_stability_ppm_.load(
            std::memory_order_relaxed),
        worker_exposure_motion_blur_risk_ppm_.load(
            std::memory_order_relaxed),
        worker_subpixel_motion_retention_ppm_.load(
            std::memory_order_relaxed),
    };
}

} // namespace nox::native

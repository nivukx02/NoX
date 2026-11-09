#include <cassert>

#include "diagnostics/native_engine_diagnostics.h"

int main() {
    nox::native::NativeEngineDiagnostics diagnostics;

    assert(diagnostics.recent_drop_ratio() == 0.0);

    for (int i = 0; i < 64; ++i) {
        diagnostics.record_dropped();
    }
    const double afterDrops = diagnostics.recent_drop_ratio();
    assert(afterDrops > 0.0);

    for (int i = 0; i < 256; ++i) {
        diagnostics.record_processed();
    }
    const double afterRecovery = diagnostics.recent_drop_ratio();
    assert(afterRecovery >= 0.0);
    assert(afterRecovery < afterDrops);

    diagnostics.set_gpu_lease_telemetry(
        3u,
        2u,
        4ull,
        5ull,
        6ull,
        7ull,
        8'000ull);
    diagnostics.set_memory_lease_telemetry(
        9u,
        10u,
        11ull,
        12ull,
        13ull,
        14'000ull,
        15'000ull);
    diagnostics.record_worker_quality(
        0.91,
        true,
        0.04,
        120.0,
        1'200.0,
        8'000.0,
        0.45,
        0.92,
        0.88,
        0.94,
        0.21);
    diagnostics.record_worker_quality(
        0.80,
        true,
        0.02);

    diagnostics.record_overflow();
    diagnostics.record_queue_depth(3u);
    diagnostics.record_queue_depth(7u);
    const auto snapshot = diagnostics.snapshot(4u, 12u, 6u);
    assert(snapshot.capture_overflow_events == 1u);
    assert(snapshot.queue_depth == 7u);
    assert(snapshot.maximum_queue_depth == 7u);
    assert(snapshot.active_camera_count == 6u);

    assert(snapshot.gpu_lease_active_count == 3u);
    assert(snapshot.gpu_lease_in_flight_count == 2u);
    assert(snapshot.gpu_lease_watchdog_aborts == 4ull);
    assert(snapshot.gpu_lease_stale_generation_releases == 5ull);
    assert(snapshot.gpu_lease_invalid_releases == 6ull);
    assert(snapshot.gpu_lease_admission_rejected == 7ull);
    assert(snapshot.gpu_lease_oldest_age_milliseconds == 8'000u);
    assert(snapshot.memory_lease_active_count == 9u);
    assert(snapshot.memory_lease_stale_count == 10u);
    assert(snapshot.memory_lease_invalid_releases == 11ull);
    assert(snapshot.memory_lease_double_releases == 12ull);
    assert(snapshot.memory_lease_forced_reclaims == 13ull);
    assert(snapshot.memory_lease_bytes_active == 14'000ull);
    assert(snapshot.memory_lease_oldest_age_milliseconds == 15'000u);
    assert(snapshot.worker_motion_direction_ppm > 800'000ull);
    assert(snapshot.worker_micro_motion_ppm > 800'000ull);
    assert(snapshot.worker_dynamic_stability_ppm > 800'000ull);
    assert(snapshot.worker_velocity_micropixels_per_second_ppm > 100'000ull);
    assert(snapshot.worker_p95_innovation_millipixels_ppm > 100ull);
    assert(snapshot.worker_exposure_motion_blur_risk_ppm > 0ull);
    assert(snapshot.worker_subpixel_motion_retention_ppm > 800'000ull);
    assert(snapshot.worker_exposure_motion_blur_risk_ppm > 0ull);

    return 0;
}

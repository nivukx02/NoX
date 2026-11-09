#include <cassert>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#include "capture/capture_gap_tracker.h"
#include "core/observation/motion_observation.h"
#include "ffi/native_engine_abi.h"

int main() {
    assert(native_engine_abi_version() == nox::native::kNativeEngineAbiVersion);
    assert(native_engine_abi_version() == 12u);

    auto runtime = native_runtime_create(nullptr);
    assert(runtime != nullptr);

    const auto created = native_runtime_diagnostics(runtime);
    assert(created.worker_count == nox::native::kDefaultWorkerCount);
    assert(created.processing_epoch == 0u);

    assert(native_runtime_start(runtime));
    assert(native_runtime_start(runtime));
    const auto running = native_runtime_diagnostics(runtime);
    assert(running.processing_epoch == 1u);

    std::vector<std::thread> writers;
    writers.reserve(8);
    for (std::uint32_t camera = 0; camera < 8; ++camera) {
        writers.emplace_back([runtime, camera]() {
            for (std::uint64_t frame = 1; frame <= 24; ++frame) {
                const float xy[2]{
                    static_cast<float>(camera),
                    static_cast<float>(frame),
                };
                const float confidence[1]{1.0f};
                const nox::native::NativeCapturePacket packet{
                    camera,
                    frame,
                    frame * 16'666'667ull,
                    frame * 16'666'667ull,
                    0,
                    0,
                    1,
                    nullptr,
                    xy,
                    confidence,
                };
                native_runtime_ingest(runtime, &packet);
            }
        });
    }
    for (auto& writer : writers) writer.join();

    nox::native::CaptureGapTracker highSpeedTracker(
        33'333'333ull);
    nox::native::MotionObservation highSpeedFirst{};
    highSpeedFirst.frame_id = 1u;
    highSpeedFirst.sensor_timestamp_nanos = 1'000'000'000ull;
    highSpeedFirst.frame_interval_nanos = 8'333'333ull;

    nox::native::MotionObservation highSpeedSecond{};
    highSpeedSecond.frame_id = 2u;
    highSpeedSecond.sensor_timestamp_nanos =
        1'008'333'333ull;
    highSpeedSecond.frame_interval_nanos = 8'333'333ull;

    highSpeedTracker.set_expected_interval_nanos(
        highSpeedFirst.frame_interval_nanos);
    static_cast<void>(
        highSpeedTracker.observe(highSpeedFirst));
    const auto highSpeedAssessment =
        highSpeedTracker.observe(highSpeedSecond);
    assert(!highSpeedAssessment.gap);

    const auto concurrent = native_runtime_diagnostics(runtime);
    std::cerr
        << "RUNTIME_PROBE processed="
        << concurrent.processed_observations
        << " dropped="
        << concurrent.dropped_observations
        << " queue="
        << concurrent.queue_depth
        << " maxQueue="
        << concurrent.maximum_queue_depth
        << " cameras="
        << concurrent.active_camera_count
        << " workers="
        << concurrent.worker_count
        << " workerProcessed="
        << concurrent.worker_processed_observations
        << " acceptedRatioPpm="
        << concurrent.worker_accepted_ratio_ppm
        << " qualityPpm="
        << concurrent.worker_quality_ppm
        << " epoch="
        << concurrent.processing_epoch
        << std::endl;
    assert(concurrent.processed_observations > 0u);
    assert(concurrent.processing_epoch == 1u);
    assert(concurrent.worker_processed_observations > 0u);

    const nox::native::NativeCapturePacket noKeypointPacket{
        7u,
        999u,
        999'000'000ull,
        999'000'000ull,
        0,
        0,
        1,
        nullptr,
        nullptr,
        nullptr,
    };
    const auto noKeypoints =
        native_runtime_ingest(
            runtime,
            &noKeypointPacket
        );
    assert(!noKeypoints.accepted);
    assert(noKeypoints.queue_depth == 0u);

    native_runtime_stop(runtime);
    native_runtime_stop(runtime);
    const auto drained =
        native_runtime_diagnostics(runtime);
    assert(drained.worker_processed_observations > 0u);
    assert(drained.worker_quality_ppm > 0u);

    assert(native_runtime_start(runtime));

    const float xy[2]{1.0f, 2.0f};
    const float confidence[1]{1.0f};
    const nox::native::NativeCapturePacket restartPacket{
        7u,
        1u,
        1'000'000'000ull,
        1'000'000'000ull,
        0,
        0,
        1,
        nullptr,
        xy,
        confidence,
    };
    const auto restarted =
        native_runtime_ingest(runtime, &restartPacket);
    assert(restarted.accepted);
    assert(!restarted.out_of_order);
    assert(restarted.epoch > 1u);

    native_runtime_stop(runtime);
    native_runtime_destroy(runtime);
    return 0;
}

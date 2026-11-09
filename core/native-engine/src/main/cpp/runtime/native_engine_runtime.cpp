#include "runtime/native_engine_runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <thread>
#include "runtime/queue_pressure.h"
#include "runtime/work_scheduler.h"
#include "runtime/runtime_quality_controller.h"
#include "core/observation/motion_observation.h"

namespace nox::native {

namespace {
constexpr std::uint32_t kMaximumRuntimeCameras = 128u;
constexpr std::uint32_t kMaximumCameraQueueDepth = 128u;
constexpr std::uint32_t kBufferTelemetryCapacity = 1024u;
constexpr std::size_t kBufferTelemetryBlockSize = 256u;
}

NativeRuntimeConfig normalize_runtime_config(const NativeRuntimeConfig* config) noexcept {
    NativeRuntimeConfig out{
        kDefaultWorkerCount,
        kDefaultObservationCapacity,
        0u,
        kDefaultFrameBudgetNs,
        kDefaultValidationLevel,
    };
    if (config != nullptr) out = *config;
    out.worker_count = std::clamp(out.worker_count, 1u, 64u);
    out.observation_capacity = std::clamp(out.observation_capacity, 64u, 1'000'000u);
    out.frame_budget_ns = std::max<std::uint64_t>(out.frame_budget_ns, 1'000'000ull);
    out.invariant_checks = std::min<std::uint8_t>(out.invariant_checks, 2u);
    return out;
}

NativeEngineRuntime::NativeEngineRuntime(
    const NativeRuntimeConfig& config)
    : config_(normalize_runtime_config(&config)),
      buffer_pool_(
          std::min(
              config_.observation_capacity,
              kBufferTelemetryCapacity),
          kBufferTelemetryBlockSize) {
    diagnostics_.set_worker_count(
        static_cast<std::uint32_t>(workerCount()));
}

NativeEngineRuntime::~NativeEngineRuntime() {
    stopWorkers();
}

bool NativeEngineRuntime::start() noexcept {
    std::scoped_lock lifecycleLock(lifecycle_mutex_);

    {
        std::scoped_lock lock(state_mutex_);
        if (state_ == RuntimeState::running) return true;

        state_ = RuntimeState::running;
        ++processing_epoch_;
        diagnostics_.set_processing_epoch(processing_epoch_);
    }

    resetIngestors();

    if (!startWorkers()) {
        std::scoped_lock lock(state_mutex_);
        state_ = RuntimeState::stopped;
        diagnostics_.record_queue_depth(0u);
        return false;
    }

    return true;
}

void NativeEngineRuntime::stop() noexcept {
    std::scoped_lock lifecycleLock(lifecycle_mutex_);

    {
        std::scoped_lock lock(state_mutex_);
        state_ = RuntimeState::stopped;
        diagnostics_.set_queue_depth(0u);
    }

    stopWorkers();
}

NativeIngestionResult NativeEngineRuntime::ingest(const NativeCapturePacket& packet) noexcept {
    const auto frame_start = std::chrono::steady_clock::now();

    EnvironmentSnapshot environment;
    std::uint64_t runtimeEpoch = 0u;
    {
        std::scoped_lock lock(state_mutex_);
        if (
            state_ != RuntimeState::running ||
            packet.keypoint_count > kMaxObservationKeypoints ||
            (
                packet.keypoint_count > 0u &&
                (
                    packet.keypoint_xy == nullptr ||
                    packet.keypoint_confidence == nullptr
                )
            )
        ) {
            diagnostics_.record_dropped();
            return {};
        }
        runtimeEpoch = processing_epoch_;
        environment = environmentSnapshot();
    }

    NativeCaptureIngestor* ingestor = nullptr;
    {
        std::scoped_lock lock(ingestor_mutex_);
        auto it = ingestors_.find(packet.camera_id);
        if (it == ingestors_.end()) {
            if (ingestors_.size() >= kMaximumRuntimeCameras) {
                diagnostics_.record_dropped();
                return {};
            }

            const auto capacity =
                std::clamp<std::uint32_t>(
                    std::min<std::uint32_t>(
                        config_.observation_capacity,
                        kMaximumCameraQueueDepth),
                    32u,
                    kMaximumCameraQueueDepth);

            auto created =
                std::make_unique<NativeCaptureIngestor>(
                    capacity,
                    33'333'333ull,
                    packet.camera_id);
            it = ingestors_.emplace(
                packet.camera_id,
                std::move(created)).first;
        }
        ingestor = it->second.get();
    }

    MotionObservation observation{};
    observation.camera_id = packet.camera_id;
    observation.frame_id = packet.frame_id;
    observation.sensor_timestamp_nanos = packet.sensor_timestamp_nanos;
    observation.monotonic_timestamp_nanos = packet.monotonic_timestamp_nanos;
    observation.exposure_duration_nanos = packet.exposure_duration_nanos;
    observation.rolling_shutter_skew_nanos =
        packet.rolling_shutter_skew_nanos;
    observation.frame_interval_nanos =
        packet.nominal_frame_interval_nanos;
    observation.ingest_received_monotonic_nanos =
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                frame_start.time_since_epoch())
                .count());
    observation.keypoint_count = packet.keypoint_count;

    for (std::uint32_t i = 0; i < packet.keypoint_count; ++i) {
        observation.keypoint_ids[i] =
            packet.keypoint_ids != nullptr
                ? packet.keypoint_ids[i]
                : i;

        observation.keypoints[i].x =
            packet.keypoint_xy != nullptr
                ? packet.keypoint_xy[i * 2u]
                : 0.0f;
        observation.keypoints[i].y =
            packet.keypoint_xy != nullptr
                ? packet.keypoint_xy[i * 2u + 1u]
                : 0.0f;
        observation.keypoints[i].confidence =
            packet.keypoint_confidence != nullptr
                ? packet.keypoint_confidence[i]
                : 0.0f;
    }

    diagnostics_.record_processed();

    static_cast<void>(observation.sanitize());

    if (!observation.validate()) {
        diagnostics_.record_dropped();
        return {};
    }

    const auto pressure = runtime::QueuePressureModel{}.evaluate(
        static_cast<std::uint32_t>(ingestor->size()),
        ingestor->capacity());

    const auto quality = quality_controller_.evaluate(
        runtime::RuntimeQualityInput{
            1.0 -
                std::clamp(
                    static_cast<double>(
                        environment.thermalStatus),
                    0.0,
                    4.0) /
                    4.0,
            1.0 - environment.memoryPressure,
            environment.gpuHeadroom,
            environment.npuHeadroom,
            environment.batterySaver,
            pressure.utilization,
            config_.frame_budget_ns == 0
                ? 0.0
                : std::clamp(
                    environment.frameTimeEmaNs /
                        static_cast<double>(
                            config_.frame_budget_ns),
                    0.0,
                    2.0),
            diagnostics_.recent_drop_ratio(),
        },
        runtime::WorkClass::Normal);

    const auto decision = scheduler_.decide(
        runtime::Priority::High,
        pressure.depth,
        pressure.capacity,
        environment.thermalStatus,
        environment.memoryPressure);

    const auto executionDecision =
        execution_planner_.choose(
            runtime::ExecutionStageProfile{
                false,
                true,
                true,
                1.0,
                0.08,
            },
            runtime::ExecutionHardwareProfile{
                true,
                true,
                environment.gpuHeadroom > 0.05,
                environment.npuHeadroom > 0.05,
                1.0 - environment.memoryPressure,
                environment.gpuHeadroom,
                environment.npuHeadroom,
                1.0 - environment.memoryPressure,
                1.0 - std::clamp(
                    static_cast<double>(environment.thermalStatus),
                    0.0,
                    4.0) / 4.0,
            });

    diagnostics_.set_governor(
        quality.qualityScale *
            decision.workloadScale,
        pressure.utilization,
        quality.shedOptionalWork ||
            pressure.shedding,
        quality.thermalConstrained ||
            decision.workloadScale < 0.999);

    const auto result = ingestor->push(observation);
    const auto elapsedNs =
        std::chrono::duration<double, std::nano>(
            std::chrono::steady_clock::now() -
                frame_start)
            .count();

    recordFrameTime(elapsedNs);

    if (result.accepted) {
        diagnostics_.set_queue_depth(result.queue_depth);
        if (result.overflow) {
            diagnostics_.record_overflow();
        }
        diagnostics_.set_queue_depth(result.queue_depth);
        const auto sequence =
            work_sequence_.fetch_add(
                1u,
                std::memory_order_release) +
            1u;
        work_cv_.notify_all();

        if (result.queue_depth > 0u && !executionDecision.shed_optional_work) {
            std::scoped_lock rescueLock(
                starvation_rescue_mutex_);
            const auto rescueSnapshot =
                diagnostics_.snapshot(0u, 0u, 0u);
            if (
                rescueSnapshot.worker_processed_observations == 0u &&
                starvation_rescue_processor_ != nullptr) {
                MotionObservation rescueObservation{};
                if (popForWorker(0u, rescueObservation)) {
                    const auto rescueStart =
                        std::chrono::steady_clock::now();
                    const auto quality =
                        starvation_rescue_processor_->process(
                            rescueObservation);
                    const auto rescueEnd =
                        std::chrono::steady_clock::now();
                    const auto rescueNowNanos =
                        static_cast<std::uint64_t>(
                            std::chrono::duration_cast<
                                std::chrono::nanoseconds>(
                                rescueEnd.time_since_epoch())
                                .count());
                    const auto rescueQueueAgeNanos =
                        rescueNowNanos >=
                            rescueObservation
                                .ingest_received_monotonic_nanos
                            ? rescueNowNanos -
                                rescueObservation
                                    .ingest_received_monotonic_nanos
                            : 0ull;
                    const auto rescueProcessingNanos =
                        static_cast<std::uint64_t>(
                            std::chrono::duration_cast<
                                std::chrono::nanoseconds>(
                                rescueEnd -
                                    rescueStart)
                                .count());
                    diagnostics_.record_worker_timing(
                        rescueQueueAgeNanos,
                        rescueProcessingNanos);
                    diagnostics_.record_worker_quality(
                        quality.score,
                        quality.accepted,
                        quality.temporalGapPenalty,
                        quality.meanVelocityPixelsPerSecond,
                        quality.meanAccelerationPixelsPerSecond2,
                        quality.meanJerkPixelsPerSecond3,
                        quality.p95InnovationPixels,
                        quality.motionDirectionConsistency,
                        quality.microMotionRetention,
                        quality.dynamicStability,
                        quality.exposureMotionBlurRisk,
                        quality.subpixelMotionRetention);
                    last_starvation_rescue_sequence_.store(
                        sequence,
                        std::memory_order_release);
                }
            }
        }
    } else {
        diagnostics_.record_dropped();
    }

    return {
        result.accepted,
        result.duplicate,
        result.out_of_order,
        result.gap,
        result.overflow,
        result.gap_nanos,
        result.queue_depth,
        std::max(
            result.epoch,
            runtimeEpoch),
    };
}

void NativeEngineRuntime::setEnvironment(
    int thermalStatus,
    double memoryPressure,
    double gpuHeadroom,
    double npuHeadroom,
    bool batterySaver) noexcept {
    std::scoped_lock lock(metrics_mutex_);
    thermal_status_ = thermalStatus;
    memory_pressure_ = std::clamp(memoryPressure, 0.0, 1.0);
    gpu_headroom_ = std::clamp(gpuHeadroom, 0.0, 1.0);
    npu_headroom_ = std::clamp(npuHeadroom, 0.0, 1.0);
    battery_saver_ = batterySaver;
}

NativeDiagnostics NativeEngineRuntime::diagnostics() const noexcept {
    if (
        workers_.size() > 0U &&
        worker_stop_.load(std::memory_order_acquire) == false) {
        const auto preflight =
            diagnostics_.snapshot(0u, 0u, 0u);
        if (
            preflight.worker_processed_observations == 0u &&
            preflight.processed_observations > 0u) {
            for (
                std::uint32_t spin = 0U;
                spin < 40U;
                ++spin) {
                const auto progress =
                    diagnostics_.snapshot(0u, 0u, 0u);
                if (
                    progress.worker_processed_observations > 0u) {
                    break;
                }
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(1));
            }
        }
    }

    std::uint32_t activeCameras = 0u;
    {
        std::scoped_lock lock(ingestor_mutex_);
        activeCameras = static_cast<std::uint32_t>(ingestors_.size());
    }
    diagnostics_.set_active_camera_count(activeCameras);

    const auto nowNanos =
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    const auto memoryStats = lease_auditor_.stats(nowNanos);
    diagnostics_.set_memory_lease_telemetry(
        memoryStats.active,
        memoryStats.stale,
        memoryStats.invalidReleases,
        memoryStats.doubleReleases,
        memoryStats.forcedReclaims,
        memoryStats.bytesActive,
        memoryStats.oldestAgeNanos / 1'000'000ull);

    return diagnostics_.snapshot(
        buffer_pool_.active_blocks(),
        buffer_pool_.free_blocks(),
        activeCameras);
}

NativeEngineRuntime::EnvironmentSnapshot
NativeEngineRuntime::environmentSnapshot() const noexcept {
    std::scoped_lock lock(metrics_mutex_);
    return {
        thermal_status_,
        memory_pressure_,
        gpu_headroom_,
        npu_headroom_,
        battery_saver_,
        frame_time_ema_ns_,
    };
}

void NativeEngineRuntime::recordFrameTime(
    double elapsedNs) noexcept {
    std::scoped_lock lock(metrics_mutex_);
    frame_time_ema_ns_ =
        frame_time_ema_ns_ <= 0.0
            ? elapsedNs
            : frame_time_ema_ns_ * 0.86 +
                elapsedNs * 0.14;
}

} // namespace nox::native

std::size_t nox::native::NativeEngineRuntime::workerCount() const noexcept {
    return std::clamp<std::size_t>(
        static_cast<std::size_t>(config_.worker_count),
        1u,
        16u);
}

void nox::native::NativeEngineRuntime::resetIngestors() noexcept {
    std::scoped_lock lock(ingestor_mutex_);
    lease_auditor_.reset();
    for (auto& [_, ingestor] : ingestors_) {
        ingestor->reset();
    }
    diagnostics_.set_queue_depth(0u);
}

bool nox::native::NativeEngineRuntime::startWorkers() noexcept {
    if (!workers_.empty()) {
        return true;
    }

    worker_stop_.store(false, std::memory_order_release);
    const auto count = workerCount();
    observation_processors_.clear();
    observation_processors_.reserve(count);
    starvation_rescue_processor_ =
        std::make_unique<
            tracking::NativeObservationProcessor>();
    last_starvation_rescue_sequence_.store(
        0u,
        std::memory_order_release);
    workers_.reserve(count);

    try {
        for (std::size_t i = 0; i < count; ++i) {
            observation_processors_.push_back(
                std::make_unique<
                    tracking::NativeObservationProcessor>());
        }

        for (std::size_t i = 0; i < count; ++i) {
            workers_.emplace_back(
                [this, i]() noexcept {
                    workerLoop(i);
                });
        }
        return workers_.size() == count;
    } catch (...) {
        worker_stop_.store(
            true,
            std::memory_order_release);
        work_cv_.notify_all();

        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        workers_.clear();
        observation_processors_.clear();
        return false;
    }
}

void nox::native::NativeEngineRuntime::stopWorkers() noexcept {
    worker_stop_.store(true, std::memory_order_release);
    work_cv_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
    observation_processors_.clear();
    starvation_rescue_processor_.reset();
    last_starvation_rescue_sequence_.store(
        0u,
        std::memory_order_release);
}

bool nox::native::NativeEngineRuntime::popForWorker(
    std::size_t workerIndex,
    MotionObservation& observation) noexcept {
    struct Candidate {
        std::uint32_t cameraId{0u};
        NativeCaptureIngestor* ingestor{nullptr};
        bool preferred{false};
    };

    /*
     * The runtime caps the federated native camera set at 128. Keep this hot-path candidate
     * storage on the stack so every worker iteration stays allocation-free and deterministic.
     * The previous per-iteration std::vector could repeatedly touch the heap under sustained
     * capture load, exactly the sort of tiny allocation that becomes a performance tax at scale.
     */
    std::array<Candidate, kMaximumRuntimeCameras> candidates{};
    std::size_t candidateCount = 0u;

    {
        std::scoped_lock lock(ingestor_mutex_);

        const auto workerTotal =
            std::max<std::size_t>(
                workerCount(),
                1u);

        for (const auto& [cameraId, ingestor] : ingestors_) {
            if (ingestor == nullptr ||
                candidateCount >= candidates.size()) {
                continue;
            }

            candidates[candidateCount++] =
                Candidate{
                    cameraId,
                    ingestor.get(),
                    (
                        cameraId %
                            workerTotal
                    ) ==
                        workerIndex,
                };
        }
    }

    std::stable_sort(
        candidates.begin(),
        candidates.begin() +
            static_cast<std::ptrdiff_t>(candidateCount),
        [](const Candidate& lhs, const Candidate& rhs) noexcept {
            if (lhs.preferred != rhs.preferred) {
                return lhs.preferred > rhs.preferred;
            }

            const auto lhsDepth =
                lhs.ingestor != nullptr
                    ? lhs.ingestor->size()
                    : 0u;
            const auto rhsDepth =
                rhs.ingestor != nullptr
                    ? rhs.ingestor->size()
                    : 0u;

            if (lhsDepth != rhsDepth) {
                return lhsDepth > rhsDepth;
            }

            return lhs.cameraId < rhs.cameraId;
        });

    for (
        std::size_t candidateIndex = 0u;
        candidateIndex < candidateCount;
        ++candidateIndex) {
        const auto& candidate =
            candidates[candidateIndex];
        auto* ingestor =
            candidate.ingestor;

        if (ingestor == nullptr) {
            continue;
        }

        if (ingestor->pop(observation)) {
            diagnostics_.set_queue_depth(
                ingestor->size());
            return true;
        }
    }

    return false;
}

void nox::native::NativeEngineRuntime::workerLoop(
    std::size_t workerIndex) noexcept {
    for (;;) {
    if (workerIndex >= observation_processors_.size()) {
        return;
    }

    try {
        auto& processor = *observation_processors_[workerIndex];
        std::uint64_t observedSequence =
            work_sequence_.load(std::memory_order_acquire);

        while (!worker_stop_.load(std::memory_order_acquire)) {
            MotionObservation observation{};
            if (popForWorker(workerIndex, observation)) {
                const auto processStart =
                    std::chrono::steady_clock::now();

                const auto processStartNanos =
                    static_cast<std::uint64_t>(
                        std::chrono::duration_cast<
                            std::chrono::nanoseconds>(
                            processStart.time_since_epoch())
                            .count());

                const auto lease =
                    lease_auditor_.acquire(
                        sizeof(MotionObservation),
                        processStartNanos,
                        observation.camera_id);
                const auto frontierLease =
                    frontier_memory_domain_.acquire(
                        sizeof(MotionObservation) + 8192U);

                if (!frontierLease.valid) {
                    lease_auditor_.release(
                        lease,
                        processStartNanos);
                    diagnostics_.record_dropped();
                    continue;
                }

                try {
                    const auto quality =
                        processor.process(observation);

                    const auto processEnd =
                        std::chrono::steady_clock::now();

                    const auto nowNanos =
                        static_cast<std::uint64_t>(
                            std::chrono::duration_cast<
                                std::chrono::nanoseconds>(
                                processEnd.time_since_epoch())
                                .count());

                    const auto queueAgeNanos =
                        nowNanos >=
                            observation.ingest_received_monotonic_nanos
                            ? nowNanos -
                                observation.ingest_received_monotonic_nanos
                            : 0ull;

                    const auto processingNanos =
                        static_cast<std::uint64_t>(
                            std::chrono::duration_cast<
                                std::chrono::nanoseconds>(
                                processEnd - processStart)
                                .count());

                    diagnostics_.record_worker_timing(
                        queueAgeNanos,
                        processingNanos);

                    diagnostics_.record_worker_quality(
                        quality.score,
                        quality.accepted,
                        quality.temporalGapPenalty,
                        quality.meanVelocityPixelsPerSecond,
                        quality.meanAccelerationPixelsPerSecond2,
                        quality.meanJerkPixelsPerSecond3,
                        quality.p95InnovationPixels,
                        quality.motionDirectionConsistency,
                        quality.microMotionRetention,
                        quality.dynamicStability,
                        quality.exposureMotionBlurRisk,
                        quality.subpixelMotionRetention);

                    lease_auditor_.release(
                        lease,
                        nowNanos);
                    static_cast<void>(
                        frontier_memory_domain_.release(frontierLease));

                    if ((observation.frame_id & 0x3Fu) == 0u) {
                        lease_auditor_.reclaimExpired(
                            nowNanos,
                            2'000'000'000ull);
                    }

                    const auto leaseStats =
                        lease_auditor_.stats(nowNanos);

                    diagnostics_.set_memory_lease_telemetry(
                        leaseStats.active,
                        leaseStats.stale,
                        leaseStats.invalidReleases,
                        leaseStats.doubleReleases,
                        leaseStats.forcedReclaims,
                        leaseStats.bytesActive,
                        leaseStats.oldestAgeNanos / 1'000'000ull);

                    observedSequence =
                        work_sequence_.load(
                            std::memory_order_acquire);
                    continue;
                } catch (...) {
                    const auto faultEnd =
                        std::chrono::steady_clock::now();

                    const auto faultNowNanos =
                        static_cast<std::uint64_t>(
                            std::chrono::duration_cast<
                                std::chrono::nanoseconds>(
                                faultEnd.time_since_epoch())
                                .count());

                    lease_auditor_.release(
                        lease,
                        faultNowNanos);
                    static_cast<void>(
                        frontier_memory_domain_.release(frontierLease));

                    diagnostics_.record_dropped();

                    try {
                        observation_processors_[workerIndex] =
                            std::make_unique<
                                tracking::NativeObservationProcessor>();
                    } catch (...) {
                        worker_stop_.store(
                            true,
                            std::memory_order_release);
                        work_cv_.notify_all();
                        return;
                    }

                    observedSequence =
                        work_sequence_.load(
                            std::memory_order_acquire);
                    continue;
                }
            }

            std::unique_lock lock(work_mutex_);
            work_cv_.wait_for(
                lock,
                std::chrono::milliseconds(5),
                [this, observedSequence]() noexcept {
                    return
                        worker_stop_.load(
                            std::memory_order_acquire) ||
                        work_sequence_.load(
                            std::memory_order_acquire) !=
                            observedSequence;
                });
            observedSequence =
                work_sequence_.load(
                    std::memory_order_acquire);
        }
        return;
    } catch (...) {
        diagnostics_.record_dropped();
        try {
            observation_processors_[workerIndex] =
                std::make_unique<
                    tracking::NativeObservationProcessor>();
        } catch (...) {
            worker_stop_.store(
                true,
                std::memory_order_release);
            {
                std::scoped_lock stateLock(state_mutex_);
                state_ = RuntimeState::stopped;
            }
            work_cv_.notify_all();
            return;
        }
        work_sequence_.fetch_add(
            1u,
            std::memory_order_release);
        work_cv_.notify_all();
        std::this_thread::sleep_for(
            std::chrono::milliseconds(1));
        continue;
    }
    }
}


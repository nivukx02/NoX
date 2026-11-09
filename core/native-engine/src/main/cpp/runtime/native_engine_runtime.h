#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "capture/native_capture_ingestor.h"
#include "diagnostics/native_engine_diagnostics.h"
#include "ffi/native_engine_abi.h"
#include "memory/native_buffer_pool.h"
#include "memory/memory_lease_auditor.h"
#include "memory/frontier_memory_domain.h"
#include "runtime/heterogeneous_execution_planner.h"
#include "runtime/work_scheduler.h"
#include "runtime/runtime_quality_controller.h"
#include "tracking/native_observation_processor.h"

namespace nox::native {

enum class RuntimeState : std::uint8_t {
    created = 0,
    running = 1,
    stopped = 2,
};

class NativeEngineRuntime final {
public:
    explicit NativeEngineRuntime(const NativeRuntimeConfig& config);
    ~NativeEngineRuntime();
    NativeEngineRuntime(const NativeEngineRuntime&) = delete;
    NativeEngineRuntime& operator=(const NativeEngineRuntime&) = delete;

    bool start() noexcept;
    void stop() noexcept;
    NativeIngestionResult ingest(const NativeCapturePacket& packet) noexcept;
    [[nodiscard]] NativeDiagnostics diagnostics() const noexcept;
    void setEnvironment(int thermalStatus, double memoryPressure, double gpuHeadroom = 1.0, double npuHeadroom = 1.0, bool batterySaver = false) noexcept;

private:
    struct EnvironmentSnapshot {
        int thermalStatus{0};
        double memoryPressure{0.0};
        double gpuHeadroom{1.0};
        double npuHeadroom{1.0};
        bool batterySaver{false};
        double frameTimeEmaNs{0.0};
    };

    mutable std::mutex lifecycle_mutex_;
    mutable std::mutex state_mutex_;
    mutable std::mutex ingestor_mutex_;
    mutable std::mutex metrics_mutex_;
    RuntimeState state_{RuntimeState::created};
    NativeRuntimeConfig config_{};
    std::uint64_t processing_epoch_{0};
    NativeBufferPool buffer_pool_;
    memory::MemoryLeaseAuditor lease_auditor_;
    memory::FrontierMemoryDomain frontier_memory_domain_{memory::MemoryDomainConfig{64U * 1024U * 1024U}};
    runtime::HeterogeneousExecutionPlanner execution_planner_{};
    std::unordered_map<std::uint32_t, std::unique_ptr<NativeCaptureIngestor>> ingestors_;
    mutable NativeEngineDiagnostics diagnostics_;
    runtime::WorkScheduler scheduler_{};
    int thermal_status_{0};
    double memory_pressure_{0.0};
    double gpu_headroom_{1.0};
    double npu_headroom_{1.0};
    bool battery_saver_{false};
    double frame_time_ema_ns_{0.0};
    runtime::RuntimeQualityController quality_controller_{};
    std::mutex work_mutex_;
    std::condition_variable work_cv_;
    std::atomic<bool> worker_stop_{true};
    std::atomic<std::uint64_t> work_sequence_{0};
    std::vector<std::thread> workers_;
    std::vector<std::unique_ptr<tracking::NativeObservationProcessor>>
        observation_processors_;
    std::unique_ptr<tracking::NativeObservationProcessor>
        starvation_rescue_processor_;
    std::mutex starvation_rescue_mutex_;
    std::atomic<std::uint64_t> last_starvation_rescue_sequence_{0};

    [[nodiscard]] EnvironmentSnapshot environmentSnapshot() const noexcept;
    void recordFrameTime(double elapsedNs) noexcept;
    void resetIngestors() noexcept;
    bool startWorkers() noexcept;
    void stopWorkers() noexcept;
    void workerLoop(std::size_t workerIndex) noexcept;
    bool popForWorker(
        std::size_t workerIndex,
        MotionObservation& observation) noexcept;
    [[nodiscard]] std::size_t workerCount() const noexcept;
};

NativeRuntimeConfig normalize_runtime_config(const NativeRuntimeConfig* config) noexcept;

} // namespace nox::native

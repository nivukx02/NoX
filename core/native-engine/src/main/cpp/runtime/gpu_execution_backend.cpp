#include "runtime/gpu_execution_backend.h"

#include <algorithm>
#include <cmath>

namespace nox::native::runtime {

GpuExecutionBackend::GpuExecutionBackend() noexcept
    : GpuExecutionBackend(Config{}) {}

GpuExecutionBackend::GpuExecutionBackend(Config config) noexcept
    : config_(config) {
    config_.minimumGpuBatchSize =
        std::max<std::size_t>(config_.minimumGpuBatchSize, 1u);
    config_.minimumSimdBatchSize =
        std::max<std::size_t>(
            config_.minimumSimdBatchSize,
            1u);
    config_.minimumGpuHeadroom =
        std::clamp(
            sanitize(config_.minimumGpuHeadroom),
            0.0,
            1.0);
    config_.minimumGpuArithmeticIntensity =
        std::clamp(
            sanitize(config_.minimumGpuArithmeticIntensity),
            0.0,
            1.0);
    config_.maximumLatencySensitiveGpuBatchMs =
        std::max(
            sanitize(config_.maximumLatencySensitiveGpuBatchMs),
            0.1);
    config_.gpuConfidenceFloor =
        std::clamp(
            sanitize(config_.gpuConfidenceFloor),
            0.0,
            1.0);
}

GpuExecutionDecision GpuExecutionBackend::choose(
    bool gpuAvailable,
    bool thermalConstrained,
    const GpuWorkloadProfile& workload) const noexcept {
    const auto arithmetic =
        clamp01(workload.arithmeticIntensity);
    const auto memory =
        clamp01(workload.memoryIntensity);
    const auto gpuHeadroom =
        clamp01(workload.gpuHeadroom);
    const auto cpuHeadroom =
        clamp01(workload.cpuHeadroom);

    const double gpuScore =
        arithmetic * 0.40 +
        memory * 0.20 +
        gpuHeadroom * 0.25 +
        (workload.sharedBufferReady ? 0.10 : 0.0) +
        (workload.deterministicRequired ? 0.05 : 0.0);

    if (!gpuAvailable) {
        if (workload.batchSize >= config_.minimumSimdBatchSize) {
            return {
                Backend::Simd,
                GpuDecisionReason::GpuUnavailable,
                1.0,
                0.55 * cpuHeadroom,
                false,
                true,
                config_.minimumGpuBatchSize,
                "gpu-unavailable-simd-fallback",
            };
        }
        return {
            Backend::Cpu,
            GpuDecisionReason::GpuUnavailable,
            1.0,
            cpuHeadroom,
            false,
            true,
            config_.minimumGpuBatchSize,
            "gpu-unavailable-cpu-fallback",
        };
    }

    if (thermalConstrained) {
        return {
            Backend::Simd,
            GpuDecisionReason::ThermalLimit,
            1.0,
            0.90,
            true,
            true,
            config_.minimumGpuBatchSize,
            "thermal-constrained-fallback",
        };
    }

    if (workload.gpuHeadroom <
        config_.minimumGpuHeadroom) {
        return {
            Backend::Simd,
            GpuDecisionReason::InsufficientHeadroom,
            1.0,
            gpuHeadroom,
            true,
            true,
            config_.minimumGpuBatchSize,
            "gpu-headroom-below-floor",
        };
    }

    if (workload.batchSize <
        config_.minimumGpuBatchSize) {
        return {
            Backend::Simd,
            GpuDecisionReason::SmallBatch,
            1.0,
            gpuScore,
            true,
            true,
            config_.minimumGpuBatchSize,
            "batch-below-gpu-threshold",
        };
    }

    if (workload.arithmeticIntensity <
        config_.minimumGpuArithmeticIntensity) {
        return {
            Backend::Simd,
            GpuDecisionReason::MemoryLimit,
            1.0,
            gpuScore,
            true,
            true,
            config_.minimumGpuBatchSize,
            "insufficient-arithmetic-intensity",
        };
    }

    if (workload.latencySensitive &&
        workload.batchSize <
            config_.minimumGpuBatchSize * 2u) {
        return {
            Backend::Simd,
            GpuDecisionReason::LatencySensitive,
            1.0,
            gpuScore,
            true,
            true,
            config_.minimumGpuBatchSize,
            "latency-sensitive-small-batch",
        };
    }

    const double confidence =
        clamp01(
            gpuScore *
            (0.70 + 0.30 * gpuHeadroom));

    if (confidence <
        config_.gpuConfidenceFloor) {
        return {
            Backend::Simd,
            GpuDecisionReason::QueuePressure,
            1.0,
            confidence,
            true,
            true,
            config_.minimumGpuBatchSize,
            "gpu-confidence-below-floor",
        };
    }

    // No fabricated benchmark claim: speedup remains 1.0 until device-specific profiling supplies
    // a measured coefficient. The policy still selects the backend using workload evidence.
    return {
        Backend::Gpu,
        GpuDecisionReason::GpuSelected,
        1.0,
        confidence,
        true,
        true,
        config_.minimumGpuBatchSize,
        "gpu-selected-by-workload-evidence",
    };
}

GpuExecutionDecision GpuExecutionBackend::choose(
    bool gpuAvailable,
    bool thermalConstrained,
    std::size_t batchSize) const noexcept {
    GpuWorkloadProfile workload{};
    workload.batchSize = batchSize;
    workload.arithmeticIntensity =
        batchSize >= config_.minimumGpuBatchSize ? 0.6 : 0.2;
    workload.memoryIntensity =
        batchSize >= config_.minimumSimdBatchSize ? 0.5 : 0.2;
    workload.gpuHeadroom = 1.0;
    workload.cpuHeadroom = 1.0;
    return choose(
        gpuAvailable,
        thermalConstrained,
        workload);
}

double GpuExecutionBackend::sanitize(double value) noexcept {
    return std::isfinite(value) ? value : 0.0;
}

double GpuExecutionBackend::clamp01(double value) noexcept {
    return std::clamp(sanitize(value), 0.0, 1.0);
}

} // namespace nox::native::runtime

#include "runtime/heterogeneous_execution_planner.h"

#include <algorithm>
#include <cmath>

namespace nox::native::runtime {

namespace {

double clamp01(double value) noexcept {
    if (!std::isfinite(value)) {
        return 0.0;
    }
    return std::clamp(value, 0.0, 1.0);
}

}  // namespace

ExecutionDecision HeterogeneousExecutionPlanner::choose(
    const ExecutionStageProfile& stage,
    const ExecutionHardwareProfile& hardware) const noexcept {
    const double memory = clamp01(hardware.memory_headroom);
    const double thermal = clamp01(hardware.thermal_headroom);
    const double gpu = clamp01(hardware.gpu_headroom);
    const double npu = clamp01(hardware.npu_headroom);
    const double cpu = clamp01(hardware.cpu_headroom);

    if (stage.optional && (memory < 0.12 || thermal < 0.12)) {
        return {
            ExecutionBackend::CPU,
            true,
            std::max(memory, thermal),
            "optional-stage-shed-under-hard-resource-pressure",
        };
    }

    if (stage.precision_sensitive) {
        if (hardware.npu_available && npu >= 0.55 && memory >= 0.35 && thermal >= 0.35) {
            return {
                ExecutionBackend::NPU,
                false,
                std::min(npu, std::min(memory, thermal)),
                "precision-sensitive-stage-routed-to-npu",
            };
        }
        if (hardware.gpu_available && gpu >= 0.55 && memory >= 0.30 && thermal >= 0.30) {
            return {
                ExecutionBackend::GPU,
                false,
                std::min(gpu, std::min(memory, thermal)),
                "precision-sensitive-stage-routed-to-gpu",
            };
        }
        if (hardware.simd_available && cpu >= 0.30 && thermal >= 0.25) {
            return {
                ExecutionBackend::SIMD,
                false,
                std::min(cpu, thermal),
                "precision-sensitive-stage-routed-to-simd",
            };
        }
    }

    if (stage.latency_sensitive && hardware.gpu_available &&
        gpu >= 0.45 && memory >= 0.25 && thermal >= 0.25) {
        return {
            ExecutionBackend::GPU,
            false,
            std::min(gpu, std::min(memory, thermal)),
            "latency-sensitive-stage-routed-to-gpu",
        };
    }

    if (hardware.npu_available && npu > cpu && npu >= 0.35 && memory >= 0.20) {
        return {
            ExecutionBackend::NPU,
            false,
            std::min(npu, memory),
            "npu-selected-by-headroom",
        };
    }

    if (hardware.simd_available && cpu >= 0.20 && thermal >= 0.20) {
        return {
            ExecutionBackend::SIMD,
            false,
            std::min(cpu, thermal),
            "simd-selected-as-deterministic-low-overhead-path",
        };
    }

    if (hardware.cpu_available) {
        return {
            ExecutionBackend::CPU,
            stage.optional && (memory < 0.20 || thermal < 0.20),
            std::min(cpu, std::min(memory, thermal)),
            "cpu-fallback-selected",
        };
    }

    return {
        ExecutionBackend::CPU,
        true,
        0.0,
        "no-supported-backend-available",
    };
}

}  // namespace nox::native::runtime

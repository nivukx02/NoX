#pragma once

#include <cstdint>
#include <string>

namespace nox::native::runtime {

enum class ExecutionBackend : std::uint8_t {
    CPU = 0,
    SIMD = 1,
    GPU = 2,
    NPU = 3,
};

struct ExecutionStageProfile {
    bool precision_sensitive{false};
    bool latency_sensitive{true};
    bool optional{false};
    double estimated_cost{1.0};
    double memory_cost_ratio{0.0};
};

struct ExecutionHardwareProfile {
    bool cpu_available{true};
    bool simd_available{true};
    bool gpu_available{false};
    bool npu_available{false};
    double cpu_headroom{1.0};
    double gpu_headroom{0.0};
    double npu_headroom{0.0};
    double memory_headroom{1.0};
    double thermal_headroom{1.0};
};

struct ExecutionDecision {
    ExecutionBackend backend{ExecutionBackend::CPU};
    bool shed_optional_work{false};
    double confidence{0.0};
    std::string reason;
};

class HeterogeneousExecutionPlanner final {
public:
    ExecutionDecision choose(
        const ExecutionStageProfile& stage,
        const ExecutionHardwareProfile& hardware) const noexcept;
};

}  // namespace nox::native::runtime

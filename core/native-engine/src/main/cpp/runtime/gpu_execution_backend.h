#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace nox::native::runtime {

enum class Backend : std::uint8_t {
    Cpu,
    Simd,
    Gpu,
};

enum class GpuDecisionReason : std::uint8_t {
    GpuSelected,
    GpuUnavailable,
    ThermalLimit,
    MemoryLimit,
    QueuePressure,
    SmallBatch,
    LatencySensitive,
    InsufficientHeadroom,
    SimdFallback,
};

struct GpuWorkloadProfile {
    std::size_t batchSize{0};
    double arithmeticIntensity{0.0};
    double memoryIntensity{0.0};
    double gpuHeadroom{1.0};
    double cpuHeadroom{1.0};
    bool latencySensitive{false};
    bool sharedBufferReady{false};
    bool deterministicRequired{false};
};

struct GpuExecutionDecision {
    Backend backend{Backend::Cpu};
    GpuDecisionReason reason{GpuDecisionReason::GpuUnavailable};
    double estimatedSpeedup{1.0};
    double confidence{0.0};
    bool available{false};
    bool cpuFallbackSafe{true};
    std::size_t minimumGpuBatchSize{64};
    std::string reasonText;
};

class GpuExecutionBackend final {
public:
    struct Config {
        std::size_t minimumGpuBatchSize{64};
        std::size_t minimumSimdBatchSize{8};
        double minimumGpuHeadroom{0.28};
        double minimumGpuArithmeticIntensity{0.35};
        double maximumLatencySensitiveGpuBatchMs{8.0};
        double gpuConfidenceFloor{0.55};
    };

    GpuExecutionBackend() noexcept;
    explicit GpuExecutionBackend(Config config) noexcept;

    [[nodiscard]] GpuExecutionDecision choose(
        bool gpuAvailable,
        bool thermalConstrained,
        const GpuWorkloadProfile& workload) const noexcept;

    [[nodiscard]] GpuExecutionDecision choose(
        bool gpuAvailable,
        bool thermalConstrained,
        std::size_t batchSize) const noexcept;

private:
    Config config_{};

    static double clamp01(double value) noexcept;
    static double sanitize(double value) noexcept;
};

} // namespace nox::native::runtime

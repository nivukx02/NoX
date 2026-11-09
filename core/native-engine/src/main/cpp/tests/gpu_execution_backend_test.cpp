#include <cassert>

#include "runtime/gpu_execution_backend.h"

using nox::native::runtime::Backend;
using nox::native::runtime::GpuDecisionReason;
using nox::native::runtime::GpuExecutionBackend;
using nox::native::runtime::GpuWorkloadProfile;

int main() {
    GpuExecutionBackend backend;

    const auto cpuFallback = backend.choose(
        false,
        false,
        GpuWorkloadProfile{
            128,
            0.80,
            0.60,
            1.0,
            0.8,
            false,
            false,
            false,
        });
    assert(cpuFallback.backend == Backend::Simd);
    assert(cpuFallback.reason == GpuDecisionReason::GpuUnavailable);
    assert(cpuFallback.cpuFallbackSafe);
    assert(cpuFallback.estimatedSpeedup == 1.0);

    const auto thermalFallback = backend.choose(
        true,
        true,
        GpuWorkloadProfile{
            128,
            0.90,
            0.70,
            0.95,
            0.8,
            false,
            true,
            false,
        });
    assert(thermalFallback.backend == Backend::Simd);
    assert(thermalFallback.reason == GpuDecisionReason::ThermalLimit);

    const auto smallBatch = backend.choose(
        true,
        false,
        GpuWorkloadProfile{
            16,
            0.90,
            0.70,
            0.95,
            0.8,
            false,
            true,
            false,
        });
    assert(smallBatch.backend == Backend::Simd);
    assert(smallBatch.reason == GpuDecisionReason::SmallBatch);

    const auto selected = backend.choose(
        true,
        false,
        GpuWorkloadProfile{
            256,
            0.90,
            0.80,
            0.95,
            0.80,
            false,
            true,
            false,
        });
    assert(selected.backend == Backend::Gpu);
    assert(selected.confidence > 0.55);
    assert(selected.cpuFallbackSafe);
    assert(selected.reason == GpuDecisionReason::GpuSelected);

    return 0;
}

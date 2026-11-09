#include <cassert>
#include "runtime/thermal_sensor_drift.h"
#include "capture/capture_settings_optimizer.h"
#include "runtime/gpu_execution_backend.h"
#include "math/simd_dispatch.h"

int main() {
    const auto thermal=nox::native::runtime::ThermalSensorDrift{}.correct(45,25,0.002);
    assert(thermal.correctedTemperature>40.0);
    assert(thermal.confidence>0.9);

    const auto settings=nox::native::capture::CaptureSettingsOptimizer{}.optimize(1.0,0.8,0.9);
    assert(settings.frameRate>=30.0);
    assert(settings.exposureMs>0.0);

    const auto gpu=nox::native::runtime::GpuExecutionBackend{}.choose(true,false,64);
    assert(gpu.backend==nox::native::runtime::Backend::Gpu);
    assert(gpu.available);

    const auto simd=nox::native::math::detectSimd();
    assert(simd==nox::native::math::SimdLevel::Scalar ||
           simd==nox::native::math::SimdLevel::SSE ||
           simd==nox::native::math::SimdLevel::AVX ||
           simd==nox::native::math::SimdLevel::NEON);
    return 0;
}

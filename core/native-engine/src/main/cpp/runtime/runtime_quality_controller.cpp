#include "runtime/runtime_quality_controller.h"

#include <algorithm>
#include <cmath>

namespace nox::native::runtime {

RuntimeQualityController::RuntimeQualityController(
    RuntimeQualityConfig config) noexcept
    : config_(config) {
    config_.minimumQualityScale =
        std::clamp(sanitize(config_.minimumQualityScale), 0.05, 1.0);
    config_.thermalFloor =
        std::clamp(sanitize(config_.thermalFloor), 0.0, 1.0);
    config_.memoryFloor =
        std::clamp(sanitize(config_.memoryFloor), 0.0, 1.0);
    config_.gpuFloor =
        std::clamp(sanitize(config_.gpuFloor), 0.0, 1.0);
    config_.npuFloor =
        std::clamp(sanitize(config_.npuFloor), 0.0, 1.0);
    config_.queueShedThreshold =
        std::clamp(sanitize(config_.queueShedThreshold), 0.0, 1.0);
    config_.frameBudgetShedThreshold =
        std::max(sanitize(config_.frameBudgetShedThreshold), 1.0);
    config_.droppedFrameShedThreshold =
        std::clamp(sanitize(config_.droppedFrameShedThreshold), 0.0, 1.0);
}

RuntimeQualityDecision RuntimeQualityController::evaluate(
    const RuntimeQualityInput& raw,
    WorkClass workClass) const noexcept {
    const RuntimeQualityInput input{
        clamp01(raw.thermalHeadroom),
        clamp01(raw.memoryHeadroom),
        clamp01(raw.gpuHeadroom),
        clamp01(raw.npuHeadroom),
        raw.batterySaver,
        clamp01(raw.queueUtilization),
        std::max(sanitize(raw.frameBudgetUtilization), 0.0),
        clamp01(raw.droppedFrameRatio),
    };

    const double thermalPressure =
        std::max(0.0, config_.thermalFloor - input.thermalHeadroom) /
        std::max(config_.thermalFloor, 1e-9);
    const double memoryPressure =
        std::max(0.0, config_.memoryFloor - input.memoryHeadroom) /
        std::max(config_.memoryFloor, 1e-9);
    const double gpuPressure =
        std::max(0.0, config_.gpuFloor - input.gpuHeadroom) /
        std::max(config_.gpuFloor, 1e-9);
    const double npuPressure =
        std::max(0.0, config_.npuFloor - input.npuHeadroom) /
        std::max(config_.npuFloor, 1e-9);
    const double queuePressure =
        std::max(0.0, input.queueUtilization - 0.55) / 0.45;
    const double framePressure =
        std::max(0.0, input.frameBudgetUtilization - 0.75) / 1.25;
    const double droppedPressure =
        input.droppedFrameRatio / std::max(config_.droppedFrameShedThreshold, 1e-9);

    double pressure =
        thermalPressure * 0.24 +
        memoryPressure * 0.16 +
        gpuPressure * 0.13 +
        npuPressure * 0.09 +
        queuePressure * 0.14 +
        framePressure * 0.12 +
        droppedPressure * 0.07 +
        (input.batterySaver ? 0.05 : 0.0);

    pressure = clamp01(pressure);

    double qualityScale =
        1.0 - pressure * (1.0 - config_.minimumQualityScale);

    if (input.batterySaver) {
        qualityScale *= 0.92;
    }

    if (workClass == WorkClass::Optional) {
        qualityScale *= 0.82;
    } else if (workClass == WorkClass::Critical) {
        qualityScale = std::max(qualityScale, config_.minimumQualityScale * 0.80);
    }

    qualityScale = std::clamp(
        qualityScale,
        config_.minimumQualityScale,
        1.0);

    const bool thermalConstrained =
        input.thermalHeadroom < config_.thermalFloor;
    const bool memoryConstrained =
        input.memoryHeadroom < config_.memoryFloor;
    const bool shedOptional =
        thermalConstrained ||
        memoryConstrained ||
        input.queueUtilization >= config_.queueShedThreshold ||
        input.frameBudgetUtilization >= config_.frameBudgetShedThreshold ||
        input.droppedFrameRatio >= config_.droppedFrameShedThreshold ||
        pressure >= 0.72 ||
        (input.batterySaver && pressure >= 0.38);

    const bool allowGpu =
        !input.batterySaver &&
        !thermalConstrained &&
        input.gpuHeadroom >= config_.gpuFloor &&
        input.frameBudgetUtilization < 1.0 &&
        input.queueUtilization < config_.queueShedThreshold;

    const double queueCapacityScale =
        std::clamp(
            1.0 - pressure * 0.55,
            0.35,
            1.0);

    const bool acceptCritical =
        workClass == WorkClass::Critical ||
        !memoryConstrained;

    return {
        qualityScale,
        pressure,
        queueCapacityScale,
        allowGpu,
        shedOptional,
        thermalConstrained,
        memoryConstrained,
        acceptCritical,
    };
}

double RuntimeQualityController::sanitize(double value) noexcept {
    return std::isfinite(value) ? value : 0.0;
}

double RuntimeQualityController::clamp01(double value) noexcept {
    return std::clamp(sanitize(value), 0.0, 1.0);
}

} // namespace nox::native::runtime

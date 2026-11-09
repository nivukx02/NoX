#include <cassert>
#include "runtime/runtime_quality_controller.h"

int main() {
    using namespace nox::native::runtime;

    RuntimeQualityController controller;

    const auto nominal = controller.evaluate(RuntimeQualityInput{
        .thermalHeadroom = 1.0,
        .memoryHeadroom = 1.0,
        .gpuHeadroom = 1.0,
        .npuHeadroom = 1.0,
        .batterySaver = false,
        .queueUtilization = 0.10,
        .frameBudgetUtilization = 0.70,
        .droppedFrameRatio = 0.0,
    });
    assert(nominal.qualityScale > 0.95);
    assert(nominal.allowGpu);
    assert(!nominal.shedOptionalWork);

    const auto constrained = controller.evaluate(RuntimeQualityInput{
        .thermalHeadroom = 0.12,
        .memoryHeadroom = 0.30,
        .gpuHeadroom = 0.20,
        .npuHeadroom = 0.25,
        .batterySaver = true,
        .queueUtilization = 0.96,
        .frameBudgetUtilization = 1.40,
        .droppedFrameRatio = 0.18,
    });
    assert(constrained.qualityScale < 0.70);
    assert(!constrained.allowGpu);
    assert(constrained.shedOptionalWork);
    assert(constrained.thermalConstrained);

    const auto critical = controller.evaluate(RuntimeQualityInput{
        .thermalHeadroom = 0.05,
        .memoryHeadroom = 0.05,
        .gpuHeadroom = 0.0,
        .npuHeadroom = 0.0,
        .batterySaver = true,
        .queueUtilization = 1.0,
        .frameBudgetUtilization = 2.0,
        .droppedFrameRatio = 1.0,
    }, WorkClass::Critical);
    assert(critical.acceptCriticalWork);
    assert(critical.qualityScale > 0.0);

    const auto optional = controller.evaluate(RuntimeQualityInput{
        .thermalHeadroom = 0.05,
        .memoryHeadroom = 0.05,
        .gpuHeadroom = 0.0,
        .npuHeadroom = 0.0,
        .batterySaver = true,
        .queueUtilization = 1.0,
        .frameBudgetUtilization = 2.0,
        .droppedFrameRatio = 1.0,
    }, WorkClass::Optional);
    assert(optional.shedOptionalWork);
    return 0;
}

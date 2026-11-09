#include <cassert>

#include "scene/adaptive_tsdf_detail_controller.h"

int main() {
    nox::native::scene::AdaptiveTsdfDetailController controller;

    const auto fine = controller.evaluate({
        0.08,
        0.10,
        0.96,
        0.95,
        0.95,
        0.96,
        0.10,
    });
    assert(fine.pixelStride == 1);
    assert(fine.preserveFineDetail);
    assert(!fine.shedDetail);
    assert(fine.detailLevel > 0.78);

    const auto pressured = controller.evaluate({
        0.92,
        0.72,
        0.30,
        0.35,
        0.20,
        0.18,
        0.22,
    });
    assert(pressured.pixelStride >= 3);
    assert(pressured.shedDetail);
    assert(pressured.observationWeightScale < 0.80);

    const auto invalid = controller.evaluate({
        2.0,
        -1.0,
        1.5,
        4.0,
        -5.0,
        1.0,
        5.0,
    });
    assert(invalid.pixelStride >= 1);
    assert(invalid.detailLevel >= 0.0);
    assert(invalid.detailLevel <= 1.0);

    const temporallyWeak =
        controller.evaluate({
            0.08,
            0.10,
            0.96,
            0.95,
            0.95,
            0.96,
            0.10,
            0.20,
            0.25,
            0.90,
            0.90,
            0.15,
        });
    assert(temporallyWeak.pixelStride >= fine.pixelStride);
    assert(temporallyWeak.reason != nullptr);

    const stableDynamic =
        controller.evaluate({
            0.22,
            0.18,
            0.94,
            0.92,
            0.90,
            0.93,
            0.24,
            0.08,
            0.94,
            0.95,
            0.90,
            0.72,
        });
    assert(stableDynamic.detailLevel > 0.45);

    const depthUncertain =
        controller.evaluate({
            0.08,
            0.10,
            0.96,
            0.95,
            0.95,
            0.96,
            0.10,
            0.95,
            0.90,
            0.90,
            0.90,
            0.10,
        });
    assert(depthUncertain.reason != nullptr);
    assert(
        depthUncertain.detailLevel <
            fine.detailLevel);
    assert(depthUncertain.observationWeightScale <= fine.observationWeightScale);

    const auto memoryBound = controller.evaluate({
        0.20,
        0.12,
        0.95,
        0.95,
        0.95,
        0.95,
        0.94,
    });
    assert(memoryBound.memoryBound);
    assert(memoryBound.shedDetail);
    assert(memoryBound.pixelStride == 4);
    assert(memoryBound.reason != nullptr);
    return 0;
}

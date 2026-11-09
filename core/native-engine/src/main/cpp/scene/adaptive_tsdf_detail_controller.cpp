#include "scene/adaptive_tsdf_detail_controller.h"

#include <algorithm>
#include <cmath>

namespace nox::native::scene {

TsdfDetailDecision AdaptiveTsdfDetailController::evaluate(
    const TsdfDetailEvidence& raw) const noexcept {
    const TsdfDetailEvidence input{
        clamp01(raw.motionEnergy),
        clamp01(raw.uncertainty),
        clamp01(raw.cameraCoverage),
        clamp01(raw.reconstructionConfidence),
        clamp01(raw.thermalHeadroom),
        clamp01(raw.gpuHeadroom),
        clamp01(raw.voxelUtilization),
        clamp01(raw.depthUncertaintyMeters / 0.25),
        clamp01(raw.temporalConsistency),
        clamp01(raw.appearanceConsistency),
        clamp01(raw.contactConsistency),
        clamp01(raw.dynamicProbability),
    };

    const double pressure =
        input.motionEnergy * 0.22 +
        input.uncertainty * 0.21 +
        (1.0 - input.thermalHeadroom) * 0.16 +
        (1.0 - input.gpuHeadroom) * 0.14 +
        (1.0 - input.reconstructionConfidence) * 0.08 +
        (1.0 - input.cameraCoverage) * 0.04 +
        input.voxelUtilization * 0.15 +
        (1.0 - input.temporalConsistency) * 0.07 +
        (1.0 - input.appearanceConsistency) * 0.06 +
        (1.0 - input.contactConsistency) * 0.02 +
        input.dynamicProbability * 0.05 +
        input.depthUncertaintyMeters * 0.08;

    const double demand =
        input.cameraCoverage * 0.20 +
        input.reconstructionConfidence * 0.20 +
        input.gpuHeadroom * 0.18 +
        input.thermalHeadroom * 0.14 +
        (1.0 - input.motionEnergy) * 0.05 +
        (1.0 - input.uncertainty) * 0.04 +
        (1.0 - input.voxelUtilization) * 0.05 +
        input.temporalConsistency * 0.05 +
        input.appearanceConsistency * 0.04 +
        input.contactConsistency * 0.01 +
        input.dynamicProbability * 0.04 +
        (1.0 - input.depthUncertaintyMeters) * 0.04;

    const double detailLevel = std::clamp(
        demand * (1.0 - pressure * 0.72),
        0.0,
        1.0);

    const bool preserveFine = detailLevel >= 0.78 && pressure < 0.28;
    const bool memoryBound =
        input.voxelUtilization >= 0.90;
    const bool shed =
        detailLevel < 0.42 ||
        pressure >= 0.68 ||
        memoryBound;

    std::uint32_t stride = 4U;
    if(memoryBound) {
        stride = 4U;
    } else if (detailLevel >= 0.78) {
        stride = 1U;
    } else if (detailLevel >= 0.60) {
        stride = 2U;
    } else if (detailLevel >= 0.42) {
        stride = 3U;
    }

    const double weightScale = std::clamp(
        0.58 + detailLevel * 0.42,
        0.50,
        1.0);

    const char* reason = "balanced";
    if (preserveFine) {
        reason = "fine_detail";
    } else if (shed) {
        reason = "compute_pressure";
    } else if (input.motionEnergy >= 0.78) {
        reason = "fast_motion";
    } else if (input.depthUncertaintyMeters >= 0.65) {
        reason = "depth_uncertainty_limited";
    } else if (input.uncertainty >= 0.65) {
        reason = "uncertainty_limited";
    } else if (input.temporalConsistency < 0.45) {
        reason = "temporal_consistency_limited";
    } else if (input.appearanceConsistency < 0.45) {
        reason = "appearance_consistency_limited";
    } else if (input.cameraCoverage < 0.45) {
        reason = "coverage_limited";
    } else if(memoryBound) {
        reason = "memory_bound";
    }

    return {
        stride,
        weightScale,
        pressure,
        detailLevel,
        preserveFine,
        shed,
        memoryBound,
        reason,
    };
}

double AdaptiveTsdfDetailController::clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

} // namespace nox::native::scene

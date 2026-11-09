#pragma once

#include <cstdint>

namespace nox::native::scene {

struct TsdfDetailEvidence {
    double motionEnergy{0.0};
    double uncertainty{0.0};
    double cameraCoverage{0.0};
    double reconstructionConfidence{0.0};
    double thermalHeadroom{1.0};
    double gpuHeadroom{1.0};
    double voxelUtilization{0.0};
    double depthUncertaintyMeters{0.0};
    double temporalConsistency{1.0};
    double appearanceConsistency{1.0};
    double contactConsistency{1.0};
    double dynamicProbability{0.0};
};

struct TsdfDetailDecision {
    std::uint32_t pixelStride{2};
    double observationWeightScale{1.0};
    double computePressure{0.0};
    double detailLevel{0.5};
    bool preserveFineDetail{false};
    bool shedDetail{false};
    bool memoryBound{false};
    const char* reason{"balanced"};
};

class AdaptiveTsdfDetailController final {
public:
    TsdfDetailDecision evaluate(
        const TsdfDetailEvidence& evidence) const noexcept;

private:
    static double clamp01(double value) noexcept;
};

} // namespace nox::native::scene

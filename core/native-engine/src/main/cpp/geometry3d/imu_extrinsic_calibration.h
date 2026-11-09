#pragma once

#include <cstddef>
#include <vector>

#include "geometry3d/imu_state.h"

namespace nox::native::imu {

struct ExtrinsicSample {
    Quaternion vision{};
    Quaternion imu{};
    double weight{1.0};
    double synchronizationConfidence{1.0};
};

struct ExtrinsicResult {
    Quaternion rotation{};
    double residual{0.0};
    double p95Residual{0.0};
    double confidence{0.0};
    double sampleConsensus{0.0};
    std::size_t acceptedSamples{0};
    std::size_t rejectedSamples{0};
    bool valid{false};
};

class ImuExtrinsicCalibration final {
public:
    [[nodiscard]] ExtrinsicResult solve(
        const Quaternion& vision,
        const Quaternion& imu) const noexcept;

    [[nodiscard]] ExtrinsicResult solveBatch(
        const std::vector<ExtrinsicSample>&,
        double maximumResidualRadians = 0.35) const noexcept;

private:
    static Quaternion inverse(
        const Quaternion&) noexcept;

    static Quaternion normalizeSafe(
        const Quaternion&) noexcept;

    static double angularResidual(
        const Quaternion&,
        const Quaternion&) noexcept;
};

} // namespace nox::native::imu

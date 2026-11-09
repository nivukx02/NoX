#pragma once

#include <cstdint>

namespace nox::native::imu {

struct BiasEstimate {
    double gx{0.0};
    double gy{0.0};
    double gz{0.0};
    double ax{0.0};
    double ay{0.0};
    double az{0.0};
    double gyroVariance{0.0};
    double accelVariance{0.0};
    double innovationRms{0.0};
    double stationaryConfidence{0.0};
    double stability{0.0};
    double confidence{0.0};
    std::uint64_t sampleCount{0};
};

class ImuBiasEstimator final {
public:
    [[nodiscard]] BiasEstimate update(
        double gx,
        double gy,
        double gz,
        double ax,
        double ay,
        double az,
        double quality) noexcept;

    [[nodiscard]] BiasEstimate updateStationary(
        double gx,
        double gy,
        double gz,
        double ax,
        double ay,
        double az,
        double stationaryConfidence,
        double dtSeconds) noexcept;

    [[nodiscard]] BiasEstimate current() const noexcept {
        return state_;
    }

private:
    BiasEstimate state_{};

    static double clamp01(
        double value) noexcept;

    static double robustWeight(
        double residual,
        double scale) noexcept;
};

} // namespace nox::native::imu

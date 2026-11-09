#pragma once

#include <cstdint>

namespace nox::native::calibration {

struct DistortedPixel {
    double x{0.0};
    double y{0.0};
};

struct BrownConradyCoefficients {
    double k1{0.0};
    double k2{0.0};
    double p1{0.0};
    double p2{0.0};
    double k3{0.0};
};

class BrownConradyDistortion final {
public:
    static DistortedPixel distort(
        double normalized_x,
        double normalized_y,
        const BrownConradyCoefficients& coefficients) noexcept;

    static DistortedPixel undistort(
        double normalized_x,
        double normalized_y,
        const BrownConradyCoefficients& coefficients,
        std::uint32_t iterations = 8u) noexcept;
};

} // namespace nox::native::calibration

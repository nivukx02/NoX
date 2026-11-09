#pragma once

#include <cstdint>

namespace nox::native::calibration {

struct LensParameters {
    double k1{0.0};
    double k2{0.0};
    double k3{0.0};
    double p1{0.0};
    double p2{0.0};
    double fx{1.0};
    double fy{1.0};
    double cx{0.0};
    double cy{0.0};
    std::uint32_t width{0};
    std::uint32_t height{0};
    double skew{0.0};
};

struct LensJacobian {
    double dxx{1.0};
    double dxy{0.0};
    double dyx{0.0};
    double dyy{1.0};
    double determinant{1.0};
    double conditioning{1.0};
};

struct LensProjectionResult {
    double x{0.0};
    double y{0.0};
    double normalizedX{0.0};
    double normalizedY{0.0};
    double radial{1.0};
    LensJacobian jacobian{};
    bool valid{false};
};

class LensModel final {
public:
    [[nodiscard]] double radialScale(
        double r2,
        const LensParameters&) const noexcept;

    [[nodiscard]] LensProjectionResult distortNormalized(
        double x,
        double y,
        const LensParameters&) const noexcept;

    [[nodiscard]] LensProjectionResult project(
        double x,
        double y,
        double z,
        const LensParameters&) const noexcept;

    [[nodiscard]] bool undistortNormalized(
        double xd,
        double yd,
        const LensParameters&,
        double& x,
        double& y,
        std::uint32_t iterations = 10U) const noexcept;

    [[nodiscard]] bool isPlausible(
        const LensParameters&) const noexcept;

private:
    static LensJacobian jacobian(
        double x,
        double y,
        const LensParameters&) noexcept;
};

} // namespace nox::native::calibration

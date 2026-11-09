#pragma once

namespace nox::native::math {

enum class RobustLossKind {
    Huber,
    Cauchy,
    Tukey,
};

class RobustLoss final {
public:
    static double weight(
        RobustLossKind kind,
        double residual,
        double scale) noexcept;

    static double cost(
        RobustLossKind kind,
        double residual,
        double scale) noexcept;

    static double adaptiveWeight(
        double residual,
        double scale,
        double trustedFraction = 0.75) noexcept;

    static double adaptiveCost(
        double residual,
        double scale,
        double trustedFraction = 0.75) noexcept;
};

} // namespace nox::native::math

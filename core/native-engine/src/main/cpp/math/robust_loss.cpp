#include "math/robust_loss.h"

#include <algorithm>
#include <cmath>

namespace nox::native::math {

double RobustLoss::weight(
    RobustLossKind kind,
    double residual,
    double scale) noexcept {
    const double normalized =
        std::abs(residual) / std::max(scale, 1e-12);
    switch (kind) {
        case RobustLossKind::Huber:
            return normalized <= 1.0 ? 1.0 : 1.0 / normalized;
        case RobustLossKind::Cauchy:
            return 1.0 / (1.0 + normalized * normalized);
        case RobustLossKind::Tukey:
            if (normalized >= 1.0) return 0.0;
            const double inner = 1.0 - normalized * normalized;
            return inner * inner;
    }
    return 1.0;
}

double RobustLoss::cost(
    RobustLossKind kind,
    double residual,
    double scale) noexcept {
    const double normalized =
        std::abs(residual) / std::max(scale, 1e-12);
    const double s2 = std::max(scale, 1e-12) * std::max(scale, 1e-12);
    switch (kind) {
        case RobustLossKind::Huber:
            return normalized <= 1.0
                ? 0.5 * residual * residual
                : s2 * (normalized - 0.5);
        case RobustLossKind::Cauchy:
            return 0.5 * s2 * std::log1p(normalized * normalized);
        case RobustLossKind::Tukey:
            if (normalized >= 1.0) return s2 / 6.0;
            const double inner = 1.0 - normalized * normalized;
            return s2 * (1.0 - inner * inner * inner) / 6.0;
    }
    return 0.0;
}

} // namespace nox::native::math


double nox::native::math::RobustLoss::adaptiveWeight(
    double residual,
    double scale,
    double trustedFraction) noexcept {
    if (!std::isfinite(residual) || residual < 0.0) return 0.0;

    const double safeScale = std::max(scale, 1e-12);
    const double fraction = std::clamp(trustedFraction, 0.0, 1.0);
    const double huber = weight(
        RobustLossKind::Huber,
        residual,
        safeScale);
    const double cauchy = weight(
        RobustLossKind::Cauchy,
        residual,
        safeScale * 1.50);
    const double tukey = weight(
        RobustLossKind::Tukey,
        residual,
        safeScale * 2.75);

    // Preserve a broad basin early in optimization, then strengthen the
    // redescending tail as correspondence confidence increases.
    const double early =
        0.52 * huber +
        0.33 * cauchy +
        0.15 * tukey;
    const double late =
        0.26 * huber +
        0.34 * cauchy +
        0.40 * tukey;

    return std::clamp(
        early * (1.0 - fraction) +
            late * fraction,
        0.0,
        1.0);
}

double nox::native::math::RobustLoss::adaptiveCost(
    double residual,
    double scale,
    double trustedFraction) noexcept {
    if (!std::isfinite(residual) || residual < 0.0) return 0.0;

    const double safeScale = std::max(scale, 1e-12);
    const double fraction = std::clamp(trustedFraction, 0.0, 1.0);

    const double huber = cost(
        RobustLossKind::Huber,
        residual,
        safeScale);
    const double cauchy = cost(
        RobustLossKind::Cauchy,
        residual,
        safeScale * 1.50);
    const double tukey = cost(
        RobustLossKind::Tukey,
        residual,
        safeScale * 2.75);

    const double early =
        0.52 * huber +
        0.33 * cauchy +
        0.15 * tukey;
    const double late =
        0.26 * huber +
        0.34 * cauchy +
        0.40 * tukey;

    return std::max(
        0.0,
        early * (1.0 - fraction) +
            late * fraction);
}

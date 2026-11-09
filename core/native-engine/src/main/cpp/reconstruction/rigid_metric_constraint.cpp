#include "reconstruction/rigid_metric_constraint.h"
#include <algorithm>
#include <cmath>

namespace nox::native::metric {
MetricConstraintResult RigidMetricConstraint::solve(
    const std::vector<PairMeasurement>& measurements) const {
    MetricConstraintResult output;
    double weightSum = 0.0;
    double scaleSum = 0.0;

    for (const auto& measurement : measurements) {
        if (measurement.observed <= 1e-9 ||
            measurement.metric <= 0.0 ||
            measurement.weight <= 0.0) {
            continue;
        }
        const double weight =
            measurement.weight * measurement.observed * measurement.observed;
        weightSum += weight;
        scaleSum += weight * (measurement.metric / measurement.observed);
    }

    if (weightSum <= 0.0) return output;
    output.scale = scaleSum / weightSum;

    double squared = 0.0;
    std::size_t used = 0;
    for (const auto& measurement : measurements) {
        if (measurement.observed <= 1e-9 ||
            measurement.metric <= 0.0 ||
            measurement.weight <= 0.0) {
            continue;
        }
        const double residual =
            output.scale * measurement.observed - measurement.metric;
        squared += measurement.weight * residual * residual;
        ++used;
    }

    output.residual = used == 0
        ? 1e9
        : std::sqrt(squared / static_cast<double>(used));
    output.uncertainty = output.residual /
        std::sqrt(static_cast<double>(std::max<std::size_t>(used, 1)));
    output.confidence = std::clamp(
        std::exp(-output.residual / std::max(0.01, output.scale * 0.02)),
        0.0,
        1.0);
    output.valid = output.confidence >= 0.55;
    return output;
}
}

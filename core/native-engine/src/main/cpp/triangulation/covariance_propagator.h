#pragma once

#include <array>

#include "triangulation/weighted_triangulator.h"

namespace nox::native::triangulation {

struct PropagatedCovariance {
    std::array<double, 9> matrix{
        1.0,0.0,0.0,
        0.0,1.0,0.0,
        0.0,0.0,1.0,
    };
    double trace{3.0};
    double determinant{1.0};
    double scalarVariance{1.0};
    double minimumVariance{1.0};
    double maximumVariance{1.0};
    double anisotropy{1.0};
    double conditionNumber{1.0};
    double p95Radius{3.0};
    double p99Radius{3.0};
    double confidence{0.0};
    double numericalRepairMagnitude{0.0};
    bool positiveDefinite{false};
    bool numericallyRepaired{false};
};

class CovariancePropagator final {
public:
    static double scalarVariance(
        const TriangulationResult&,
        double observationSigma) noexcept;

    static PropagatedCovariance propagate(
        const TriangulationResult&,
        double observationSigma) noexcept;

private:
    static bool eigenvaluesSymmetric(
        const std::array<double, 9>& matrix,
        std::array<double, 3>& eigenvalues) noexcept;
};

} // namespace nox::native::triangulation

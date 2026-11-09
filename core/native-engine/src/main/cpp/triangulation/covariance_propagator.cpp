#include "triangulation/covariance_propagator.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nox::native::triangulation {

namespace {

double determinant(
    const std::array<double, 9>& matrix) noexcept {
    return
        matrix[0] *
            (
                matrix[4] * matrix[8] -
                matrix[5] * matrix[7]
            ) -
        matrix[1] *
            (
                matrix[3] * matrix[8] -
                matrix[5] * matrix[6]
            ) +
        matrix[2] *
            (
                matrix[3] * matrix[7] -
                matrix[4] * matrix[6]
            );
}

std::array<double, 9> symmetrize(
    const std::array<double, 9>& matrix) noexcept {
    return {
        matrix[0],
        0.5 * (matrix[1] + matrix[3]),
        0.5 * (matrix[2] + matrix[6]),
        0.5 * (matrix[3] + matrix[1]),
        matrix[4],
        0.5 * (matrix[5] + matrix[7]),
        0.5 * (matrix[6] + matrix[2]),
        0.5 * (matrix[7] + matrix[5]),
        matrix[8],
    };
}

} // namespace

bool CovariancePropagator::eigenvaluesSymmetric(
    const std::array<double, 9>& input,
    std::array<double, 3>& eigenvalues) noexcept {
    const auto matrix =
        symmetrize(input);

    double a = matrix[0];
    double b = matrix[1];
    double c = matrix[2];
    double d = matrix[4];
    double e = matrix[5];
    double f = matrix[8];

    const double p1 =
        b * b +
        c * c +
        e * e;

    if (p1 < 1e-24) {
        eigenvalues = {
            a,
            d,
            f,
        };
        return true;
    }

    const double q =
        (a + d + f) / 3.0;

    const double pa =
        a - q;
    const double pd =
        d - q;
    const double pf =
        f - q;

    const double p2 =
        pa * pa +
        pd * pd +
        pf * pf +
        2.0 * p1;

    const double p =
        std::sqrt(
            std::max(
                0.0,
                p2 / 6.0));

    if (!std::isfinite(p) || p < 1e-15) {
        eigenvalues = {
            q, q, q,
        };
        return true;
    }

    const std::array<double, 9> B{
        pa / p, b / p, c / p,
        b / p, pd / p, e / p,
        c / p, e / p, pf / p,
    };

    const double detB =
        determinant(B);
    const double r =
        std::clamp(
            detB / 2.0,
            -1.0,
            1.0);

    constexpr double kPi =
        3.14159265358979323846;
    const double phi =
        std::acos(r) / 3.0;

    eigenvalues[0] =
        q +
        2.0 * p *
        std::cos(phi);
    eigenvalues[2] =
        q +
        2.0 * p *
        std::cos(
            phi +
            2.0 * kPi / 3.0);
    eigenvalues[1] =
        3.0 * q -
        eigenvalues[0] -
        eigenvalues[2];

    return
        std::isfinite(eigenvalues[0]) &&
        std::isfinite(eigenvalues[1]) &&
        std::isfinite(eigenvalues[2]);
}

double CovariancePropagator::scalarVariance(
    const TriangulationResult& result,
    double observationSigma) noexcept {
    return std::max(
        1e-12,
        result.covariance *
            observationSigma *
            observationSigma);
}

PropagatedCovariance CovariancePropagator::propagate(
    const TriangulationResult& result,
    double observationSigma) noexcept {
    const double observationVariance =
        std::max(
            1e-12,
            observationSigma *
                observationSigma);

    PropagatedCovariance output;

    for (std::size_t index = 0;
         index < 9;
         ++index) {
        output.matrix[index] =
            result.covarianceMatrix[index] *
            observationVariance;
    }

    output.matrix =
        symmetrize(
            output.matrix);

    // Numerical covariance repair is deliberately conservative:
    // small negative eigenvalues caused by floating-point noise are floored,
    // while materially indefinite matrices remain unqualified.
    {
        std::array<double, 3> rawEigenvalues{};
        if (eigenvaluesSymmetric(output.matrix, rawEigenvalues)) {
            const double scale =
                std::max(
                    1e-12,
                    std::abs(output.trace) / 3.0);
            double repairMagnitude = 0.0;
            bool hasMaterialNegative = false;
            for (double eigenvalue : rawEigenvalues) {
                if (eigenvalue < -scale * 1e-6) {
                    hasMaterialNegative = true;
                    break;
                }
            }

            if (!hasMaterialNegative) {
                std::array<double, 9> repaired = output.matrix;
                // Preserve the diagonal-dominant matrix when the only issue
                // is sub-ULP/rounding-level negative curvature.
                const double floor = scale * 1e-10;
                for (int index = 0; index < 3; ++index) {
                    if (rawEigenvalues[index] < floor) {
                        repairMagnitude +=
                            floor - rawEigenvalues[index];
                    }
                }
                if (repairMagnitude > 0.0) {
                    for (int index = 0; index < 3; ++index) {
                        repaired[index * 3 + index] +=
                            repairMagnitude / 3.0;
                    }
                    output.matrix =
                        symmetrize(repaired);
                    output.numericallyRepaired = true;
                    output.numericalRepairMagnitude =
                        repairMagnitude;
                }
            }
        }
    }

    output.trace =
        output.matrix[0] +
        output.matrix[4] +
        output.matrix[8];

    output.determinant =
        determinant(
            output.matrix);

    output.scalarVariance =
        output.trace / 3.0;

    std::array<double, 3> eigenvalues{};
    if (
        eigenvaluesSymmetric(
            output.matrix,
            eigenvalues)) {
        std::sort(
            eigenvalues.begin(),
            eigenvalues.end());

        output.minimumVariance =
            eigenvalues[0];
        output.maximumVariance =
            eigenvalues[2];

        const double safeMinimum =
            std::max(
                std::abs(
                    output.minimumVariance),
                1e-12);
        const double safeMaximum =
            std::max(
                std::abs(
                    output.maximumVariance),
                safeMinimum);

        output.conditionNumber =
            safeMaximum /
            safeMinimum;

        output.anisotropy =
            safeMaximum <= 1e-12
                ? 0.0
                : std::clamp(
                    1.0 -
                        safeMinimum /
                            safeMaximum,
                    0.0,
                    1.0);

        output.positiveDefinite =
            output.minimumVariance >
                0.0 &&
            output.maximumVariance >
                0.0;

        const double radius95 =
            std::sqrt(
                safeMaximum) *
            1.96;

        const double radius99 =
            std::sqrt(
                safeMaximum) *
            2.576;

        output.p95Radius =
            radius95;
        output.p99Radius =
            radius99;

        const double conditioningConfidence =
            std::clamp(
                1.0 /
                    (
                        1.0 +
                        std::max(
                            0.0,
                            output.conditionNumber - 1.0) /
                            25.0
                    ),
                0.0,
                1.0);

        const double volumeConfidence =
            std::clamp(
                output.determinant <= 0.0
                    ? 0.0
                    : 1.0 /
                        (
                            1.0 +
                            output.scalarVariance
                        ),
                0.0,
                1.0);

        output.confidence =
            std::clamp(
                conditioningConfidence * 0.60 +
                volumeConfidence * 0.40,
                0.0,
                1.0);
    } else {
        output.minimumVariance =
            output.scalarVariance;
        output.maximumVariance =
            output.scalarVariance;
        output.conditionNumber =
            std::numeric_limits<double>::infinity();
        output.anisotropy = 1.0;
        output.p95Radius =
            std::sqrt(
                std::max(
                    0.0,
                    output.trace)) *
            1.96;
        output.p99Radius =
            std::sqrt(
                std::max(
                    0.0,
                    output.trace)) *
            2.576;
        output.confidence = 0.0;
    }

    return output;
}

} // namespace nox::native::triangulation

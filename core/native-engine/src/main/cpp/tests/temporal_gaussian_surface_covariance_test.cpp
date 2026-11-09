#include <cassert>
#include <cmath>

#include "scene/temporal_gaussian_surface_state.h"

int main() {
    using namespace nox::native::scene;

    TemporalGaussianSurfaceState state({
        .cellSizeMeters = 0.05,
        .maximumAssociationMahalanobis = 16.0,
        .maximumSpatialResidualMeters = 0.20,
        .minimumObservationConfidence = 0.01,
        .dynamicSuppression = 0.0,
        .temporalHalfLifeSeconds = 1.0,
        .velocityDecay = 0.90,
        .confidenceDecay = 0.95,
        .minimumVarianceMeters2 = 1e-7,
        .maximumVarianceMeters2 = 0.25,
        .innovationVarianceInflation = 1.0,
        .maximumPrimitives = 128,
    });

    assert(
        state.integrate({
            {0.0, 0.0, 2.0},
            {0, 0, 1},
            {0, 0, 0},
            {0.001, 0.001, 0.001},
            1.0,
            0.0,
            1.0,
            1'000'000'000ULL,
        }));

    for (int i = 1; i <= 12; ++i) {
        const double d = 0.008 * static_cast<double>(i);
        assert(
            state.integrate({
                {d, d, 2.0 + d},
                {0, 0, 1},
                {0, 0, 0},
                {0.0005, 0.0005, 0.0005},
                1.0,
                0.0,
                1.0,
                1'000'000'000ULL +
                    static_cast<std::uint64_t>(i) *
                    10'000'000ULL,
            }));
    }

    const auto snapshot = state.snapshot();
    assert(!snapshot.empty());

    const auto& primitive = snapshot.front();

    assert(std::isfinite(primitive.covariance[3]));
    assert(std::isfinite(primitive.covariance[4]));
    assert(std::isfinite(primitive.covariance[5]));

    assert(
        std::abs(primitive.covariance[3]) > 1e-8 ||
        std::abs(primitive.covariance[4]) > 1e-8 ||
        std::abs(primitive.covariance[5]) > 1e-8);

    assert(
        primitive.covariance[3] >=
            -std::sqrt(
                primitive.covariance[0] *
                primitive.covariance[1]) &&
        primitive.covariance[3] <=
            std::sqrt(
                primitive.covariance[0] *
                primitive.covariance[1]));

    assert(
        primitive.covariance[4] >=
            -std::sqrt(
                primitive.covariance[0] *
                primitive.covariance[2]) &&
        primitive.covariance[4] <=
            std::sqrt(
                primitive.covariance[0] *
                primitive.covariance[2]));

    assert(
        primitive.covariance[5] >=
            -std::sqrt(
                primitive.covariance[1] *
                primitive.covariance[2]) &&
        primitive.covariance[5] <=
            std::sqrt(
                primitive.covariance[1] *
                primitive.covariance[2]));


    // A covariance-aware posterior should let an oblique residual move the mean across coupled axes.
    // This catches regressions back to independent scalar XYZ averaging.
    TemporalGaussianSurfaceState coupledState({
        .cellSizeMeters = 0.05,
        .maximumAssociationMahalanobis = 64.0,
        .maximumSpatialResidualMeters = 0.50,
        .minimumObservationConfidence = 0.01,
        .dynamicSuppression = 0.0,
        .temporalHalfLifeSeconds = 1.0,
        .velocityDecay = 0.90,
        .confidenceDecay = 0.95,
        .minimumVarianceMeters2 = 1e-8,
        .maximumVarianceMeters2 = 0.25,
        .innovationVarianceInflation = 2.0,
        .crossCovarianceRetention = 0.25,
        .maximumCorrelationMagnitude = 0.96,
        .maximumPrimitives = 64,
    });

    assert(
        coupledState.integrate({
            {0.0, 0.0, 2.0},
            {0, 0, 1},
            {0, 0, 0},
            {0.0004, 0.0004, 0.0004},
            1.0,
            0.0,
            1.0,
            2'000'000'000ULL,
        }));

    for (int i = 1; i <= 8; ++i) {
        const double d =
            0.01 * static_cast<double>(i);
        assert(
            coupledState.integrate({
                {d, d, 2.0},
                {0, 0, 1},
                {0, 0, 0},
                {0.0004, 0.0004, 0.0004},
                1.0,
                0.0,
                1.0,
                2'000'000'000ULL +
                    static_cast<std::uint64_t>(i) *
                    10'000'000ULL,
            }));
    }

    const auto beforeCoupled =
        coupledState.snapshot().front();

    assert(
        coupledState.integrate({
            {beforeCoupled.mean.x,
             beforeCoupled.mean.y + 0.04,
             beforeCoupled.mean.z},
            {0, 0, 1},
            {0, 0, 0},
            {0.0004, 0.0004, 0.0004},
            1.0,
            0.0,
            1.0,
            2'100'000'000ULL,
        }));

    const auto afterCoupled =
        coupledState.snapshot().front();

    assert(
        std::abs(
            afterCoupled.mean.x -
                beforeCoupled.mean.x) >
            1e-9);

    return 0;
}

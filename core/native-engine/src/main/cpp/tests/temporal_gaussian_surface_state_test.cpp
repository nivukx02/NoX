#include <cassert>
#include <cmath>
#include <vector>

#include "scene/temporal_gaussian_surface_state.h"

int main() {
    {
        nox::native::scene::GaussianDeformationGraph graph(
            nox::native::scene::GaussianDeformationOptions{
                0.10,
                128,
                4,
                0.50,
                1.50,
                0.35,
                0.01,
                0.72,
            });

        auto makePrimitive =
            [](double variance, double x) {
                nox::native::scene::GaussianSurfacePrimitive p;
                p.mean = {x, 0.0, 1.0};
                p.covariance = {
                    variance,
                    variance,
                    variance,
                    0.0,
                    0.0,
                    0.0,
                };
                p.confidence = 0.90;
                p.temporalStability = 0.90;
                p.opacity = 1.0;
                return p;
            };

        const auto tight = makePrimitive(0.0001, 0.0);
        const auto noisy = makePrimitive(0.40, 0.0);

        assert(graph.buildCanonicalGraph({tight}));
        const auto tightStats = graph.stats();

        graph.reset();

        assert(graph.buildCanonicalGraph({noisy}));
        const auto noisyStats = graph.stats();

        assert(
            tightStats.meanBindingConfidence >
            noisyStats.meanBindingConfidence);
    }


    using namespace nox::native::scene;

    TemporalGaussianSurfaceState highNormalState;
    TemporalGaussianSurfaceState lowNormalState;

    assert(
        highNormalState.integrate({
            {0.0,0.0,1.0},
            {0.8,0.0,0.6},
            {0,0,0},
            {0.0004,0.0004,0.0004},
            1.0,
            0.0,
            1.0,
            1'000'000'000ULL,
            1.0,
        }));
    assert(
        highNormalState.integrate({
            {0.001,0.0,1.0},
            {0.8,0.0,0.6},
            {0,0,0},
            {0.0004,0.0004,0.0004},
            1.0,
            0.0,
            1.0,
            1'010'000'000ULL,
            1.0,
        }));

    assert(
        lowNormalState.integrate({
            {0.0,0.0,1.0},
            {0.8,0.0,0.6},
            {0,0,0},
            {0.0004,0.0004,0.0004},
            1.0,
            0.0,
            1.0,
            1'000'000'000ULL,
            1.0,
        }));
    assert(
        lowNormalState.integrate({
            {0.001,0.0,1.0},
            {0.8,0.0,0.6},
            {0,0,0},
            {0.0004,0.0004,0.0004},
            1.0,
            0.0,
            1.0,
            1'010'000'000ULL,
            0.0,
        }));

    const auto highNormalSnapshot =
        highNormalState.snapshot();
    const auto lowNormalSnapshot =
        lowNormalState.snapshot();
    assert(highNormalSnapshot.size() == 1U);
    assert(lowNormalSnapshot.size() == 1U);
    assert(
        highNormalSnapshot.front().normal.x >
        lowNormalSnapshot.front().normal.x);
    
    TemporalGaussianSurfaceState state({
        .cellSizeMeters = 0.02,
        .maximumAssociationMahalanobis = 9.0,
        .maximumSpatialResidualMeters = 0.06,
        .minimumObservationConfidence = 0.05,
        .dynamicSuppression = 0.55,
        .temporalHalfLifeSeconds = 0.80,
        .velocityDecay = 0.88,
        .confidenceDecay = 0.92,
        .minimumVarianceMeters2 = 1e-7,
        .maximumVarianceMeters2 = 0.25,
        .maximumPrimitives = 4096,
        .maximumAccelerationMetersPerSecond2 = 120.0,
        .maximumJerkMetersPerSecond3 = 900.0,
        .accelerationDecay = 0.82,
        .jerkDecay = 0.68,
    });

    assert(
        state.integrate({
            {0.0,0.0,1.0},
            {0,0,1},
            {0.10,0.00,0.00},
            {0.0004,0.0004,0.0006},
            0.96,
            0.03,
            0.98,
            1'000'000'000ULL,
        }));

    assert(
        state.integrate({
            {0.038,0.001,1.001},
            {0,0,1},
            {0.10,0.00,0.00},
            {0.0008,0.0008,0.0008},
            0.93,
            0.04,
            0.95,
            1'012'000'000ULL,
        }));

    assert(
        state.integrate({
            {0.001,0.0005,1.0005},
            {0,0,1},
            {0.11,0.00,0.00},
            {0.0005,0.0005,0.0007},
            0.94,
            0.04,
            0.96,
            1'016'000'000ULL,
        }));

    // Non-finite velocity must never poison persistent Gaussian state.
    // Non-finite higher-order motion evidence must never poison persistent state.
    assert(
        state.integrate({
            {0.075,0.0,1.0},
            {0,0,1},
            {0,0,0},
            {0.001,0.001,0.001},
            0.91,
            0.04,
            0.94,
            1'013'000'000ULL,
            1.0,
            {NAN, 0.0, 0.0},
            {0.0, NAN, 0.0},
        }));
    for (const auto& primitive : state.snapshot()) {
        assert(std::isfinite(primitive.acceleration.x));
        assert(std::isfinite(primitive.acceleration.y));
        assert(std::isfinite(primitive.acceleration.z));
        assert(std::isfinite(primitive.jerk.x));
        assert(std::isfinite(primitive.jerk.y));
        assert(std::isfinite(primitive.jerk.z));
    }

    assert(
        state.integrate({
            {0.09,0.0,1.0},
            {0,0,1},
            {NAN, 0.0, 0.0},
            {0.001,0.001,0.001},
            0.90,
            0.10,
            0.90,
            1'014'000'000ULL,
        }));

    assert(
        state.integrate({
            {0.11,0.0,1.0},
            {0,0,1},
            {1.50,0.0,0.0},
            {0.001,0.001,0.001},
            0.88,
            0.90,
            0.55,
            1'016'000'000ULL,
        }));

    assert(
        !state.integrate({
            {0.001,0.0002,1.0003},
            {0,0,1},
            {0.0,0.0,0.0},
            {0.0005,0.0005,0.0005},
            0.90,
            0.05,
            0.90,
            1'015'000'000ULL,
        }));

    const auto before =
        state.stats();
    assert(before.primitiveCount >= 2U);
    assert(before.meanConfidence > 0.50);
    assert(before.meanUncertaintyMeters > 0.0);
    assert(before.trustedPrimitiveRatio >= 0.0);
    assert(before.trustedPrimitiveRatio <= 1.0);
    assert(before.dynamicPrimitives >= 1U);

    assert(state.predict(1'033'000'000ULL));
    {
        const predicted = state.snapshot();
        bool foundHigherOrderState = false;
        for (const auto& primitive : predicted) {
            if (
                std::isfinite(primitive.acceleration.x) &&
                std::isfinite(primitive.acceleration.y) &&
                std::isfinite(primitive.acceleration.z) &&
                std::isfinite(primitive.jerk.x) &&
                std::isfinite(primitive.jerk.y) &&
                std::isfinite(primitive.jerk.z)
            ) {
                foundHigherOrderState = true;
                break;
            }
        }
        assert(foundHigherOrderState);
    }

    const auto after =
        state.stats();
    assert(
        after.meanUncertaintyMeters >=
        before.meanUncertaintyMeters);

    const auto snapshot =
        state.snapshot();
    assert(!snapshot.empty());

    for (const auto& primitive :
         snapshot) {
        assert(std::isfinite(primitive.mean.x));
        assert(std::isfinite(primitive.mean.y));
        assert(std::isfinite(primitive.mean.z));
        assert(primitive.confidence >= 0.0);
        assert(primitive.confidence <= 1.0);
        assert(primitive.dynamicProbability >= 0.0);
        assert(primitive.dynamicProbability <= 1.0);
        assert(primitive.temporalStability >= 0.0);
        assert(primitive.temporalStability <= 1.0);
        assert(primitive.covariance[0] > 0.0);
        assert(primitive.covariance[1] > 0.0);
        assert(primitive.covariance[2] > 0.0);
    }

    state.prune(10'000'000'000ULL);
    assert(state.snapshot().size() < 4096U);

    // Capacity-eviction regression: a weak primitive at the end of the dense array must
    // be evicted cleanly without leaving a stale spatial index behind.
    GaussianSurfaceOptions evictionOptions;
    evictionOptions.cellSizeMeters = 0.02;
    evictionOptions.maximumAssociationMahalanobis = 9.0;
    evictionOptions.maximumSpatialResidualMeters = 0.02;
    evictionOptions.minimumObservationConfidence = 0.05;
    evictionOptions.temporalHalfLifeSeconds = 100.0;
    evictionOptions.maximumPrimitives = 1024U;
    evictionOptions.minimumEvictionUtility = 0.12;

    TemporalGaussianSurfaceState evictionState(evictionOptions);

    for (std::size_t index = 0U; index < 1023U; ++index) {
        const double x =
            0.05 *
            static_cast<double>(index);
        assert(
            evictionState.integrate({
                {x, 0.0, 0.0},
                {0,0,1},
                {0,0,0},
                {0.0004,0.0004,0.0004},
                0.95,
                0.02,
                0.95,
                1'000'000'000ULL,
            }));
    }

    const double weakPosition = 60.0;
    assert(
        evictionState.integrate({
            {weakPosition, 0.0, 0.0},
            {0,0,1},
            {0,0,0},
            {0.25,0.25,0.25},
            0.05,
            0.90,
            0.10,
            1'000'000'000ULL,
        }));

    const auto saturated =
        evictionState.stats();
    assert(saturated.primitiveCount == 1024U);

    assert(
        evictionState.integrate({
            {weakPosition + 0.05, 0.0, 0.0},
            {0,0,1},
            {0,0,0},
            {0.0004,0.0004,0.0004},
            0.95,
            0.02,
            0.95,
            1'500'000'000ULL,
        }));

    const auto evicted =
        evictionState.snapshot();
    assert(evicted.size() == 1024U);

    bool weakStillPresent = false;
    bool replacementPresent = false;
    for (const auto& primitive : evicted) {
        const double dxWeak =
            primitive.mean.x -
            weakPosition;
        const double dxReplacement =
            primitive.mean.x -
            (weakPosition + 0.05);
        if (std::abs(dxWeak) < 1e-9) {
            weakStillPresent = true;
        }
        if (std::abs(dxReplacement) < 1e-9) {
            replacementPresent = true;
        }
    }
    assert(!weakStillPresent);
    assert(replacementPresent);

    evictionState.reset();
    assert(evictionState.snapshot().empty());

    state.reset();
    assert(state.snapshot().empty());
    assert(state.stats().primitiveCount == 0U);

    return 0;
}

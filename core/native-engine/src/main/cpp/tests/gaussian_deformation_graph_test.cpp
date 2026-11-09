#include <cassert>
#include <cmath>
#include <vector>

#include "scene/gaussian_deformation_graph.h"

int main() {
    using namespace nox::native::scene;

    std::vector<GaussianSurfacePrimitive> primitives{
        {
            {0.00,0.00,1.00},
            {0,0,1},
            {0.0,0.0,0.0},
            {0.0004,0.0004,0.0004},
            0.95,
            0.03,
            0.95,
            0.90,
            5,
            1'000'000'000ULL,
        },
        {
            {0.04,0.00,1.00},
            {0,0,1},
            {0.0,0.0,0.0},
            {0.0005,0.0005,0.0005},
            0.92,
            0.04,
            0.94,
            0.88,
            6,
            1'000'000'000ULL,
        },
        {
            {0.00,0.05,1.00},
            {0,0,1},
            {0.0,0.0,0.0},
            {0.0005,0.0005,0.0005},
            0.93,
            0.05,
            0.93,
            0.87,
            7,
            1'000'000'000ULL,
        },
    };

    GaussianDeformationGraph graph({
        .nodeSpacingMeters = 0.03,
        .maximumNodes = 64,
        .maximumInfluencesPerGaussian = 3,
        .maximumNodeTranslationMeters = 0.50,
        .maximumNodeRotationRadians = 1.50,
        .maximumEdgeStrainRatio = 0.35,
        .minimumBindingConfidence = 0.10,
        .temporalSmoothing = 0.60,
    });

    assert(graph.buildCanonicalGraph(primitives));

    assert(
        graph.updateNodePose(
            0,
            nox::native::math::SE3d{
                nox::native::math::Quaterniond::fromRotationVector(
                    {0.0,0.0,0.08}),
                {0.01,0.0,0.0},
            },
            {0.10,0.0,0.0},
            0.96,
            0.95));

    auto deformed =
        graph.deform(primitives);

    assert(deformed.size() == primitives.size());
    assert(
        std::isfinite(
            deformed.front().mean.x));
    assert(
        deformed.front().confidence >
        0.50);

    const auto baselineStats = graph.stats();

    assert(
        graph.updateNodePose(
            0,
            nox::native::math::SE3d{
                nox::native::math::Quaterniond::fromRotationVector(
                    {0.0,0.0,0.40}),
                {0.0,0.0,0.0},
            },
            {0.0,0.0,0.0},
            0.96,
            0.95));

    const rotatedStats = graph.stats();

    assert(
        std::abs(
            rotatedStats.maximumEdgeStrainRatio -
            baselineStats.maximumEdgeStrainRatio) <
        1e-9);

    const auto stats =
        graph.stats();

    assert(stats.nodeCount > 0U);
    assert(stats.bindingCount > 0U);
    assert(stats.meanNodeConfidence > 0.50);
    assert(stats.meanBindingConfidence > 0.50);
    assert(stats.temporalStability > 0.50);
    assert(stats.p95EdgeStrainRatio >= 0.0);
    assert(stats.maximumEdgeStrainRatio >=
           stats.p95EdgeStrainRatio);
    assert(
        stats.trustedBindingRatio >= 0.0 &&
        stats.trustedBindingRatio <= 1.0);

    graph.reset();
    assert(
        graph.stats().nodeCount == 0U);
    assert(
        graph.stats().bindingCount == 0U);

    return 0;
}

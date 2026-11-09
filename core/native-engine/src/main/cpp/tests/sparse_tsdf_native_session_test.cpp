#include <cassert>
#include <cmath>
#include "scene/sparse_tsdf_native_session.h"

int main() {
    nox::native::scene::SparseTsdfNativeSession session({
        0.05,
        0.12,
        32.0F,
        8,
    });

    for (int x = -4; x <= 4; ++x) {
        for (int y = -4; y <= 4; ++y) {
            const double px = (x + 0.5) * 0.05;
            const double py = (y + 0.5) * 0.05;
            session.integrateSurfacePoint(
                {0,0,0},
                {px,py,1.0},
                1.0,
                1.0);
        }
    }

    assert(session.voxelCount() > 0);
    const auto mesh = session.extract(0.0F, 5000);
    assert(!mesh.triangles.empty());
    assert(mesh.vertices.size() <= mesh.triangles.size() * 3);
    assert(mesh.vertices.size() > 0);
    assert(mesh.normals.size() == mesh.vertices.size());
    assert(mesh.vertexUncertaintyMeters.size() == mesh.vertices.size());
    for (const auto& triangle : mesh.triangles) {
        assert(triangle[0] < mesh.vertices.size());
        assert(triangle[1] < mesh.vertices.size());
        assert(triangle[2] < mesh.vertices.size());
        assert(triangle[0] != triangle[1]);
        assert(triangle[1] != triangle[2]);
        assert(triangle[0] != triangle[2]);
    }
    assert(mesh.meanConfidence > 0.0);

    std::vector<float> depth(32 * 32, 1.0F);
    nox::native::scene::SparseTsdfNativeSession frameSession({0.05,0.12,32.0F,8});
    frameSession.integrateDepthFrame(
        depth.data(),32,32,100.0,100.0,15.5,15.5,
        {{1,0,0,0},{0,0,0}},2,1.0);
    assert(frameSession.voxelCount()>0);

    nox::native::scene::SparseTsdfNativeSession highEndSession(
        {0.025,0.08,32.0F,8});
    highEndSession.integrateDepthFrameHighEnd(
        depth.data(),
        32,
        32,
        100.0,
        100.0,
        15.5,
        15.5,
        {{1,0,0,0},{0,0,0}},
        2,
        1.0,
        0.001,
        0.96,
        0.94,
        0.92,
        0.03);
    assert(highEndSession.voxelCount()>0);

    assert(
        highEndSession.integrateGaussianObservation({
            {0.0,0.0,1.0},
            {0,0,1},
            {0.1,0.0,0.0},
            {0.0004,0.0004,0.0006},
            0.96,
            0.02,
            0.98,
            1'000'000'000ULL,
        }));
    assert(
        highEndSession.integrateGaussianObservation({
            {0.001,0.0005,1.0002},
            {0,0,1},
            {0.1,0.0,0.0},
            {0.0005,0.0005,0.0007},
            0.94,
            0.03,
            0.97,
            1'016'000'000ULL,
        }));
    const auto gaussianStats =
        highEndSession.gaussianSurfaceStats();
    assert(gaussianStats.primitiveCount >= 1U);
    assert(gaussianStats.meanConfidence > 0.50);
    assert(gaussianStats.meanUncertaintyMeters > 0.0);


    assert(
        highEndSession.rebuildGaussianDeformationGraph());

    assert(
        highEndSession.updateGaussianDeformationNode(
            0,
            nox::native::math::SE3d{
                nox::native::math::Quaterniond::fromRotationVector(
                    {0.0,0.0,0.05}),
                {0.005,0.0,0.0},
            },
            {0.10,0.0,0.0},
            0.95,
            0.96));

    const auto deformed =
        highEndSession.deformGaussianSurface();
    assert(
        deformed.size() ==
        highEndSession.gaussianSurfaceSnapshot().size());

    const auto deformationStats =
        highEndSession.gaussianDeformationStats();
    assert(deformationStats.nodeCount > 0U);
    assert(deformationStats.bindingCount > 0U);
    assert(deformationStats.meanNodeConfidence > 0.50);

    const auto batchAccepted =
        highEndSession.integrateGaussianObservations({
            {
                {0.01,0.0,1.0},
                {0,0,1},
                {0,0,0},
                {0.0005,0.0005,0.0005},
                0.94,
                0.02,
                0.96,
                1'033'000'000ULL,
            },
            {
                {0.02,0.0,1.0},
                {0,0,1},
                {0,0,0},
                {0.0006,0.0006,0.0006},
                0.93,
                0.03,
                0.95,
                1'033'000'000ULL,
            },
        });
    assert(batchAccepted >= 1U);
    const auto batchGaussianStats =
        highEndSession.gaussianSurfaceStats();
    assert(batchGaussianStats.primitiveCount >= 1U);
    assert(batchGaussianStats.temporalStability >= 0.0);


    const auto highEndMesh =
        highEndSession.extract(0.0F,5000);
    assert(!highEndMesh.triangles.empty());
    assert(highEndMesh.vertexDynamicProbability.size() ==
           highEndMesh.vertices.size());
    assert(highEndMesh.p95UncertaintyMeters > 0.0);

    nox::native::scene::SparseTsdfNativeSession zDepthSession(
        {0.05,0.12,32.0F,8});
    std::vector<float> zDepth(3 * 1, 1.0F);
    zDepthSession.integrateDepthFrame(
        zDepth.data(),3,1,1.0,1.0,0.0,0.0,
        {{1,0,0,0},{0,0,0}},
        1,
        1.0,
        nox::native::scene::DepthSampleConvention::OpticalAxisZ);
    assert(zDepthSession.voxelCount()>0);

    return 0;
}


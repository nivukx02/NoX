#include <cassert>
#include <cmath>
#include <vector>

#include "reconstruction/body_surface_deformation.h"
#include "reconstruction/depth_uncertainty_fusion.h"
#include "scene/mesh_temporal_coherence.h"
#include "scene/scene_flow_estimator.h"
#include "scene/volumetric_fusion.h"

int main() {
    using nox::native::reconstruction::BodySurfaceDeformation;
    using nox::native::reconstruction::DepthEstimate;
    using nox::native::reconstruction::DepthUncertaintyFusion;
    using nox::native::reconstruction::ContactAnchor;
    using nox::native::reconstruction::DeformationOptions;
    using nox::native::reconstruction::Displacement;
    using nox::native::reconstruction::SurfaceEdge;
    using nox::native::reconstruction::Vertex;
    using nox::native::scene::FlowSample;
    using nox::native::scene::MeshTemporalCoherence;
    using nox::native::scene::MeshTemporalOptions;
    using nox::native::scene::SceneFlowEstimator;
    using nox::native::scene::SceneFlowOptions;
    using nox::native::scene::VolumetricFusion;
    using nox::native::scene::VolumetricFusionOptions;
    using nox::native::scene::VoxelObservation;

    BodySurfaceDeformation deformation;
    const std::vector<Vertex> base{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    const std::vector<Displacement> displacement{
        {0.010, 0.000, 0.000, 0.95, 0.001},
        {0.000, 0.020, 0.000, 0.90, 0.0015},
        {0.000, 0.000, 0.015, 0.92, 0.001},
        {-0.008, 0.000, 0.000, 0.88, 0.002},
    };
    const std::vector<SurfaceEdge> edges{
        {0, 1, 1.0},
        {0, 2, 1.0},
        {0, 3, 1.0},
        {1, 2, 0.9},
        {1, 3, 0.9},
        {2, 3, 0.9},
    };
    const std::vector<ContactAnchor> anchors{
        {0, 0.002, 0.0, 0.0, 0.95, 0.8},
    };

    const auto deformationResult =
        deformation.solve(
            base,
            displacement,
            edges,
            anchors,
            DeformationOptions{
                .dataWeight = 1.0,
                .smoothnessWeight = 0.20,
                .contactWeight = 0.35,
                .strainWeight = 0.30,
                .damping = 0.72,
                .maximumDisplacementMeters = 0.20,
                .maximumEdgeStrain = 0.15,
                .convergenceToleranceMeters = 0.00001,
                .iterations = 16,
                .maximumVertices = 1024,
            });

    assert(deformationResult.vertices.size() == base.size());
    assert(std::isfinite(deformationResult.p95DisplacementMeters));
    assert(std::isfinite(deformationResult.maxEdgeStrain));
    assert(deformationResult.confidence > 0.0);
    assert(deformationResult.accepted);

    SceneFlowEstimator flowEstimator;
    const auto first =
        FlowSample{
            0.0, 0.0, 0.0,
            1'000'000'000ULL,
            0.95,
        };
    const auto second =
        FlowSample{
            0.010, 0.002, 0.000,
            1'016'666'667ULL,
            0.94,
        };
    const auto highEndFlow =
        flowEstimator.estimateHighEnd(
            first,
            second,
            nullptr,
            SceneFlowOptions{});

    assert(highEndFlow.confidence > 0.0);
    assert(highEndFlow.speedMetersPerSecond > 0.0);
    assert(std::isfinite(highEndFlow.uncertaintyMeters));
    assert(!highEndFlow.outlier);

    const auto sequence =
        flowEstimator.estimateSequence(
            std::vector<FlowSample>{
                first,
                second,
                {0.022, 0.004, 0.001, 1'033'333'334ULL, 0.94},
                {0.035, 0.006, 0.001, 1'050'000'001ULL, 0.93},
            });
    assert(sequence.samples.size() == 3U);
    assert(std::isfinite(sequence.p95SpeedMetersPerSecond));
    assert(std::isfinite(sequence.p95InnovationMeters));

    VolumetricFusion volume;
    const VolumetricFusionOptions volumeOptions{};
    assert(
        volume.integrateHighEnd(
            0,
            0,
            0,
            VoxelObservation{
                0.05f,
                0.95f,
                0.001f,
                0.98f,
                0.96f,
                0.95f,
                1'000'000'000ULL,
            },
            volumeOptions));

    assert(
        volume.integrateHighEnd(
            0,
            0,
            0,
            VoxelObservation{
                0.048f,
                0.94f,
                0.001f,
                0.97f,
                0.95f,
                0.96f,
                1'016'000'000ULL,
            },
            volumeOptions));

    const auto voxel =
        volume.sample(0, 0, 0);
    assert(voxel.weight > 0.0f);
    assert(voxel.variance > 0.0f);
    assert(voxel.observationCount >= 2U);

    const auto stats =
        volume.stats(volumeOptions);
    assert(stats.voxelCount == 1U);
    assert(stats.integratedObservations >= 2U);
    assert(std::isfinite(stats.confidence));

    volume.decay(
        1'600'000'000ULL,
        volumeOptions);
    assert(volume.voxelCount() == 1U);

    MeshTemporalCoherence temporalMesh;
    const auto temporalMeshResult =
        temporalMesh.stabilizeHighEnd(
            std::vector<nox::native::scene::Vertex>{
                {0.0, 0.0, 0.0, 0.96},
                {1.0, 0.0, 0.0, 0.95},
                {0.0, 1.0, 0.0, 0.94},
            },
            std::vector<nox::native::scene::Vertex>{
                {0.006, 0.001, 0.0, 0.93},
                {1.006, 0.000, 0.0, 0.92},
                {0.001, 1.005, 0.0, 0.91},
            },
            MeshTemporalOptions{
                .blendAlpha = 0.45,
                .maxDisplacementMeters = 0.05,
                .confidenceFloor = 0.20,
                .maximumMatchDistanceMeters = 0.08,
                .spatialCellSizeMeters = 0.02,
                .innovationScaleMeters = 0.008,
                .confidenceDamping = 0.82,
                .velocityDamping = 0.70,
                .searchRadiusCells = 2,
                .maximumVertices = 1024,
            });

    assert(
        temporalMeshResult.correspondenceCoverage >
        0.95);
    assert(
        temporalMeshResult.p95Displacement <
        0.02);
    assert(
        temporalMeshResult.temporalStability >
        0.80);
    assert(temporalMeshResult.accepted);

    DepthUncertaintyFusion depthFusion;
    const auto depthResult =
        depthFusion.fuseRobust({
            DepthEstimate{
                2.000,
                0.0004,
                0.96,
                0.000,
                0.98,
                0.96,
                0.05,
            },
            DepthEstimate{
                2.003,
                0.0009,
                0.91,
                0.000,
                0.94,
                0.95,
                0.10,
            },
            DepthEstimate{
                1.999,
                0.0007,
                0.93,
                0.000,
                0.97,
                0.94,
                0.04,
            },
            DepthEstimate{
                2.80,
                0.0005,
                0.95,
                0.000,
                0.40,
                0.25,
                0.30,
            },
        });

    assert(depthResult.valid);
    assert(depthResult.acceptedSamples >= 3U);
    assert(depthResult.rejectedSamples >= 1U);
    assert(std::abs(depthResult.depth - 2.0) < 0.02);
    assert(depthResult.p95ResidualMeters < 0.03);
    assert(depthResult.confidence > 0.50);
    assert(depthResult.consensus > 0.50);

    return 0;
}

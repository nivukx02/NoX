#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>
#include "face/face_geometry_fusion.h"
#include "face/face_surface_alignment.h"

int main(){
    using namespace nox::native::face;

    std::vector<FaceLandmarkObservation> observations={
        {1,0,{640,360},{0,0,0},{0,0,1},1.0},
        {1,1,{620,360},{1,0,0},{-1,0,1},1.0},
        {1,2,{640,340},{0,1,0},{0,-1,1},0.9},

        {2,0,{640,360},{0,0,0},{0.1,0.0,1.0},1.0},
        {2,1,{640,360},{1,0,0},{-0.1,0.0,1.0},1.0},
        {2,2,{640,360},{0,1,0},{0.0,-0.1,1.0},0.9},

        {3,0,{640,360},{0,0,0},{-0.1,0.1,1.0},1.0},
        {3,1,{640,360},{1,0,0},{-0.3,0.1,1.0},1.0},
        {3,2,{640,360},{0,1,0},{-0.1,-0.2,1.0},0.9},
    };

    const auto state=FaceGeometryFusion{}.solve(observations,10);
    assert(state.metric);
    assert(state.landmarks.size()==3);
    for(const auto& landmark:state.landmarks){
        assert(landmark.supporting_pairs>=2);
        assert(landmark.confidence>0.3);
        assert(landmark.triangulation_conditioning>0.05);
        assert(std::isfinite(landmark.uncertainty));
    }



    auto permuted=observations;
    std::reverse(permuted.begin(),permuted.end());
    const auto permutedState =
        FaceGeometryFusion{}.solve(permuted,10);
    assert(permutedState.landmarks.size()==state.landmarks.size());
    for(std::size_t i=0;i<state.landmarks.size();++i){
        assert(permutedState.landmarks[i].id==state.landmarks[i].id);
        assert(std::abs(
            permutedState.landmarks[i].confidence -
                state.landmarks[i].confidence) < 1e-9);
    }

    std::vector<FaceLandmarkObservation> nearlyParallel={
        {9,0,{0,0},{0,0,0},{0,0,1},1.0},
        {9,1,{0,0},{1,0,0},{-0.999,0,0.063},1.0},
        {9,2,{0,0},{0,1,0},{0,-0.999,0.063},0.9},
    };

    const auto conditioned =
        FaceGeometryFusion{}.solve(
            nearlyParallel,
            20);

    if(!conditioned.landmarks.empty()){
        assert(
            conditioned.landmarks.front()
                .triangulation_conditioning < 0.50);
    }
    assert(std::isfinite(state.confidence));
    assert(state.robust_support_ratio >= 0.0);
    assert(state.robust_support_ratio <= 1.0);
    assert(state.conditioning_score >= 0.0);
    assert(state.conditioning_score <= 1.0);
    assert(state.mean_landmark_uncertainty >= 0.0);

    auto temporalObservations = observations;
    for (auto& observation : temporalObservations) {
        observation.ray_origin.x += 0.0008;
        observation.ray_origin.y += 0.0004;
    }

    const auto temporal =
        FaceGeometryFusion{}.solveTemporal(
            temporalObservations,
            10'008'333'333ULL,
            state,
            1.0 / 120.0,
            FaceTemporalOptions{
                .maximumLandmarkVelocityMetersPerSecond = 12.0,
                .maximumInnovationMeters = 0.05,
                .innovationScaleMeters = 0.01,
                .minimumTemporalConfidence = 0.30,
                .maximumDtSeconds = 0.25,
            });

    assert(temporal.temporal_confidence > 0.50);
    assert(temporal.temporal_consistency > 0.50);
    assert(temporal.temporal_innovation_p95 >= 0.0);
    assert(temporal.temporal_innovation_max >= temporal.temporal_innovation_p95);
    assert(temporal.temporal_velocity_meters_per_second >= 0.0);

    auto temporalJump = temporalObservations;
    for (auto& observation : temporalJump) {
        observation.ray_origin.x += 0.50;
    }

    const auto rejectedTemporal =
        FaceGeometryFusion{}.solveTemporal(
            temporalJump,
            10'016'666'666ULL,
            state,
            1.0 / 120.0,
            FaceTemporalOptions{
                .maximumLandmarkVelocityMetersPerSecond = 2.0,
                .maximumInnovationMeters = 0.03,
                .innovationScaleMeters = 0.005,
                .minimumTemporalConfidence = 0.40,
                .maximumDtSeconds = 0.25,
            });

    assert(
        (rejectedTemporal.failure_flags &
            FACE_FAILURE_HIGH_UNCERTAINTY) != 0u ||
        rejectedTemporal.confidence < temporal.confidence);


    std::vector<FaceLandmarkObservation> contaminated={
        {11,0,{0,0},{0,0,0},{0,0,1},1.0},
        {11,1,{0,0},{1,0,0},{-1,0,1},1.0},
        {11,2,{0,0},{0,1,0},{0,-1,1},1.0},
        {11,3,{0,0},{4,4,4},{-0.9,-0.9,0.2},0.95},
    };
    const auto robust =
        FaceGeometryFusion{}.solve(
            contaminated,
            30);
    if(!robust.landmarks.empty()){
        assert(
            robust.landmarks.front().uncertainty >= 0.0);
        assert(
            robust.landmarks.front().confidence <= 1.0);
        assert(
            robust.rejected_pair_count >= 1u ||
            robust.robust_support_ratio <= 0.75);
    }
    assert(FaceSurfaceAlignment::shapeConsistency(state)>=0.0);
    assert(
        (robust.failure_flags &
            FACE_FAILURE_NO_MULTI_VIEW_SUPPORT) == 0u ||
        robust.robust_support_ratio >= 0.0);

    FaceGeometryFusion bounded({
        128,
        8,
        4,
        0.10,
    });

    std::vector<FaceLandmarkObservation> flood;
    flood.reserve(10'000);
    for(std::uint32_t index=0; index<10'000; ++index) {
        const auto id=index%16u;
        const auto camera=index%32u;
        flood.push_back({
            id,
            camera,
            {static_cast<double>(index),0.0},
            {static_cast<double>(camera),0.0,0.0},
            {0.0,0.0,1.0},
            0.95,
        });
    }

    const auto boundedState =
        bounded.solve(flood,40);
    assert(boundedState.landmarks.size() <= 8u);
    for(const auto& landmark:boundedState.landmarks) {
        assert(landmark.id < 16u);
        assert(std::isfinite(landmark.confidence));
        assert(std::isfinite(landmark.uncertainty));
    }

    FaceLandmarkObservation malformed{
        99u,
        0u,
        {0.0,0.0},
        {0.0,0.0,0.0},
        {0.0,0.0,std::numeric_limits<double>::quiet_NaN()},
        1.0,
    };
    const auto malformedState =
        bounded.solve({malformed},50);
    assert(malformedState.landmarks.empty());
    return 0;
}

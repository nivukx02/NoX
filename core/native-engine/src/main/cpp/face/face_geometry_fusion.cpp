#include "face/face_geometry_fusion.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>
#include <numeric>

namespace nox::native::face {
namespace {
Vec3 add(Vec3 a,Vec3 b){return{a.x+b.x,a.y+b.y,a.z+b.z};}
Vec3 sub(Vec3 a,Vec3 b){return{a.x-b.x,a.y-b.y,a.z-b.z};}
Vec3 mul(Vec3 a,double s){return{a.x*s,a.y*s,a.z*s};}
double dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
double norm(Vec3 a){return std::sqrt(dot(a,a));}
Vec3 cross(Vec3 a,Vec3 b){return{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
bool finiteVec(Vec3 v) {
    return std::isfinite(v.x) &&
        std::isfinite(v.y) &&
        std::isfinite(v.z);
}
Vec3 normalize(Vec3 v){
    if(!finiteVec(v))return{0,0,1};
    const double n=norm(v);
    return !std::isfinite(n)||n<1e-12
        ? Vec3{0,0,1}
        : mul(v,1.0/n);
}
struct Candidate {
    Vec3 point{};
    double pair_error{0.0};
    double weight{0.0};
    double conditioning{0.0};
};

bool closest(
    const FaceLandmarkObservation& a,
    const FaceLandmarkObservation& b,
    Vec3& out,
    double& error,
    double& conditioningOut) {
    if(
        !finiteVec(a.ray_origin)||
        !finiteVec(a.ray_direction)||
        !finiteVec(b.ray_origin)||
        !finiteVec(b.ray_direction)||
        !std::isfinite(a.confidence)||
        !std::isfinite(b.confidence)) {
        return false;
    }

    const auto u=normalize(a.ray_direction);
    const auto v=normalize(b.ray_direction);
    const Vec3 w=sub(b.ray_origin,a.ray_origin);
    const double uv=dot(u,v);
    const double denominator=1.0-uv*uv;
    if(std::abs(denominator)<1e-8)return false;
    const double angularConditioning=
        std::sqrt(
            std::clamp(
                denominator,
                0.0,
                1.0));
    if(angularConditioning<0.05)return false;
    const double s=
        (dot(w,u)-uv*dot(w,v))/denominator;
    const double t=
        (uv*dot(w,u)-dot(w,v))/denominator;
    if(s<=0.0||t<=0.0)return false;
    const Vec3 p1=add(a.ray_origin,mul(u,s));
    const Vec3 p2=add(b.ray_origin,mul(v,t));
    out=mul(add(p1,p2),0.5);
    error=norm(sub(p1,p2));
    const double baseline =
        norm(sub(
            b.ray_origin,
            a.ray_origin));
    const double meanDepth =
        0.5 * (s + t);
    const double depthConditioning =
        baseline <= 1e-9
            ? angularConditioning
            : std::clamp(
                meanDepth /
                    (2.0 * baseline),
                0.0,
                1.0);
    conditioningOut =
        std::min(
            angularConditioning,
            depthConditioning);
    return std::isfinite(error);
}
double distance(Vec3 a,Vec3 b){return norm(sub(a,b));}
}

FaceGeometryFusion::FaceGeometryFusion() noexcept
    : FaceGeometryFusion(Config{}) {}

FaceGeometryFusion::FaceGeometryFusion(
    Config config) noexcept
    : config_(config) {
    config_.maxInputObservations =
        std::clamp<std::size_t>(
            config_.maxInputObservations,
            128U,
            32'768U);
    config_.maxTrackedLandmarks =
        std::clamp<std::size_t>(
            config_.maxTrackedLandmarks,
            32U,
            2'048U);
    config_.maxObservationsPerLandmark =
        std::clamp<std::size_t>(
            config_.maxObservationsPerLandmark,
            2U,
            64U);
    config_.minimumConfidence =
        std::clamp(
            std::isfinite(config_.minimumConfidence)
                ? config_.minimumConfidence
                : 0.05,
            0.0,
            1.0);
}

bool FaceGeometryFusion::finite(
    const FaceLandmarkObservation& observation) noexcept {
    return
        std::isfinite(observation.confidence) &&
        std::isfinite(observation.pixel.x) &&
        std::isfinite(observation.pixel.y) &&
        finiteVec(observation.ray_origin) &&
        finiteVec(observation.ray_direction);
}

bool FaceGeometryFusion::betterCandidate(
    const FaceLandmarkObservation& lhs,
    const FaceLandmarkObservation& rhs) noexcept {
    if(lhs.confidence != rhs.confidence) {
        return lhs.confidence > rhs.confidence;
    }
    if(lhs.camera_id != rhs.camera_id) {
        return lhs.camera_id < rhs.camera_id;
    }
    if(lhs.ray_origin.x != rhs.ray_origin.x) {
        return lhs.ray_origin.x < rhs.ray_origin.x;
    }
    if(lhs.ray_origin.y != rhs.ray_origin.y) {
        return lhs.ray_origin.y < rhs.ray_origin.y;
    }
    return lhs.ray_origin.z < rhs.ray_origin.z;
}

FaceGeometryState FaceGeometryFusion::solve(
    const std::vector<FaceLandmarkObservation>& observations,
    std::uint64_t timestampNanos) const {
    FaceGeometryState output;
    output.timestamp_nanos=timestampNanos;

    std::unordered_map<
        std::uint32_t,
        std::vector<FaceLandmarkObservation>
    > grouped;
    grouped.reserve(config_.maxTrackedLandmarks);

    const auto inputLimit =
        std::min(
            observations.size(),
            config_.maxInputObservations);

    for(std::size_t inputIndex=0;
        inputIndex<inputLimit;
        ++inputIndex) {
        const auto& observation=observations[inputIndex];
        if(
            !finite(observation)||
            observation.confidence<=config_.minimumConfidence) {
            continue;
        }

        auto found=grouped.find(observation.id);
        if(
            found==grouped.end() &&
            grouped.size()>=config_.maxTrackedLandmarks) {
            continue;
        }

        auto& bucket=grouped[observation.id];
        if(bucket.size()<config_.maxObservationsPerLandmark) {
            bucket.push_back(observation);
            continue;
        }

        auto weakest=std::min_element(
            bucket.begin(),
            bucket.end(),
            [](const auto& lhs,const auto& rhs) {
                return FaceGeometryFusion::betterCandidate(
                    rhs,lhs);
            });
        if(
            weakest!=bucket.end() &&
            betterCandidate(observation,*weakest)) {
            *weakest=observation;
        }
    }

    double confidenceSum=0.0;
    double reprojectionSum=0.0;
    double conditioningSum=0.0;
    std::size_t validLandmarks=0;
    std::uint32_t pairAttempts=0;
    std::uint32_t pairRejected=0;
    std::uint32_t totalResidualRejected=0;

    std::vector<std::uint32_t> landmarkIds;
    landmarkIds.reserve(grouped.size());
    for(const auto& entry:grouped){
        landmarkIds.push_back(entry.first);
    }
    std::sort(landmarkIds.begin(),landmarkIds.end());

    for(const auto id:landmarkIds){
        auto& items=grouped[id];
        std::sort(
            items.begin(),
            items.end(),
            [](const FaceLandmarkObservation& a,
               const FaceLandmarkObservation& b){
                if(a.camera_id!=b.camera_id){
                    return a.camera_id<b.camera_id;
                }
                if(a.confidence!=b.confidence){
                    return a.confidence>b.confidence;
                }
                if(a.ray_origin.x!=b.ray_origin.x){
                    return a.ray_origin.x<b.ray_origin.x;
                }
                if(a.ray_origin.y!=b.ray_origin.y){
                    return a.ray_origin.y<b.ray_origin.y;
                }
                return a.ray_origin.z<b.ray_origin.z;
            });
        if(items.size()<2)continue;

        std::vector<Candidate> candidates;
        candidates.reserve(items.size()*(items.size()-1)/2);
        double minimumObservedConditioning=1.0;

        for(std::size_t i=0;i<items.size();++i){
            for(std::size_t j=i+1;j<items.size();++j){
                if(items[i].camera_id==items[j].camera_id)continue;
                ++pairAttempts;
                const auto ui=normalize(items[i].ray_direction);
                const auto uj=normalize(items[j].ray_direction);
                const double pairCosine =
                    std::clamp(dot(ui,uj),-1.0,1.0);
                const double pairConditioning =
                    std::sqrt(
                        std::clamp(
                            1.0-pairCosine*pairCosine,
                            0.0,
                            1.0));
                if(std::isfinite(pairConditioning)){
                    minimumObservedConditioning=
                        std::min(
                            minimumObservedConditioning,
                            pairConditioning);
                }
                Vec3 point{};
                double error=0.0;
                double conditioning=0.0;
                if(!closest(
                    items[i],
                    items[j],
                    point,
                    error,
                    conditioning)){
                    ++pairRejected;
                    continue;
                }

                if (
                    !std::isfinite(error) ||
                    error > 0.25) {
                    ++pairRejected;
                    continue;
                }

                const double weight=
                    std::clamp(items[i].confidence,0.0,1.0)*
                    std::clamp(items[j].confidence,0.0,1.0)*
                    conditioning/
                    (1.0+error);
                if(weight<=0.0)continue;
                candidates.push_back({
                    point,
                    error,
                    weight,
                    conditioning});
            }
        }

        if(candidates.empty())continue;

        Vec3 seed{};
        double seedWeight=0.0;
        for(const auto& candidate:candidates){
            seed=add(seed,mul(candidate.point,candidate.weight));
            seedWeight+=candidate.weight;
        }
        if(seedWeight<=0.0)continue;
        seed=mul(seed,1.0/seedWeight);

        std::vector<double> residuals;
        residuals.reserve(candidates.size());
        for(const auto& candidate:candidates){
            residuals.push_back(distance(candidate.point,seed));
        }
        auto sorted=residuals;
        std::sort(sorted.begin(),sorted.end());
        const double median =
            sorted[sorted.size()/2];

        std::vector<double> absoluteDeviations;
        absoluteDeviations.reserve(
            residuals.size());
        for(const double residual:residuals){
            absoluteDeviations.push_back(
                std::abs(residual-median));
        }
        std::sort(
            absoluteDeviations.begin(),
            absoluteDeviations.end());
        const double mad =
            absoluteDeviations[
                absoluteDeviations.size()/2];
        const double robustScale =
            std::max(
                1e-6,
                1.4826*mad);
        const double gate =
            std::max(
                0.010,
                median+3.0*robustScale);

        Vec3 fused{};
        double fusedWeight=0.0;
        double fusedError=0.0;
        double minimumFusedConditioning=1.0;
        std::uint32_t support=0;
        std::uint32_t residualRejected=0;
        double allWeight=0.0;

        for(std::size_t index=0;index<candidates.size();++index){
            const auto& candidate=candidates[index];
            allWeight+=candidate.weight;
            if(residuals[index]>gate){
                ++residualRejected;
                continue;
            }
            fused=add(fused,mul(candidate.point,candidate.weight));
            fusedWeight+=candidate.weight;
            fusedError+=candidate.weight*(
                candidate.pair_error+residuals[index]);
            minimumFusedConditioning =
                std::min(
                    minimumFusedConditioning,
                    candidate.conditioning);
            ++support;
        }

        totalResidualRejected += residualRejected;

        if(fusedWeight<=0.0)continue;

        FaceLandmarkState state;
        state.id=id;
        state.position=mul(fused,1.0/fusedWeight);
        state.uncertainty=std::clamp(
            fusedError/fusedWeight +
                robustScale,
            0.0,
            1.0);
        state.triangulation_conditioning=
            std::clamp(
                std::min(
                    minimumObservedConditioning,
                    minimumFusedConditioning),
                0.0,
                1.0);
        state.confidence=std::clamp(
            (fusedWeight/std::max(allWeight,1e-9))*
            std::min(1.0,
                static_cast<double>(support)/
                static_cast<double>(std::max<std::size_t>(candidates.size(),1))),
            0.0,
            1.0);
        state.supporting_pairs=support;
        state.metric=true;

        output.landmarks.push_back(state);
        confidenceSum+=state.confidence;
        reprojectionSum+=state.uncertainty;
        conditioningSum+=state.triangulation_conditioning;
        ++validLandmarks;
    }

    output.rejected_pair_count=
        pairRejected +
        totalResidualRejected;
    output.robust_support_ratio=
        pairAttempts==0
            ? 0.0
            : std::clamp(
                1.0 -
                    static_cast<double>(
                        pairRejected +
                        totalResidualRejected) /
                    static_cast<double>(pairAttempts),
                0.0,
                1.0);
    if(validLandmarks==0){
        output.failure_flags=FACE_FAILURE_NO_MULTI_VIEW_SUPPORT;
        return output;
    }

    output.head_center={};
    for(const auto& state:output.landmarks){
        output.head_center=add(
            output.head_center,
            state.position);
    }
    output.head_center=mul(
        output.head_center,
        1.0/static_cast<double>(validLandmarks));
    output.confidence=std::clamp(
        confidenceSum/static_cast<double>(validLandmarks),
        0.0,
        1.0) *
        std::clamp(
            output.robust_support_ratio +
                0.20 *
                    std::clamp(
                        1.0 -
                            static_cast<double>(pairRejected) /
                                static_cast<double>(
                                    std::max<std::uint32_t>(
                                        pairAttempts,1U)),
                        0.0,
                        1.0),
            0.0,
            1.0);
    output.reprojection_rms=
        reprojectionSum/static_cast<double>(validLandmarks);
    output.mean_landmark_uncertainty=
        output.reprojection_rms;
    output.conditioning_score=
        std::clamp(
            conditioningSum/static_cast<double>(validLandmarks),
            0.0,
            1.0);
    output.metric=true;

    output.failure_flags=FACE_FAILURE_NONE;
    const double totalRejected =
        static_cast<double>(
            pairRejected + totalResidualRejected);
    const double outlierRatio =
        pairAttempts==0
            ? 1.0
            : totalRejected /
                static_cast<double>(pairAttempts);
    if(outlierRatio>0.35){
        output.failure_flags|=
            FACE_FAILURE_HIGH_OUTLIER_RATIO;
    }
    if(output.conditioning_score<0.35){
        output.failure_flags|=
            FACE_FAILURE_LOW_CONDITIONING;
    }
    if(output.mean_landmark_uncertainty>0.08){
        output.failure_flags|=
            FACE_FAILURE_HIGH_UNCERTAINTY;
    }
    if(output.confidence<0.45){
        output.failure_flags|=
            FACE_FAILURE_LOW_CONFIDENCE;
    }

    if(output.landmarks.size()>=3){
        Vec3 forward{};
        for(std::size_t i=1;i<output.landmarks.size();++i){
            const Vec3 a=sub(
                output.landmarks[i-1].position,
                output.head_center);
            const Vec3 b=sub(
                output.landmarks[i].position,
                output.head_center);
            forward=add(forward,cross(a,b));
        }
        output.head_forward=normalize(forward);
    }

    return output;
}

FaceGeometryState FaceGeometryFusion::solveTemporal(
    const std::vector<FaceLandmarkObservation>& observations,
    std::uint64_t timestampNanos,
    const FaceGeometryState& previous,
    double dtSeconds,
    const FaceTemporalOptions& options) const {
    FaceGeometryState current =
        solve(
            observations,
            timestampNanos);

    if (
        !std::isfinite(dtSeconds) ||
        dtSeconds <= 0.0 ||
        dtSeconds > options.maximumDtSeconds) {
        current.confidence *= 0.35;
        current.failure_flags |=
            FACE_FAILURE_HIGH_UNCERTAINTY;
        return current;
    }

    if (
        previous.landmarks.empty() ||
        current.landmarks.empty()) {
        return current;
    }

    const double maximumVelocity =
        std::max(
            options.maximumLandmarkVelocityMetersPerSecond,
            0.01);

    const double maximumInnovation =
        std::max(
            options.maximumInnovationMeters,
            1e-5);

    const double innovationScale =
        std::max(
            options.innovationScaleMeters,
            1e-5);

    std::vector<double> innovations;
    innovations.reserve(
        current.landmarks.size());

    double evidenceSum = 0.0;
    double velocitySum = 0.0;
    std::size_t matched = 0U;
    std::size_t rejected = 0U;

    for (auto& landmark :
         current.landmarks) {
        const auto previousIt =
            std::find_if(
                previous.landmarks.begin(),
                previous.landmarks.end(),
                [&landmark](const FaceLandmarkState& candidate) {
                    return candidate.id == landmark.id;
                });

        if (
            previousIt ==
            previous.landmarks.end()) {
            continue;
        }

        const double innovation =
            distance(
                landmark.position,
                previousIt->position);

        const double velocity =
            innovation /
            dtSeconds;

        const double innovationEvidence =
            std::exp(
                -innovation /
                innovationScale);

        const double velocityEvidence =
            std::clamp(
                1.0 -
                    std::max(
                        0.0,
                        velocity -
                            maximumVelocity) /
                    maximumVelocity,
                0.0,
                1.0);

        innovations.push_back(
            innovation);
        velocitySum +=
            velocity;
        evidenceSum +=
            std::clamp(
                landmark.confidence,
                0.0,
                1.0) *
            innovationEvidence *
            velocityEvidence;

        ++matched;

        if (
            innovation >
                maximumInnovation ||
            velocity >
                maximumVelocity) {
            ++rejected;
        }
    }

    if (
        innovations.empty()) {
        current.confidence *= 0.40;
        current.failure_flags |=
            FACE_FAILURE_HIGH_UNCERTAINTY;
        return current;
    }

    std::sort(
        innovations.begin(),
        innovations.end());

    const double p95Position =
        0.95 *
        static_cast<double>(
            innovations.size() - 1U);

    const auto lower =
        static_cast<std::size_t>(
            std::floor(p95Position));

    const auto upper =
        std::min(
            innovations.size() - 1U,
            lower + 1U);

    const double alpha =
        p95Position -
        static_cast<double>(
            lower);

    const double meanInnovation =
        std::accumulate(
            innovations.begin(),
            innovations.end(),
            0.0) /
        static_cast<double>(
            innovations.size());

    const double p95Innovation =
        innovations[lower] +
        (
            innovations[upper] -
            innovations[lower]
        ) *
        alpha;

    const double maximumInnovationObserved =
        innovations.back();

    const double meanVelocity =
        velocitySum /
        static_cast<double>(
            matched);

    const double matchRatio =
        std::clamp(
            static_cast<double>(
                matched) /
            static_cast<double>(
                std::max(
                    current.landmarks.size(),
                    std::size_t{1})),
            0.0,
            1.0);

    const double evidence =
        std::clamp(
            evidenceSum /
            static_cast<double>(
                std::max(
                    matched,
                    std::size_t{1})),
            0.0,
            1.0);

    const double innovationEvidence =
        std::exp(
            -p95Innovation /
            innovationScale);

    const double rejectionEvidence =
        1.0 -
        std::clamp(
            static_cast<double>(
                rejected) /
            static_cast<double>(
                std::max(
                    matched,
                    std::size_t{1})),
            0.0,
            1.0);

    const double temporalConfidence =
        std::clamp(
            matchRatio * 0.25 +
            evidence * 0.30 +
            innovationEvidence * 0.30 +
            rejectionEvidence * 0.15,
            0.0,
            1.0);

    current.temporal_confidence =
        temporalConfidence;
    current.temporal_consistency =
        std::clamp(
            0.55 * innovationEvidence +
            0.25 * rejectionEvidence +
            0.20 * matchRatio,
            0.0,
            1.0);
    current.temporal_innovation_p95 =
        p95Innovation;
    current.temporal_innovation_max =
        maximumInnovationObserved;
    current.temporal_velocity_meters_per_second =
        meanVelocity;

    current.confidence =
        std::clamp(
            current.confidence *
                (
                    0.55 +
                    0.45 *
                        temporalConfidence
                ),
            0.0,
            1.0);

    current.mean_landmark_uncertainty =
        std::clamp(
            current.mean_landmark_uncertainty +
                p95Innovation *
                    0.50,
            0.0,
            1.0);

    if (
        p95Innovation >
            maximumInnovation ||
        velocitySum /
                static_cast<double>(
                    matched) >
            maximumVelocity ||
        temporalConfidence <
            options.minimumTemporalConfidence) {
        current.failure_flags |=
            FACE_FAILURE_HIGH_UNCERTAINTY;
        current.confidence *= 0.60;

        for (auto& landmark :
             current.landmarks) {
            const auto previousIt =
                std::find_if(
                    previous.landmarks.begin(),
                    previous.landmarks.end(),
                    [&landmark](const FaceLandmarkState& candidate) {
                        return candidate.id == landmark.id;
                    });

            if (
                previousIt ==
                previous.landmarks.end()) {
                continue;
            }

            const double innovation =
                distance(
                    landmark.position,
                    previousIt->position);

            if (
                innovation >
                maximumInnovation) {
                landmark.confidence *=
                    0.50;
            }
        }
    }

    (void)meanInnovation;

    return current;
}

} // namespace nox::native::face

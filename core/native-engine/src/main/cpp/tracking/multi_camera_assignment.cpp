#include "tracking/multi_camera_assignment.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace nox::native::tracking {
namespace {

double clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}
std::vector<int> hungarian(
    const std::vector<std::vector<double>>& costs) {
    const std::size_t rows=costs.size();
    const std::size_t cols=rows==0?0:costs.front().size();
    const std::size_t n=std::max(rows,cols);
    if(n==0)return{};
    const double penalty=1e6;

    std::vector<double> u(n+1,0.0),v(n+1,0.0);
    std::vector<std::size_t> p(n+1,0),way(n+1,0);

    for(std::size_t i=1;i<=n;++i){
        p[0]=i;
        std::size_t j0=0;
        std::vector<double> minv(n+1,std::numeric_limits<double>::infinity());
        std::vector<bool> used(n+1,false);

        do{
            used[j0]=true;
            const std::size_t i0=p[j0];
            double delta=std::numeric_limits<double>::infinity();
            std::size_t j1=0;

            for(std::size_t j=1;j<=n;++j){
                if(used[j])continue;
                const double cost=
                    i0<=rows&&j<=cols
                        ? costs[i0-1][j-1]
                        : penalty;
                const double cur=
                    cost-u[i0]-v[j];
                if(cur<minv[j]){
                    minv[j]=cur;
                    way[j]=j0;
                }
                if(minv[j]<delta){
                    delta=minv[j];
                    j1=j;
                }
            }

            for(std::size_t j=0;j<=n;++j){
                if(used[j]){
                    u[p[j]]+=delta;
                    v[j]-=delta;
                }else{
                    minv[j]-=delta;
                }
            }
            j0=j1;
        }while(p[j0]!=0);

        do{
            const std::size_t j1=way[j0];
            p[j0]=p[j1];
            j0=j1;
        }while(j0!=0);
    }

    std::vector<int> assignment(rows,-1);
    for(std::size_t j=1;j<=n;++j){
        if(p[j]==0||p[j]>rows||j>cols)continue;
        assignment[p[j]-1]=static_cast<int>(j-1);
    }
    return assignment;
}
}

std::vector<Assignment> MultiCameraAssignment::solve(
    const std::vector<GeometricIdentityTrack>& tracks,
    const std::vector<Vec3>& predicted,
    const std::vector<DetectionCandidate>& detections,
    const CameraVisibilityGraph* visibility) const {
    (void)visibility;

    const std::size_t trackCount=
        std::min(tracks.size(),predicted.size());
    if(trackCount==0||detections.empty())return{};

    std::vector<std::vector<double>> costs(
        trackCount,
        std::vector<double>(detections.size(),
            std::numeric_limits<double>::infinity()));

    for(std::size_t track=0;track<trackCount;++track){
        for(std::size_t detection=0;
            detection<detections.size();
            ++detection){
            costs[track][detection]=tracks[track].cost(
                predicted[track],
                detections[detection].position,
                detections[detection].timestampDelta,
                1.0,
                detections[detection].confidence).total;
        }
    }

    const auto assignment=hungarian(costs);
    std::vector<Assignment> output;
    output.reserve(trackCount);

    for(std::size_t track=0;track<trackCount;++track){
        const int matched=assignment[track];
        if(matched<0||
           static_cast<std::size_t>(matched)>=detections.size()){
            continue;
        }

        const double best=costs[track][static_cast<std::size_t>(matched)];
        double secondBest=std::numeric_limits<double>::infinity();
        for(std::size_t detection=0;detection<detections.size();++detection){
            if(static_cast<int>(detection)==matched)continue;
            secondBest=std::min(secondBest,costs[track][detection]);
        }

        const double margin=
            std::isfinite(secondBest)
                ? std::max(0.0,secondBest-best)
                : 1e6;
        output.push_back({
            tracks[track].id(),
            detections[static_cast<std::size_t>(matched)].detectionId,
            best,
            std::exp(-best),
            margin,
            best<1.0&&margin>0.05,
        });
    }

    return output;
}


std::vector<Assignment>
MultiCameraAssignment::solveHighEnd(
    const std::vector<GeometricIdentityTrack>& tracks,
    const std::vector<Vec3>& predicted,
    const std::vector<Vec3>& predictedVelocity,
    const std::vector<DetectionCandidate>& detections,
    const CameraVisibilityGraph* visibility) const {
    (void)visibility;

    const std::size_t trackCount =
        std::min(
            {
                tracks.size(),
                predicted.size(),
                predictedVelocity.size(),
            });

    if (
        trackCount == 0U ||
        detections.empty()) {
        return {};
    }

    std::vector<std::vector<double>> costs(
        trackCount,
        std::vector<double>(
            detections.size(),
            std::numeric_limits<double>::infinity()));

    for (
        std::size_t track = 0;
        track < trackCount;
        ++track) {
        for (
            std::size_t detection = 0;
            detection < detections.size();
            ++detection) {
            const auto& candidate =
                detections[detection];

            costs[track][detection] =
                tracks[track]
                    .costMultimodal(
                        predicted[track],
                        candidate.position,
                        predictedVelocity[track],
                        candidate.velocity,
                        candidate.timestampDelta,
                        candidate.visibility,
                        candidate.confidence,
                        candidate.appearanceSimilarity,
                        candidate.faceConsistency,
                        candidate.handConsistency,
                        candidate.contactConsistency,
                        candidate.scaleRatio)
                    .total +
                clamp01(
                    candidate.occlusionRisk) *
                    0.10;
        }
    }

    const auto assignment =
        hungarian(
            costs);

    std::vector<Assignment> output;
    output.reserve(
        trackCount);

    for (
        std::size_t track = 0;
        track < trackCount;
        ++track) {
        const int matched =
            assignment[track];

        if (
            matched < 0 ||
            static_cast<std::size_t>(
                matched) >=
                detections.size()) {
            continue;
        }

        const std::size_t matchedIndex =
            static_cast<std::size_t>(
                matched);

        const double best =
            costs[track][matchedIndex];

        double secondBest =
            std::numeric_limits<double>::infinity();

        for (
            std::size_t detection = 0;
            detection < detections.size();
            ++detection) {
            if (
                detection ==
                matchedIndex) {
                continue;
            }

            secondBest =
                std::min(
                    secondBest,
                    costs[track][detection]);
        }

        const double margin =
            std::isfinite(
                secondBest)
                ? std::max(
                    0.0,
                    secondBest -
                        best)
                : 1e6;

        const double ambiguity =
            std::isfinite(
                secondBest)
                ? std::clamp(
                    margin /
                        std::max(
                            secondBest,
                            0.001),
                    0.0,
                    1.0)
                : 1.0;

        const bool accepted =
            std::isfinite(best) &&
            best < 1.35 &&
            ambiguity > 0.05;

        output.push_back({
            tracks[track].id(),
            detections[matchedIndex].detectionId,
            best,
            std::clamp(
                std::exp(
                    -best),
                0.0,
                1.0),
            margin,
            accepted,
        });
    }

    return output;
}

}  // namespace nox::native::tracking

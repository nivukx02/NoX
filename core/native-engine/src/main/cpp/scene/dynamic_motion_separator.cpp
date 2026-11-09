#include "scene/dynamic_motion_separator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <bit>

namespace nox::native::scene {
namespace {

double clampConfidence(double value) noexcept {
    if(!std::isfinite(value)) return 0.0;
    return std::clamp(value,0.0,1.0);
}
double distanceSquared(const Vec3& a,const Vec3& b) noexcept {
    const double dx=a.x-b.x;
    const double dy=a.y-b.y;
    const double dz=a.z-b.z;
    return dx*dx+dy*dy+dz*dz;
}
double dot(const Vec3& a,const Vec3& b) noexcept {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
double norm(const Vec3& v) noexcept {
    return std::sqrt(dot(v,v));
}
double normalAngle(const Vec3& a,const Vec3& b) noexcept {
    const double na=norm(a),nb=norm(b);
    if(na<1e-9||nb<1e-9)return 0.0;
    return std::acos(std::clamp(dot(a,b)/(na*nb),-1.0,1.0));
}
}

DynamicMotionResult DynamicMotionSeparator::classify(
    std::vector<Point>& current,
    const std::vector<Point>& previous,
    const DynamicMotionOptions& options) const {
    DynamicMotionResult result;
    if(current.empty()||previous.empty())return result;

    const double threshold=
        std::max(
            options.displacementThresholdMeters,
            1e-6);

    const double searchRadius=
        std::max(
            options.maximumMatchDistanceMeters,
            threshold*4.0);

    const double cellSize=searchRadius;
    const double radiusSquared=
        searchRadius*searchRadius;

    struct Cell {
        long long x{0};
        long long y{0};
        long long z{0};

        bool operator==(const Cell& other) const noexcept {
            return x==other.x &&
                y==other.y &&
                z==other.z;
        }
    };

    struct CellHash {
        std::size_t operator()(const Cell& cell) const noexcept {
            const auto mix=[](std::uint64_t value){
                value^=value>>30U;
                value*=0xbf58476d1ce4e5b9ULL;
                value^=value>>27U;
                value*=0x94d049bb133111ebULL;
                value^=value>>31U;
                return value;
            };
            return static_cast<std::size_t>(
                mix(static_cast<std::uint64_t>(cell.x))^
                std::rotl(
                    mix(static_cast<std::uint64_t>(cell.y)),
                    21) ^
                std::rotl(
                    mix(static_cast<std::uint64_t>(cell.z)),
                    42));
        }
    };

    const auto cellFor=[&](const Vec3& point){
        return Cell{
            static_cast<long long>(
                std::floor(point.x/cellSize)),
            static_cast<long long>(
                std::floor(point.y/cellSize)),
            static_cast<long long>(
                std::floor(point.z/cellSize)),
        };
    };

    std::unordered_map<Cell,std::vector<std::size_t>,CellHash>
        spatialIndex;
    spatialIndex.reserve(previous.size()*2U);

    for(std::size_t index=0;
        index<previous.size();
        ++index) {
        spatialIndex[
            cellFor(previous[index].p)
        ].push_back(index);
    }

    double confidenceSum=0.0;
    std::size_t matched=0;

    for(auto& point:current){
        const auto center=cellFor(point.p);

        const Point* nearest=nullptr;
        double nearestDistanceSquared=
            radiusSquared;

        for(long long dz=-1;dz<=1;++dz){
            for(long long dy=-1;dy<=1;++dy){
                for(long long dx=-1;dx<=1;++dx){
                    const auto iterator=
                        spatialIndex.find(
                            Cell{
                                center.x+dx,
                                center.y+dy,
                                center.z+dz,
                            });
                    if(iterator==spatialIndex.end()){
                        continue;
                    }

                    for(const auto index:iterator->second){
                        const auto& candidate=
                            previous[index];

                        const double distance=
                            distanceSquared(
                                point.p,
                                candidate.p);

                        if(distance<
                           nearestDistanceSquared) {
                            nearestDistanceSquared=
                                distance;
                            nearest=&candidate;
                        }
                    }
                }
            }
        }

        ++result.evaluated;

        if(nearest==nullptr){
            ++result.unmatched;
            point.dynamic=false;
            point.confidence=
                std::max(
                    options.confidenceFloor,
                    clampConfidence(point.confidence)*
                        0.92);
            continue;
        }

        ++matched;

        const auto timestampDelta=
            point.lastSeen>nearest->lastSeen
                ? point.lastSeen-nearest->lastSeen
                : nearest->lastSeen-point.lastSeen;

        const bool historyValid=
            timestampDelta<=
            options.maxHistoryGapNanos;

        if(!historyValid){
            ++result.unmatched;
            point.dynamic=false;
            point.confidence=
                std::max(
                    options.confidenceFloor,
                    clampConfidence(point.confidence)*
                        0.94);
            continue;
        }

        const double distance=
            std::sqrt(
                std::max(
                    0.0,
                    nearestDistanceSquared));
        const double angle=
            normalAngle(
                point.normal,
                nearest->normal);

        const double displacementEvidence=
            std::clamp(
                distance/threshold,
                0.0,
                1.0);

        const double normalEvidence=
            std::clamp(
                angle/
                    std::max(
                        options.normalChangeRadians,
                        1e-6),
                0.0,
                1.0);

        const double motionEvidence=
            0.65*displacementEvidence+
            0.35*normalEvidence;

        const bool strongMotion=
            motionEvidence>=
                options.dynamicConfidenceThreshold;

        const bool dynamic=
            strongMotion ||
            nearestDistanceSquared>
                threshold*threshold;

        point.dynamic=dynamic;

        if(dynamic){
            ++result.dynamic;
            point.confidence=
                clampConfidence(
                    point.confidence*
                    (0.88+0.12*motionEvidence));
        }else{
            point.confidence=
                clampConfidence(
                    point.confidence*
                    (0.96+0.04*
                        (1.0-motionEvidence)));
        }

        const double correspondenceConfidence=
            std::exp(
                -distance/
                std::max(
                    threshold*3.0,
                    1e-4));

        confidenceSum+=
            correspondenceConfidence*
            (0.50+
                0.50*
                clampConfidence(
                    nearest->confidence));
    }

    result.dynamicRatio=
        static_cast<double>(result.dynamic)/
        static_cast<double>(
            std::max<std::size_t>(
                result.evaluated,
                1U));

    result.matchRatio=
        static_cast<double>(matched)/
        static_cast<double>(
            std::max<std::size_t>(
                result.evaluated,
                1U));

    const double correspondenceConfidence=
        matched==0
            ? 0.0
            : confidenceSum/
                static_cast<double>(matched);

    const double dynamicSeparationConfidence=
        result.evaluated==0
            ? 0.0
            : (
                result.dynamicRatio<0.85
                    ? 1.0
                    : 0.75);

    result.confidence=
        std::clamp(
            0.55*result.matchRatio+
            0.35*correspondenceConfidence+
            0.10*dynamicSeparationConfidence,
            0.0,
            1.0);

    return result;
}

} // namespace nox::native::scene

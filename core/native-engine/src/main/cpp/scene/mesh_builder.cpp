#include "scene/mesh_builder.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_set>

namespace nox::native::scene {
namespace {
constexpr double kPi=3.1415926535897932384626433832795;
constexpr double kEps=1e-12;

double deg(double r) noexcept { return r*180.0/kPi; }

double acosSafe(double v) noexcept {
    return std::acos(std::clamp(v,-1.0,1.0));
}

std::uint64_t mix64(std::uint64_t v) noexcept {
    v^=v>>30U;
    v*=0xbf58476d1ce4e5b9ULL;
    v^=v>>27U;
    v*=0x94d049bb133111ebULL;
    v^=v>>31U;
    return v;
}
}

MeshBuilder::MeshBuilder(MeshBuildConfig config) noexcept : config_(config) {
    config_.minimumPointConfidence=clamp01(config_.minimumPointConfidence);
    config_.minimumNeighborDistance=std::max(config_.minimumNeighborDistance,1e-8);
    config_.neighborhoodRadius=std::max(
        config_.neighborhoodRadius,
        config_.minimumNeighborDistance*8.0);
    config_.maximumEdgeLength=std::max(
        config_.maximumEdgeLength,
        config_.minimumNeighborDistance*2.0);
    config_.maximumSurfaceDeviation=std::max(
        config_.maximumSurfaceDeviation,1e-6);
    config_.minimumTriangleArea=std::max(
        config_.minimumTriangleArea,1e-12);
    config_.minimumInteriorAngleDegrees=std::clamp(
        config_.minimumInteriorAngleDegrees,0.1,30.0);
    config_.maximumNormalAngleDegrees=std::clamp(
        config_.maximumNormalAngleDegrees,1.0,89.0);
    config_.maximumTriangleAspectRatio=std::max(
        config_.maximumTriangleAspectRatio,1.0);
    config_.maximumVertexUncertaintyMeters =
        std::max(
            config_.maximumVertexUncertaintyMeters,
            1e-5);
    config_.minimumVertexTemporalConsistency =
        clamp01(
            config_.minimumVertexTemporalConsistency);
    config_.maximumVertexDynamicProbability =
        clamp01(
            config_.maximumVertexDynamicProbability);
    config_.uncertaintyQualityScaleMeters =
        std::max(
            config_.uncertaintyQualityScaleMeters,
            1e-5);
    config_.maximumNeighborsPerVertex=std::max<std::size_t>(
        config_.maximumNeighborsPerVertex,3);
    config_.maximumTriangles=std::max<std::size_t>(
        config_.maximumTriangles,1);
    config_.maximumIncidentTrianglesPerEdge=std::clamp<std::size_t>(
        config_.maximumIncidentTrianglesPerEdge,1,8);
}

std::size_t MeshBuilder::CellHash::operator()(const Cell& c) const noexcept {
    const std::uint64_t x=mix64(static_cast<std::uint64_t>(c.x));
    const std::uint64_t y=mix64(static_cast<std::uint64_t>(c.y));
    const std::uint64_t z=mix64(static_cast<std::uint64_t>(c.z));
    const std::uint64_t h=x^std::rotl(y,21)^std::rotl(z,42);
    return static_cast<std::size_t>(h^(h>>32U));
}

double MeshBuilder::dot(const Vec3d&a,const Vec3d&b) noexcept {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

MeshBuilder::Vec3d MeshBuilder::cross(
    const Vec3d&a,const Vec3d&b) noexcept {
    return {
        a.y*b.z-a.z*b.y,
        a.z*b.x-a.x*b.z,
        a.x*b.y-a.y*b.x
    };
}

double MeshBuilder::norm(const Vec3d&v) noexcept {
    return std::sqrt(dot(v,v));
}

MeshBuilder::Vec3d MeshBuilder::normalize(const Vec3d&v) noexcept {
    const double n=norm(v);
    if(n<=kEps||!std::isfinite(n)) return {0.0,0.0,1.0};
    return mul(v,1.0/n);
}

MeshBuilder::Vec3d MeshBuilder::sub(const Vec3d&a,const Vec3d&b) noexcept {
    return {a.x-b.x,a.y-b.y,a.z-b.z};
}

MeshBuilder::Vec3d MeshBuilder::add(const Vec3d&a,const Vec3d&b) noexcept {
    return {a.x+b.x,a.y+b.y,a.z+b.z};
}

MeshBuilder::Vec3d MeshBuilder::mul(
    const Vec3d&a,double s) noexcept {
    return {a.x*s,a.y*s,a.z*s};
}

MeshBuilder::Vec3d MeshBuilder::toVec3d(
    const Vec3&v) noexcept {
    return {v.x,v.y,v.z};
}

Vec3 MeshBuilder::toVec3(const Vec3d&v) noexcept {
    return {v.x,v.y,v.z};
}

double MeshBuilder::clamp01(double v) noexcept {
    if(!std::isfinite(v)) return 0.0;
    return std::clamp(v,0.0,1.0);
}

MeshBuilder::Cell MeshBuilder::cellFor(const Vec3d&v) const noexcept {
    const double s=config_.neighborhoodRadius;
    return {
        static_cast<long long>(std::floor(v.x/s)),
        static_cast<long long>(std::floor(v.y/s)),
        static_cast<long long>(std::floor(v.z/s))
    };
}

std::vector<MeshBuilder::CandidatePoint>
MeshBuilder::filterAndCollapse(
    const std::vector<Point>&points,
    MeshBuildDiagnostics&d) const {
    d.inputPoints=points.size();

    std::vector<CandidatePoint> out;
    out.reserve(points.size());

    std::unordered_map<Cell,std::vector<std::uint32_t>,CellHash> buckets;
    buckets.reserve(points.size());

    const double merge=std::max(
        config_.minimumNeighborDistance,
        config_.neighborhoodRadius*0.01);
    const double merge2=merge*merge;

    for(std::size_t source=0;source<points.size();++source){
        const Point&p=points[source];

        if(config_.rejectDynamicPoints&&p.dynamic){
            ++d.rejectedDynamic;
            continue;
        }

        if(!std::isfinite(p.p.x)||!std::isfinite(p.p.y)||
           !std::isfinite(p.p.z)||
           clamp01(p.confidence)<config_.minimumPointConfidence){
            ++d.rejectedLowConfidence;
            continue;
        }

        const double uncertainty =
            std::isfinite(p.uncertaintyMeters)
                ? std::max(0.0, p.uncertaintyMeters)
                : std::numeric_limits<double>::infinity();

        if (
            uncertainty >
            config_.maximumVertexUncertaintyMeters) {
            ++d.rejectedHighUncertainty;
            continue;
        }

        const double dynamicProbability =
            std::isfinite(p.dynamicProbability)
                ? clamp01(p.dynamicProbability)
                : (
                    p.dynamic
                        ? 1.0
                        : 0.0
                );

        if (
            dynamicProbability >
            config_.maximumVertexDynamicProbability) {
            ++d.rejectedDynamicProbability;
            continue;
        }

        const double temporalConsistency =
            std::isfinite(p.temporalConsistency)
                ? clamp01(p.temporalConsistency)
                : 1.0;

        if (
            temporalConsistency <
            config_.minimumVertexTemporalConsistency) {
            ++d.rejectedTemporalInconsistency;
            continue;
        }

        const Vec3d pos=toVec3d(p.p);
        const Cell center=cellFor(pos);
        bool duplicate=false;

        for(long long z=-1;z<=1&&!duplicate;++z)
        for(long long y=-1;y<=1&&!duplicate;++y)
        for(long long x=-1;x<=1&&!duplicate;++x){
            const auto it=buckets.find(
                Cell{center.x+x,center.y+y,center.z+z});
            if(it==buckets.end()) continue;

            for(const std::uint32_t candidate:it->second){
                const Vec3d q=toVec3d(out[candidate].point.p);
                const Vec3d delta=sub(pos,q);
                if(dot(delta,delta)<=merge2){
                    if(p.confidence>out[candidate].point.confidence){
                        out[candidate]=CandidatePoint{p,source};
                    }
                    duplicate=true;
                    break;
                }
            }
        }

        if(duplicate){
            ++d.rejectedDuplicate;
            continue;
        }

        const auto index=
            static_cast<std::uint32_t>(out.size());
        out.push_back({p,source});
        buckets[center].push_back(index);
    }

    if(config_.deterministicOrdering){
        std::sort(
            out.begin(),
            out.end(),
            [](const CandidatePoint&a,const CandidatePoint&b){
                if(a.point.p.x!=b.point.p.x) return a.point.p.x<b.point.p.x;
                if(a.point.p.y!=b.point.p.y) return a.point.p.y<b.point.p.y;
                if(a.point.p.z!=b.point.p.z) return a.point.p.z<b.point.p.z;
                if(a.point.confidence!=b.point.confidence)
                    return a.point.confidence>b.point.confidence;
                return a.sourceIndex<b.sourceIndex;
            });
    }

    d.acceptedVertices=out.size();
    return out;
}

std::vector<MeshBuilder::Neighbor>
MeshBuilder::collectNeighbors(
    std::uint32_t centerIndex,
    const std::vector<CandidatePoint>&vertices,
    const std::unordered_map<Cell,std::vector<std::uint32_t>,CellHash>&index)
    const {
    if(centerIndex>=vertices.size()) return {};

    const Vec3d center=
        toVec3d(vertices[centerIndex].point.p);
    const Cell cc=cellFor(center);
    const double radius2=
        config_.neighborhoodRadius*config_.neighborhoodRadius;
    const double minDistance2=
        config_.minimumNeighborDistance*
        config_.minimumNeighborDistance;

    std::vector<std::uint32_t> ids;

    for(long long z=-1;z<=1;++z)
    for(long long y=-1;y<=1;++y)
    for(long long x=-1;x<=1;++x){
        const auto it=index.find(
            Cell{cc.x+x,cc.y+y,cc.z+z});
        if(it==index.end()) continue;
        ids.insert(ids.end(),it->second.begin(),it->second.end());
    }

    std::sort(ids.begin(),ids.end());
    ids.erase(std::unique(ids.begin(),ids.end()),ids.end());

    const Vec3d n0=normalize(
        toVec3d(vertices[centerIndex].point.normal));

    std::vector<Neighbor> out;
    out.reserve(ids.size());

    for(const std::uint32_t id:ids){
        if(id==centerIndex) continue;

        const Vec3d delta=
            sub(toVec3d(vertices[id].point.p),center);
        const double d2=dot(delta,delta);

        if(d2<=minDistance2||d2>radius2) continue;

        const double distance=std::sqrt(d2);
        if(distance>config_.maximumEdgeLength) continue;

        const Vec3d n=normalize(
            toVec3d(vertices[id].point.normal));
        const double agreement=
            clamp01((dot(n0,n)+1.0)*0.5);
        const double normalAngle=
            deg(acosSafe(std::abs(dot(n0,n))));

        if(config_.requireNormalAgreement&&
           normalAngle>config_.maximumNormalAngleDegrees){
            continue;
        }

        out.push_back({
            id,
            distance,
            clamp01(vertices[id].point.confidence),
            agreement,
            0.0
        });
    }

    const Vec3d axis=
        std::abs(n0.z)<0.85?
        Vec3d{0.0,0.0,1.0}:Vec3d{0.0,1.0,0.0};
    const Vec3d tx=normalize(cross(axis,n0));
    const Vec3d ty=normalize(cross(n0,tx));

    for(auto&neighbor:out){
        const Vec3d delta=
            sub(
                toVec3d(vertices[neighbor.index].point.p),
                center);
        neighbor.tangentAngle=
            std::atan2(dot(delta,ty),dot(delta,tx));
    }

    std::sort(
        out.begin(),
        out.end(),
        [](const Neighbor&a,const Neighbor&b){
            if(a.tangentAngle!=b.tangentAngle)
                return a.tangentAngle<b.tangentAngle;
            if(a.distance!=b.distance)
                return a.distance<b.distance;
            return a.index<b.index;
        });

    if(out.size()>config_.maximumNeighborsPerVertex){
        out.resize(config_.maximumNeighborsPerVertex);
    }

    return out;
}

MeshBuilder::Vec3d MeshBuilder::normalFor(
    std::uint32_t index,
    const std::vector<CandidatePoint>&vertices,
    const std::vector<Neighbor>&neighbors) const {
    if(index>=vertices.size()) return {0.0,0.0,1.0};

    const Vec3d supplied=
        toVec3d(vertices[index].point.normal);
    if(norm(supplied)>kEps) return normalize(supplied);

    if(neighbors.size()<2) return {0.0,0.0,1.0};

    const Vec3d center=
        toVec3d(vertices[index].point.p);
    Vec3d total{};

    const std::size_t limit=
        std::min<std::size_t>(neighbors.size(),12);

    for(std::size_t i=0;i<limit;++i){
        const std::size_t j=(i+1U)%limit;
        if(i==j) continue;

        const Vec3d a=sub(
            toVec3d(vertices[neighbors[i].index].point.p),
            center);
        const Vec3d b=sub(
            toVec3d(vertices[neighbors[j].index].point.p),
            center);
        const Vec3d n=cross(a,b);
        const double weight=norm(n);

        if(weight>kEps){
            total=add(
                total,
                mul(normalize(n),weight));
        }
    }

    return normalize(total);
}

double MeshBuilder::angleDeg(
    const Vec3d&a,const Vec3d&b,const Vec3d&c) noexcept {
    const Vec3d ba=sub(a,b);
    const Vec3d bc=sub(c,b);
    const double na=norm(ba);
    const double nc=norm(bc);

    if(na<=kEps||nc<=kEps) return 0.0;

    return deg(
        acosSafe(dot(ba,bc)/(na*nc)));
}

bool MeshBuilder::acceptTriangle(
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t c,
    const std::vector<CandidatePoint>&vertices,
    double&quality,
    double&edgeLength,
    double&normalAgreement) const {
    if(a==b||b==c||a==c||
       a>=vertices.size()||
       b>=vertices.size()||
       c>=vertices.size()){
        return false;
    }

    const Vec3d pa=toVec3d(vertices[a].point.p);
    const Vec3d pb=toVec3d(vertices[b].point.p);
    const Vec3d pc=toVec3d(vertices[c].point.p);

    const Vec3d ab=sub(pb,pa);
    const Vec3d ac=sub(pc,pa);
    const Vec3d bc=sub(pc,pb);

    const double lab=norm(ab);
    const double lac=norm(ac);
    const double lbc=norm(bc);

    edgeLength=std::max({lab,lac,lbc});

    if(!std::isfinite(edgeLength)||
       lab>config_.maximumEdgeLength||
       lac>config_.maximumEdgeLength||
       lbc>config_.maximumEdgeLength){
        return false;
    }

    const Vec3d rawFace=cross(ab,ac);
    const double twiceArea=norm(rawFace);

    if(!std::isfinite(twiceArea)||
       twiceArea<2.0*config_.minimumTriangleArea){
        return false;
    }

    const double smallestEdge=
        std::max(
            config_.minimumNeighborDistance,
            std::min({lab,lac,lbc}));
    const double aspect=edgeLength/smallestEdge;

    if(aspect>config_.maximumTriangleAspectRatio){
        return false;
    }

    const double minAngle=std::min({
        angleDeg(pb,pa,pc),
        angleDeg(pa,pb,pc),
        angleDeg(pa,pc,pb)
    });

    if(minAngle<config_.minimumInteriorAngleDegrees){
        return false;
    }

    const Vec3d face=normalize(rawFace);
    const Vec3d na=normalFor(a,vertices);
    const Vec3d nb=normalFor(b,vertices);
    const Vec3d nc=normalFor(c,vertices);

    const double qa=std::abs(dot(face,na));
    const double qb=std::abs(dot(face,nb));
    const double qc=std::abs(dot(face,nc));

    normalAgreement=(qa+qb+qc)/3.0;

    const double threshold=std::cos(
        config_.maximumNormalAngleDegrees*kPi/180.0);

    if(config_.requireNormalAgreement&&
       std::min({qa,qb,qc})<threshold){
        return false;
    }

    const Vec3d centroid=
        mul(add(add(pa,pb),pc),1.0/3.0);
    const Vec3d reference=
        normalize(add(add(na,nb),nc));
    const double deviation=
        std::abs(dot(sub(centroid,pa),reference));

    if(!std::isfinite(deviation)||
       deviation>config_.maximumSurfaceDeviation){
        return false;
    }

    const double areaQ=clamp01(
        twiceArea/
        std::max(
            2.0*config_.minimumTriangleArea,
            config_.maximumEdgeLength*
            config_.maximumEdgeLength));

    const double angleQ=clamp01(
        (minAngle-config_.minimumInteriorAngleDegrees)/
        std::max(
            1.0,
            60.0-config_.minimumInteriorAngleDegrees));

    const double aspectQ=clamp01(
        1.0-(aspect-1.0)/
        std::max(
            1.0,
            config_.maximumTriangleAspectRatio-1.0));

    const double planeQ=clamp01(
        1.0-deviation/
        config_.maximumSurfaceDeviation);

    const double confidenceQ=(
        clamp01(vertices[a].point.confidence)+
        clamp01(vertices[b].point.confidence)+
        clamp01(vertices[c].point.confidence))/3.0;

    const double uncertaintyA =
        std::isfinite(
            vertices[a].point.uncertaintyMeters)
            ? std::max(
                0.0,
                vertices[a].point.uncertaintyMeters)
            : config_.maximumVertexUncertaintyMeters;

    const double uncertaintyB =
        std::isfinite(
            vertices[b].point.uncertaintyMeters)
            ? std::max(
                0.0,
                vertices[b].point.uncertaintyMeters)
            : config_.maximumVertexUncertaintyMeters;

    const double uncertaintyC =
        std::isfinite(
            vertices[c].point.uncertaintyMeters)
            ? std::max(
                0.0,
                vertices[c].point.uncertaintyMeters)
            : config_.maximumVertexUncertaintyMeters;

    const double dynamicA =
        clamp01(
            vertices[a].point.dynamicProbability);
    const double dynamicB =
        clamp01(
            vertices[b].point.dynamicProbability);
    const double dynamicC =
        clamp01(
            vertices[c].point.dynamicProbability);

    const double temporalA =
        clamp01(
            vertices[a].point.temporalConsistency);
    const double temporalB =
        clamp01(
            vertices[b].point.temporalConsistency);
    const double temporalC =
        clamp01(
            vertices[c].point.temporalConsistency);

    const double triangleUncertainty =
        (
            uncertaintyA +
            uncertaintyB +
            uncertaintyC
        ) /
        3.0;

    const double triangleDynamicRisk =
        (
            dynamicA +
            dynamicB +
            dynamicC
        ) /
        3.0;

    const double triangleTemporalConsistency =
        (
            temporalA +
            temporalB +
            temporalC
        ) /
        3.0;

    const double uncertaintyQ =
        std::exp(
            -triangleUncertainty /
            config_.uncertaintyQualityScaleMeters);

    const double dynamicQ =
        1.0 -
        triangleDynamicRisk;

    const double temporalQ =
        triangleTemporalConsistency;

    quality=
        areaQ*0.12+
        angleQ*0.14+
        aspectQ*0.12+
        normalAgreement*0.20+
        planeQ*0.10+
        confidenceQ*0.10+
        uncertaintyQ*0.08+
        dynamicQ*0.06+
        temporalQ*0.08;

    return std::isfinite(quality)&&quality>0.0;
}

std::uint64_t MeshBuilder::edgeKey(
    std::uint32_t a,
    std::uint32_t b) noexcept {
    if(a>b) std::swap(a,b);
    return (static_cast<std::uint64_t>(a)<<32U) |
        static_cast<std::uint64_t>(b);
}

std::uint64_t MeshBuilder::triangleKey(
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t c) noexcept {
    std::array<std::uint32_t,3> ids{a,b,c};
    std::sort(ids.begin(),ids.end());

    std::uint64_t hash=1469598103934665603ULL;
    for(const auto id:ids){
        hash^=static_cast<std::uint64_t>(id);
        hash*=1099511628211ULL;
        hash=mix64(hash);
    }
    return hash;
}

MeshBuilder::Result MeshBuilder::buildDetailed(
    const std::vector<Point>&points) const {
    Result result;
    auto&diagnostics=result.diagnostics;

    const auto vertices=
        filterAndCollapse(points,diagnostics);

    if(vertices.size()<3) return result;

    std::unordered_map<
        Cell,
        std::vector<std::uint32_t>,
        CellHash> index;

    index.reserve(vertices.size());

    for(std::uint32_t i=0;
        i<vertices.size();
        ++i){
        index[
            cellFor(
                toVec3d(vertices[i].point.p))
        ].push_back(i);
    }

    std::unordered_set<std::uint64_t> seen;
    seen.reserve(
        std::min<std::size_t>(
            config_.maximumTriangles*2U,
            vertices.size()*8U));

    std::vector<TriangleCandidate> candidates;
    candidates.reserve(
        std::min<std::size_t>(
            config_.maximumTriangles*2U,
            vertices.size()*4U));

    for(std::uint32_t center=0;
        center<vertices.size();
        ++center){
        const auto neighbors=
            collectNeighbors(center,vertices,index);

        diagnostics.candidateEdges+=neighbors.size();
        if(neighbors.size()<2) continue;

        for(std::size_t i=0;
            i<neighbors.size();
            ++i){
            const std::size_t j=
                (i+1U)%neighbors.size();

            const std::uint32_t a=center;
            const std::uint32_t b=neighbors[i].index;
            const std::uint32_t c=neighbors[j].index;

            if(a==b||a==c||b==c) continue;

            if(!seen.insert(
                triangleKey(a,b,c)).second){
                continue;
            }

            double quality=0.0;
            double edge=0.0;
            double normal=0.0;

            if(!acceptTriangle(
                a,b,c,vertices,
                quality,edge,normal)){
                ++diagnostics.rejectedTriangles;
                continue;
            }

            std::uint32_t orientedA=a;
            std::uint32_t orientedB=b;
            std::uint32_t orientedC=c;

            if(config_.orientTrianglesToNormals){
                const Vec3d pa=
                    toVec3d(vertices[orientedA].point.p);
                const Vec3d pb=
                    toVec3d(vertices[orientedB].point.p);
                const Vec3d pc=
                    toVec3d(vertices[orientedC].point.p);
                const Vec3d face=
                    normalize(
                        cross(
                            sub(pb,pa),
                            sub(pc,pa)));
                const Vec3d reference=
                    normalize(
                        add(
                            add(
                                normalFor(orientedA,vertices),
                                normalFor(orientedB,vertices)),
                            normalFor(orientedC,vertices)));
                if(dot(face,reference)<0.0){
                    std::swap(orientedB,orientedC);
                }
            }

            candidates.push_back({
                orientedA,
                orientedB,
                orientedC,
                quality,
                edge,
                normal});
        }
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const TriangleCandidate&a,
           const TriangleCandidate&b){
            if(a.quality!=b.quality)
                return a.quality>b.quality;
            if(a.a!=b.a) return a.a<b.a;
            if(a.b!=b.b) return a.b<b.b;
            return a.c<b.c;
        });

    if(candidates.size()>config_.maximumTriangles){
        candidates.resize(config_.maximumTriangles);
    }

    if(config_.enforceManifoldEdges&&!candidates.empty()){
        std::unordered_map<std::uint64_t,std::uint8_t> edgeIncidence;
        edgeIncidence.reserve(candidates.size()*2U);

        std::vector<TriangleCandidate> filtered;
        filtered.reserve(candidates.size());

        for(const auto&candidate:candidates){
            const std::uint64_t edges[3]={
                edgeKey(candidate.a,candidate.b),
                edgeKey(candidate.b,candidate.c),
                edgeKey(candidate.c,candidate.a),
            };

            bool acceptable=true;
            for(const auto edgeId:edges){
                const auto it=edgeIncidence.find(edgeId);
                const std::size_t count=
                    it==edgeIncidence.end()?
                    0U:
                    static_cast<std::size_t>(it->second);
                if(count>=config_.maximumIncidentTrianglesPerEdge){
                    acceptable=false;
                    break;
                }
            }

            if(!acceptable){
                ++diagnostics.rejectedNonManifold;
                continue;
            }

            for(const auto edgeId:edges){
                auto&count=edgeIncidence[edgeId];
                if(count<255U) ++count;
            }
            filtered.push_back(candidate);
        }

        candidates.swap(filtered);

        diagnostics.nonManifoldEdgeCount=0;
        std::size_t boundaryEdges=0;
        for(const auto&entry:edgeIncidence){
            if(entry.second>config_.maximumIncidentTrianglesPerEdge){
                ++diagnostics.nonManifoldEdgeCount;
            }
            if(entry.second==1U){
                ++boundaryEdges;
            }
        }
        diagnostics.boundaryEdgeRatio=
            edgeIncidence.empty()?
            0.0:
            static_cast<double>(boundaryEdges)/
            static_cast<double>(edgeIncidence.size());
    }

    const std::uint32_t invalid=
        std::numeric_limits<std::uint32_t>::max();

    std::vector<std::uint32_t> remap(
        vertices.size(),invalid);
    std::vector<std::uint32_t> used;
    used.reserve(candidates.size()*3U);

    for(const auto&candidate:candidates){
        for(const std::uint32_t id:{
            candidate.a,candidate.b,candidate.c}){
            if(remap[id]==invalid){
                remap[id]=
                    static_cast<std::uint32_t>(
                        used.size());
                used.push_back(id);
            }
        }
    }

    result.mesh.vertices.reserve(used.size());
    result.mesh.vertexConfidence.reserve(used.size());
    result.mesh.vertexNormals.reserve(used.size());
    result.mesh.vertexUncertaintyMeters.reserve(used.size());
    result.mesh.vertexDynamicProbability.reserve(used.size());
    result.mesh.vertexTemporalConsistency.reserve(used.size());

    for(const std::uint32_t id:used){
        result.mesh.vertices.push_back(
            vertices[id].point.p);

        result.mesh.vertexConfidence.push_back(
            static_cast<float>(
                clamp01(
                    vertices[id]
                        .point
                        .confidence)));

        result.mesh.vertexNormals.push_back(
            toVec3(
                normalFor(id,vertices)));

        result.mesh.vertexUncertaintyMeters.push_back(
            static_cast<float>(
                std::isfinite(
                    vertices[id].point.uncertaintyMeters)
                    ? std::max(
                        0.0,
                        vertices[id].point.uncertaintyMeters)
                    : config_.maximumVertexUncertaintyMeters));

        result.mesh.vertexDynamicProbability.push_back(
            static_cast<float>(
                clamp01(
                    vertices[id]
                        .point
                        .dynamicProbability)));

        result.mesh.vertexTemporalConsistency.push_back(
            static_cast<float>(
                clamp01(
                    vertices[id]
                        .point
                        .temporalConsistency)));
    }

    result.mesh.indices.reserve(
        candidates.size()*3U);
    result.mesh.triangleQuality.reserve(
        candidates.size());

    double qualitySum=0.0;
    double edgeSum=0.0;
    double normalSum=0.0;
    double uncertaintySum=0.0;
    double dynamicSum=0.0;
    double temporalSum=0.0;
    std::vector<double> vertexUncertainties;
    vertexUncertainties.reserve(
        used.size());

    for(const auto&candidate:candidates){
        result.mesh.indices.push_back(
            remap[candidate.a]);
        result.mesh.indices.push_back(
            remap[candidate.b]);
        result.mesh.indices.push_back(
            remap[candidate.c]);

        result.mesh.triangleQuality.push_back(
            static_cast<float>(
                clamp01(
                    candidate.quality)));

        qualitySum+=candidate.quality;
        edgeSum+=candidate.edgeLength;
        normalSum+=candidate.normalAgreement;
    }

    diagnostics.acceptedTriangles=
        candidates.size();

    for (const std::uint32_t id :
         used) {
        const double uncertainty =
            std::isfinite(
                vertices[id].point.uncertaintyMeters)
                ? std::max(
                    0.0,
                    vertices[id].point.uncertaintyMeters)
                : config_.maximumVertexUncertaintyMeters;

        uncertaintySum +=
            uncertainty;
        dynamicSum +=
            clamp01(
                vertices[id]
                    .point
                    .dynamicProbability);
        temporalSum +=
            clamp01(
                vertices[id]
                    .point
                    .temporalConsistency);
        vertexUncertainties.push_back(
            uncertainty);
    }

    if (!used.empty()) {
        diagnostics.meanVertexUncertaintyMeters =
            uncertaintySum /
            static_cast<double>(
                used.size());
        diagnostics.p95VertexUncertaintyMeters =
            [&vertexUncertainties]() {
                if (vertexUncertainties.empty()) return 0.0;
                std::sort(
                    vertexUncertainties.begin(),
                    vertexUncertainties.end());
                const double position =
                    0.95 *
                    static_cast<double>(
                        vertexUncertainties.size() - 1U);
                const auto lower =
                    static_cast<std::size_t>(
                        std::floor(position));
                const auto upper =
                    std::min(
                        vertexUncertainties.size() - 1U,
                        lower + 1U);
                const double alpha =
                    position -
                    static_cast<double>(
                        lower);
                return vertexUncertainties[lower] +
                    (
                        vertexUncertainties[upper] -
                        vertexUncertainties[lower]
                    ) *
                    alpha;
            }();

        diagnostics.meanDynamicProbability =
            dynamicSum /
            static_cast<double>(
                used.size());

        diagnostics.meanTemporalConsistency =
            temporalSum /
            static_cast<double>(
                used.size());

        std::size_t trusted = 0U;
        for (const std::uint32_t id :
             used) {
            const double uncertainty =
                std::isfinite(
                    vertices[id].point.uncertaintyMeters)
                    ? std::max(
                        0.0,
                        vertices[id].point.uncertaintyMeters)
                    : config_.maximumVertexUncertaintyMeters;

            if (
                uncertainty <=
                    config_.maximumVertexUncertaintyMeters * 0.50 &&
                clamp01(
                    vertices[id]
                        .point
                        .dynamicProbability) <=
                    config_.maximumVertexDynamicProbability * 0.50 &&
                clamp01(
                    vertices[id]
                        .point
                        .temporalConsistency) >=
                    std::max(
                        0.50,
                        config_.minimumVertexTemporalConsistency)) {
                ++trusted;
            }
        }

        diagnostics.trustedVertexRatio =
            static_cast<double>(
                trusted) /
            static_cast<double>(
                used.size());
    }

    if(!candidates.empty()){
        diagnostics.meanTriangleQuality=
            qualitySum/
            static_cast<double>(
                candidates.size());

        diagnostics.meanEdgeLength=
            edgeSum/
            static_cast<double>(
                candidates.size());

        diagnostics.meanNormalAgreement=
            normalSum/
            static_cast<double>(
                candidates.size());
    }

    const double coverage=
        points.empty()?0.0:
        static_cast<double>(
            used.size())/
        static_cast<double>(
            points.size());

    if(!config_.enforceManifoldEdges){
        std::unordered_map<std::uint64_t,std::uint8_t> edgeIncidence;
        edgeIncidence.reserve(candidates.size()*2U);
        for(const auto&candidate:candidates){
            ++edgeIncidence[edgeKey(candidate.a,candidate.b)];
            ++edgeIncidence[edgeKey(candidate.b,candidate.c)];
            ++edgeIncidence[edgeKey(candidate.c,candidate.a)];
        }
        std::size_t boundaryEdges=0;
        std::size_t nonManifoldEdges=0;
        for(const auto&entry:edgeIncidence){
            if(entry.second==1U) ++boundaryEdges;
            if(entry.second>2U) ++nonManifoldEdges;
        }
        diagnostics.boundaryEdgeRatio=
            edgeIncidence.empty()?
            0.0:
            static_cast<double>(boundaryEdges)/
            static_cast<double>(edgeIncidence.size());
        diagnostics.nonManifoldEdgeCount=nonManifoldEdges;
    }

    const double topology=
        candidates.empty()?0.0:
        std::min(
            1.0,
            static_cast<double>(
                candidates.size())/
            std::max(
                1.0,
                static_cast<double>(
                    used.size())*0.5));

    const double manifoldQuality=
        diagnostics.nonManifoldEdgeCount>0U?
        0.0:
        (config_.maximumIncidentTrianglesPerEdge<=2U?
            1.0:
            0.96);

    const double boundaryQuality=
        1.0-std::min(
            0.20,
            diagnostics.boundaryEdgeRatio*0.08);

    result.mesh.confidence=clamp01(
        coverage*0.18+
        diagnostics.meanTriangleQuality*0.40+
        diagnostics.meanNormalAgreement*0.20+
        topology*0.12+
        manifoldQuality*0.06+
        boundaryQuality*0.04);

    diagnostics.confidence=
        result.mesh.confidence;

    return result;
}

Mesh MeshBuilder::build(
    const std::vector<Point>&points) const {
    return buildDetailed(points).mesh;
}

} // namespace nox::native::scene

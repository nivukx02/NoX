#include "scene/sparse_tsdf_volume.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nox::native::scene {

namespace {

struct Corner {
    std::array<std::int32_t,3> voxel{};
    math::Vec3d position{};
    float value{1.0F};
    float weight{0.0F};
};

struct VertexKeyHash {
    std::size_t operator()(
        const std::array<std::int64_t,3>& key) const noexcept {
        const auto x=static_cast<std::uint64_t>(key[0])*0x9E3779B97F4A7C15ULL;
        const auto y=static_cast<std::uint64_t>(key[1])*0xC2B2AE3D27D4EB4FULL;
        const auto z=static_cast<std::uint64_t>(key[2])*0x165667B19E3779F9ULL;
        return static_cast<std::size_t>(x^(y<<1)^(z<<7));
    }
};

struct VertexAccumulator {
    math::Vec3d position{};
    math::Vec3d normal{};
    double uncertaintyMeters{0.0};
    double dynamicProbability{0.0};
    double confidenceWeight{0.0};
};

constexpr std::array<std::array<int,3>,8> CornerOffsets{{
    {{0,0,0}}, {{1,0,0}}, {{1,1,0}}, {{0,1,0}},
    {{0,0,1}}, {{1,0,1}}, {{1,1,1}}, {{0,1,1}},
}};

constexpr std::array<std::array<int,4>,6> Tetrahedra{{
    {{0,5,1,6}}, {{0,1,2,6}}, {{0,2,3,6}},
    {{0,3,7,6}}, {{0,7,4,6}}, {{0,4,5,6}},
}};

double dot(const math::Vec3d& a,const math::Vec3d& b) noexcept {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

double voxelDynamic(
    const SparseTsdfVolume& volume,
    const Corner& corner) noexcept {
    const auto voxel =
        volume.sample(
            corner.voxel[0],
            corner.voxel[1],
            corner.voxel[2]);
    return std::clamp(
        static_cast<double>(voxel.dynamicProbability),
        0.0,
        1.0);
}

math::Vec3d normalize(math::Vec3d value) noexcept {
    const double n=std::sqrt(dot(value,value));
    if(n<1e-12)return {0.0,0.0,1.0};
    return value*(1.0/n);
}

math::Vec3d interpolate(
    const math::Vec3d& a,
    const math::Vec3d& b,
    float va,
    float vb,
    float iso) noexcept {
    const double denominator=static_cast<double>(vb-va);
    if(std::abs(denominator)<1e-7)return (a+b)*0.5;
    const double t=std::clamp(
        (static_cast<double>(iso-va))/denominator,0.0,1.0);
    return a+(b-a)*t;
}

std::array<std::int64_t,3> quantizeVertex(
    const math::Vec3d& point) noexcept {
    constexpr double kResolution=1e-5;
    return {
        static_cast<std::int64_t>(std::llround(point.x/kResolution)),
        static_cast<std::int64_t>(std::llround(point.y/kResolution)),
        static_cast<std::int64_t>(std::llround(point.z/kResolution)),
    };
}

math::Vec3d tsdfGradient(
    const SparseTsdfVolume& volume,
    const std::array<std::int32_t,3>& voxel) noexcept {
    const auto center=volume.sample(
        voxel[0],voxel[1],voxel[2]);

    if(center.weight<=0.0F) return {};

    const auto xm=volume.sample(voxel[0]-1,voxel[1],voxel[2]);
    const auto xp=volume.sample(voxel[0]+1,voxel[1],voxel[2]);
    const auto ym=volume.sample(voxel[0],voxel[1]-1,voxel[2]);
    const auto yp=volume.sample(voxel[0],voxel[1]+1,voxel[2]);
    const auto zm=volume.sample(voxel[0],voxel[1],voxel[2]-1);
    const auto zp=volume.sample(voxel[0],voxel[1],voxel[2]+1);

    auto derivative=[
        &center](
        TsdfVoxel negative,
        TsdfVoxel positive) noexcept {
        if(negative.weight>0.0F && positive.weight>0.0F) {
            return static_cast<double>(
                positive.distance-negative.distance);
        }
        if(positive.weight>0.0F) {
            return static_cast<double>(
                positive.distance-center.distance);
        }
        if(negative.weight>0.0F) {
            return static_cast<double>(
                center.distance-negative.distance);
        }
        return 0.0;
    };

    const auto gradient=math::Vec3d{
        derivative(xm,xp),
        derivative(ym,yp),
        derivative(zm,zp),
    };
    if(dot(gradient,gradient)<1e-12) return {};
    return normalize(gradient);
}

std::uint32_t addOrAccumulateVertex(
    std::vector<VertexAccumulator>& accumulators,
    std::unordered_map<std::array<std::int64_t,3>,
        std::uint32_t,VertexKeyHash>& indices,
    const SparseTsdfVolume& volume,
    const Corner& corner,
    const math::Vec3d& fallbackNormal,
    double uncertaintyMeters) {
    const auto key=quantizeVertex(corner.position);
    const auto found=indices.find(key);
    const auto gradient=tsdfGradient(volume,corner.voxel);
    const auto normal = (gradient.x*gradient.x+
                         gradient.y*gradient.y+
                         gradient.z*gradient.z)>1e-12
        ? gradient
        : fallbackNormal;
    const double contribution=
        std::max(0.01,static_cast<double>(corner.weight));

    if(found!=indices.end()) {
        auto& accumulated=accumulators[found->second];
        const double nextWeight=accumulated.confidenceWeight+contribution;
        accumulated.normal=
            accumulated.normal*(accumulated.confidenceWeight/nextWeight)+
            normal*(contribution/nextWeight);
        accumulated.position=
            accumulated.position*(accumulated.confidenceWeight/nextWeight)+
            corner.position*(contribution/nextWeight);
        accumulated.uncertaintyMeters=
            std::min(
                accumulated.uncertaintyMeters,
                uncertaintyMeters);
        accumulated.dynamicProbability=
            (
                accumulated.dynamicProbability *
                    accumulated.confidenceWeight +
                voxelDynamic(
                    volume,
                    corner) *
                    contribution
            ) /
            std::max(
                nextWeight,
                1e-9);
        accumulated.confidenceWeight=nextWeight;
        return found->second;
    }

    const auto index=static_cast<std::uint32_t>(accumulators.size());
    indices.emplace(key,index);
    accumulators.push_back({
        corner.position,
        normal,
        uncertaintyMeters,
        voxelDynamic(
            volume,
            corner),
        contribution,
    });
    return index;
}

void appendTriangle(
    TsdfMesh& mesh,
    std::vector<VertexAccumulator>& accumulators,
    std::unordered_map<std::array<std::int64_t,3>,
        std::uint32_t,VertexKeyHash>& indices,
    const SparseTsdfVolume& volume,
    const Corner& a,
    const Corner& b,
    const Corner& c,
    double uncertaintyMeters,
    const SparseTsdfConfig& config,
    std::size_t maxTriangles) {
    if(mesh.triangles.size()>=maxTriangles)return;

    const auto edge1=b.position-a.position;
    const auto edge2=c.position-a.position;
    const auto edge3=c.position-b.position;

    const double edge1LengthSquared =
        edge1.x*edge1.x +
        edge1.y*edge1.y +
        edge1.z*edge1.z;
    const double edge2LengthSquared =
        edge2.x*edge2.x +
        edge2.y*edge2.y +
        edge2.z*edge2.z;
    const double edge3LengthSquared =
        edge3.x*edge3.x +
        edge3.y*edge3.y +
        edge3.z*edge3.z;
    const double maximumEdgeSquared =
        std::max(
            edge1LengthSquared,
            std::max(
                edge2LengthSquared,
                edge3LengthSquared));

    const auto crossProduct=math::Vec3d{
        edge1.y*edge2.z-edge1.z*edge2.y,
        edge1.z*edge2.x-edge1.x*edge2.z,
        edge1.x*edge2.y-edge1.y*edge2.x,
    };
    const double areaDouble =
        0.5 *
        std::sqrt(
            std::max(
                0.0,
                crossProduct.x*crossProduct.x +
                    crossProduct.y*crossProduct.y +
                    crossProduct.z*crossProduct.z));

    if(
        !std::isfinite(maximumEdgeSquared) ||
        maximumEdgeSquared >
            config.maximumTriangleEdgeMeters *
            config.maximumTriangleEdgeMeters ||
        !std::isfinite(areaDouble) ||
        areaDouble*areaDouble <
            config.minimumTriangleAreaSqMeters ||
        !std::isfinite(uncertaintyMeters) ||
        uncertaintyMeters >
            config.maximumTriangleUncertaintyMeters) {
        ++mesh.rejectedTriangleCount;
        return;
    }

    const auto fallbackNormal=normalize(crossProduct);
    const auto ia=addOrAccumulateVertex(
        accumulators,indices,volume,a,fallbackNormal,uncertaintyMeters);
    const auto ib=addOrAccumulateVertex(
        accumulators,indices,volume,b,fallbackNormal,uncertaintyMeters);
    const auto ic=addOrAccumulateVertex(
        accumulators,indices,volume,c,fallbackNormal,uncertaintyMeters);
    if(ia==ib||ib==ic||ia==ic)return;
    mesh.triangles.push_back({ia,ib,ic});
}

}

std::size_t SparseTsdfVolume::KeyHash::operator()(
    const std::array<std::int32_t,3>& key) const noexcept {
    const std::uint64_t x=
        static_cast<std::uint32_t>(key[0])*0x9E3779B1u;
    const std::uint64_t y=
        static_cast<std::uint32_t>(key[1])*0x85EBCA77u;
    const std::uint64_t z=
        static_cast<std::uint32_t>(key[2])*0xC2B2AE3Du;
    return static_cast<std::size_t>(x^(y<<1)^(z<<7));
}

SparseTsdfVolume::SparseTsdfVolume(SparseTsdfConfig config)
    : config_(config) {
    config_.voxelSize=std::max(config_.voxelSize,1e-4);
    config_.truncationDistance=
        std::max(config_.truncationDistance,config_.voxelSize);
    config_.maxWeight=std::max(config_.maxWeight,1.0F);
    config_.blockResolution=
        std::max<std::int32_t>(config_.blockResolution,1);
    config_.maxResidentVoxels=
        std::max<std::size_t>(config_.maxResidentVoxels,1024u);
    config_.maxResidentMemoryBytes=
        std::max<std::size_t>(
            config_.maxResidentMemoryBytes,
            1u * 1024u * 1024u);
    config_.estimatedBytesPerVoxel=
        std::max<std::size_t>(
            config_.estimatedBytesPerVoxel,
            sizeof(TsdfVoxel));
    config_.minimumTriangleAreaSqMeters=
        std::max(0.0, config_.minimumTriangleAreaSqMeters);
    config_.maximumTriangleEdgeMeters=
        std::max(
            config_.voxelSize * 2.0,
            config_.maximumTriangleEdgeMeters);
    config_.maximumTriangleUncertaintyMeters=
        std::max(
            config_.voxelSize,
            config_.maximumTriangleUncertaintyMeters);
}

void SparseTsdfVolume::reset() {
    voxels_.clear();
    last_extracted_mesh_.reset();
    rejected_voxel_allocations_=0;
}

void SparseTsdfVolume::releaseMemory() noexcept {
    decltype(voxels_) empty;
    voxels_.swap(empty);

    // The extracted mesh can dominate resident memory after a high-detail reconstruction.
    // Releasing the TSDF volume must release that cache too; otherwise a memory-pressure trim
    // appears successful while hundreds of megabytes remain retained by the last mesh.
    last_extracted_mesh_.reset();

    rejected_voxel_allocations_=0;
}

std::array<std::int32_t,3> SparseTsdfVolume::worldToVoxel(
    const math::Vec3d& point) const noexcept {
    return {
        static_cast<std::int32_t>(
            std::floor(point.x/config_.voxelSize)),
        static_cast<std::int32_t>(
            std::floor(point.y/config_.voxelSize)),
        static_cast<std::int32_t>(
            std::floor(point.z/config_.voxelSize)),
    };
}

math::Vec3d SparseTsdfVolume::voxelToWorld(
    std::int32_t x,
    std::int32_t y,
    std::int32_t z) const noexcept {
    return {
        (static_cast<double>(x)+0.5)*config_.voxelSize,
        (static_cast<double>(y)+0.5)*config_.voxelSize,
        (static_cast<double>(z)+0.5)*config_.voxelSize,
    };
}

void SparseTsdfVolume::integrateVoxel(
    std::int32_t x,
    std::int32_t y,
    std::int32_t z,
    float signedDistance,
    float weight) noexcept {
    const std::array<std::int32_t,3> key{x,y,z};
    auto iterator=voxels_.find(key);
    if(iterator==voxels_.end()) {
        const auto memoryLimited =
            voxels_.size() >= config_.maxResidentVoxels ||
            voxels_.size() >=
                config_.maxResidentMemoryBytes /
                std::max<std::size_t>(
                    config_.estimatedBytesPerVoxel,
                    sizeof(TsdfVoxel));
        if(memoryLimited) {
            ++rejected_voxel_allocations_;
            return;
        }
        iterator=voxels_.emplace(
            key,
            TsdfVoxel{}).first;
    }

    auto& voxel=iterator->second;

    const double residual =
        std::abs(
            static_cast<double>(voxel.distance) -
            static_cast<double>(signedDistance));
    const double huberScale = 0.18;
    const double robustScale =
        residual <= huberScale
            ? 1.0
            : std::clamp(
                huberScale /
                    std::max(residual,1e-9),
                0.04,
                1.0);

    const float safeWeight=
        std::clamp(
            weight *
                static_cast<float>(robustScale),
            0.0F,
            config_.maxWeight);
    const float total=
        std::min(config_.maxWeight,voxel.weight+safeWeight);
    if(total<=0.0F)return;
    const double priorWeight =
        static_cast<double>(voxel.weight);
    const double measurementWeight =
        static_cast<double>(safeWeight);

    voxel.distance =
        static_cast<float>(
            (
                static_cast<double>(voxel.distance) *
                    priorWeight +
                static_cast<double>(signedDistance) *
                    measurementWeight
            ) /
            std::max(
                static_cast<double>(total),
                1e-9));

    voxel.weight =
        total;

    voxel.variance =
        std::max(
            1e-6F,
            voxel.variance *
                static_cast<float>(
                    priorWeight /
                    std::max(
                        static_cast<double>(total),
                        1e-9)));
    voxel.confidence =
        std::clamp(
            0.65F *
                voxel.confidence +
            0.35F *
                std::clamp(
                    safeWeight /
                        std::max(
                            config_.maxWeight,
                            1.0F),
                    0.0F,
                    1.0F),
            0.0F,
            1.0F);
    voxel.observationCount =
        std::min<std::uint64_t>(
            std::uint64_t{1'000'000'000ULL},
            voxel.observationCount + std::uint64_t{1ULL});
}

void SparseTsdfVolume::integrateSurfacePointHighEnd(
    const math::Vec3d& origin,
    const math::Vec3d& direction,
    double surfaceDepth,
    double observationWeight,
    double uncertaintyMeters,
    double temporalConsistency,
    double appearanceConsistency,
    double contactConsistency,
    double dynamicProbability) {
    if (
        !std::isfinite(surfaceDepth) ||
        surfaceDepth <= 0.0 ||
        !std::isfinite(observationWeight) ||
        observationWeight <= 0.0) {
        return;
    }

    const auto ray =
        normalize(direction);

    const double safeUncertainty =
        std::clamp(
            std::isfinite(uncertaintyMeters)
                ? uncertaintyMeters
                : 0.05,
            0.0005,
            2.0);

    const double temporal =
        std::clamp(
            std::isfinite(temporalConsistency)
                ? temporalConsistency
                : 0.0,
            0.0,
            1.0);
    const double appearance =
        std::clamp(
            std::isfinite(appearanceConsistency)
                ? appearanceConsistency
                : 0.0,
            0.0,
            1.0);
    const double contact =
        std::clamp(
            std::isfinite(contactConsistency)
                ? contactConsistency
                : 0.0,
            0.0,
            1.0);
    const double dynamic =
        std::clamp(
            std::isfinite(dynamicProbability)
                ? dynamicProbability
                : 0.0,
            0.0,
            1.0);

    const double evidence =
        temporal * 0.42 +
        appearance * 0.23 +
        contact * 0.20 +
        (1.0 - dynamic) * 0.15;

    const double precision =
        1.0 /
        (
            safeUncertainty *
            safeUncertainty
        );

    const double effectiveWeight =
        observationWeight *
        std::clamp(
            precision / 100.0,
            0.05,
            4.0) *
        std::clamp(
            0.35 +
                0.65 * evidence,
            0.20,
            1.0) *
        (
            1.0 -
            0.80 * dynamic
        );

    const double start =
        std::max(
            0.0,
            surfaceDepth -
                config_.truncationDistance);
    const double end =
        surfaceDepth +
        config_.truncationDistance;
    const double step =
        config_.voxelSize *
        0.70;

    for (
        double t = start;
        t <= end;
        t += step) {
        const auto point =
            origin +
            ray * t;

        const auto index =
            worldToVoxel(point);

        const double signedDistance =
            std::clamp(
                (
                    surfaceDepth -
                    t
                ) /
                config_.truncationDistance,
                -1.0,
                1.0);

        if (!std::isfinite(signedDistance)) {
            continue;
        }

        const float weight =
            static_cast<float>(
                std::clamp(
                    effectiveWeight *
                        std::exp(
                            -std::abs(
                                signedDistance) *
                            0.65),
                    0.0,
                    static_cast<double>(
                        config_.maxWeight)));

        const auto priorIterator =
            voxels_.find(index);
        const float priorWeight =
            priorIterator == voxels_.end()
                ? 0.0F
                : priorIterator->second.weight;
        const float priorVariance =
            priorIterator == voxels_.end()
                ? 1.0F
                : std::max(
                    priorIterator->second.variance,
                    1e-6F);
        const float priorDynamic =
            priorIterator == voxels_.end()
                ? 0.0F
                : priorIterator->second.dynamicProbability;
        const float priorConfidence =
            priorIterator == voxels_.end()
                ? 0.0F
                : priorIterator->second.confidence;

        integrateVoxel(
            index[0],
            index[1],
            index[2],
            static_cast<float>(
                signedDistance),
            weight);

        auto iterator =
            voxels_.find(index);
        if (iterator != voxels_.end()) {
            auto& voxel =
                iterator->second;
            const float measurementVariance =
                static_cast<float>(
                    std::clamp(
                        safeUncertainty *
                            safeUncertainty,
                        1e-6,
                        4.0));
            const float priorPrecision =
                priorWeight > 0.0F
                    ? 1.0F /
                        priorVariance
                    : 0.0F;
            const float measurementPrecision =
                weight > 0.0F
                    ? 1.0F /
                        measurementVariance
                    : 0.0F;
            const float posteriorPrecision =
                priorPrecision +
                measurementPrecision;

            if (posteriorPrecision > 0.0F) {
                voxel.variance =
                    std::clamp(
                        1.0F /
                            posteriorPrecision,
                        1e-6F,
                        4.0F);
            }

            const double dynamicBlend =
                priorWeight <= 0.0F
                    ? dynamic
                    : (
                        static_cast<double>(
                            priorDynamic) *
                            priorWeight +
                        dynamic *
                            static_cast<double>(
                                weight)
                    ) /
                        std::max(
                            static_cast<double>(
                                priorWeight +
                                weight),
                            1e-9);

            const double confidenceBlend =
                priorWeight <= 0.0F
                    ? evidence
                    : (
                        static_cast<double>(
                            priorConfidence) *
                            priorWeight +
                        evidence *
                            static_cast<double>(
                                weight)
                    ) /
                        std::max(
                            static_cast<double>(
                                priorWeight +
                                weight),
                            1e-9);

            voxel.dynamicProbability =
                static_cast<float>(
                    std::clamp(
                        dynamicBlend,
                        0.0,
                        1.0));

            voxel.confidence =
                static_cast<float>(
                    std::clamp(
                        confidenceBlend,
                        0.0,
                        1.0));
        }
    }
}

void SparseTsdfVolume::integrateSurfacePoint(
    const math::Vec3d& origin,
    const math::Vec3d& direction,
    double surfaceDepth,
    double observationWeight) {
    const auto ray=normalize(direction);
    if(surfaceDepth<=0.0)return;
    const double start=
        std::max(0.0,surfaceDepth-config_.truncationDistance);
    const double end=
        surfaceDepth+config_.truncationDistance;
    const double step=config_.voxelSize*0.75;
    for(double t=start;t<=end;t+=step){
        const auto point=origin+ray*t;
        const auto index=worldToVoxel(point);
        const double signedDistance=
            std::clamp(
                (surfaceDepth-t)/config_.truncationDistance,
                -1.0,1.0);
        if(!std::isfinite(signedDistance))continue;

        // The ray may already be expressed in world coordinates. Do not use
        // a world-space axis as an implicit camera-incidence term: that would
        // bias fusion when the camera rig rotates. Incidence/visibility
        // weighting belongs at the observation boundary.
        integrateVoxel(
            index[0],index[1],index[2],
            static_cast<float>(signedDistance),
            static_cast<float>(std::clamp(observationWeight,0.0,1.0)));
    }
}

TsdfVoxel SparseTsdfVolume::sample(
    std::int32_t x,
    std::int32_t y,
    std::int32_t z) const noexcept {
    const auto iterator=voxels_.find({x,y,z});
    return iterator==voxels_.end()?TsdfVoxel{}:iterator->second;
}

TsdfMesh SparseTsdfVolume::extractSurface(
    float isoLevel,
    std::size_t maxTriangles) const {
    if(
        voxels_.empty() &&
        last_extracted_mesh_.has_value()) {
        return *last_extracted_mesh_;
    }

    TsdfMesh mesh;
    if(voxels_.empty()||maxTriangles==0)return mesh;

    std::vector<VertexAccumulator> accumulators;
    accumulators.reserve(std::min<std::size_t>(
        voxels_.size()*2,
        maxTriangles*3));
    std::unordered_map<std::array<std::int64_t,3>,
        std::uint32_t,VertexKeyHash> indices;
    indices.reserve(std::min<std::size_t>(
        voxels_.size()*2,
        maxTriangles*3));

    std::vector<std::array<std::int32_t,3>> orderedKeys;
    orderedKeys.reserve(voxels_.size());
    for(const auto& entry:voxels_) {
        orderedKeys.push_back(entry.first);
    }
    std::sort(
        orderedKeys.begin(),
        orderedKeys.end(),
        [](const auto& a,const auto& b) {
            if(a[0]!=b[0])return a[0]<b[0];
            if(a[1]!=b[1])return a[1]<b[1];
            return a[2]<b[2];
        });

    for(const auto& index:orderedKeys){
        if(mesh.triangles.size()>=maxTriangles)break;
        const auto voxelIterator=voxels_.find(index);
        if(voxelIterator==voxels_.end())continue;

        std::array<Corner,8> corners{};
        for(std::size_t cornerIndex=0;cornerIndex<8;++cornerIndex){
            const auto offset=CornerOffsets[cornerIndex];
            const auto x=index[0]+offset[0];
            const auto y=index[1]+offset[1];
            const auto z=index[2]+offset[2];
            const auto value=sample(x,y,z);
            corners[cornerIndex]={
                {x,y,z},
                voxelToWorld(x,y,z),
                value.distance,
                value.weight,
            };
        }

        for(const auto& tetra:Tetrahedra){
            std::array<int,4> inside{};
            int insideCount=0;
            for(int i=0;i<4;++i){
                inside[i]=
                    corners[tetra[i]].value<isoLevel?1:0;
                insideCount+=inside[i];
            }
            if(insideCount==0||insideCount==4)continue;

            std::array<math::Vec3d,6> intersections{};
            int intersectionCount=0;
            for(int a=0;a<4;++a){
                for(int b= a+1;b<4;++b){
                    if(inside[a]==inside[b])continue;
                    const auto&ca=corners[tetra[a]];
                    const auto&cb=corners[tetra[b]];
                    intersections[intersectionCount++]=interpolate(
                        ca.position,cb.position,
                        ca.value,cb.value,isoLevel);
                }
            }

            double averageWeight=0.0;
            for(int i=0;i<4;++i)averageWeight+=corners[tetra[i]].weight;
            averageWeight*=0.25;
            const double uncertaintyMeters=
                config_.truncationDistance/
                std::sqrt(
                    std::max(
                        averageWeight,
                        0.25))
                *
                (
                    1.0 +
                    0.50 *
                        std::max(
                            0.0,
                            1.0 -
                                averageWeight /
                                    16.0)
                );

            if(intersectionCount==3||intersectionCount==4){
                auto makeInterpolatedCorner = [&](int slot) {
                    Corner corner;
                    corner.position=intersections[slot];
                    corner.voxel={
                        static_cast<std::int32_t>(
                            std::floor(corner.position.x/config_.voxelSize)),
                        static_cast<std::int32_t>(
                            std::floor(corner.position.y/config_.voxelSize)),
                        static_cast<std::int32_t>(
                            std::floor(corner.position.z/config_.voxelSize)),
                    };
                    corner.value=isoLevel;
                    corner.weight=static_cast<float>(averageWeight);
                    return corner;
                };

                const auto c0=makeInterpolatedCorner(0);
                const auto c1=makeInterpolatedCorner(1);
                const auto c2=makeInterpolatedCorner(2);
                appendTriangle(
                    mesh,accumulators,indices,*this,
                    c0,c1,c2,uncertaintyMeters,maxTriangles);

                if(intersectionCount==4){
                    const auto c3=makeInterpolatedCorner(3);
                    appendTriangle(
                        mesh,accumulators,indices,*this,
                        c0,c2,c3,uncertaintyMeters,maxTriangles);
                }
            }
        }
    }

    mesh.vertices.reserve(accumulators.size());
    mesh.normals.reserve(accumulators.size());
    mesh.vertexUncertaintyMeters.reserve(accumulators.size());
    mesh.vertexDynamicProbability.reserve(
        accumulators.size());

    std::vector<double> uncertainties;
    uncertainties.reserve(
        accumulators.size());

    double confidenceNumerator=0.0;
    double confidenceDenominator=0.0;
    double uncertaintySum=0.0;
    double dynamicSum=0.0;
    std::size_t trustedVertices=0U;

    for(const auto& accumulator:accumulators){
        mesh.vertices.push_back(
            accumulator.position);
        mesh.normals.push_back(
            normalize(
                accumulator.normal));
        mesh.vertexUncertaintyMeters.push_back(
            accumulator.uncertaintyMeters);
        mesh.vertexDynamicProbability.push_back(
            std::clamp(
                accumulator.dynamicProbability,
                0.0,
                1.0));

        confidenceNumerator +=
            std::min(
                1.0,
                accumulator.confidenceWeight / 8.0);
        confidenceDenominator += 1.0;

        uncertaintySum +=
            std::max(
                0.0,
                accumulator.uncertaintyMeters);
        uncertainties.push_back(
            std::max(
                0.0,
                accumulator.uncertaintyMeters));

        dynamicSum +=
            std::clamp(
                accumulator.dynamicProbability,
                0.0,
                1.0);

        if(
            accumulator.uncertaintyMeters <=
                config_.voxelSize * 0.75 &&
            accumulator.confidenceWeight >=
                1.0 &&
            accumulator.dynamicProbability <=
                0.35) {
            ++trustedVertices;
        }
    }

    mesh.meanConfidence =
        confidenceDenominator > 0.0
            ? std::clamp(
                confidenceNumerator /
                    confidenceDenominator,
                0.0,
                1.0)
            : 0.0;

    mesh.meanUncertaintyMeters =
        confidenceDenominator > 0.0
            ? uncertaintySum /
                confidenceDenominator
            : 0.0;

    if(!uncertainties.empty()){
        std::sort(
            uncertainties.begin(),
            uncertainties.end());
        const double position =
            0.95 *
            static_cast<double>(
                uncertainties.size()-1U);
        const auto lower =
            static_cast<std::size_t>(
                std::floor(position));
        const auto upper =
            std::min(
                uncertainties.size()-1U,
                lower+1U);
        const double alpha =
            position -
            static_cast<double>(lower);
        mesh.p95UncertaintyMeters =
            uncertainties[lower] +
            (
                uncertainties[upper] -
                uncertainties[lower]
            ) *
            alpha;
    }

    mesh.meanDynamicProbability =
        confidenceDenominator > 0.0
            ? dynamicSum /
                confidenceDenominator
            : 0.0;

    mesh.trustedVertexRatio =
        confidenceDenominator > 0.0
            ? static_cast<double>(
                trustedVertices) /
                confidenceDenominator
            : 0.0;

    last_extracted_mesh_ = mesh;
    return mesh;
}

std::size_t SparseTsdfVolume::activeVoxelCount() const noexcept {
    return voxels_.size();
}

std::size_t SparseTsdfVolume::residentMemoryBytes() const noexcept {
    if(voxels_.empty()) return 0u;
    const auto estimatedVoxelBytes =
        std::max<std::size_t>(
            config_.estimatedBytesPerVoxel,
            sizeof(TsdfVoxel));
    const auto nodeBytes =
        voxels_.size() * estimatedVoxelBytes;
    const auto bucketBytes =
        voxels_.bucket_count() * sizeof(void*);
    return nodeBytes + bucketBytes;
}

double SparseTsdfVolume::voxelUtilization() const noexcept {
    const auto voxelLimit =
        std::min(
            config_.maxResidentVoxels,
            config_.maxResidentMemoryBytes /
                std::max<std::size_t>(
                    config_.estimatedBytesPerVoxel,
                    sizeof(TsdfVoxel)));
    if(voxelLimit==0) return 1.0;
    return std::clamp(
        static_cast<double>(voxels_.size()) /
            static_cast<double>(voxelLimit),
        0.0,
        1.0);
}

std::size_t SparseTsdfVolume::rejectedVoxelAllocations() const noexcept {
    return rejected_voxel_allocations_;
}

std::size_t SparseTsdfVolume::rejectedTriangleCount() const noexcept {
    if (!last_extracted_mesh_.has_value()) return 0u;
    return last_extracted_mesh_->rejectedTriangleCount;
}

} // namespace nox::native::scene

#include <cassert>
#include <cmath>
#include "scene/sparse_tsdf_volume.h"

int main() {
    nox::native::scene::SparseTsdfVolume volume({
        0.05,
        0.12,
        32.0F,
        8,
        256,
        1'048'576,
        96,
    });

    for (int x=-4;x<=4;++x) {
        for (int y=-4;y<=4;++y) {
            const double px=(x+0.5)*0.05;
            const double py=(y+0.5)*0.05;
            volume.integrateSurfacePoint(
                {0,0,0},
                {px,py,1.0},
                1.0,
                1.0);
        }
    }

    assert(volume.activeVoxelCount()>0);
    const auto mesh=volume.extractSurface(0.0F,5000);
    assert(!mesh.triangles.empty());
    assert(mesh.vertices.size()<=mesh.triangles.size()*3);
    assert(mesh.vertices.size()>0);
    assert(mesh.normals.size()==mesh.vertices.size());
    assert(mesh.vertexUncertaintyMeters.size()==mesh.vertices.size());
    for(const auto& triangle:mesh.triangles){
        assert(triangle[0]<mesh.vertices.size());
        assert(triangle[1]<mesh.vertices.size());
        assert(triangle[2]<mesh.vertices.size());
        assert(triangle[0]!=triangle[1]);
        assert(triangle[1]!=triangle[2]);
        assert(triangle[0]!=triangle[2]);
    }
    std::size_t unitNormals=0;
    double averageAbsZ=0.0;
    for(const auto& normal:mesh.normals){
        const double length=std::sqrt(
            normal.x*normal.x+
            normal.y*normal.y+
            normal.z*normal.z);
        if(std::abs(length-1.0)<1e-6)++unitNormals;
        averageAbsZ += std::abs(normal.z);
    }
    assert(unitNormals==mesh.normals.size());
    averageAbsZ /= static_cast<double>(mesh.normals.size());
    assert(averageAbsZ>0.70);
    assert(mesh.meanConfidence>0.0);
    assert(mesh.vertexDynamicProbability.size()==
           mesh.vertices.size());
    assert(mesh.meanUncertaintyMeters>0.0);
    assert(mesh.p95UncertaintyMeters>=
           mesh.meanUncertaintyMeters);
    assert(mesh.meanDynamicProbability>=0.0);
    assert(mesh.meanDynamicProbability<=1.0);
    assert(mesh.trustedVertexRatio>=0.0);
    assert(mesh.trustedVertexRatio<=1.0);
    assert(mesh.rejectedTriangleCount>=0u);
    assert(volume.rejectedTriangleCount()==mesh.rejectedTriangleCount);

    assert(volume.voxelUtilization() < 1.0);
    assert(volume.residentMemoryBytes() <= 1'048'576);
    assert(volume.rejectedVoxelAllocations() == 0);

    volume.releaseMemory();
    assert(volume.activeVoxelCount() == 0u);
    assert(volume.residentMemoryBytes() == 0u);
    assert(volume.rejectedVoxelAllocations() == 0u);

    const auto sample=volume.sample(0,0,19);
    assert(std::isfinite(sample.distance));
    nox::native::scene::SparseTsdfVolume robustVolume({
        0.05,
        0.12,
        32.0F,
        8,
    });

    robustVolume.integrateSurfacePoint(
        {0,0,0},
        {0,0,1},
        1.0,
        1.0);
    const auto beforeOutlier =
        robustVolume.sample(0,0,19);

    robustVolume.integrateSurfacePoint(
        {0,0,0},
        {0,0,1},
        1.08,
        1.0);
    const auto afterOutlier =
        robustVolume.sample(0,0,19);

    assert(
        std::abs(afterOutlier.distance) >
        std::abs(beforeOutlier.distance) * 0.40);
    assert(
        std::abs(afterOutlier.distance) -
        std::abs(beforeOutlier.distance) < 0.30);

    robustVolume.integrateSurfacePointHighEnd(
        {0,0,0},
        {0,0,1},
        1.0,
        1.0,
        0.001,
        0.98,
        0.95,
        0.92,
        0.02);
    const auto highEndVoxel =
        robustVolume.sample(0,0,19);
    assert(highEndVoxel.variance > 0.0F);
    assert(highEndVoxel.confidence > 0.0F);
    assert(highEndVoxel.dynamicProbability >= 0.0F);
    assert(highEndVoxel.dynamicProbability <= 1.0F);



    const auto second=volume.extractSurface(0.0F,5000);
    assert(second.triangles.size()==mesh.triangles.size());
    assert(second.vertices.size()==mesh.vertices.size());
    assert(second.normals.size()==mesh.normals.size());
    assert(second.vertexUncertaintyMeters.size()==
           mesh.vertexUncertaintyMeters.size());
    assert(second.rejectedTriangleCount==
           mesh.rejectedTriangleCount);
    for(std::size_t i=0;i<mesh.vertices.size();++i){
        assert(std::abs(
            second.vertices[i].x-mesh.vertices[i].x)<1e-12);
        assert(std::abs(
            second.vertices[i].y-mesh.vertices[i].y)<1e-12);
        assert(std::abs(
            second.vertices[i].z-mesh.vertices[i].z)<1e-12);
        assert(second.normals[i].x==mesh.normals[i].x);
        assert(second.normals[i].y==mesh.normals[i].y);
        assert(second.normals[i].z==mesh.normals[i].z);
    }
    for(std::size_t i=0;i<mesh.triangles.size();++i){
        assert(second.triangles[i]==mesh.triangles[i]);
    }
    return 0;
}


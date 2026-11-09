#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "scene/mesh_builder.h"

namespace {
using namespace nox::native::scene;

Point p(
    double x,
    double y,
    double z,
    double confidence=1.0,
    bool dynamic=false,
    double uncertainty=0.002,
    double dynamicProbability=0.0,
    double temporalConsistency=0.96){
    Point value;
    value.p={x,y,z};
    value.normal={0.0,0.0,1.0};
    value.confidence=confidence;
    value.dynamic=dynamic;
    value.uncertaintyMeters=uncertainty;
    value.dynamicProbability=dynamicProbability;
    value.temporalConsistency=temporalConsistency;
    return value;
}

void assertFinite(const MeshBuilder::Result&r){
    assert(std::isfinite(r.mesh.confidence));
    assert(r.mesh.confidence>=0.0&&r.mesh.confidence<=1.0);
    assert(r.mesh.indices.size()%3U==0U);
    for(const auto&v:r.mesh.vertices){
        assert(std::isfinite(v.x));
        assert(std::isfinite(v.y));
        assert(std::isfinite(v.z));
    }
}

void planarGridProducesRealTriangles(){
    MeshBuilder builder;
    std::vector<Point> points;
    for(int y=0;y<3;++y){
        for(int x=0;x<3;++x){
            points.push_back(p(0.1*x,0.1*y,0.0));
        }
    }
    const auto result=builder.buildDetailed(points);
    assertFinite(result);
    assert(!result.mesh.vertices.empty());
    assert(!result.mesh.indices.empty());
    assert(result.diagnostics.acceptedTriangles>0U);
    assert(result.diagnostics.meanNormalAgreement>0.90);
}

void collinearDataCannotFabricateTopology(){
    MeshBuilder builder;
    const std::vector<Point> points{
        p(0,0,0),p(0.1,0,0),p(0.2,0,0),
        p(0.3,0,0),p(0.4,0,0)
    };
    const auto result=builder.buildDetailed(points);
    assertFinite(result);
    assert(result.mesh.indices.empty());
}

void dynamicAndWeakObservationsAreNotMeshed(){
    MeshBuilder builder;
    const std::vector<Point> points{
        p(0,0,0),p(0.1,0,0),p(0,0.1,0),
        p(0.1,0.1,0),p(0.2,0.2,0,0.03),
        p(0.3,0.3,0,1.0,true)
    };
    const auto result=builder.buildDetailed(points);
    assertFinite(result);
    assert(result.diagnostics.rejectedLowConfidence>=1U);
    assert(result.diagnostics.rejectedDynamic>=1U);
}

void duplicateObservationsCollapse(){
    MeshBuilder builder;
    const std::vector<Point> points{
        p(0,0,0,0.3),p(0.00001,0.00001,0,0.95),
        p(0.1,0,0),p(0,0.1,0)
    };
    const auto result=builder.buildDetailed(points);
    assertFinite(result);
    assert(result.diagnostics.rejectedDuplicate>=1U);
    assert(result.mesh.vertices.size()<=3U);
}


void uncertaintyAndTemporalGatesRejectWeakSurfaceEvidence(){
    MeshBuildConfig config;
    config.maximumVertexUncertaintyMeters=0.005;
    config.minimumVertexTemporalConsistency=0.80;
    config.maximumVertexDynamicProbability=0.30;

    MeshBuilder builder(config);

    const std::vector<Point> points{
        p(0.0,0.0,0.0,1.0,false,0.001,0.02,0.98),
        p(0.1,0.0,0.0,1.0,false,0.001,0.02,0.98),
        p(0.0,0.1,0.0,1.0,false,0.001,0.02,0.98),
        p(0.1,0.1,0.0,1.0,false,0.020,0.02,0.98),
        p(0.2,0.0,0.0,1.0,false,0.001,0.55,0.98),
        p(0.2,0.1,0.0,1.0,false,0.001,0.02,0.20),
    };

    const auto result =
        builder.buildDetailed(points);

    assertFinite(result);
    assert(result.diagnostics.rejectedHighUncertainty >= 1U);
    assert(result.diagnostics.rejectedDynamicProbability >= 1U);
    assert(result.diagnostics.rejectedTemporalInconsistency >= 1U);
    assert(!result.mesh.vertexUncertaintyMeters.empty());
    assert(
        result.mesh.vertexUncertaintyMeters.size() ==
        result.mesh.vertices.size());
    assert(
        result.mesh.vertexDynamicProbability.size() ==
        result.mesh.vertices.size());
    assert(
        result.mesh.vertexTemporalConsistency.size() ==
        result.mesh.vertices.size());
    assert(result.diagnostics.meanVertexUncertaintyMeters >= 0.0);
    assert(result.diagnostics.p95VertexUncertaintyMeters >= 0.0);
    assert(result.diagnostics.trustedVertexRatio >= 0.0);
    assert(result.diagnostics.trustedVertexRatio <= 1.0);
}

void inputPermutationDoesNotChangeCardinality(){
    MeshBuilder builder;
    std::vector<Point> a{
        p(0,0,0),p(0.1,0,0),p(0,0.1,0),
        p(0.1,0.1,0),p(0.2,0,0),p(0.2,0.1,0)
    };
    auto b=a;
    std::reverse(b.begin(),b.end());

    const auto ra=builder.buildDetailed(a);
    const auto rb=builder.buildDetailed(b);

    assertFinite(ra);
    assertFinite(rb);
    assert(ra.mesh.vertices.size()==rb.mesh.vertices.size());
    assert(ra.mesh.indices.size()==rb.mesh.indices.size());
    assert(ra.diagnostics.acceptedTriangles==
           rb.diagnostics.acceptedTriangles);
}

void triangleWindingFollowsSurfaceNormals(){
    MeshBuilder builder;
    const std::vector<Point> points{
        p(0,0,0),p(0.1,0,0),p(0,0.1,0)
    };
    const auto result=builder.buildDetailed(points);
    assertFinite(result);
    assert(result.mesh.indices.size()>=3U);
    const auto a=result.mesh.vertices[result.mesh.indices[0]];
    const auto b=result.mesh.vertices[result.mesh.indices[1]];
    const auto c=result.mesh.vertices[result.mesh.indices[2]];
    const double nx=(b.y-a.y)*(c.z-a.z)-(b.z-a.z)*(c.y-a.y);
    const double ny=(b.z-a.z)*(c.x-a.x)-(b.x-a.x)*(c.z-a.z);
    const double nz=(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
    assert((nx*0.0+ny*0.0+nz*1.0)>0.0);
}

void nonManifoldEdgeGuardCanBeTightened(){
    MeshBuildConfig config;
    config.maximumIncidentTrianglesPerEdge=1;
    config.enforceManifoldEdges=true;
    MeshBuilder builder(config);

    std::vector<Point> points{
        p(0,0,0),p(0.1,0,0),p(0,0.1,0),
        p(0.1,0.1,0),p(0.2,0,0),p(0.2,0.1,0)
    };
    const auto result=builder.buildDetailed(points);
    assertFinite(result);
    assert(result.diagnostics.rejectedNonManifold>0U);
    assert(result.diagnostics.nonManifoldEdgeCount==0U);
}

void edgeEnvelopeIsRespected(){
    MeshBuildConfig config;
    config.maximumEdgeLength=0.15;
    config.neighborhoodRadius=0.30;
    MeshBuilder builder(config);

    const std::vector<Point> points{
        p(0,0,0),p(0.1,0,0),p(0,0.1,0),
        p(1.0,1.0,0)
    };

    const auto result=builder.buildDetailed(points);
    assertFinite(result);

    for(std::size_t i=0;i<result.mesh.indices.size();i+=3U){
        assert(result.mesh.indices[i]<
               result.mesh.vertices.size());
        assert(result.mesh.indices[i+1U]<
               result.mesh.vertices.size());
        assert(result.mesh.indices[i+2U]<
               result.mesh.vertices.size());
    }
}
}

int main(){
    planarGridProducesRealTriangles();
    collinearDataCannotFabricateTopology();
    dynamicAndWeakObservationsAreNotMeshed();
    duplicateObservationsCollapse();
    uncertaintyAndTemporalGatesRejectWeakSurfaceEvidence();
    inputPermutationDoesNotChangeCardinality();
    edgeEnvelopeIsRespected();
    triangleWindingFollowsSurfaceNormals();
    nonManifoldEdgeGuardCanBeTightened();
    return 0;
}

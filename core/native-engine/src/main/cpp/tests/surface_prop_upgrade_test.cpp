#include <cassert>
#include <vector>
#include "reconstruction/depth_uncertainty_fusion.h"
#include "scene/mesh_temporal_coherence.h"
#include "reconstruction/body_surface_deformation.h"
#include "tracking/rigid_prop_tracker.h"

int main() {
    const auto depth =
        nox::native::reconstruction::DepthUncertaintyFusion{}.fuse({2.0,0.1,0.9},{2.1,0.2,0.8});
    assert(depth.depth>1.9);
    assert(depth.confidence>0.5);

    const auto mesh=nox::native::scene::MeshTemporalCoherence{}.stabilize(
        {{0,0,0}},{{1,0,0}},0.5);
    assert(mesh.size()==1);
    assert(mesh[0].x>0.4&&mesh[0].x<0.6);

    const std::vector<nox::native::reconstruction::Vertex> base={{0,0,0}};
    const std::vector<nox::native::reconstruction::Displacement> displacement={{1,0,0,0.5}};
    const auto body=nox::native::reconstruction::BodySurfaceDeformation{}.apply(base,displacement);
    assert(body.size()==1&&body[0].x>0.4);

    const auto prop=nox::native::tracking::RigidPropTracker{}.update({},1,1,1,0.9);
    assert(prop.confidence>0.2);
    return 0;
}

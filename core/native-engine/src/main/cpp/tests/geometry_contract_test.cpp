#include <cassert>
#include "scene/volumetric_fusion.h"
#include "geometry3d/world_pose_stabilizer.h"
#include "scene/scene_flow_estimator.h"
#include "core/coordinate_frame_registry.h"

int main() {
    nox::native::scene::VolumetricFusion volume;
    volume.integrate(1,1,1,0.03f,0.9f);
    assert(volume.sample(1,1,1).weight>0.8f);

    const auto pose=nox::native::geometry::WorldPoseStabilizer{}.update(
        {},{1,0,0},1.0);
    assert(pose.x>0);

    const auto flow=nox::native::scene::SceneFlowEstimator{}.estimate(
        0,0,0,0.1,0,0,1);
    assert(flow.dx>0);

    nox::native::core::CoordinateFrameRegistry frames;
    frames.registerFrame("camera","world");
    assert(frames.isCompatible("camera","world"));
    return 0;
}

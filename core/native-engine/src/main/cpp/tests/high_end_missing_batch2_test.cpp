#include <cassert>
#include <vector>
#include "kinematics/skeleton_retargeter.h"
#include "kinematics/foot_lock_solver.h"
#include "capture/rolling_shutter_compensator.h"
#include "lighting/exposure_vignette_model.h"
#include "memory/allocation_telemetry.h"
#include "export/schema_negotiator.h"
#include "replay/replay_matrix_validator.h"
#include "calibration/robust_bundle_adjustment.h"
#include "calibration/lens_model.h"
#include "lighting/camera_color_normalizer.h"
#include "reconstruction/depth_uncertainty_fusion.h"
#include "scene/mesh_temporal_coherence.h"
#include "tracking/rigid_prop_tracker.h"
#include "capture/capture_trigger.h"
#include "runtime/network_health.h"
#include "synchronization/jitter_buffer.h"
#include "capture/capture_settings_optimizer.h"
#include "runtime/gpu_execution_backend.h"
#include "math/simd_dispatch.h"
#include "scene/volumetric_fusion.h"
#include "geometry3d/world_pose_stabilizer.h"
#include "scene/scene_flow_estimator.h"
#include "core/coordinate_frame_registry.h"

int main() {
    using namespace nox::native;
    const auto ret=kinematics::SkeletonRetargeter{}.solve(
        {1,2},{1,2},{{0,0,1},{1,1,1}});
    assert(ret.valid);
    assert(kinematics::FootLockSolver{}.solve(.01,.2,.01,.2).leftLocked);

    const auto rolling=capture::RollingShutterCompensator{}.correct({1,1},.6,1,.01);
    assert(rolling.x!=1);

    assert(lighting::ExposureVignetteModel{}.correctedLuma(1.0,0.0,.4)>0);

    memory::AllocationTelemetryRecorder allocation;
    allocation.recordAllocation(10);
    assert(allocation.snapshot().bytesAllocated==10);

    const auto negotiated=exporting::SchemaNegotiator{}.negotiate({}, {});
    assert(negotiated.compatible);

    calibration::LensParameters lens{};
    assert(calibration::LensModel{}.radialScale(.1,lens)==1.0);

    std::vector<calibration::BundlePoint> points={{0,0,0}};
    const std::vector<calibration::BundleObservation> observations={{0,{1,0,0},1}};
    const auto bundle=calibration::RobustBundleAdjustment{}.solve(points,observations);
    assert(bundle.rms>0);

    const auto color=lighting::CameraColorNormalizer{}.normalize({2,2,2},{2,2,2});
    assert(color.r==1);

    const auto depth=reconstruction::DepthUncertaintyFusion{}.fuse({2,.1,.9},{2.1,.2,.8});
    assert(depth.confidence>.5);

    const auto mesh=scene::MeshTemporalCoherence{}.stabilize({{0,0,0}},{{1,0,0}});
    assert(mesh.front().x>0);

    const auto prop=tracking::RigidPropTracker{}.update({},1,1,1,.9);
    assert(prop.confidence>.2);

    assert(capture::CaptureTrigger{}.evaluate(true,.9,1).fired);
    assert(runtime::NetworkHealthModel{}.evaluate(10,.01,2).healthy);

    sync::JitterBuffer buffer(2);
    assert(buffer.push({1}));
    sync::JitterSample sample{};
    assert(buffer.pop(sample));

    assert(capture::CaptureSettingsOptimizer{}.optimize(.5,.8,.9).frameRate==60);
    assert(runtime::GpuExecutionBackend{}.choose(true,false,64).available);

    const auto simd=math::detectSimd();
    assert(simd==math::SimdLevel::Scalar||simd==math::SimdLevel::SSE||
           simd==math::SimdLevel::AVX||simd==math::SimdLevel::NEON);

    scene::VolumetricFusion volume;
    volume.integrate(0,0,0,.2f,1.0f);
    assert(volume.voxelCount()==1);

    const auto pose=geometry::WorldPoseStabilizer{}.update({}, {1,0,0},.8);
    assert(pose.x>0);

    assert(scene::SceneFlowEstimator{}.estimate(0,0,0,1,0,0,1).dx>0);

    core::CoordinateFrameRegistry frames;
    frames.registerFrame("camera","world");
    assert(frames.isCompatible("camera","world"));
    return 0;
}

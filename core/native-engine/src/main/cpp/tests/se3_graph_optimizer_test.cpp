#include <cassert>
#include <cmath>
#include "math/se3.h"
#include "math/robust_loss.h"
#include "calibration/global_camera_graph_optimizer.h"

int main() {
    using nox::native::math::SE3d;
    using nox::native::math::Quaterniond;
    using nox::native::math::Vec3d;
    using nox::native::math::Twist6d;

    const Twist6d motion{{0,0,0.25},{1.0,0.2,-0.1}};
    const auto pose=SE3d::exp(motion);
    const auto recovered=pose.log();

    assert(std::abs(recovered.rotation.z-0.25)<1e-6);
    assert(std::abs(recovered.translation.x-1.0)<1e-6);

    const auto identity=pose.compose(pose.inverse());
    assert(std::abs(identity.translation.x)<1e-8);
    assert(std::abs(identity.translation.y)<1e-8);
    assert(std::abs(identity.translation.z)<1e-8);

    const auto huber=nox::native::math::RobustLoss::weight(
        nox::native::math::RobustLossKind::Huber,5.0,1.0);
    assert(huber<0.25);

    const auto graph=nox::native::calibration::GlobalCameraGraphOptimizer{}.solve(
        2,
        {{"C0","C1",{1.0,0.0,0.0,Quaterniond::fromRotationVector({0,0,0.5})},1.0,0.02,0.03}},
        64,
        0.30);
    assert(graph.poses.size()==2);
    assert(graph.initializationEdges==1);
    assert(graph.residual<0.02);
    assert(graph.confidence>0.95);
    assert(graph.cycleConfidence>0.95);
    assert(graph.cycleFailures==0);
    assert(graph.poses[0].x==0.0);
    assert(graph.poses[0].y==0.0);
    assert(graph.poses[0].z==0.0);
    assert(graph.poses[1].x>0.9);
    assert(std::isfinite(graph.residual));

    const auto graphWithCycle =
        nox::native::calibration::GlobalCameraGraphOptimizer{}.solve(
            3,
            {
                {"C0","C1",
                    {1.0,0.0,0.0,
                        Quaterniond::fromRotationVector({0,0,0.2})},
                    1.0,0.02,0.03},
                {"C1","C2",
                    {1.0,0.5,0.0,
                        Quaterniond::fromRotationVector({0,0,0.25})},
                    1.0,0.02,0.03},
                {"C0","C2",
                    {2.0,0.5,0.0,
                        Quaterniond::fromRotationVector({0,0,0.45})},
                    1.2,0.03,0.04},
            },
            96,
            0.25);

    assert(graphWithCycle.poses.size()==3);
    assert(graphWithCycle.initializationEdges>=2);
    assert(graphWithCycle.cycleConfidence>0.80);
    assert(graphWithCycle.cycleFailures<=1);
    assert(graphWithCycle.poses[0].x==0.0);
    assert(std::isfinite(graphWithCycle.confidence));
    const auto rotationVector=graph.poses[1].rotation.toRotationVector();
    assert(std::abs(rotationVector.z-0.5)<0.03);
    return 0;
}

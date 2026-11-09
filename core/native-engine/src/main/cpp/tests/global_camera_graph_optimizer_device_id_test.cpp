#include <cassert>
#include <string>
#include <vector>

#include "calibration/global_camera_graph_optimizer.h"

int main() {
    using namespace nox::native::calibration;

    const std::vector<std::string> cameraIds{
        "rig-alpha-left",
        "studio-west-17",
        "mobile-node-042",
    };

    const auto result =
        GlobalCameraGraphOptimizer{}.solve(
            cameraIds,
            {
                {"rig-alpha-left", "studio-west-17", {1, 0, 0}, 1},
                {"studio-west-17", "mobile-node-042", {1, 0, 0}, 1},
            });

    assert(result.poses.size() == 3u);
    assert(result.initializationEdges == 2u);
    assert(result.connected);
    assert(result.disconnectedCameraCount == 0u);
    assert(result.converged || result.residual < 0.5);

    const auto disconnected =
        GlobalCameraGraphOptimizer{}.solve(
            cameraIds,
            {
                {"rig-alpha-left", "studio-west-17", {1, 0, 0}, 1},
            });

    assert(!disconnected.connected);
    assert(disconnected.disconnectedCameraCount == 1u);
    assert(!disconnected.converged);
    assert(disconnected.confidence == 0.0);

    return 0;
}

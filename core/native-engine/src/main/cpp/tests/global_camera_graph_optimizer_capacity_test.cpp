#include <cassert>
#include <vector>
#include "calibration/global_camera_graph_optimizer.h"

int main() {
    using nox::native::calibration::GlobalCameraGraphOptimizer;
    using nox::native::calibration::GraphEdge;

    std::vector<GraphEdge> edges;
    edges.push_back({"C0", "C1", {}, 1.0, 1.0, 1.0});

    const auto normal =
        GlobalCameraGraphOptimizer{}.solve(
            128,
            edges,
            4,
            0.25);

    assert(normal.poses.size() == 128);

    const auto oversized =
        GlobalCameraGraphOptimizer{}.solve(
            129,
            edges,
            4,
            0.25);

    assert(oversized.poses.empty());
    assert(oversized.confidence == 0.0);

    return 0;
}

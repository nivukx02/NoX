#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "calibration/global_camera_graph_optimizer.h"

namespace nox::native::calibration {

struct GraphCycleResult {
    double rmsTranslation{0.0};
    double rmsRotation{0.0};
    double confidence{0.0};
    std::size_t cyclesChecked{0};
    std::size_t inconsistentEdges{0};
    bool consistent{false};
};

class GraphCycleValidator final {
public:
    GraphCycleResult evaluate(
        const std::vector<GraphPose>& poses,
        const std::vector<GraphEdge>& edges) const;

    GraphCycleResult evaluate(
        const std::vector<GraphPose>& poses,
        const std::vector<std::string>& cameraIds,
        const std::vector<GraphEdge>& edges) const;
};

} // namespace nox::native::calibration

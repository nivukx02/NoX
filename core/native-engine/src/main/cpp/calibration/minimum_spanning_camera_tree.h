#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "calibration/global_camera_graph_optimizer.h"

namespace nox::native::calibration {

struct CameraTreeResult {
    std::vector<std::size_t> selectedEdgeIndices;
    double totalCost{0.0};
    bool connected{false};
};

class MinimumSpanningCameraTree final {
public:
    CameraTreeResult solve(
        std::size_t cameraCount,
        const std::vector<GraphEdge>& edges) const;

    CameraTreeResult solve(
        const std::vector<std::string>& cameraIds,
        const std::vector<GraphEdge>& edges) const;
};

} // namespace nox::native::calibration

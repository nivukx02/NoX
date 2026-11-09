#pragma once

#include <cstddef>
#include <vector>

#include "calibration/global_camera_graph_optimizer.h"

namespace nox::native::calibration {

struct RigObservabilityResult {
    double baselineCoverage{0.0};
    double minimumBaselineMeters{0.0};
    double p10BaselineMeters{0.0};
    double p95BaselineMeters{0.0};
    double angularDiversity{0.0};
    double edgeConfidence{0.0};
    double p95EdgeUncertainty{0.0};
    double redundancy{0.0};
    double connectivity{0.0};
    double leverageDiversity{0.0};
    double score{0.0};
    std::size_t connectedComponents{0};
    std::size_t weakCameraCount{0};
    std::size_t isolatedCameraCount{0};
    std::size_t validEdgeCount{0};
    bool observable{false};
    bool referenceGradeObservable{false};
};

class RigObservabilityAnalyzer final {
public:
    [[nodiscard]] RigObservabilityResult analyze(
        std::size_t cameraCount,
        const std::vector<GraphEdge>& edges) const;

private:
    static double percentile(
        std::vector<double>,
        double) noexcept;

    static double clamp01(
        double) noexcept;
};

} // namespace nox::native::calibration

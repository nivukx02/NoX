#pragma once
#include <string>
#include <vector>
#include "math/se3.h"

namespace nox::native::calibration {

struct GraphPose {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    math::Quaterniond rotation{};
};

struct GraphEdge {
    std::string from;
    std::string to;
    GraphPose relative{};
    double weight{1.0};
    double translationSigma{1.0};
    double rotationSigma{1.0};
};

struct GraphOptimizationResult {
    std::vector<GraphPose> poses;
    double residual{0.0};
    double confidence{0.0};
    bool converged{false};
    bool connected{false};
    std::size_t initializationEdges{0};
    std::size_t disconnectedCameraCount{0};
    double cycleConfidence{0.0};
    std::size_t cycleFailures{0};
};

class GlobalCameraGraphOptimizer final {
public:
    static constexpr std::size_t kMaximumCameras = 128u;
    static constexpr std::size_t kMaximumEdges = 4096u;

    GraphOptimizationResult solve(
        std::size_t cameraCount,
        const std::vector<GraphEdge>& edges,
        std::size_t iterations=48,
        double step=0.25) const;

    /*
     * Production overload for real distributed camera identities. Output pose index follows
     * cameraIds order. The legacy count-based overload remains available for existing C0...C127
     * integrations.
     */
    GraphOptimizationResult solve(
        const std::vector<std::string>& cameraIds,
        const std::vector<GraphEdge>& edges,
        std::size_t iterations=48,
        double step=0.25) const;
};

} // namespace nox::native::calibration

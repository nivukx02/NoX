#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nox::native::scene {

struct Vertex {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double confidence{1.0};
};

struct MeshTemporalOptions {
    double blendAlpha{0.35};
    double maxDisplacementMeters{0.10};
    double confidenceFloor{0.20};
    double maximumMatchDistanceMeters{0.08};
    double spatialCellSizeMeters{0.025};
    double innovationScaleMeters{0.008};
    double confidenceDamping{0.82};
    double velocityDamping{0.70};
    std::uint32_t searchRadiusCells{2};
    std::uint32_t maximumVertices{500'000};
};

struct MeshTemporalResult {
    std::vector<Vertex> vertices;
    double maxDisplacement{0.0};
    double p95Displacement{0.0};
    double p95Innovation{0.0};
    double confidence{0.0};
    double correspondenceCoverage{0.0};
    double temporalStability{0.0};
    double meanInnovation{0.0};
    bool accepted{false};
};

class MeshTemporalCoherence final {
public:
    std::vector<Vertex> stabilize(
        const std::vector<Vertex>& previous,
        const std::vector<Vertex>& current,
        double alpha=.35) const;

    MeshTemporalResult stabilizeHighEnd(
        const std::vector<Vertex>& previous,
        const std::vector<Vertex>& current,
        const MeshTemporalOptions& options={}) const;
};

} // namespace nox::native::scene

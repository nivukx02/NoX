#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nox::native::reconstruction {

struct Vertex {
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

struct Displacement {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double confidence{0.0};
    double uncertaintyMeters{0.01};
};

struct SurfaceEdge {
    std::uint32_t a{0};
    std::uint32_t b{0};
    double confidence{1.0};
};

struct ContactAnchor {
    std::uint32_t vertex{0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double confidence{1.0};
    double stiffness{1.0};
};

struct DeformationOptions {
    double dataWeight{1.0};
    double smoothnessWeight{0.22};
    double contactWeight{0.40};
    double strainWeight{0.35};
    double damping{0.72};
    double maximumDisplacementMeters{0.20};
    double maximumEdgeStrain{0.15};
    double convergenceToleranceMeters{0.0001};
    std::uint32_t iterations{12};
    std::uint32_t maximumVertices{500'000};
};

struct DeformationResult {
    std::vector<Vertex> vertices;
    double meanDisplacementMeters{0.0};
    double p95DisplacementMeters{0.0};
    double maxDisplacementMeters{0.0};
    double meanEdgeStrain{0.0};
    double maxEdgeStrain{0.0};
    double confidence{0.0};
    double objective{0.0};
    std::uint32_t iterations{0};
    bool converged{false};
    bool accepted{false};
};

class BodySurfaceDeformation final {
public:
    [[nodiscard]] std::vector<Vertex> apply(
        const std::vector<Vertex>& base,
        const std::vector<Displacement>& displacement) const;

    [[nodiscard]] DeformationResult solve(
        const std::vector<Vertex>& base,
        const std::vector<Displacement>& displacement,
        const std::vector<SurfaceEdge>& edges,
        const std::vector<ContactAnchor>& anchors,
        const DeformationOptions& options = {}) const;

private:
    static double clamp01(double value) noexcept;
    static bool finite(double value) noexcept;
    static double distance(const Vertex& a, const Vertex& b) noexcept;
    static double percentile95(std::vector<double> values) noexcept;
};

} // namespace nox::native::reconstruction

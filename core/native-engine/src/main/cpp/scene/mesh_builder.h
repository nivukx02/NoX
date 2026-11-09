#pragma once
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "scene/scene_point_map.h"

namespace nox::native::scene {

struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<float> vertexConfidence;
    std::vector<Vec3> vertexNormals;
    std::vector<float> triangleQuality;
    std::vector<float> vertexUncertaintyMeters;
    std::vector<float> vertexDynamicProbability;
    std::vector<float> vertexTemporalConsistency;
    double confidence{0.0};
};

struct MeshBuildConfig {
    double minimumPointConfidence{0.10};
    double neighborhoodRadius{0.20};
    double minimumNeighborDistance{1e-5};
    double maximumEdgeLength{0.30};
    double maximumSurfaceDeviation{0.04};
    double minimumTriangleArea{1e-7};
    double minimumInteriorAngleDegrees{8.0};
    double maximumNormalAngleDegrees{55.0};
    double maximumTriangleAspectRatio{8.0};
    std::size_t maximumNeighborsPerVertex{24};
    std::size_t maximumTriangles{250000};
    std::size_t maximumIncidentTrianglesPerEdge{2};
    bool requireNormalAgreement{true};
    bool enforceManifoldEdges{true};
    bool orientTrianglesToNormals{true};
    bool rejectDynamicPoints{true};
    double maximumVertexUncertaintyMeters{0.020};
    double minimumVertexTemporalConsistency{0.40};
    double maximumVertexDynamicProbability{0.45};
    double uncertaintyQualityScaleMeters{0.010};
    bool deterministicOrdering{true};
};

struct MeshBuildDiagnostics {
    std::size_t inputPoints{0};
    std::size_t rejectedDynamic{0};
    std::size_t rejectedLowConfidence{0};
    std::size_t rejectedDuplicate{0};
    std::size_t acceptedVertices{0};
    std::size_t candidateEdges{0};
    std::size_t acceptedTriangles{0};
    std::size_t rejectedTriangles{0};
    std::size_t rejectedNonManifold{0};
    std::size_t nonManifoldEdgeCount{0};
    double boundaryEdgeRatio{0.0};
    double meanTriangleQuality{0.0};
    double meanEdgeLength{0.0};
    double meanNormalAgreement{0.0};
    double meanVertexUncertaintyMeters{0.0};
    double p95VertexUncertaintyMeters{0.0};
    double meanDynamicProbability{0.0};
    double meanTemporalConsistency{0.0};
    double trustedVertexRatio{0.0};
    std::size_t rejectedHighUncertainty{0};
    std::size_t rejectedDynamicProbability{0};
    std::size_t rejectedTemporalInconsistency{0};
    double confidence{0.0};
};

class MeshBuilder final {
public:
    struct Result {
        Mesh mesh;
        MeshBuildDiagnostics diagnostics;
    };

    explicit MeshBuilder(MeshBuildConfig config = {}) noexcept;
    Result buildDetailed(const std::vector<Point>& points) const;
    Mesh build(const std::vector<Point>& points) const;
    const MeshBuildConfig& config() const noexcept { return config_; }

private:
    struct CandidatePoint { Point point; std::size_t sourceIndex{0}; };
    struct Cell {
        long long x{0}, y{0}, z{0};
        bool operator==(const Cell& other) const noexcept {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    struct CellHash {
        std::size_t operator()(const Cell& cell) const noexcept;
    };
    struct Neighbor {
        std::uint32_t index{0};
        double distance{0.0};
        double confidence{0.0};
        double normalAgreement{0.0};
        double tangentAngle{0.0};
    };
    struct TriangleCandidate {
        std::uint32_t a{0}, b{0}, c{0};
        double quality{0.0};
        double edgeLength{0.0};
        double normalAgreement{0.0};
    };
    struct Vec3d { double x{0.0}, y{0.0}, z{0.0}; };

    static double dot(const Vec3d&, const Vec3d&) noexcept;
    static Vec3d cross(const Vec3d&, const Vec3d&) noexcept;
    static double norm(const Vec3d&) noexcept;
    static Vec3d normalize(const Vec3d&) noexcept;
    static Vec3d sub(const Vec3d&, const Vec3d&) noexcept;
    static Vec3d add(const Vec3d&, const Vec3d&) noexcept;
    static Vec3d mul(const Vec3d&, double) noexcept;
    static Vec3d toVec3d(const Vec3&) noexcept;
    static Vec3 toVec3(const Vec3d&) noexcept;
    static double clamp01(double) noexcept;
    static std::uint64_t triangleKey(
        std::uint32_t, std::uint32_t, std::uint32_t) noexcept;
    static std::uint64_t edgeKey(
        std::uint32_t, std::uint32_t) noexcept;

    Cell cellFor(const Vec3d&) const noexcept;
    std::vector<CandidatePoint> filterAndCollapse(
        const std::vector<Point>&, MeshBuildDiagnostics&) const;
    std::vector<Neighbor> collectNeighbors(
        std::uint32_t,
        const std::vector<CandidatePoint>&,
        const std::unordered_map<Cell, std::vector<std::uint32_t>, CellHash>&) const;
    Vec3d normalFor(
        std::uint32_t,
        const std::vector<CandidatePoint>&,
        const std::vector<Neighbor>& = {}) const;
    bool acceptTriangle(
        std::uint32_t, std::uint32_t, std::uint32_t,
        const std::vector<CandidatePoint>&,
        double&, double&, double&) const;
    static double angleDeg(
        const Vec3d&, const Vec3d&, const Vec3d&) noexcept;

    MeshBuildConfig config_;
};

} // namespace nox::native::scene

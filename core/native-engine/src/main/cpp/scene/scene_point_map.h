#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nox::native::scene {

struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

struct Point {
    Vec3 p{};
    Vec3 normal{0,0,1};
    double confidence{0};
    std::uint64_t lastSeen{0};
    bool dynamic{false};
    double uncertaintyMeters{0.02};
    double dynamicProbability{0.0};
    double normalConsistency{1.0};
    double temporalConsistency{1.0};
    std::uint64_t observationCount{1};
};

class ScenePointMap final {
public:
    explicit ScenePointMap(
        double cellSizeMeters = 0.025,
        std::size_t maxPoints = 1'000'000);

    void integrate(
        const Vec3&,
        const Vec3&,
        double,
        std::uint64_t);

    void integrateEvidence(
        const Vec3& position,
        const Vec3& normal,
        double confidence,
        std::uint64_t timestampNanos,
        double uncertaintyMeters,
        double dynamicProbability,
        double temporalConsistency = 1.0);

    [[nodiscard]] std::vector<Point> snapshot() const;

    void reset() noexcept;

    [[nodiscard]] std::size_t size() const noexcept {
        return points_.size();
    }

private:
    struct CellHash {
        std::size_t operator()(
            const std::array<std::int64_t,3>&) const noexcept;
    };

    double cellSizeMeters_{0.025};
    std::size_t maxPoints_{1'000'000};
    std::vector<Point> points_;
    std::vector<std::array<std::int64_t,3>> pointCells_;

    std::unordered_map<
        std::array<std::int64_t,3>,
        std::vector<std::size_t>,
        CellHash> buckets_;

    [[nodiscard]] std::array<std::int64_t,3> cellFor(
        const Vec3&) const noexcept;

    static double distance(
        const Vec3&,
        const Vec3&) noexcept;

    static double normalizeLength(
        const Vec3&) noexcept;

    static Vec3 normalized(
        const Vec3&) noexcept;

    static double normalAgreement(
        const Vec3&,
        const Vec3&) noexcept;

    std::size_t findBestNeighbor(
        const Vec3&,
        const std::array<std::int64_t,3>&,
        double) const noexcept;

    void removeFromBucket(
        std::size_t index);

    void addToBucket(
        std::size_t index);

    void reindexPoint(
        std::size_t index,
        const std::array<std::int64_t,3>& newCell);
};

} // namespace nox::native::scene

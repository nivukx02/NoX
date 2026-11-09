#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nox::native::calibration {

struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

struct Quaternion {
    double w{1.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};

    Quaternion normalized() const noexcept;
    Quaternion conjugate() const noexcept;
    static Quaternion multiply(const Quaternion& a, const Quaternion& b) noexcept;
    Vec3 rotate(Vec3 v) const noexcept;
};

struct RigidPose {
    Quaternion rotation{};
    Vec3 translation{};

    RigidPose compose(const RigidPose& next) const noexcept;
    RigidPose inverse() const noexcept;
    Vec3 transform(Vec3 point) const noexcept;
};

struct RigEdge {
    std::string from;
    std::string to;
    RigidPose from_to{};
};

class RigGraph final {
public:
    void addEdge(RigEdge edge);
    std::optional<RigidPose> resolve(
        const std::string& reference,
        const std::string& target) const;

private:
    struct Neighbor {
        std::string camera;
        RigidPose pose{};
    };

    std::unordered_map<std::string, std::vector<Neighbor>> adjacency_;
};

} // namespace nox::native::calibration

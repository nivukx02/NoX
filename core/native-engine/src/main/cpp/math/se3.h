#pragma once
#include <array>
#include <cmath>

namespace nox::native::math {

struct Vec3d {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    Vec3d operator+(const Vec3d& other) const noexcept {
        return {x + other.x, y + other.y, z + other.z};
    }
    Vec3d operator-(const Vec3d& other) const noexcept {
        return {x - other.x, y - other.y, z - other.z};
    }
    Vec3d operator*(double scalar) const noexcept {
        return {x * scalar, y * scalar, z * scalar};
    }
};

struct Quaterniond {
    double w{1.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};

    [[nodiscard]] Quaterniond normalized() const noexcept;
    [[nodiscard]] Quaterniond conjugate() const noexcept;
    static Quaterniond multiply(const Quaterniond&, const Quaterniond&) noexcept;
    static Quaterniond slerp(const Quaterniond&, const Quaterniond&, double) noexcept;
    static Quaterniond fromRotationVector(const Vec3d&) noexcept;
    [[nodiscard]] Vec3d rotate(const Vec3d&) const noexcept;
    [[nodiscard]] Vec3d toRotationVector() const noexcept;
};

struct Twist6d {
    Vec3d rotation{};
    Vec3d translation{};
};

struct SE3d {
    Quaterniond rotation{};
    Vec3d translation{};

    [[nodiscard]] SE3d compose(const SE3d&) const noexcept;
    [[nodiscard]] SE3d inverse() const noexcept;
    [[nodiscard]] Vec3d transform(const Vec3d&) const noexcept;
    [[nodiscard]] Twist6d log() const noexcept;
    static SE3d exp(const Twist6d&) noexcept;
};

} // namespace nox::native::math

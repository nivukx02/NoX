#pragma once
#include <cstdint>
namespace nox::native::tracking {
struct Vec3{double x{0},y{0},z{0};};
struct JointObservation{std::uint32_t id{0};Vec3 position{};double confidence{0};};
struct JointState{
    std::uint32_t id{0};
    Vec3 position{};
    Vec3 velocity{};
    Vec3 acceleration{};
    Vec3 jerk{};
    Vec3 innovation{};
    double innovationMagnitude{0.0};
    double velocityMagnitude{0.0};
    double accelerationMagnitude{0.0};
    double jerkMagnitude{0.0};
    double confidence{0};
    double uncertainty{1};
    double predictionAgeSeconds{0.0};
    bool visible{false};
    bool predicted{false};
    bool rejected{false};
};
class JointTrack final{
public:
    explicit JointTrack(std::uint32_t id):id_(id){}
    JointState update(const JointObservation&,std::uint64_t timestampNanos,double trustScale=1.0) noexcept;
    JointState predict(std::uint64_t timestampNanos,double trustScale=1.0) noexcept;
    [[nodiscard]] JointState state() const noexcept { return state_; }
private:
    std::uint32_t id_;
    JointState state_{};
    std::uint64_t lastTimestampNanos_{0};
    bool initialized_{false};
};
} // namespace nox::native::tracking

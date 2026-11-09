#pragma once
#include <cstdint>
namespace nox::native::kinematics {
struct FootLockState {
    bool leftLocked{false};
    bool rightLocked{false};
    double leftError{0};
    double rightError{0};
    struct Vec3 {
        double x{0},y{0},z{0};
    };
    Vec3 leftAnchor{};
    Vec3 rightAnchor{};
    double leftAnchorDrift{0};
    double rightAnchorDrift{0};
    Vec3 leftCorrection{};
    Vec3 rightCorrection{};
    bool leftHasAnchor{false};
    bool rightHasAnchor{false};
};
class FootLockSolver final {
public:
    FootLockState solve(
        double leftSpeed,
        double rightSpeed,
        double leftHeight,
        double rightHeight,
        double speedThreshold=.05,
        double heightThreshold=.03) const noexcept;

    FootLockState updateAnchored(
        const FootLockState& previous,
        const FootLockState::Vec3& leftPosition,
        const FootLockState::Vec3& rightPosition,
        double leftSpeed,
        double rightSpeed,
        double leftHeight,
        double rightHeight,
        double speedThreshold=.05,
        double heightThreshold=.03,
        double releaseSpeedThreshold=.10,
        double releaseHeightThreshold=.06) const noexcept;
};
}
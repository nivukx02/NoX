#include "tracking/temporal_motion_state.h"
#include <algorithm>

namespace nox::native::tracking {
void TemporalMotionState::reset(
    std::uint64_t timestamp, const std::array<double, 3>& position, double confidence) noexcept {
    state_ = {};
    state_.position = position;
    const double variance = std::max(0.01, 1.0 - confidence);
    state_.variance = {variance, variance, variance};
    timestamp_ = timestamp;
    confidence_ = std::clamp(confidence, 0.0, 1.0);
    last_interval_seconds_ = 1.0 / 60.0;
    initialized_ = true;
}

filtering::MotionState3 TemporalMotionState::predict(std::uint64_t timestamp) noexcept {
    if (!initialized_) return state_;
    if (timestamp <= timestamp_) {
        if (timestamp < timestamp_) {
            confidence_ *= 0.90;
        }
        return state_;
    }

    const double dt =
        std::clamp(
            static_cast<double>(
                timestamp - timestamp_) *
                1e-9,
            1e-3,
            0.5);

    last_interval_seconds_ = dt;
    state_ = filtering::RobustStateVector::predict(
        state_,
        dt);
    timestamp_ = timestamp;
    confidence_ = std::max(0.0, confidence_ * (1.0 - 0.05 * dt));
    return state_;
}

filtering::MotionState3 TemporalMotionState::correct(
    std::uint64_t timestamp, const std::array<double, 3>& position, double confidence) noexcept {
    if (initialized_ && timestamp <= timestamp_) {
        confidence_ *= 0.94;
        return state_;
    }
    predict(timestamp);
    std::array<double, 3> innovation{};
    for (int i = 0; i < 3; ++i) {
        innovation[i] = position[i] - state_.position[i];
    }
    std::array<double, 3> innovationVariance{};
    for (int i = 0; i < 3; ++i) {
        innovationVariance[i] = state_.variance[i] + 0.02;
    }
    const auto decision = gate_.accept(innovation, innovationVariance);
    if (!decision.accepted) {
        confidence_ *= 0.92;
        return state_;
    }
    state_ = filtering::RobustStateVector::correct(
        state_,
        position,
        {0.0004, 0.0004, 0.0004},
        confidence,
        last_interval_seconds_);
    confidence_ = std::clamp(
        0.65 * confidence_ + 0.35 * confidence * decision.trust, 0.0, 1.0);
    return state_;
}
}

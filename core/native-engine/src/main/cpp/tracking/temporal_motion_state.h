#pragma once
#include <array>
#include <cstdint>
#include "filtering/innovation_gate.h"
#include "filtering/robust_state_vector.h"

namespace nox::native::tracking {
class TemporalMotionState final {
public:
    void reset(std::uint64_t timestamp, const std::array<double, 3>& position, double confidence) noexcept;
    filtering::MotionState3 predict(std::uint64_t timestamp) noexcept;
    filtering::MotionState3 correct(std::uint64_t timestamp, const std::array<double, 3>& position, double confidence) noexcept;
    [[nodiscard]] filtering::MotionState3 state() const noexcept { return state_; }
    [[nodiscard]] double confidence() const noexcept { return confidence_; }
    [[nodiscard]] double lastIntervalSeconds() const noexcept { return last_interval_seconds_; }
private:
    filtering::MotionState3 state_{};
    std::uint64_t timestamp_{0};
    double confidence_{0.0};
    bool initialized_{false};
    double last_interval_seconds_{1.0 / 60.0};
    filtering::InnovationGate gate_{};
};
}

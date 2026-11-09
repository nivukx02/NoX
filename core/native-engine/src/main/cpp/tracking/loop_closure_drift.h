#pragma once
#include <cstdint>
namespace nox::native::tracking {
struct LoopClosureInput { double positionError{0}; double rotationError{0}; double confidence{0}; std::uint64_t frame{0}; };
struct DriftCorrection { double translationScale{0}; double rotationScale{0}; double confidence{0}; bool applied{false}; };
class LoopClosureDriftCorrector final {
public:
    DriftCorrection solve(const LoopClosureInput&,double positionThreshold=0.25,double rotationThreshold=0.15) const noexcept;
};
}
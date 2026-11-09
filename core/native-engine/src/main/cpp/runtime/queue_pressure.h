#pragma once
#include <cstdint>
namespace nox::native::runtime { struct QueuePressure { double utilization{0}; std::uint32_t depth{0},capacity{0}; bool shedding{false}; }; class QueuePressureModel final { public: QueuePressure evaluate(std::uint32_t,std::uint32_t) const noexcept; }; }
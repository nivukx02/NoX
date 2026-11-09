#pragma once
#include <cstdint>
namespace nox::native::runtime { enum class Priority{Critical=0,High=1,Normal=2,Low=3}; struct WorkDecision{bool accepted{false};Priority priority{Priority::Normal};double workloadScale{1};bool shed{false};}; class WorkScheduler final { public: WorkDecision decide(Priority,std::uint32_t,std::uint32_t,int,double) const noexcept; }; }
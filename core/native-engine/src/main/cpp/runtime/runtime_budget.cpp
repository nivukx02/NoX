#include "runtime/runtime_budget.h"
namespace nox::native::runtime {
BudgetSample RuntimeBudget::beginFrame(std::uint64_t now) noexcept { start_=now; return {start_,budget_,0,false}; }
BudgetSample RuntimeBudget::endFrame(std::uint64_t now) noexcept { const auto e=now>=start_?now-start_:0; return {start_,budget_,e,e>budget_}; }
std::uint64_t RuntimeBudget::remainingNanos(std::uint64_t now) const noexcept { if(now<=start_) return budget_; const auto e=now-start_; return e>=budget_?0:budget_-e; }
}
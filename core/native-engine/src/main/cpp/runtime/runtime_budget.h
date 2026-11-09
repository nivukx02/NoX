#pragma once
#include <cstdint>
namespace nox::native::runtime {
struct BudgetSample { std::uint64_t frameStart{0}, budgetNanos{0}, elapsedNanos{0}; bool missed{false}; };
class RuntimeBudget final {
public: explicit RuntimeBudget(std::uint64_t frameBudgetNanos):budget_(frameBudgetNanos){}
    BudgetSample beginFrame(std::uint64_t now) noexcept;
    BudgetSample endFrame(std::uint64_t now) noexcept;
    std::uint64_t remainingNanos(std::uint64_t now) const noexcept;
private: std::uint64_t budget_; std::uint64_t start_{0};
};
}
#pragma once
#include <array>
namespace nox::native::filtering{struct InnovationDecision{bool accepted{false};double normalizedResidual{0};double trust{0};};class InnovationGate final{public:InnovationDecision accept(const std::array<double,3>&,const std::array<double,3>&,double=6.0)const noexcept;};}
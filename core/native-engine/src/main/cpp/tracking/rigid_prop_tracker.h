#pragma once
#include <cstdint>
namespace nox::native::tracking { struct PropState{double x{0},y{0},z{0};double confidence{0};}; class RigidPropTracker final{public:PropState update(const PropState&previous,double x,double y,double z,double measurementConfidence)const noexcept;};}
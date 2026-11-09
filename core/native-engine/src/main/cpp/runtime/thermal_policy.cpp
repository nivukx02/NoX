#include "runtime/thermal_policy.h"
#include <algorithm>
namespace nox::native::runtime { ThermalPolicyResult ThermalPolicy::evaluate(int s,double m) const noexcept { double scale=1.0; if(s>=4) scale=.5; else if(s>=3) scale=.75; if(m>.8) scale=std::min(scale,.65); return {scale,scale<.999}; } }
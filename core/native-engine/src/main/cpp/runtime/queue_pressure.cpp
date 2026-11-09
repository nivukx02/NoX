#include "runtime/queue_pressure.h"
#include <algorithm>
namespace nox::native::runtime { QueuePressure QueuePressureModel::evaluate(std::uint32_t d,std::uint32_t c) const noexcept { if(c==0) return {1.0,d,c,true}; const double u=std::clamp(static_cast<double>(d)/c,0.0,1.0); return {u,d,c,u>=0.85}; } }
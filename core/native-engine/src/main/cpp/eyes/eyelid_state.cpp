#include "eyes/eyelid_state.h"
#include <algorithm>
namespace nox::native::eyes{EyelidState EyelidStateEstimator::estimate(double l,double r,double c)const noexcept{return{1.0-std::clamp(l,0.0,1.0),1.0-std::clamp(r,0.0,1.0),std::clamp(c,0.0,1.0)};}}
#include "tracking/rigid_prop_tracker.h"
#include <algorithm>
namespace nox::native::tracking { PropState RigidPropTracker::update(const PropState&p,double x,double y,double z,double c)const noexcept{const double a=.2+.7*std::clamp(c,0.0,1.0);return{p.x*(1-a)+x*a,p.y*(1-a)+y*a,p.z*(1-a)+z*a,.7*p.confidence+.3*c};} }
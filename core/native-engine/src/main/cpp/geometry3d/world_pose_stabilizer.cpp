#include "geometry3d/world_pose_stabilizer.h"
#include <algorithm>
namespace nox::native::geometry { Pose WorldPoseStabilizer::update(const Pose&p,const Pose&m,double c)const noexcept{double a=std::clamp(.08+.7*c,.08,.8);return{p.x*(1-a)+m.x*a,p.y*(1-a)+m.y*a,p.z*(1-a)+m.z*a};} }
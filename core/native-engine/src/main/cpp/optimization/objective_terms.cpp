#include "optimization/objective_terms.h"
#include <algorithm>
#include <cmath>
namespace nox::native::optimization {
namespace { double dist(kinematics::Vec3 a,kinematics::Vec3 b){const double x=a.x-b.x,y=a.y-b.y,z=a.z-b.z;return std::sqrt(x*x+y*y+z*z);} }
ObjectiveBreakdown ObjectiveTerms::evaluate(const kinematics::Pose& c,const kinematics::Pose& o,const std::vector<kinematics::Bone>& bones,double floor) const {
    ObjectiveBreakdown r;
    const auto n=std::min(c.joints.size(),o.joints.size());
    for(std::size_t i=0;i<n;++i){
        const double d=dist(c.joints[i],o.joints[i]);
        r.reprojection+=d*d;
        r.multiView+=0.5*d*d;
        r.temporal+=0.25*d*d;
        if(c.joints[i].y<floor){const double e=floor-c.joints[i].y;r.contact+=e*e;}
    }
    for(std::size_t i=0;i<n && i<bones.size();++i){
        const int p=bones[i].parent;
        if(p>=0 && static_cast<std::size_t>(p)<n){
            const double e=dist(c.joints[i],c.joints[p])-bones[i].length;
            r.bone+=e*e;
        }
    }
    r.total=r.bone+r.jointLimit+r.contact+r.temporal+r.reprojection+r.multiView+r.faceHead+r.handWrist;
    return r;
}
}
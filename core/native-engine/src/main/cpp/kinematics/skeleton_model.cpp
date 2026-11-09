#include "kinematics/skeleton_model.h"
#include <cmath>
namespace nox::native::kinematics {
namespace {
Vec3 add(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
Vec3 mul(Vec3 a, double s) { return {a.x*s,a.y*s,a.z*s}; }
Vec3 norm(Vec3 v) { const double n=std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z); return n<1e-12?Vec3{0,0,1}:mul(v,1.0/n); }
}
Pose SkeletonModel::forwardKinematics(const Vec3& root,const std::vector<Vec3>& directions,const std::vector<Bone>& bones) const {
    Pose out; out.joints.resize(bones.size());
    for(std::size_t i=0;i<bones.size();++i){
        const int p=bones[i].parent;
        if(p<0 || static_cast<std::size_t>(p)>=out.joints.size()) out.joints[i]=root;
        else out.joints[i]=add(out.joints[p],mul(norm(directions[i]),bones[i].length));
    }
    return out;
}
}
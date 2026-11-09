#include <cassert>
#include <cmath>
#include "tracking/imu_vision_fusion.h"
int main(){using namespace nox::native::imu;ImuVisionFusion f;auto s=f.propagate(0,0,1,0,9.8,0,.01);double n=std::sqrt(s.orientation.w*s.orientation.w+s.orientation.x*s.orientation.x+s.orientation.y*s.orientation.y+s.orientation.z*s.orientation.z);assert(std::abs(n-1)<1e-9);auto c=f.correct({1,0,0,0},.9);assert(c.confidence>.2);f.reset();assert(f.biases().confidence==0);}
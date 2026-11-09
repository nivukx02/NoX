#include "filtering/innovation_gate.h"
#include <algorithm>
#include <cmath>
namespace nox::native::filtering{InnovationDecision InnovationGate::accept(const std::array<double,3>&i,const std::array<double,3>&v,double h)const noexcept{double s=0;for(int k=0;k<3;++k)s+=(i[k]*i[k])/std::max(v[k],1e-9);double r=std::sqrt(s);return{r<=h,r,std::exp(-.5*r)};}}
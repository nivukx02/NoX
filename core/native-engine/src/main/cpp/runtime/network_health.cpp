#include "runtime/network_health.h"
#include <algorithm>
#include <cmath>
namespace nox::native::runtime { NetworkHealth NetworkHealthModel::evaluate(double l,double loss,double j)const noexcept{double s=std::clamp(std::exp(-l/80.0)*std::exp(-loss*12.0)*std::exp(-j/20.0),0.0,1.0);return{l,loss,j,s,s>=.6};} }
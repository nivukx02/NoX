#include "runtime/thermal_sensor_drift.h"
#include <algorithm>
namespace nox::native::runtime { ThermalDrift ThermalSensorDrift::correct(double m,double r,double age)const noexcept{const double d=(m-r)*std::min(1.0,age/60.0);return{m-d,d,std::max(0.0,1.0-age/300.0)};} }
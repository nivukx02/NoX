#include "lighting/exposure_vignette_model.h"
#include <algorithm>
#include <cmath>
namespace nox::native::lighting { double ExposureVignetteModel::correctedLuma(double l,double e,double r,double s)const noexcept{const double v=1.0-std::clamp(s*r*r,0.0,.8);return l*std::max(v,1e-3)*std::pow(2.0,-e);} }
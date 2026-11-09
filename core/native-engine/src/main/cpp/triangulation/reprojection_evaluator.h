#pragma once
#include <vector>
#include "triangulation/weighted_triangulator.h"
namespace nox::native::triangulation{struct ReprojectionResidual{double rms{0};std::vector<double> perView;};class ReprojectionEvaluator final{public:ReprojectionResidual evaluate(const Vec3&,const std::vector<RayObservation>&)const;};}
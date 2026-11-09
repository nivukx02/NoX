#pragma once
#include "triangulation/weighted_triangulator.h"
namespace nox::native::triangulation{class CheiralityGate final{public:static bool allInFront(const Vec3&point,const std::vector<RayObservation>&rays,double minimumDepth=1e-5)noexcept;};}
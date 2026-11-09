#pragma once
#include <vector>
#include "scene/scene_point_map.h"
namespace nox::native::scene{class NormalEstimator final{public:static Vec3 estimate(const Point&,const Point&,const Point&)noexcept;};}
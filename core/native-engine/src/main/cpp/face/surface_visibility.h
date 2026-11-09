#pragma once
#include <vector>
namespace nox::native::face {
struct SurfaceVertex { double x{0},y{0},z{0}; double nx{0},ny{0},nz{1}; double confidence{0}; bool visible{false}; };
class SurfaceVisibility final { public: std::vector<SurfaceVertex> evaluate(const std::vector<SurfaceVertex>&,double vx,double vy,double vz) const; };
}
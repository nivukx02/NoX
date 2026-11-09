#pragma once
#include <vector>
namespace nox::native::metric {
struct PairMeasurement { double observed{0}; double metric{0}; double weight{1}; };
struct MetricConstraintResult { double scale{1}; double uncertainty{1}; double residual{0}; double confidence{0}; bool valid{false}; };
class RigidMetricConstraint final { public: MetricConstraintResult solve(const std::vector<PairMeasurement>&) const; };
}
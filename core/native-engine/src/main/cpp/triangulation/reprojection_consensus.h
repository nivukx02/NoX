#pragma once
#include <vector>
namespace nox::native::triangulation {
struct ConsensusResult { double median{0}; double mad{0}; double robustRms{0}; double confidence{0}; bool consistent{false}; };
class ReprojectionConsensus final {
public: ConsensusResult evaluate(const std::vector<double>& residuals,double rejectSigma=3.0) const;
};
}
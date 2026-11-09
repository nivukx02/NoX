#include "triangulation/reprojection_consensus.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace nox::native::triangulation {
ConsensusResult ReprojectionConsensus::evaluate(
    const std::vector<double>& residuals, double sigma) const {
    ConsensusResult output;
    if (residuals.empty()) return output;

    std::vector<double> sorted = residuals;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];

    std::vector<double> deviations;
    deviations.reserve(sorted.size());
    for (double value : sorted) {
        deviations.push_back(std::abs(value - median));
    }
    std::sort(deviations.begin(), deviations.end());
    output.median = median;
    output.mad = deviations[deviations.size() / 2];

    const double scale = std::max(1e-9, 1.4826 * output.mad);
    double squared = 0.0;
    std::size_t used = 0;
    for (double value : residuals) {
        const double z = std::abs(value - median) / scale;
        if (z <= sigma) {
            squared += value * value;
            ++used;
        }
    }

    output.robustRms = used == 0
        ? std::numeric_limits<double>::infinity()
        : std::sqrt(squared / static_cast<double>(used));
    output.confidence = std::clamp(
        std::exp(-output.robustRms / 3.0) *
        (static_cast<double>(used) / residuals.size()),
        0.0,
        1.0);
    output.consistent = output.confidence >= 0.55;
    return output;
}
}

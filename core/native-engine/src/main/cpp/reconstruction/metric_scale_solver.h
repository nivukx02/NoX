#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "reconstruction/scale_evidence.h"

namespace nox::native::scale {

struct ScaleSourceSummary {
    Source source{Source::RelativeDepthOnly};
    std::size_t supportingEvidence{0};
    double weightedSupport{0.0};
};

struct Result {
    double scale{1.0};
    double uncertainty{1.0};
    double p95Disagreement{1.0};
    double medianDisagreement{1.0};
    double confidence{0.0};
    double consensusSupport{0.0};
    double sourceDiversity{0.0};
    double traceability{0.0};
    Source source{Source::RelativeDepthOnly};
    std::vector<ScaleSourceSummary> sourceSummaries;
    bool valid{false};
    bool metricReady{false};
};

class MetricScaleSolver final {
public:
    [[nodiscard]] Result solve(
        const std::vector<ScaleEvidence>& evidence) const;

private:
    static double clamp01(double value) noexcept;
    static double median(
        std::vector<double> values) noexcept;
    static double percentile(
        std::vector<double> values,
        double probability) noexcept;
};

} // namespace nox::native::scale

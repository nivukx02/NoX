#include <cassert>
#include <cmath>

#include "reconstruction/metric_scale_solver.h"

int main() {
    using namespace nox::native::scale;

    const auto strong =
        MetricScaleSolver{}.solve({
            ScaleEvidence{
                Source::MeasuredRig,
                2.000,
                0.010,
                "rig#1",
                0.98,
                0.95,
                0.99,
            },
            ScaleEvidence{
                Source::KnownSceneDimension,
                2.005,
                0.020,
                "scene#2",
                0.94,
                0.85,
                0.95,
            },
            ScaleEvidence{
                Source::StableBaseline,
                1.998,
                0.015,
                "baseline#3",
                0.92,
                0.90,
                0.97,
            },
            ScaleEvidence{
                Source::RelativeDepthOnly,
                0.50,
                0.01,
                "relative",
            },
        });

    assert(strong.valid);
    assert(strong.metricReady);
    assert(strong.scale > 1.98);
    assert(strong.scale < 2.02);
    assert(strong.consensusSupport > 0.80);
    assert(strong.sourceDiversity > 0.60);
    assert(strong.traceability > 0.40);
    assert(std::isfinite(strong.p95Disagreement));

    const auto outlier =
        MetricScaleSolver{}.solve({
            ScaleEvidence{
                Source::MeasuredRig,
                2.000,
                0.010,
                "rig#1",
                0.98,
                0.95,
                0.99,
            },
            ScaleEvidence{
                Source::KnownSceneDimension,
                2.010,
                0.020,
                "scene#2",
                0.90,
                0.80,
                0.90,
            },
            ScaleEvidence{
                Source::StableBaseline,
                0.900,
                0.010,
                "bad#3",
                0.40,
                0.20,
                0.20,
            },
        });

    assert(std::isfinite(outlier.scale));
    assert(outlier.scale > 1.90);
    assert(outlier.scale < 2.08);
    assert(outlier.consensusSupport < strong.consensusSupport);

    const auto bad =
        MetricScaleSolver{}.solve({
            ScaleEvidence{
                Source::RelativeDepthOnly,
                0.5,
                0.01,
                "relative",
            },
        });

    assert(!bad.valid);
    assert(!bad.metricReady);
    return 0;
}

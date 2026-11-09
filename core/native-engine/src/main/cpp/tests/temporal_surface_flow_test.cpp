#include "reconstruction/temporal_surface_flow.h"
#include <cassert>
#include <vector>

int main() {
    nox::native::reconstruction::TemporalSurfaceFlow flow;
    std::vector<nox::native::reconstruction::SurfaceSample> previous{
        {1u, 0.0, 0.0, 1.0, 0.95},
        {2u, 0.5, 0.0, 1.0, 0.90},
        {3u, 0.0, 0.5, 1.0, 0.92},
    };
    std::vector<nox::native::reconstruction::SurfaceSample> current{
        {11u, 0.01, 0.0, 1.0, 0.95},
        {12u, 0.51, 0.0, 1.0, 0.90},
        {13u, 0.0, 0.51, 1.0, 0.92},
    };

    const auto result =
        flow.estimate(previous, current, 1.0 / 60.0);

    assert(result.matchedCount == 3u);
    assert(result.correspondenceCoverage > 0.9);
    assert(result.temporalStability > 0.5);
    assert(result.p95ResidualMeters < 0.02);
    return 0;
}

#include <cassert>
#include <cmath>
#include <vector>

#include "reconstruction/depth_uncertainty_fusion.h"

using namespace nox::native::reconstruction;

int main() {
    DepthUncertaintyFusion fusion;

    const auto independent =
        fusion.fuseRobust({
            DepthEstimate{
                2.000, 0.0004, 0.98, 0.0, 0.98, 0.98, 0.0, 0.0,
            },
            DepthEstimate{
                2.002, 0.0004, 0.98, 0.0, 0.98, 0.98, 0.0, 0.0,
            },
            DepthEstimate{
                1.998, 0.0004, 0.98, 0.0, 0.98, 0.98, 0.0, 0.0,
            },
        });

    const auto correlated =
        fusion.fuseRobust({
            DepthEstimate{
                2.000, 0.0004, 0.98, 0.0, 0.98, 0.98, 0.95, 0.0,
            },
            DepthEstimate{
                2.002, 0.0004, 0.98, 0.0, 0.98, 0.98, 0.95, 0.0,
            },
            DepthEstimate{
                1.998, 0.0004, 0.98, 0.0, 0.98, 0.98, 0.95, 0.0,
            },
        });

    assert(independent.valid);
    assert(correlated.valid);
    assert(
        correlated.effectiveSampleCount <
        independent.effectiveSampleCount);
    assert(
        correlated.variance >=
        independent.variance * 0.80);

    return 0;
}

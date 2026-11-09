#include "tracking/dense_visibility_contact_fusion.h"
#include <cassert>
#include <vector>

int main() {
    using nox::native::tracking::DenseVisibilityContactFusion;
    using nox::native::tracking::LandmarkEvidence;

    DenseVisibilityContactFusion fusion;
    std::vector<std::vector<LandmarkEvidence>> views{
        {
            {0u, 0.95, 0.98, 0.7, true, false, false},
            {1u, 0.90, 0.90, 0.4, true, false, true},
        },
        {
            {0u, 0.93, 0.95, 0.9, true, false, false},
            {1u, 0.88, 0.91, 0.5, true, false, true},
        },
    };

    const auto result = fusion.fuse(views);
    assert(result.evaluatedLandmarks == 2u);
    assert(result.visibleCoverage > 0.9);
    assert(result.crossViewSupport > 0.45);
    assert(result.contactCoverage > 0.0);
    assert(result.confidence > 0.75);
    return 0;
}

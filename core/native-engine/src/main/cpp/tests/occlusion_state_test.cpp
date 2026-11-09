#include "tracking/occlusion_state.h"

#include <cassert>

int main() {
    using namespace nox::native::occlusion;

    OcclusionStateTracker tracker;

    const auto visible =
        tracker.visible(
            0.96,
            0.08,
            0.95,
            0.20,
            0.96);

    assert(visible.state == State::Visible);
    assert(visible.crossViewSupport > 0.90);
    assert(visible.temporalStability > 0.90);

    const auto predicted =
        tracker.predict(
            0.92,
            0.10,
            0.040,
            0.200,
            0.85,
            0.75,
            0.90);

    assert(predicted.state == State::Predicted);
    assert(predicted.confidence > 0.20);
    assert(predicted.contactProbability > 0.50);
    assert(predicted.uncertainty > 0.20);

    const auto fused =
        tracker.fuse({
            visible,
            tracker.partiallyOccluded(
                0.72,
                0.30,
                0.65,
                0.70,
                0.72),
            predicted,
        });

    assert(fused.state == State::Visible);
    assert(fused.confidence > 0.55);
    assert(fused.crossViewSupport > 0.50);
    assert(fused.contactProbability > 0.40);

    const auto unsupported =
        tracker.predict(
            0.90,
            0.10,
            0.50,
            0.20);

    assert(unsupported.state == State::Unsupported);
    assert(unsupported.confidence == 0.0);

    return 0;
}

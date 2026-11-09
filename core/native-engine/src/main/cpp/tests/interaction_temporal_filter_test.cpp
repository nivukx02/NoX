#include <cassert>
#include <cmath>

#include "tracking/interaction_temporal_filter.h"

int main() {
    using namespace nox::native::tracking;

    InteractionTemporalFilter filter(
        InteractionTemporalConfig{
            0.12,
            0.45,
            0.25,
            6,
            0.80,
            0.35,
            0.12,
            0.90,
            0.28,
        });

    const Interaction sample{
        1, 2, 0.08, -0.10, 1.0, true,
    };

    const auto first = filter.update({sample}, 0.1);
    assert(!first.empty());
    assert(first.front().interacting);
    assert(first.front().contactAgeSeconds > 0.0);
    assert(first.front().peakApproachRate >= 0.10);

    const Interaction weak{
        1, 2, 0.11, 0.0, 0.10, false,
    };
    const auto second = filter.update({weak}, 0.01);
    assert(!second.empty());
    assert(second.front().interacting);
    assert(second.front().contactProbability > 0.25);

    const Interaction impact{
        1, 2, 0.04, -1.50, 0.92, true,
    };
    const auto third = filter.update({impact}, 0.016);
    assert(!third.empty());
    assert(third.front().impactScore > 0.50);
    assert(third.front().peakApproachRate >= 1.0);

    const auto missing =
        filter.update({}, 0.016);
    assert(!missing.empty());
    assert(missing.front().interacting);
    assert(missing.front().contactProbability > 0.20);

    const auto recovered =
        filter.update({sample}, 0.016);
    assert(!recovered.empty());
    assert(recovered.front().interactionConfidence > 0.0);

    const auto stale =
        filter.update({sample}, -0.25);
    assert(!stale.empty());
    assert(stale.front().staleEvidenceRejected);
    assert(stale.front().interacting);

    filter.reset();
    assert(
        filter.update({}, 0.016).empty());

    return 0;
}

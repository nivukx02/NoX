#include <cassert>
#include <cmath>
#include <vector>

#include "validation/paired_reference_statistics.h"

int main() {
    using namespace nox::native::validation;

    std::vector<PairedReferenceSample> samples;
    samples.reserve(1600);

    for (int i = 0; i < 1600; ++i) {
        const double nox =
            0.00035 +
            static_cast<double>(i % 7) * 0.00001;
        const double reference =
            0.00085 +
            static_cast<double>(i % 5) * 0.00001;

        samples.push_back({
            nox,
            reference,
        });
    }

    const auto result =
        PairedReferenceStatisticsEngine{}.compare(
            samples,
            0x9e3779b97f4a7c15ULL);

    assert(result.sampleCount == 1600);
    assert(result.meanPairedDifferenceMeters < 0.0);
    assert(result.p95NoXErrorMeters < result.p95ReferenceErrorMeters);
    assert(result.probabilityNoXBetter > 0.99);
    assert(
        result.winner ==
        PairedReferenceWinner::NOX);
    assert(result.confidenceIntervalUpperMeters < 0.0);
    assert(
        result.repeatabilityConfidence > 0.90);

    return 0;
}

#include <cassert>
#include <cmath>
#include <vector>

#include "validation/dynamic_metrology_protocol.h"

int main() {
    using namespace nox::native::validation;

    DynamicMetrologyProtocol protocol({
        .referenceLengthMeters = 0.32088,
        .maximumGapSeconds = 0.050,
        .targetP95ErrorMillimeters = 1.0,
        .targetRmseMillimeters = 0.75,
        .targetMaxErrorMillimeters = 2.0,
        .minimumConfidence = 0.50,
    });

    std::vector<MetrologyEndpointSample> samples;
    samples.reserve(120);

    for (
        std::uint64_t frame = 0;
        frame < 120;
        ++frame) {
        const double t =
            static_cast<double>(frame) /
            120.0;

        const double motion =
            0.05 *
            std::sin(
                t *
                16.0);

        samples.push_back({
            1'000'000'000ULL +
                frame *
                8'333'333ULL,
            motion,
            0.0,
            0.0,
            motion +
                0.32055 +
                0.00015 *
                    std::sin(
                        t *
                        40.0),
            0.0,
            0.0,
            0.98,
        });
    }

    const auto report =
        protocol.evaluate(
            samples);

    assert(report.sampleCount == 120U);
    assert(report.acceptedSamples == 120U);
    assert(report.dynamicCoverageValid);
    assert(report.p95ErrorMillimeters < 1.0);
    assert(report.rmseMillimeters < 0.75);
    assert(report.maxErrorMillimeters < 1.0);
    assert(std::isfinite(report.driftMillimetersPerMinute));
    assert(report.confidence > 0.70);

    return 0;
}

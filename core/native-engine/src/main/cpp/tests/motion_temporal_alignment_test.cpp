#include <cassert>
#include <cmath>
#include <vector>

#include "synchronization/motion_temporal_alignment.h"

int main() {
    using namespace nox::native::sync;

    std::vector<MotionTemporalSample> reference;
    std::vector<MotionTemporalSample> remote;

    for (std::size_t i = 0; i < 96U; ++i) {
        const double x =
            static_cast<double>(i);
        const double pulse =
            std::exp(
                -(
                    x -
                    42.0
                ) * (
                    x -
                    42.0
                ) /
                18.0);

        reference.push_back({
            1'000'000'000ULL +
                static_cast<std::uint64_t>(i) *
                    8'333'333ULL,
            pulse,
            1.0,
        });

        const double remoteX =
            x - 4.25;
        const double remotePulse =
            std::exp(
                -(
                    remoteX -
                    42.0
                ) * (
                    remoteX -
                    42.0
                ) /
                18.0);

        remote.push_back({
            1'100'000'000ULL +
                static_cast<std::uint64_t>(i) *
                    8'333'333ULL,
            remotePulse,
            0.98,
        });
    }

    const auto result =
        MotionTemporalAlignment{}.estimate(
            reference,
            remote,
            8'333'333.0);

    assert(result.valid);
    assert(result.offsetSamples > 3.5);
    assert(result.offsetSamples < 5.0);
    assert(result.confidence > 0.60);
    assert(result.supportRatio > 0.80);
    assert(result.evaluatedLags > 0U);

    const auto strict =
        MotionTemporalAlignment{}.estimate(
            reference,
            remote,
            8'333'333.0,
            MotionTemporalAlignmentOptions{
                .minimumSamples = 16U,
                .maximumLags = 64U,
                .minimumCorrelation = 0.55,
                .minimumConfidence = 0.70,
                .minimumPeakSeparation = 0.07,
                .differentiateSignal = true,
            });

    assert(strict.valid);
    assert(std::abs(strict.offsetNanos) > 20'000'000.0);

    return 0;
}

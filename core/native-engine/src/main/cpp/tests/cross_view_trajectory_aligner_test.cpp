#include <cassert>
#include <cmath>
#include <vector>

#include "synchronization/cross_view_trajectory_aligner.h"

int main() {
    using namespace nox::native::sync;

    std::vector<TrajectorySample> a;
    std::vector<TrajectorySample> b;

    for (
        std::uint64_t frame = 0;
        frame < 240;
        ++frame) {
        const double time =
            static_cast<double>(frame) /
            120.0;

        const double x =
            0.4 *
            std::sin(
                time *
                7.0);
        const double y =
            0.1 *
            std::sin(
                time *
                13.0);
        const double z =
            2.0 +
            0.05 *
            std::cos(
                time *
                5.0);

        a.push_back({
            1'000'000'000ULL +
                frame *
                8'333'333ULL,
            x,
            y,
            z,
            0.98,
        });
    }

    const std::uint64_t offsetFrames = 4ULL;

    for (
        std::uint64_t frame = 0;
        frame < 240;
        ++frame) {
        const std::uint64_t source =
            frame >= offsetFrames
                ? frame - offsetFrames
                : 0ULL;

        const auto& sample =
            a[
                static_cast<std::size_t>(
                    source)
            ];

        b.push_back({
            2'000'000'000ULL +
                frame *
                8'333'333ULL,
            sample.x,
            sample.y,
            sample.z,
            0.97,
        });
    }

    const auto result =
        CrossViewTrajectoryAligner{}.align(
            a,
            b);

    assert(result.valid);
    assert(result.comparedSamples > 100U);
    assert(result.correlation > 0.45);
    assert(std::abs(result.offsetSamples + 4.0) < 0.6);
    assert(result.confidence > 0.60);
    assert(result.curvature >= 0.0);
    assert(result.sideLobeCorrelation <= result.correlation + 1e-9);
    assert(result.peakSeparation >= 0.0);
    assert(result.peakSeparation <= 1.0);
    assert(result.overlapRatio > 0.50);
    assert(result.offsetUncertaintySeconds >= 0.0);

    return 0;
}

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "synchronization/synchronization_estimator.h"
#include "synchronization/subframe_alignment.h"
#include "synchronization/event_correlation.h"

int main() {
    using namespace nox::native::sync;

    std::vector<std::uint64_t> r, q;
    constexpr double off = 2'500'000.0;
    constexpr double slope = 1.000120;

    for(int i=0;i<32;++i){
        const auto t =
            1'000'000'000ull +
            static_cast<std::uint64_t>(i)*33'333'333ull;
        r.push_back(t);
        q.push_back(
            static_cast<std::uint64_t>(
                static_cast<double>(t)*slope+off));
    }

    auto x=SynchronizationEstimator{}.estimate(r,q);
    assert(x.valid);
    assert(std::abs(x.drift_ppm-120)<.5);
    assert(std::abs(x.offset_nanos-off)<2000);
    assert(x.jitter_nanos<2000);
    assert(x.sampleCount==32);
    assert(x.residual_p95_nanos<5000);
    assert(x.residualMaxNanos<5000);
    assert(x.clockSupportRatio>0.90);

    std::vector<std::uint64_t> badQ=q;
    badQ[7]+=12'000'000ull;
    badQ[18]-=9'000'000ull;

    auto robust =
        SynchronizationEstimator{}.estimate(r,badQ);
    assert(robust.sampleCount==32);
    assert(robust.temporal_confidence < x.temporal_confidence);
    assert(robust.outlier_ratio > 0.0);
    assert(robust.valid);

    auto mismatch =
        SynchronizationEstimator{}.estimate(
            {1,2,3},
            {1,2}
        );
    assert(!mismatch.valid);
    assert(mismatch.sampleCount==0);

    auto reversed =
        SynchronizationEstimator{}.estimate(
            {3,2,4},
            {3,2,4}
        );
    assert(!reversed.valid);
    assert(reversed.monotonicInput==false);

    auto d=SubframeAlignment::refineParabolicPeak(
        .8,1,.9);
    assert(d>0&&d<.5);

    const auto peak =
        SubframeAlignment::refinePeak(
            .8,
            1.0,
            .9);
    assert(peak.valid);
    assert(peak.confidence>0.25);

    std::vector<double> eventA(64);
    std::vector<double> eventB(64);
    for(std::size_t i=0;i<eventA.size();++i){
        const double center =
            static_cast<double>(i) -
            18.25;
        eventA[i] =
            std::exp(
                -center*center/20.0);
        const double shifted =
            static_cast<double>(i) -
            22.25;
        eventB[i] =
            std::exp(
                -shifted*shifted/20.0);
    }

    const auto correlation =
        EventCorrelation{}.estimate(
            eventA,
            eventB);
    assert(correlation.valid);
    assert(correlation.offset_samples>3.5);
    assert(correlation.offset_samples<4.5);
    assert(correlation.confidence>0.60);
    assert(correlation.sharpness>0.0);

    const auto strict =
        SynchronizationEstimator{}.estimate(
            r,
            q,
            eventA,
            eventB,
            1'000'000.0,
            SynchronizationOptions::highPrecision());
    assert(strict.valid);
    assert(strict.temporal_confidence >= 0.85);
    assert(strict.clockSupportRatio >= 0.90);
    assert(strict.outlier_ratio <= 0.05);
    assert(strict.subframe_confidence >= 0.65);

    std::vector<MotionTemporalSample> motionReference;
    std::vector<MotionTemporalSample> motionRemote;
    for (std::size_t i = 0; i < 64U; ++i) {
        const double x = static_cast<double>(i);
        const double pulse =
            std::exp(
                -(x - 24.0) *
                (x - 24.0) /
                12.0);
        motionReference.push_back({
            2'000'000'000ULL +
                static_cast<std::uint64_t>(i) *
                8'333'333ULL,
            pulse,
            1.0,
        });
        motionRemote.push_back({
            2'100'000'000ULL +
                static_cast<std::uint64_t>(i) *
                8'333'333ULL,
            std::exp(
                -(x - 28.0) *
                (x - 28.0) /
                12.0),
            0.98,
        });
    }

    const auto fused =
        SynchronizationEstimator{}.estimateWithMotionSignals(
            r,
            q,
            motionReference,
            motionRemote,
            8'333'333.0);

    assert(fused.motionConfidence > 0.60);
    assert(fused.motionOffsetNanos > 20'000'000.0);
    assert(std::isfinite(fused.fusedOffsetNanos));
    assert(fused.temporal_confidence > 0.50);

    return 0;
}

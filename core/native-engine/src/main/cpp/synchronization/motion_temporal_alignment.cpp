#include "synchronization/motion_temporal_alignment.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace nox::native::sync {

namespace {

std::vector<double> preprocess(
    const std::vector<MotionTemporalSample>& samples,
    bool differentiate) {
    std::vector<double> signal;
    signal.reserve(
        samples.size());

    for (const auto& sample :
         samples) {
        if (
            sample.timestampNanos == 0ULL ||
            !std::isfinite(
                sample.motionEnergy) ||
            !std::isfinite(
                sample.confidence) ||
            sample.confidence <= 0.0) {
            signal.push_back(
                0.0);
            continue;
        }

        signal.push_back(
            sample.motionEnergy *
            std::clamp(
                sample.confidence,
                0.0,
                1.0));
    }

    if (
        signal.size() < 3U ||
        !differentiate) {
        return signal;
    }

    std::vector<double> difference(
        signal.size(),
        0.0);

    for (
        std::size_t index = 1U;
        index < signal.size();
        ++index) {
        difference[index] =
            signal[index] -
            signal[index - 1U];
    }

    return difference;
}

void robustNormalize(
    std::vector<double>& values) {
    if (values.empty()) return;

    const double mean =
        std::accumulate(
            values.begin(),
            values.end(),
            0.0) /
        static_cast<double>(
            values.size());

    double variance = 0.0;
    for (const double value :
         values) {
        const double delta =
            value -
            mean;
        variance +=
            delta * delta;
    }

    variance /=
        static_cast<double>(
            std::max<std::size_t>(
                values.size(),
                1U));

    const double scale =
        std::sqrt(
            std::max(
                variance,
                1e-12));

    for (double& value :
         values) {
        value =
            (
                value -
                mean
            ) /
            scale;
    }
}

} // namespace

double MotionTemporalAlignment::clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

double MotionTemporalAlignment::correlationAtLag(
    const std::vector<double>& a,
    const std::vector<double>& b,
    int lag,
    double& support) noexcept {
    support = 0.0;

    double sumA = 0.0;
    double sumB = 0.0;
    double count = 0.0;

    for (
        std::size_t index = 0;
        index < a.size();
        ++index) {
        const long long other =
            static_cast<long long>(
                index) +
            static_cast<long long>(
                lag);

        if (
            other < 0 ||
            other >=
                static_cast<long long>(
                    b.size())) {
            continue;
        }

        sumA +=
            a[index];
        sumB +=
            b[
                static_cast<std::size_t>(
                    other)
            ];
        count +=
            1.0;
    }

    if (count < 4.0) {
        return
            -std::numeric_limits<double>::infinity();
    }

    const double meanA =
        sumA / count;
    const double meanB =
        sumB / count;

    double numerator = 0.0;
    double varianceA = 0.0;
    double varianceB = 0.0;

    for (
        std::size_t index = 0;
        index < a.size();
        ++index) {
        const long long other =
            static_cast<long long>(
                index) +
            static_cast<long long>(
                lag);

        if (
            other < 0 ||
            other >=
                static_cast<long long>(
                    b.size())) {
            continue;
        }

        const double x =
            a[index] -
            meanA;
        const double y =
            b[
                static_cast<std::size_t>(
                    other)
            ] -
            meanB;

        const double taper =
            0.80 +
            0.20 *
                (
                    1.0 -
                    std::abs(
                        2.0 *
                            (
                                static_cast<double>(
                                    index) /
                                    static_cast<double>(
                                        std::max<std::size_t>(
                                            a.size() - 1U,
                                            1U))
                            ) -
                        1.0)
                );

        numerator +=
            taper *
            x *
            y;
        varianceA +=
            taper *
            x *
            x;
        varianceB +=
            taper *
            y *
            y;
    }

    const double denominator =
        std::sqrt(
            std::max(
                varianceA *
                    varianceB,
                0.0));

    if (
        denominator <=
        1e-12) {
        return
            -std::numeric_limits<double>::infinity();
    }

    support =
        count /
        static_cast<double>(
            std::max(
                a.size(),
                b.size()));

    return numerator /
        denominator;
}

double MotionTemporalAlignment::parabolicOffset(
    double left,
    double center,
    double right) noexcept {
    const double denominator =
        left -
        2.0 *
            center +
        right;

    if (
        !std::isfinite(
            denominator) ||
        denominator >=
            -1e-12) {
        return 0.0;
    }

    return std::clamp(
        0.5 *
            (
                left -
                right
            ) /
            denominator,
        -0.5,
        0.5);
}

MotionTemporalAlignmentResult
MotionTemporalAlignment::estimate(
    const std::vector<MotionTemporalSample>& reference,
    const std::vector<MotionTemporalSample>& remote,
    double sampleIntervalNanos,
    const MotionTemporalAlignmentOptions& options) const {
    MotionTemporalAlignmentResult result;

    if (
        reference.size() <
            std::max(
                std::size_t{3},
                options.minimumSamples) ||
        remote.size() <
            std::max(
                std::size_t{3},
                options.minimumSamples) ||
        !std::isfinite(sampleIntervalNanos) ||
        sampleIntervalNanos <= 0.0) {
        return result;
    }

    auto a =
        preprocess(
            reference,
            options.differentiateSignal);

    auto b =
        preprocess(
            remote,
            options.differentiateSignal);

    robustNormalize(a);
    robustNormalize(b);

    const int maximumLag =
        static_cast<int>(
            std::min(
                options.maximumLags,
                std::max(
                    a.size(),
                    b.size()) -
                    1U));

    if (maximumLag <= 0) {
        return result;
    }

    double best =
        -std::numeric_limits<double>::infinity();
    double secondBest =
        -std::numeric_limits<double>::infinity();
    int bestLag = 0;
    double bestSupport = 0.0;

    std::vector<double> correlation(
        static_cast<std::size_t>(
            maximumLag * 2 +
            1),
        -std::numeric_limits<double>::infinity());

    for (
        int lag = -maximumLag;
        lag <= maximumLag;
        ++lag) {
        double support = 0.0;
        const double value =
            correlationAtLag(
                a,
                b,
                lag,
                support);

        correlation[
            static_cast<std::size_t>(
                lag + maximumLag)
        ] = value;

        if (
            value >
            best) {
            secondBest =
                best;
            best =
                value;
            bestLag =
                lag;
            bestSupport =
                support;
        } else if (
            value >
                secondBest &&
            std::abs(
                lag -
                bestLag) >
                1) {
            secondBest =
                value;
        }
    }

    if (
        !std::isfinite(best)) {
        return result;
    }

    const auto at = [&correlation,maximumLag](int lag) {
        if (
            lag <
                -maximumLag ||
            lag >
                maximumLag) {
            return
                -std::numeric_limits<double>::infinity();
        }
        return correlation[
            static_cast<std::size_t>(
                lag + maximumLag)
        ];
    };

    const double left =
        at(
            bestLag - 1);
    const double right =
        at(
            bestLag + 1);

    const double fractional =
        parabolicOffset(
            left,
            best,
            right);

    result.coarseOffsetSamples =
        static_cast<double>(
            bestLag);
    result.offsetSamples =
        result.coarseOffsetSamples +
        fractional;
    result.offsetNanos =
        result.offsetSamples *
        sampleIntervalNanos;
    result.peakCorrelation =
        best;
    result.sideLobeCorrelation =
        std::isfinite(
            secondBest)
            ? secondBest
            : best;

    result.sharpness =
        clamp01(
            result.peakCorrelation -
            result.sideLobeCorrelation);

    const double normalizedPeak =
        clamp01(
            (
                result.peakCorrelation +
                1.0
            ) /
            2.0);

    const double separation =
        clamp01(
            (
                result.peakCorrelation -
                result.sideLobeCorrelation
            ) /
            0.35);

    const double supportScore =
        clamp01(
            bestSupport);

    result.confidence =
        clamp01(
            normalizedPeak * 0.48 +
            separation * 0.30 +
            supportScore * 0.22);

    result.supportRatio =
        bestSupport;
    result.evaluatedLags =
        correlation.size();

    result.valid =
        result.peakCorrelation >=
            options.minimumCorrelation &&
        result.confidence >=
            options.minimumConfidence &&
        result.supportRatio >=
            0.40 &&
        (
            std::isfinite(
                secondBest)
                ? (
                    result.peakCorrelation -
                    result.sideLobeCorrelation
                ) >=
                    options.minimumPeakSeparation
                : true
        );

    return result;
}

} // namespace nox::native::sync

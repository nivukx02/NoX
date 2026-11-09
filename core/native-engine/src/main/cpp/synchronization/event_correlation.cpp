#include "synchronization/event_correlation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "synchronization/subframe_alignment.h"

namespace nox::native::sync {
namespace {

struct Candidate {
    double correlation{
        -std::numeric_limits<double>::infinity()};
    int lag{0};
};

double overlapCorrelation(
    const std::vector<double>& a,
    const std::vector<double>& b,
    int lag) {
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

    if (
        count <
        3.0) {
        return
            -std::numeric_limits<double>::infinity();
    }

    const double meanA =
        sumA /
        count;
    const double meanB =
        sumB /
        count;

    double dot = 0.0;
    double aa = 0.0;
    double bb = 0.0;

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

        // Mild center weighting reduces the edge-dominated bias that
        // otherwise appears when only a short overlap remains.
        const double normalizedIndex =
            count <= 1.0
                ? 0.5
                : static_cast<double>(
                    index) /
                    static_cast<double>(
                        std::max(
                            std::size_t{1},
                            a.size() - 1U));

        const double taper =
            0.75 +
            0.25 *
                (
                    1.0 -
                    std::abs(
                        2.0 *
                            normalizedIndex -
                        1.0)
                );

        dot +=
            taper *
            x *
            y;
        aa +=
            taper *
            x *
            x;
        bb +=
            taper *
            y *
            y;
    }

    const double denominator =
        std::sqrt(
            std::max(
                aa * bb,
                0.0));

    if (
        denominator <=
        1e-12) {
        return
            -std::numeric_limits<double>::infinity();
    }

    return
        dot /
        denominator;
}

} // namespace

CorrelationPeak EventCorrelation::estimate(
    const std::vector<double>& a,
    const std::vector<double>& b) const {
    CorrelationPeak output;

    if (
        a.size() < 5U ||
        b.size() < 5U) {
        return output;
    }

    Candidate best;

    const int minimumLag =
        -static_cast<int>(
            b.size()) +
        1;

    const int maximumLag =
        static_cast<int>(
            a.size()) -
        1;

    std::vector<double> correlationValues;
    correlationValues.reserve(
        static_cast<std::size_t>(
            maximumLag -
            minimumLag +
            1));

    std::vector<int> lagValues;
    lagValues.reserve(
        correlationValues.capacity());

    for (
        int lag = minimumLag;
        lag <= maximumLag;
        ++lag) {
        const double value =
            overlapCorrelation(
                a,
                b,
                lag);

        correlationValues.push_back(
            value);
        lagValues.push_back(
            lag);

        if (
            value >
            best.correlation) {
            best.correlation =
                value;
            best.lag =
                lag;
        }
    }

    if (
        !std::isfinite(
            best.correlation)) {
        return output;
    }

    double left =
        best.correlation;
    double right =
        best.correlation;

    if (
        best.lag > minimumLag) {
        left =
            overlapCorrelation(
                a,
                b,
                best.lag - 1);
    }

    if (
        best.lag < maximumLag) {
        right =
            overlapCorrelation(
                a,
                b,
                best.lag + 1);
    }

    const auto subframe =
        SubframeAlignment::refinePeak(
            left,
            best.correlation,
            right);

    double secondBest =
        -std::numeric_limits<double>::infinity();

    for (
        std::size_t index = 0;
        index < correlationValues.size();
        ++index) {
        const int lag =
            lagValues[index];

        if (
            std::abs(
                lag -
                best.lag) <=
            1) {
            continue;
        }

        secondBest =
            std::max(
                secondBest,
                correlationValues[index]);
    }

    output.coarse_offset_samples =
        static_cast<double>(
            best.lag);

    output.offset_samples =
        static_cast<double>(
            best.lag) +
        subframe.offset_samples;

    output.peak_value =
        best.correlation;

    output.side_lobe_value =
        std::isfinite(
            secondBest)
            ? secondBest
            : best.correlation;

    output.sharpness =
        std::clamp(
            output.peak_value -
                output.side_lobe_value,
            0.0,
            1.0);

    const double normalizedPeak =
        std::clamp(
            (
                output.peak_value +
                1.0
            ) *
            0.5,
            0.0,
            1.0);

    const double peakSeparation =
        std::isfinite(
            secondBest)
            ? std::clamp(
                output.peak_value -
                    secondBest,
                0.0,
                1.0)
            : 1.0;

    output.confidence =
        std::clamp(
            normalizedPeak *
                0.65 +
            peakSeparation *
                0.25 +
            (
                subframe.valid
                    ? subframe.confidence
                    : 0.0
            ) *
                0.10,
            0.0,
            1.0);

    output.valid =
        output.peak_value >
            0.55 &&
        output.confidence >
            0.60 &&
        (
            !subframe.valid ||
            subframe.confidence >
                0.15
        );

    return output;
}

} // namespace nox::native::sync

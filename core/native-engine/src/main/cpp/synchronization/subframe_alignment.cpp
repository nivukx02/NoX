#include "synchronization/subframe_alignment.h"

#include <algorithm>
#include <cmath>

namespace nox::native::sync {

double SubframeAlignment::refineParabolicPeak(
    double left,
    double center,
    double right) noexcept {
    const double denominator =
        left -
        2.0 * center +
        right;

    if (
        !std::isfinite(
            denominator) ||
        denominator >=
            -1e-12 ||
        !std::isfinite(
            left) ||
        !std::isfinite(
            center) ||
        !std::isfinite(
            right)) {
        return 0.0;
    }

    const double offset =
        0.5 *
        (
            left -
            right
        ) /
        denominator;

    return std::clamp(
        offset,
        -0.5,
        0.5);
}

double SubframeAlignment::samplesToNanoseconds(
    double samples,
    double sampleIntervalNanoseconds) noexcept {
    if (
        !std::isfinite(samples) ||
        !std::isfinite(
            sampleIntervalNanoseconds) ||
        sampleIntervalNanoseconds <=
            0.0) {
        return 0.0;
    }

    return
        samples *
        sampleIntervalNanoseconds;
}

SubframePeak SubframeAlignment::refinePeak(
    double left,
    double center,
    double right) noexcept {
    SubframePeak result;

    if (
        !std::isfinite(left) ||
        !std::isfinite(center) ||
        !std::isfinite(right)) {
        return result;
    }

    const double denominator =
        left -
        2.0 * center +
        right;

    const double offset =
        refineParabolicPeak(
            left,
            center,
            right);

    const double curvature =
        -denominator;

    const double neighborAverage =
        0.5 *
        (
            left +
            right
        );

    const double peakContrast =
        center -
        neighborAverage;

    const double sharpness =
        std::abs(
            curvature) /
        std::max(
            std::abs(center) +
                std::abs(neighborAverage) +
                1e-9,
            1e-9);

    result.offset_samples =
        offset;
    result.curvature =
        curvature;

    const double curvatureEvidence =
        std::clamp(
            sharpness * 32.0,
            0.0,
            1.0);
    const double contrastEvidence =
        std::clamp(
            peakContrast /
                (
                    std::abs(center) +
                    1e-9
                ) *
                24.0,
            0.0,
            1.0);
    result.confidence =
        std::clamp(
            0.55 * curvatureEvidence +
            0.45 * contrastEvidence,
            0.0,
            1.0);

    result.valid =
        curvature >
            1e-12 &&
        center >=
            left &&
        center >=
            right &&
        result.confidence >
            0.05;

    return result;
}

} // namespace nox::native::sync

#include "synchronization/cross_view_trajectory_aligner.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nox::native::sync {

namespace {

} // namespace

CrossViewTrajectoryAligner::CrossViewTrajectoryAligner(
    TrajectoryAlignmentOptions options) noexcept
    : options_(options) {
    options_.maximumOffsetSeconds =
        std::clamp(
            options_.maximumOffsetSeconds,
            0.005,
            2.0);
    options_.sampleIntervalSeconds =
        std::clamp(
            options_.sampleIntervalSeconds,
            1e-4,
            0.5);
    options_.minimumCorrelation =
        std::clamp(
            options_.minimumCorrelation,
            0.0,
            0.99);
    options_.velocitySmoothing =
        std::clamp(
            options_.velocitySmoothing,
            0.0,
            1.0);
    options_.minimumSamples =
        std::clamp<std::size_t>(
            options_.minimumSamples,
            8U,
            100'000U);
}

double CrossViewTrajectoryAligner::norm(
    double x,
    double y,
    double z) noexcept {
    return std::hypot(
        std::hypot(
            x,
            y),
        z);
}

double CrossViewTrajectoryAligner::clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

std::vector<double>
CrossViewTrajectoryAligner::motionEnergy(
    const std::vector<TrajectorySample>& samples,
    double smoothing) {
    if (
        samples.size() < 2U) {
        return {};
    }

    std::vector<double> energy;
    energy.reserve(
        samples.size());

    double smoothed =
        0.0;

    for (
        std::size_t index = 1U;
        index < samples.size();
        ++index) {
        const auto& previous =
            samples[index - 1U];
        const auto& current =
            samples[index];

        if (
            current.timestampNanos <=
                previous.timestampNanos) {
            continue;
        }

        const double dt =
            static_cast<double>(
                current.timestampNanos -
                previous.timestampNanos) /
            1e9;

        if (
            dt <= 0.0 ||
            dt > 0.25) {
            continue;
        }

        const double dx =
            current.x -
            previous.x;
        const double dy =
            current.y -
            previous.y;
        const double dz =
            current.z -
            previous.z;

        const double speed =
            norm(
                dx,
                dy,
                dz) /
            dt;

        const double confidence =
            std::sqrt(
                clamp01(
                    previous.confidence) *
                clamp01(
                    current.confidence));

        const double weightedSpeed =
            speed *
            confidence;

        smoothed =
            smoothing *
                smoothed +
            (
                1.0 -
                smoothing
            ) *
                weightedSpeed;

        energy.push_back(
            smoothed);
    }

    return energy;
}

double CrossViewTrajectoryAligner::correlationAt(
    const std::vector<double>& a,
    const std::vector<double>& b,
    int lag) {
    double sumA = 0.0;
    double sumB = 0.0;
    std::size_t count = 0U;

    for (
        std::size_t index = 0;
        index < a.size();
        ++index) {
        const long long remoteIndex =
            static_cast<long long>(
                index) +
            static_cast<long long>(
                lag);

        if (
            remoteIndex < 0 ||
            remoteIndex >=
                static_cast<long long>(
                    b.size())) {
            continue;
        }

        sumA +=
            a[index];
        sumB +=
            b[
                static_cast<std::size_t>(
                    remoteIndex)
            ];
        ++count;
    }

    if (count < 8U) {
        return
            -std::numeric_limits<double>::infinity();
    }

    const double meanA =
        sumA /
        static_cast<double>(count);
    const double meanB =
        sumB /
        static_cast<double>(count);

    double dot = 0.0;
    double aa = 0.0;
    double bb = 0.0;

    for (
        std::size_t index = 0;
        index < a.size();
        ++index) {
        const long long remoteIndex =
            static_cast<long long>(
                index) +
            static_cast<long long>(
                lag);

        if (
            remoteIndex < 0 ||
            remoteIndex >=
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
                    remoteIndex)
            ] -
            meanB;

        dot +=
            x * y;
        aa +=
            x * x;
        bb +=
            y * y;
    }

    if (
        count <
        8U) {
        return
            -std::numeric_limits<double>::infinity();
    }

    const double denominator =
        std::sqrt(
            std::max(
                aa *
                    bb,
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

double CrossViewTrajectoryAligner::subSampleOffset(
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

TrajectoryAlignmentResult
CrossViewTrajectoryAligner::align(
    const std::vector<TrajectorySample>& reference,
    const std::vector<TrajectorySample>& remote) const {
    TrajectoryAlignmentResult result;

    const auto referenceEnergy =
        motionEnergy(
            reference,
            options_.velocitySmoothing);
    const auto remoteEnergy =
        motionEnergy(
            remote,
            options_.velocitySmoothing);

    if (
        referenceEnergy.size() <
            options_.minimumSamples ||
        remoteEnergy.size() <
            options_.minimumSamples) {
        return result;
    }

    const int maximumLag =
        static_cast<int>(
            std::floor(
                options_.maximumOffsetSeconds /
                options_.sampleIntervalSeconds));

    int bestLag =
        0;

    double bestCorrelation =
        -std::numeric_limits<double>::infinity();

    double secondBestCorrelation =
        -std::numeric_limits<double>::infinity();

    std::size_t bestOverlap = 0U;

    for (
        int lag = -maximumLag;
        lag <= maximumLag;
        ++lag) {
        const double correlation =
            correlationAt(
                referenceEnergy,
                remoteEnergy,
                lag);

        if (!std::isfinite(correlation)) {
            continue;
        }

        const std::size_t approximateOverlap =
            lag >= 0
                ? (
                    remoteEnergy.size() >
                        static_cast<std::size_t>(lag)
                        ? std::min(
                            referenceEnergy.size(),
                            remoteEnergy.size() -
                                static_cast<std::size_t>(lag))
                        : 0U
                )
                : (
                    referenceEnergy.size() >
                        static_cast<std::size_t>(-lag)
                        ? std::min(
                            referenceEnergy.size() -
                                static_cast<std::size_t>(-lag),
                            remoteEnergy.size())
                        : 0U
                );

        if (
            correlation >
            bestCorrelation) {
            if (
                std::isfinite(
                    bestCorrelation) &&
                std::abs(
                    lag -
                    bestLag) >
                1) {
                secondBestCorrelation =
                    bestCorrelation;
            }

            bestCorrelation =
                correlation;
            bestLag =
                lag;
            bestOverlap =
                approximateOverlap;
        } else if (
            std::abs(
                lag -
                bestLag) >
            1) {
            secondBestCorrelation =
                std::max(
                    secondBestCorrelation,
                    correlation);
        }
    }

    if (
        !std::isfinite(
            bestCorrelation)) {
        return result;
    }

    const double left =
        correlationAt(
            referenceEnergy,
            remoteEnergy,
            bestLag - 1);
    const double right =
        correlationAt(
            referenceEnergy,
            remoteEnergy,
            bestLag + 1);

    const double fractional =
        std::isfinite(left) &&
        std::isfinite(right)
            ? subSampleOffset(
                left,
                bestCorrelation,
                right)
            : 0.0;

    result.offsetSamples =
        -(
            static_cast<double>(
                bestLag) +
            fractional);

    result.offsetSeconds =
        result.offsetSamples *
        options_.sampleIntervalSeconds;

    result.correlation =
        bestCorrelation;

    const double curvature =
        std::max(
            0.0,
            -(
                left -
                2.0 *
                    bestCorrelation +
                right
            ));

    result.curvature =
        curvature;

    result.sideLobeCorrelation =
        std::isfinite(
            secondBestCorrelation)
            ? secondBestCorrelation
            : bestCorrelation;

    result.peakSeparation =
        std::clamp(
            bestCorrelation -
                result.sideLobeCorrelation,
            0.0,
            1.0);

    result.overlapRatio =
        std::min(
            referenceEnergy.size(),
            remoteEnergy.size()) == 0U
            ? 0.0
            : static_cast<double>(
                bestOverlap) /
                static_cast<double>(
                    std::min(
                        referenceEnergy.size(),
                        remoteEnergy.size()));

    const double sharpness =
        curvature /
        std::max(
            std::abs(
                bestCorrelation) +
                1e-9,
            1e-9);

    result.comparedSamples =
        std::min(
            referenceEnergy.size(),
            remoteEnergy.size());

    const double peakScore =
        clamp01(
            (
                bestCorrelation +
                1.0
            ) *
            0.5);

    const double overlapScore =
        clamp01(
            result.overlapRatio);

    const double separationScore =
        clamp01(
            result.peakSeparation *
            6.0);

    const double curvatureScore =
        clamp01(
            sharpness *
            2.0);

    result.confidence =
        clamp01(
            peakScore * 0.35 +
            overlapScore * 0.20 +
            separationScore * 0.25 +
            curvatureScore * 0.20);

    result.offsetUncertaintySeconds =
        options_.sampleIntervalSeconds /
        std::max(
            std::sqrt(
                std::max(
                    result.curvature,
                    1e-9)),
            1e-3);

    result.valid =
        bestCorrelation >=
            options_.minimumCorrelation &&
        result.confidence >=
            0.60 &&
        std::abs(
            result.offsetSeconds) <=
            options_.maximumOffsetSeconds;

    return result;
}

} // namespace nox::native::sync

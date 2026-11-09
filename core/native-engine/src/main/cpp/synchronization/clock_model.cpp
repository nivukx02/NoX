#include "synchronization/clock_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace nox::native::sync {
namespace {

struct Fit {
    double slope{1.0};
    double intercept{0.0};
    bool valid{false};
};

double median(
    std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }

    const auto middle =
        values.begin() +
        values.size() / 2U;

    std::nth_element(
        values.begin(),
        middle,
        values.end());

    const double upper =
        *middle;

    if (
        values.size() %
            2U !=
        0U) {
        return upper;
    }

    const auto lowerMiddle =
        middle - 1;

    std::nth_element(
        values.begin(),
        lowerMiddle,
        middle);

    return
        0.5 *
        (
            *lowerMiddle +
            upper
        );
}

double medianAbsoluteDeviation(
    const std::vector<double>& values,
    double center) {
    std::vector<double> deviations;
    deviations.reserve(
        values.size());

    for (const double value :
         values) {
        deviations.push_back(
            std::abs(
                value -
                center));
    }

    return median(
        std::move(
            deviations));
}

Fit weightedLeastSquares(
    const std::vector<std::uint64_t>& reference,
    const std::vector<std::uint64_t>& remote,
    const std::vector<double>& weights) {
    if (
        reference.size() !=
            remote.size() ||
        reference.size() !=
            weights.size() ||
        reference.size() < 2U) {
        return {};
    }

    const double origin =
        static_cast<double>(
            reference.front());

    double sumW = 0.0;
    double sumX = 0.0;
    double sumY = 0.0;
    double sumXX = 0.0;
    double sumXY = 0.0;

    for (
        std::size_t index = 0;
        index < reference.size();
        ++index) {
        const double weight =
            std::max(
                0.0,
                weights[index]);

        if (
            weight <= 0.0) {
            continue;
        }

        const double x =
            static_cast<double>(
                reference[index]) -
            origin;
        const double y =
            static_cast<double>(
                remote[index]);

        sumW +=
            weight;
        sumX +=
            weight * x;
        sumY +=
            weight * y;
        sumXX +=
            weight * x * x;
        sumXY +=
            weight * x * y;
    }

    const double denominator =
        sumW * sumXX -
        sumX * sumX;

    if (
        sumW <= 1e-9 ||
        std::abs(
            denominator) <
            1e-6) {
        return {};
    }

    const double slope =
        (
            sumW * sumXY -
            sumX * sumY
        ) /
        denominator;

    const double intercept =
        (
            sumY -
            slope * sumX
        ) /
        sumW;

    if (
        !std::isfinite(slope) ||
        !std::isfinite(intercept)) {
        return {};
    }

    return {
        slope,
        intercept,
        true,
    };
}

Fit robustTheilSenSeed(
    const std::vector<std::uint64_t>& reference,
    const std::vector<std::uint64_t>& remote) {
    if (
        reference.size() !=
            remote.size() ||
        reference.size() < 3U) {
        return {};
    }

    std::vector<double> slopes;
    slopes.reserve(
        std::min<std::size_t>(
            reference.size() *
                (reference.size() - 1U) /
                2U,
            4096U));

    for (
        std::size_t i = 0;
        i < reference.size() &&
            slopes.size() < 4096U;
        ++i) {
        for (
            std::size_t j = i + 1U;
            j < reference.size() &&
                slopes.size() < 4096U;
            ++j) {
            const double dx =
                static_cast<double>(
                    reference[j]) -
                static_cast<double>(
                    reference[i]);

            if (
                std::abs(dx) <
                1.0) {
                continue;
            }

            const double dy =
                static_cast<double>(
                    remote[j]) -
                static_cast<double>(
                    remote[i]);

            slopes.push_back(
                dy / dx);
        }
    }

    if (slopes.empty()) {
        return {};
    }

    return {
        median(
            std::move(
                slopes)),
        0.0,
        true,
    };
}

double percentile(
    std::vector<double> values,
    double p) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end());

    const double position =
        std::clamp(
            p,
            0.0,
            1.0) *
        static_cast<double>(
            values.size() - 1U);

    const auto lower =
        static_cast<std::size_t>(
            std::floor(position));

    const auto upper =
        std::min(
            values.size() - 1U,
            lower + 1U);

    const double alpha =
        position -
        static_cast<double>(
            lower);

    return
        values[lower] +
        (
            values[upper] -
            values[lower]
        ) *
        alpha;
}

} // namespace

ClockEstimate ClockModel::fit(
    const std::vector<std::uint64_t>& reference,
    const std::vector<std::uint64_t>& remote) const {
    ClockEstimate output;

    if (
        reference.size() !=
            remote.size() ||
        reference.size() < 3U) {
        return output;
    }

    for (
        std::size_t index = 1;
        index < reference.size();
        ++index) {
        if (
            reference[index] <=
            reference[index - 1U]) {
            return output;
        }
    }

    output.sample_count =
        reference.size();

    const auto seed =
        robustTheilSenSeed(
            reference,
            remote);

    if (!seed.valid) {
        return output;
    }

    const double origin =
        static_cast<double>(
            reference.front());

    double intercept =
        static_cast<double>(
            remote.front()) -
        seed.slope *
            (
                static_cast<double>(
                    reference.front()) -
                origin
            );

    double slope =
        seed.slope;

    std::vector<double> residuals(
        reference.size(),
        0.0);
    std::vector<double> weights(
        reference.size(),
        1.0);

    constexpr std::uint32_t
        kMaxIterations =
            12U;

    for (
        std::uint32_t iteration = 0;
        iteration < kMaxIterations;
        ++iteration) {
        for (
            std::size_t index = 0;
            index < reference.size();
            ++index) {
            const double x =
                static_cast<double>(
                    reference[index]) -
                origin;

            residuals[index] =
                static_cast<double>(
                    remote[index]) -
                (
                    intercept +
                    slope * x
                );
        }

        const double residualMedian =
            median(
                residuals);

        const double mad =
            medianAbsoluteDeviation(
                residuals,
                residualMedian);

        const double robustSigma =
            std::max(
                1.0,
                1.4826 *
                    mad);

        const double tuning =
            std::max(
                5.0,
                2.5 *
                    robustSigma);

        std::size_t inlierCount = 0U;

        for (
            std::size_t index = 0;
            index < residuals.size();
            ++index) {
            const double absolute =
                std::abs(
                    residuals[index] -
                    residualMedian);

            weights[index] =
                absolute <=
                    tuning
                    ? 1.0
                    : tuning /
                        std::max(
                            absolute,
                            1e-12);

            if (
                absolute <=
                tuning) {
                ++inlierCount;
            }
        }

        const auto fit =
            weightedLeastSquares(
                reference,
                remote,
                weights);

        if (!fit.valid) {
            break;
        }

        const double slopeDelta =
            std::abs(
                fit.slope -
                slope);

        const double interceptDelta =
            std::abs(
                fit.intercept -
                intercept);

        slope =
            fit.slope;
        intercept =
            fit.intercept;

        output.iterations =
            iteration + 1U;

        if (
            slopeDelta <
                1e-12 &&
            interceptDelta <
                1e-3) {
            break;
        }

        (void)inlierCount;
    }

    double weightedSquared =
        0.0;
    double sumWeight =
        0.0;
    std::size_t inliers =
        0U;

    residuals.clear();
    residuals.reserve(
        reference.size());

    for (
        std::size_t index = 0;
        index < reference.size();
        ++index) {
        const double x =
            static_cast<double>(
                reference[index]) -
            origin;

        const double residual =
            static_cast<double>(
                remote[index]) -
            (
                intercept +
                slope * x
            );

        residuals.push_back(
            std::abs(
                residual));
    }

    const double finalMedian =
        median(
            residuals);

    const double finalMad =
        medianAbsoluteDeviation(
            residuals,
            finalMedian);

    const double finalRobustScale =
        std::max(
            5.0,
            2.5 *
                std::max(
                    1.0,
                    1.4826 *
                        finalMad));

    for (
        const double absolute :
        residuals) {
        const double weight =
            absolute <=
                finalRobustScale
                ? 1.0
                : finalRobustScale /
                    std::max(
                        absolute,
                        1e-12);

        weightedSquared +=
            weight *
            absolute *
            absolute;
        sumWeight +=
            weight;

        if (
            absolute <=
            finalRobustScale) {
            ++inliers;
        }
    }

    if (
        sumWeight <=
        1e-9) {
        return output;
    }

    output.offset_nanos =
        intercept -
        slope * origin;

    output.drift_ppm =
        (
            slope -
            1.0
        ) *
        1e6;

    output.jitter_rms_nanos =
        std::sqrt(
            weightedSquared /
            sumWeight);

    std::vector<double> inlierResiduals;
    inlierResiduals.reserve(
        residuals.size());

    for (const double absolute :
         residuals) {
        const double weight =
            absolute <=
                finalRobustScale
                ? 1.0
                : finalRobustScale /
                    std::max(
                        absolute,
                        1e-12);

        if (
            weight >= 0.50) {
            inlierResiduals.push_back(
                absolute);
        }
    }

    output.residual_p95_nanos =
        percentile(
            inlierResiduals,
            0.95);

    output.residual_max_nanos =
        residuals.empty()
            ? 0.0
            : *std::max_element(
                residuals.begin(),
                residuals.end());

    output.support_ratio =
        static_cast<double>(
            inliers) /
        static_cast<double>(
            reference.size());

    output.outlier_ratio =
        1.0 -
        output.support_ratio;

    // Approximate parameter uncertainty from robust residual scale and
    // the temporal lever arm of the samples. This is intentionally a
    // conservative diagnostic, not a metrology certificate.
    double centeredXX = 0.0;
    double meanX = 0.0;

    for (
        const auto timestamp :
        reference) {
        meanX +=
            static_cast<double>(
                timestamp) -
            origin;
    }

    meanX /=
        static_cast<double>(
            reference.size());

    for (
        const auto timestamp :
        reference) {
        const double x =
            static_cast<double>(
                timestamp) -
            origin -
            meanX;
        centeredXX +=
            x * x;
    }

    const double degreesOfFreedom =
        std::max(
            1.0,
            static_cast<double>(
                reference.size()) -
                2.0);

    const double residualVariance =
        weightedSquared /
        std::max(
            degreesOfFreedom,
            1.0);

    output.slope_uncertainty_ppm =
        centeredXX <=
            1e-9
            ? std::numeric_limits<double>::infinity()
            : std::sqrt(
                residualVariance /
                centeredXX) *
                1e6;

    output.offset_uncertainty_nanos =
        std::sqrt(
            std::max(
                0.0,
                residualVariance *
                    (
                        1.0 /
                            static_cast<double>(
                                reference.size()) +
                        (
                            origin +
                            meanX
                        ) *
                        (
                            origin +
                            meanX
                        ) /
                            std::max(
                                centeredXX,
                                1e-9)
                    )));

    const double supportScore =
        std::clamp(
            output.support_ratio,
            0.0,
            1.0);

    const double jitterScore =
        std::exp(
            -output.jitter_rms_nanos /
            2'000'000.0);

    const double driftScore =
        std::exp(
            -std::abs(
                output.drift_ppm) /
            1000.0);

    const double residualScore =
        std::exp(
            -output.residual_p95_nanos /
            5'000'000.0);

    const double sampleScore =
        1.0 -
        std::exp(
            -static_cast<double>(
                reference.size()) /
            3.0);

    output.confidence =
        std::clamp(
            supportScore *
            jitterScore *
            driftScore *
            residualScore *
            sampleScore,
            0.0,
            1.0);

    output.valid =
        output.confidence >=
            0.50 &&
        std::isfinite(
            output.offset_nanos) &&
        std::isfinite(
            output.drift_ppm) &&
        std::isfinite(
            output.residual_p95_nanos) &&
        output.support_ratio >=
            0.70;

    return output;
}

} // namespace nox::native::sync

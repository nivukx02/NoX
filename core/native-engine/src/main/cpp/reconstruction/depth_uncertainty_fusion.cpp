#include "reconstruction/depth_uncertainty_fusion.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nox::native::reconstruction {

double DepthUncertaintyFusion::clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

double DepthUncertaintyFusion::median(
    std::vector<double> values) noexcept {
    if (values.empty()) return 0.0;

    std::sort(
        values.begin(),
        values.end());

    const auto middle =
        values.size() / 2U;

    if ((values.size() & 1U) != 0U) {
        return values[middle];
    }

    return 0.5 *
        (
            values[middle - 1U] +
            values[middle]
        );
}

double DepthUncertaintyFusion::percentile95(
    std::vector<double> values) noexcept {
    if (values.empty()) return 0.0;

    std::sort(
        values.begin(),
        values.end());

    const double index =
        0.95 *
        static_cast<double>(
            values.size() - 1U);

    const auto lower =
        static_cast<std::size_t>(
            std::floor(index));
    const auto upper =
        std::min(
            values.size() - 1U,
            lower + 1U);

    const double alpha =
        index -
        static_cast<double>(lower);

    return values[lower] +
        (
            values[upper] -
            values[lower]
        ) *
        alpha;
}

DepthEstimate DepthUncertaintyFusion::fuse(
    const DepthEstimate& a,
    const DepthEstimate& b) const noexcept {
    const double varianceA =
        std::max(
            a.variance,
            1e-9);
    const double varianceB =
        std::max(
            b.variance,
            1e-9);

    const double correlation =
        clamp01(
            0.5 *
                (
                    a.sensorCorrelation +
                    b.sensorCorrelation
                ));

    const double correlationInflation =
        1.0 +
        correlation;

    const double weightA =
        (
            clamp01(a.confidence) *
            clamp01(a.temporalConsistency) *
            clamp01(a.geometricConsistency)
        ) /
        varianceA;

    const double weightB =
        (
            clamp01(b.confidence) *
            clamp01(b.temporalConsistency) *
            clamp01(b.geometricConsistency)
        ) /
        varianceB;

    const double totalWeight =
        weightA +
        weightB;

    if (totalWeight <= 0.0) {
        return {};
    }

    const double depth =
        (
            weightA *
                (a.depth + a.biasMeters) +
            weightB *
                (b.depth + b.biasMeters)
        ) /
        totalWeight;

    const double disagreement =
        std::abs(
            (
                a.depth +
                a.biasMeters
            ) -
            (
                b.depth +
                b.biasMeters
            ));

    const double robustPenalty =
        clamp01(
            disagreement /
            0.20);

    const double variance =
        std::max(
            1e-9,
            1.0 /
                totalWeight *
                (
                    1.0 +
                    robustPenalty * 2.0 +
                    correlationInflation
                ));

    return {
        depth,
        variance,
        clamp01(
            (
                weightA *
                    clamp01(a.confidence) +
                weightB *
                    clamp01(b.confidence)
            ) /
            totalWeight *
            (
                1.0 -
                robustPenalty * 0.45
            )),
        0.0,
        clamp01(
            1.0 -
                robustPenalty),
        clamp01(1.0 - robustPenalty),
        correlation,
    };
}

DepthFusionResult DepthUncertaintyFusion::fuseRobust(
    const std::vector<DepthEstimate>& estimates,
    double huberThresholdSigma,
    double maximumDepthResidualMeters) const {
    DepthFusionResult result;

    if (estimates.empty()) {
        return result;
    }

    const double thresholdSigma =
        std::max(
            0.25,
            huberThresholdSigma);
    const double maximumResidual =
        std::max(
            1e-4,
            maximumDepthResidualMeters);

    std::vector<DepthEstimate> valid;
    valid.reserve(
        estimates.size());

    for (const auto& estimate :
         estimates) {
        if (
            !std::isfinite(estimate.depth) ||
            estimate.depth <= 0.0 ||
            !std::isfinite(estimate.variance) ||
            estimate.variance <= 0.0 ||
            !std::isfinite(estimate.confidence)) {
            ++result.rejectedSamples;
            continue;
        }

        if (
            clamp01(estimate.confidence) <= 0.01) {
            ++result.rejectedSamples;
            continue;
        }

        valid.push_back(
            estimate);
    }

    if (valid.empty()) {
        return result;
    }

    std::vector<double> centers;
    centers.reserve(valid.size());
    for (const auto& estimate :
         valid) {
        centers.push_back(
            estimate.depth +
            estimate.biasMeters);
    }

    double robustCenter =
        median(centers);

    for (std::uint32_t iteration = 0;
         iteration < 6U;
         ++iteration) {
        double numerator = 0.0;
        double denominator = 0.0;

        for (const auto& estimate :
             valid) {
            const double effectiveVariance =
                estimate.variance /
                std::max(
                    clamp01(
                        estimate.confidence *
                        estimate.temporalConsistency *
                        estimate.geometricConsistency),
                    0.05);

            const double residual =
                std::abs(
                    (
                        estimate.depth +
                        estimate.biasMeters
                    ) -
                    robustCenter);

            const double sigma =
                std::sqrt(
                    std::max(
                        effectiveVariance,
                        1e-9));

            const double normalized =
                residual /
                std::max(
                    sigma,
                    1e-6);

            const double huber =
                normalized <= thresholdSigma
                    ? 1.0
                    : thresholdSigma /
                        std::max(
                            normalized,
                            thresholdSigma);

            const double correlationPenalty =
                1.0 -
                0.65 *
                    clamp01(
                        estimate.sensorCorrelation);

            const double weight =
                (
                    clamp01(
                        estimate.confidence) *
                    clamp01(
                        estimate.temporalConsistency) *
                    clamp01(
                        estimate.geometricConsistency) *
                    correlationPenalty
                ) /
                std::max(
                    estimate.variance,
                    1e-9) *
                huber;

            numerator +=
                weight *
                (
                    estimate.depth +
                    estimate.biasMeters
                );
            denominator += weight;
        }

        if (denominator <= 0.0) break;

        const double next =
            numerator /
            denominator;

        if (
            std::abs(
                next -
                robustCenter) <
            1e-8) {
            robustCenter =
                next;
            break;
        }

        robustCenter =
            next;
    }

    double totalWeight = 0.0;
    double squaredWeightSum = 0.0;
    double weightedVariance = 0.0;
    double confidenceSum = 0.0;
    double correlationSum = 0.0;

    std::vector<double> residuals;
    residuals.reserve(
        valid.size());

    for (const auto& estimate :
         valid) {
        const double measured =
            estimate.depth +
            estimate.biasMeters;

        const double residual =
            std::abs(
                measured -
                robustCenter);

        const double effectiveVariance =
            estimate.variance /
            std::max(
                clamp01(
                    estimate.confidence *
                    estimate.temporalConsistency *
                    estimate.geometricConsistency),
                0.05);

        const double sigma =
            std::sqrt(
                std::max(
                    effectiveVariance,
                    1e-9));

        const double normalized =
            residual /
            std::max(
                sigma,
                1e-6);

        if (
            residual >
                maximumResidual &&
            normalized >
                thresholdSigma) {
            ++result.rejectedSamples;
            continue;
        }

        const double huber =
            normalized <= thresholdSigma
                ? 1.0
                : thresholdSigma /
                    std::max(
                        normalized,
                        thresholdSigma);

        const double correlationPenalty =
            1.0 -
            0.65 *
                clamp01(
                    estimate.sensorCorrelation);

        const double weight =
            (
                clamp01(
                    estimate.confidence) *
                clamp01(
                    estimate.temporalConsistency) *
                clamp01(
                    estimate.geometricConsistency) *
                correlationPenalty
            ) /
            std::max(
                estimate.variance,
                1e-9) *
            huber;

        if (weight <= 0.0) {
            ++result.rejectedSamples;
            continue;
        }

        totalWeight +=
            weight;
        squaredWeightSum +=
            weight *
            weight;

        weightedVariance +=
            weight *
            effectiveVariance;

        confidenceSum +=
            weight *
            clamp01(
                estimate.confidence);

        correlationSum +=
            weight *
            clamp01(
                estimate.sensorCorrelation);

        residuals.push_back(
            residual);

        ++result.acceptedSamples;
    }

    if (
        totalWeight <= 0.0 ||
        result.acceptedSamples == 0U) {
        return result;
    }

    result.depth =
        robustCenter;

    const double effectiveSamples =
        std::max(
            1.0,
            (
                totalWeight *
                totalWeight
            ) /
                std::max(
                    squaredWeightSum,
                    1e-12));

    const double correlationInflation =
        1.0 +
        clamp01(
            correlationSum /
            totalWeight);

    result.variance =
        std::max(
            1e-9,
            (
                weightedVariance /
                totalWeight
            ) /
            effectiveSamples *
            correlationInflation);

    result.confidence =
        clamp01(
            confidenceSum /
            totalWeight *
            (
                1.0 -
                clamp01(
                    static_cast<double>(
                        result.rejectedSamples) /
                        static_cast<double>(
                            estimates.size())) *
                    0.50
            ));

    result.p95ResidualMeters =
        percentile95(
            residuals);

    const double medianResidual =
        median(
            residuals);

    result.consensus =
        clamp01(
            1.0 -
                medianResidual /
                maximumResidual);

    result.effectiveSampleCount =
        squaredWeightSum <= 0.0
            ? 0.0
            : (
                totalWeight *
                totalWeight
            ) /
                squaredWeightSum;

    result.correlationPenalty =
        clamp01(
            correlationSum /
            totalWeight);

    result.valid =
        std::isfinite(result.depth) &&
        std::isfinite(result.variance) &&
        result.depth > 0.0 &&
        result.confidence >= 0.25 &&
        result.consensus >= 0.35;

    return result;
}

} // namespace nox::native::reconstruction

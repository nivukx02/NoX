#include "reconstruction/metric_scale_solver.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <vector>

namespace nox::native::scale {

double MetricScaleSolver::clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

double MetricScaleSolver::median(
    std::vector<double> values) noexcept {
    if (values.empty()) return 0.0;
    std::sort(
        values.begin(),
        values.end());
    const std::size_t middle =
        values.size() / 2U;
    if ((values.size() & 1U) != 0U) {
        return values[middle];
    }
    return
        0.5 *
        (
            values[middle - 1U] +
            values[middle]
        );
}

double MetricScaleSolver::percentile(
    std::vector<double> values,
    double probability) noexcept {
    if (values.empty()) return 0.0;
    std::sort(
        values.begin(),
        values.end());
    const double p =
        clamp01(probability);
    const double index =
        p *
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
    return
        values[lower] +
        (
            values[upper] -
            values[lower]
        ) *
        alpha;
}

Result MetricScaleSolver::solve(
    const std::vector<ScaleEvidence>& evidence) const {
    Result output;

    struct Candidate {
        const ScaleEvidence* evidence{nullptr};
        double baseWeight{0.0};
    };

    std::vector<Candidate> candidates;
    candidates.reserve(
        evidence.size());

    for (const auto& item :
         evidence) {
        if (
            item.source ==
                Source::RelativeDepthOnly ||
            !std::isfinite(item.scale) ||
            !std::isfinite(item.uncertainty) ||
            item.scale <= 0.0 ||
            item.uncertainty <= 0.0) {
            continue;
        }

        const double confidence =
            clamp01(
                item.confidence);
        const double stability =
            clamp01(
                item.temporalStability);
        const double traceability =
            clamp01(
                item.traceability);

        const double variance =
            item.uncertainty *
            item.uncertainty;

        const double precision =
            1.0 /
            std::max(
                variance,
                1e-12);

        const double weight =
            precision *
            (
                0.45 +
                0.30 * confidence +
                0.15 * stability +
                0.10 * traceability
            );

        if (
            !std::isfinite(weight) ||
            weight <= 0.0) {
            continue;
        }

        candidates.push_back({
            &item,
            weight,
        });
    }

    if (candidates.empty()) {
        return output;
    }

    double weightedScale = 0.0;
    double weightSum = 0.0;

    for (const auto& candidate :
         candidates) {
        weightedScale +=
            candidate.evidence->scale *
            candidate.baseWeight;
        weightSum +=
            candidate.baseWeight;
    }

    if (weightSum <= 0.0) {
        return output;
    }

    double estimate =
        weightedScale /
        weightSum;

    // Iteratively down-weight disagreement.
    for (std::uint32_t iteration = 0;
         iteration < 8U;
         ++iteration) {
        double numerator = 0.0;
        double denominator = 0.0;

        for (const auto& candidate :
             candidates) {
            const double residual =
                std::abs(
                    candidate.evidence->scale -
                    estimate);

            const double scale =
                std::max(
                    candidate.evidence->uncertainty *
                        2.0,
                    1e-5);

            const double normalized =
                residual /
                scale;

            const double robustWeight =
                normalized <= 1.0
                    ? 1.0
                    : 1.0 /
                        normalized;

            numerator +=
                candidate.evidence->scale *
                candidate.baseWeight *
                robustWeight;
            denominator +=
                candidate.baseWeight *
                robustWeight;
        }

        if (denominator <= 0.0) break;

        const double next =
            numerator /
            denominator;

        if (
            std::abs(next - estimate) <
            std::max(
                estimate * 1e-8,
                1e-9)) {
            estimate = next;
            break;
        }
        estimate = next;
    }

    output.scale = estimate;

    std::vector<double> normalizedResiduals;
    std::vector<double> absoluteResiduals;
    normalizedResiduals.reserve(
        candidates.size());
    absoluteResiduals.reserve(
        candidates.size());

    double robustWeightSum = 0.0;
    double robustPrecisionSum = 0.0;
    double consensusWeight = 0.0;
    double traceabilityWeighted = 0.0;

    std::map<Source, ScaleSourceSummary> sourceMap;

    for (const auto& candidate :
         candidates) {
        const auto& item =
            *candidate.evidence;

        const double residual =
            std::abs(
                item.scale -
                output.scale);

        const double uncertainty =
            std::max(
                item.uncertainty,
                1e-6);

        const double normalized =
            residual /
            uncertainty;

        const double robustWeight =
            normalized <= 1.0
                ? 1.0
                : 1.0 /
                    normalized;

        absoluteResiduals.push_back(
            residual);
        normalizedResiduals.push_back(
            normalized);

        robustWeightSum +=
            candidate.baseWeight *
            robustWeight;
        robustPrecisionSum +=
            (
                candidate.baseWeight *
                robustWeight
            ) /
            (
                uncertainty *
                uncertainty
            );

        const double consensusContribution =
            clamp01(
                std::exp(
                    -0.5 *
                    normalized *
                    normalized));
        consensusWeight +=
            candidate.baseWeight *
            consensusContribution;

        traceabilityWeighted +=
            candidate.baseWeight *
            clamp01(
                item.traceability);

        auto& summary =
            sourceMap[item.source];
        summary.source =
            item.source;
        summary.supportingEvidence += 1U;
        summary.weightedSupport +=
            candidate.baseWeight;
    }

    output.uncertainty =
        robustPrecisionSum > 0.0
            ? std::sqrt(
                1.0 /
                robustPrecisionSum)
            : 1.0;

    output.medianDisagreement =
        median(
            normalizedResiduals);

    output.p95Disagreement =
        percentile(
            normalizedResiduals,
            0.95);

    output.consensusSupport =
        robustWeightSum > 0.0
            ? clamp01(
                consensusWeight /
                robustWeightSum)
            : 0.0;

    output.traceability =
        robustWeightSum > 0.0
            ? clamp01(
                traceabilityWeighted /
                robustWeightSum)
            : 0.0;

    for (const auto& [
             source,
             summary] :
         sourceMap) {
        output.sourceSummaries.push_back(
            summary);
    }

    output.sourceDiversity =
        clamp01(
            static_cast<double>(
                output.sourceSummaries.size()) /
            3.0);

    const double precisionConfidence =
        clamp01(
            std::exp(
                -output.p95Disagreement /
                4.0));

    output.confidence =
        clamp01(
            output.consensusSupport * 0.38 +
            precisionConfidence * 0.24 +
            output.sourceDiversity * 0.12 +
            output.traceability * 0.16 +
            clamp01(
                1.0 /
                (
                    1.0 +
                    output.uncertainty
                )
            ) * 0.10);

    Source strongestSource =
        Source::RelativeDepthOnly;
    double strongestWeight = 0.0;
    for (const auto& entry :
         sourceMap) {
        if (
            entry.second.weightedSupport >
            strongestWeight) {
            strongestWeight =
                entry.second.weightedSupport;
            strongestSource =
                entry.first;
        }
    }
    output.source =
        strongestSource;

    output.metricReady =
        candidates.size() >= 2U &&
        output.consensusSupport >= 0.70 &&
        output.p95Disagreement <= 2.5 &&
        output.confidence >= 0.72 &&
        output.traceability >= 0.20 &&
        output.uncertainty <
            std::max(
                output.scale * 0.05,
                0.0005);

    output.valid =
        output.metricReady &&
        std::isfinite(output.scale) &&
        output.scale > 0.0;

    return output;
}

} // namespace nox::native::scale

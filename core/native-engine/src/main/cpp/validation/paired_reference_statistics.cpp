#include "validation/paired_reference_statistics.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace nox::native::validation {

namespace {

double percentile(
    std::vector<double> values,
    double probability) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::sort(values.begin(), values.end());

    const double position =
        std::clamp(probability, 0.0, 1.0) *
        static_cast<double>(values.size() - 1);

    const auto lower =
        static_cast<std::size_t>(position);
    const auto upper =
        std::min(
            values.size() - 1,
            lower + 1);
    const double fraction =
        position -
        static_cast<double>(lower);

    return values[lower] * (1.0 - fraction) +
           values[upper] * fraction;
}

std::uint64_t splitMix64(
    std::uint64_t& state) noexcept {
    std::uint64_t z =
        (state += 0x9e3779b97f4a7c15ULL);
    z =
        (z ^ (z >> 30U)) *
        0xbf58476d1ce4e5b9ULL;
    z =
        (z ^ (z >> 27U)) *
        0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

} // namespace

PairedReferenceStatisticsEngine::PairedReferenceStatisticsEngine() noexcept
    : PairedReferenceStatisticsEngine(Config{}) {}

PairedReferenceStatisticsEngine::PairedReferenceStatisticsEngine(
    Config config) noexcept
    : config_(config) {
    config_.bootstrapIterations =
        std::clamp<std::uint32_t>(
            config_.bootstrapIterations,
            128U,
            16'384U);
    config_.tieFraction =
        std::clamp(
            config_.tieFraction,
            0.0,
            1.0);
    config_.minimumSamples =
        std::max<std::uint64_t>(
            1ULL,
            config_.minimumSamples);
    config_.minimumConfidenceForDecision =
        std::clamp(
            config_.minimumConfidenceForDecision,
            0.50,
            0.999);
}

PairedReferenceStatistics
PairedReferenceStatisticsEngine::compare(
    const std::vector<PairedReferenceSample>& samples,
    std::uint64_t seed) const noexcept {
    PairedReferenceStatistics result{};
    result.sampleCount =
        static_cast<std::uint64_t>(samples.size());

    if (samples.empty()) {
        return result;
    }

    std::vector<double> noxErrors;
    std::vector<double> referenceErrors;
    std::vector<double> differences;
    noxErrors.reserve(samples.size());
    referenceErrors.reserve(samples.size());
    differences.reserve(samples.size());

    std::uint64_t betterCount = 0ULL;
    std::uint64_t comparableCount = 0ULL;

    for (const auto& sample : samples) {
        if (
            !std::isfinite(sample.noxErrorMeters) ||
            !std::isfinite(sample.referenceErrorMeters) ||
            sample.noxErrorMeters < 0.0 ||
            sample.referenceErrorMeters < 0.0
        ) {
            continue;
        }

        noxErrors.push_back(sample.noxErrorMeters);
        referenceErrors.push_back(
            sample.referenceErrorMeters);

        const double difference =
            sample.noxErrorMeters -
            sample.referenceErrorMeters;
        differences.push_back(difference);

        ++comparableCount;
        if (difference < 0.0) {
            ++betterCount;
        }
    }

    if (differences.empty()) {
        return result;
    }

    result.meanPairedDifferenceMeters =
        0.0;
    for (const double difference : differences) {
        result.meanPairedDifferenceMeters +=
            difference;
    }
    result.meanPairedDifferenceMeters /=
        static_cast<double>(differences.size());

    result.medianPairedDifferenceMeters =
        percentile(
            differences,
            0.50);

    result.p95NoXErrorMeters =
        percentile(noxErrors, 0.95);
    result.p95ReferenceErrorMeters =
        percentile(referenceErrors, 0.95);
    result.p99NoXErrorMeters =
        percentile(noxErrors, 0.99);
    result.p99ReferenceErrorMeters =
        percentile(referenceErrors, 0.99);

    result.probabilityNoXBetter =
        static_cast<double>(betterCount) /
        static_cast<double>(
            std::max<std::uint64_t>(
                1ULL,
                comparableCount));

    /*
     * Deterministic non-parametric bootstrap over paired samples. This preserves the matched
     * nature of the comparison while making the result reproducible for a stored qualification
     * artifact.
     */
    std::vector<double> bootstrapMeans;
    bootstrapMeans.reserve(
        config_.bootstrapIterations);

    const std::size_t count =
        differences.size();

    std::uint64_t state =
        seed == 0ULL
            ? 0x6a09e667f3bcc909ULL
            : seed;

    for (
        std::uint32_t iteration = 0U;
        iteration < config_.bootstrapIterations;
        ++iteration) {
        double sum = 0.0;

        for (
            std::size_t sampleIndex = 0U;
            sampleIndex < count;
            ++sampleIndex) {
            const std::uint64_t randomValue =
                splitMix64(state);

            const std::size_t index =
                static_cast<std::size_t>(
                    randomValue %
                    static_cast<std::uint64_t>(
                        count));

            sum += differences[index];
        }

        bootstrapMeans.push_back(
            sum /
            static_cast<double>(count));
    }

    result.confidenceIntervalLowerMeters =
        percentile(
            bootstrapMeans,
            0.025);
    result.confidenceIntervalUpperMeters =
        percentile(
            bootstrapMeans,
            0.975);

    const double signConsistency =
        std::max(
            result.probabilityNoXBetter,
            1.0 -
                result.probabilityNoXBetter);

    const double sampleConfidence =
        std::clamp(
            static_cast<double>(
                comparableCount) /
                static_cast<double>(
                    config_.minimumSamples),
            0.0,
            1.0);

    result.repeatabilityConfidence =
        0.55 * sampleConfidence +
        0.45 * signConsistency;

    if (
        comparableCount <
            config_.minimumSamples ||
        !std::isfinite(
            result.confidenceIntervalLowerMeters) ||
        !std::isfinite(
            result.confidenceIntervalUpperMeters)
    ) {
        result.winner =
            PairedReferenceWinner::INDETERMINATE;
        return result;
    }

    const double p95Reference =
        std::max(
            result.p95ReferenceErrorMeters,
            1e-12);

    const bool p95Tie =
        std::abs(
            result.p95NoXErrorMeters -
            result.p95ReferenceErrorMeters) <=
        p95Reference *
            config_.tieFraction;

    const bool ciStrictlyNegative =
        result.confidenceIntervalUpperMeters <
        0.0;

    const bool ciStrictlyPositive =
        result.confidenceIntervalLowerMeters >
        0.0;

    const bool confidenceReady =
        result.repeatabilityConfidence >=
        config_.minimumConfidenceForDecision;

    if (
        p95Tie &&
        !ciStrictlyNegative &&
        !ciStrictlyPositive
    ) {
        result.winner =
            PairedReferenceWinner::TIE;
    } else if (
        ciStrictlyNegative &&
        result.probabilityNoXBetter >=
            config_.minimumConfidenceForDecision &&
        confidenceReady
    ) {
        result.winner =
            PairedReferenceWinner::NOX;
    } else if (
        ciStrictlyPositive &&
        result.probabilityNoXBetter <=
            1.0 -
                config_.minimumConfidenceForDecision &&
        confidenceReady
    ) {
        result.winner =
            PairedReferenceWinner::REFERENCE;
    } else {
        result.winner =
            PairedReferenceWinner::INDETERMINATE;
    }

    return result;
}

} // namespace nox::native::validation

#include "validation/reference_system_comparison.h"

#include <algorithm>
#include <cmath>

namespace nox::native::validation {

ReferenceSystemComparisonEngine::ReferenceSystemComparisonEngine() noexcept
    : ReferenceSystemComparisonEngine(Config{}) {}

ReferenceSystemComparisonEngine::ReferenceSystemComparisonEngine(
    Config config) noexcept
    : config_(config) {
    config_.p95TieFraction =
        std::clamp(config_.p95TieFraction, 0.0, 1.0);
    config_.p99TieFraction =
        std::clamp(config_.p99TieFraction, 0.0, 1.0);
    config_.maximumDropoutDifference =
        std::max(0.0, config_.maximumDropoutDifference);
    config_.maximumLagDifferenceMillis =
        std::max(0.0, config_.maximumLagDifferenceMillis);
    config_.minimumIndependentSessions =
        std::max<std::uint32_t>(
            1U,
            config_.minimumIndependentSessions);
    config_.minimumMatchedSamples =
        std::max<std::uint64_t>(
            1ULL,
            config_.minimumMatchedSamples);
    config_.maximumReferenceUncertaintyMeters =
        std::max(
            0.0,
            config_.maximumReferenceUncertaintyMeters);
}

ReferenceSystemComparison
ReferenceSystemComparisonEngine::compare(
    const MotionAccuracyReport& nox,
    const MotionAccuracyReport& reference,
    std::uint32_t noxIndependentSessions,
    std::uint32_t referenceIndependentSessions,
    std::uint64_t matchedSamplesPerSystem,
    double referenceUncertaintyMeters) const noexcept {
    ReferenceSystemComparison result{};

    result.noxP95Meters = nox.p95ErrorMeters;
    result.referenceP95Meters = reference.p95ErrorMeters;
    result.noxP99Meters = nox.p99ErrorMeters;
    result.referenceP99Meters = reference.p99ErrorMeters;
    result.noxDropoutRate = nox.dropoutRate;
    result.referenceDropoutRate = reference.dropoutRate;
    result.noxTemporalLagP95Millis =
        nox.temporalOffsetP95Millis;
    result.referenceTemporalLagP95Millis =
        reference.temporalOffsetP95Millis;
    result.noxJitterRmsMeters = nox.jitterRmsMeters;
    result.referenceJitterRmsMeters =
        reference.jitterRmsMeters;
    result.noxIndependentSessions =
        noxIndependentSessions;
    result.referenceIndependentSessions =
        referenceIndependentSessions;

    result.statisticallyComparable =
        noxIndependentSessions >=
            config_.minimumIndependentSessions &&
        referenceIndependentSessions >=
            config_.minimumIndependentSessions &&
        matchedSamplesPerSystem >=
            config_.minimumMatchedSamples &&
        std::isfinite(nox.p95ErrorMeters) &&
        std::isfinite(reference.p95ErrorMeters) &&
        std::isfinite(nox.p99ErrorMeters) &&
        std::isfinite(reference.p99ErrorMeters);

    /*
     * The reference system's own uncertainty is included in its adjusted error. This prevents a
     * measurement rig from being treated as exact ground truth simply because it is labeled "reference".
     */
    const double referenceUncertainty =
        std::min(
            config_.maximumReferenceUncertaintyMeters,
            std::max(
                0.0,
                std::isfinite(referenceUncertaintyMeters)
                    ? referenceUncertaintyMeters
                    : config_.maximumReferenceUncertaintyMeters));

    result.adjustedNoxP95Meters =
        std::hypot(
            nox.p95ErrorMeters,
            referenceUncertainty);
    result.adjustedReferenceP95Meters =
        std::hypot(
            reference.p95ErrorMeters,
            referenceUncertainty);
    result.adjustedNoxP99Meters =
        std::hypot(
            nox.p99ErrorMeters,
            referenceUncertainty);
    result.adjustedReferenceP99Meters =
        std::hypot(
            reference.p99ErrorMeters,
            referenceUncertainty);

    const double referenceP95 =
        std::max(
            result.adjustedReferenceP95Meters,
            1e-12);
    const double referenceP99 =
        std::max(
            result.adjustedReferenceP99Meters,
            1e-12);

    result.relativeP95Advantage =
        (
            result.adjustedReferenceP95Meters -
            result.adjustedNoxP95Meters
        ) / referenceP95;

    result.relativeP99Advantage =
        (
            result.adjustedReferenceP99Meters -
            result.adjustedNoxP99Meters
        ) / referenceP99;

    const double sampleConfidence =
        std::clamp(
            static_cast<double>(matchedSamplesPerSystem) /
                static_cast<double>(
                    config_.minimumMatchedSamples),
            0.0,
            1.0);
    const double sessionConfidence =
        std::clamp(
            static_cast<double>(
                std::min(
                    noxIndependentSessions,
                    referenceIndependentSessions)) /
                static_cast<double>(
                    config_.minimumIndependentSessions),
            0.0,
            1.0);

    result.repeatabilityConfidence =
        0.60 * sampleConfidence +
        0.40 * sessionConfidence;

    if (!result.statisticallyComparable) {
        result.winner =
            ReferenceWinner::INDETERMINATE;
        result.absoluteQualificationEvidenceAvailable = false;
        return result;
    }

    const double p95Difference =
        std::abs(
            result.adjustedNoxP95Meters -
            result.adjustedReferenceP95Meters);
    const double p99Difference =
        std::abs(
            result.adjustedNoxP99Meters -
            result.adjustedReferenceP99Meters);

    const bool p95Tie =
        p95Difference <=
        result.adjustedReferenceP95Meters *
            config_.p95TieFraction;

    const bool p99Tie =
        p99Difference <=
        result.adjustedReferenceP99Meters *
            config_.p99TieFraction;

    const bool operationalTie =
        std::abs(
            nox.dropoutRate -
            reference.dropoutRate) <=
            config_.maximumDropoutDifference &&
        std::abs(
            nox.temporalOffsetP95Millis -
            reference.temporalOffsetP95Millis) <=
            config_.maximumLagDifferenceMillis;

    if (p95Tie && p99Tie && operationalTie) {
        result.winner = ReferenceWinner::TIE;
    } else if (
        result.relativeP95Advantage > 0.0 &&
        result.relativeP99Advantage >=
            -config_.p99TieFraction) {
        result.winner = ReferenceWinner::NOX;
    } else if (
        result.relativeP95Advantage < 0.0 &&
        result.relativeP99Advantage <=
            config_.p99TieFraction) {
        result.winner = ReferenceWinner::REFERENCE;
    } else {
        result.winner =
            ReferenceWinner::INDETERMINATE;
    }

    result.absoluteQualificationEvidenceAvailable =
        result.repeatabilityConfidence >= 0.90 &&
        reference.temporalOffsetP95Millis <=
            config_.maximumLagDifferenceMillis * 2.0;

    return result;
}


ReferenceSystemComparison
ReferenceSystemComparisonEngine::comparePaired(
    const MotionAccuracyReport& nox,
    const MotionAccuracyReport& reference,
    std::uint32_t noxIndependentSessions,
    std::uint32_t referenceIndependentSessions,
    std::uint64_t matchedSamplesPerSystem,
    double referenceUncertaintyMeters,
    const std::vector<PairedReferenceSample>& pairedSamples,
    std::uint64_t bootstrapSeed) const noexcept {
    auto result =
        compare(
            nox,
            reference,
            noxIndependentSessions,
            referenceIndependentSessions,
            matchedSamplesPerSystem,
            referenceUncertaintyMeters);

    const auto paired =
        PairedReferenceStatisticsEngine{}
            .compare(
                pairedSamples,
                bootstrapSeed);

    result.pairedComparisonAvailable =
        paired.sampleCount > 0 &&
        paired.winner !=
            PairedReferenceWinner::INDETERMINATE;
    result.pairedWinner =
        paired.winner;
    result.pairedMeanDifferenceMeters =
        paired.meanPairedDifferenceMeters;
    result.pairedConfidenceIntervalLowerMeters =
        paired.confidenceIntervalLowerMeters;
    result.pairedConfidenceIntervalUpperMeters =
        paired.confidenceIntervalUpperMeters;
    result.pairedProbabilityNoXBetter =
        paired.probabilityNoXBetter;

    if (
        result.pairedComparisonAvailable &&
        result.statisticallyComparable
    ) {
        const bool aggregateAgrees =
            (
                paired.winner ==
                    PairedReferenceWinner::NOX &&
                result.winner ==
                    ReferenceWinner::NOX
            ) ||
            (
                paired.winner ==
                    PairedReferenceWinner::REFERENCE &&
                result.winner ==
                    ReferenceWinner::REFERENCE
            ) ||
            (
                paired.winner ==
                    PairedReferenceWinner::TIE &&
                result.winner ==
                    ReferenceWinner::TIE
            );

        if (!aggregateAgrees) {
            result.winner =
                ReferenceWinner::INDETERMINATE;
        }

        result.absoluteQualificationEvidenceAvailable =
            result.absoluteQualificationEvidenceAvailable &&
            aggregateAgrees;
    }

    return result;
}

} // namespace nox::native::validation

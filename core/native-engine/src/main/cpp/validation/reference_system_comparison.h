#pragma once

#include <cstdint>

#include "validation/motion_accuracy_evaluator.h"
#include "validation/paired_reference_statistics.h"

namespace nox::native::validation {

enum class ReferenceWinner {
    NOX,
    REFERENCE,
    TIE,
    INDETERMINATE,
};

struct ReferenceSystemComparison {
    ReferenceWinner winner{ReferenceWinner::INDETERMINATE};
    double noxP95Meters{0.0};
    double referenceP95Meters{0.0};
    double noxP99Meters{0.0};
    double referenceP99Meters{0.0};
    double noxDropoutRate{1.0};
    double referenceDropoutRate{1.0};
    double noxTemporalLagP95Millis{0.0};
    double referenceTemporalLagP95Millis{0.0};
    double noxJitterRmsMeters{0.0};
    double referenceJitterRmsMeters{0.0};
    double adjustedNoxP95Meters{0.0};
    double adjustedReferenceP95Meters{0.0};
    double adjustedNoxP99Meters{0.0};
    double adjustedReferenceP99Meters{0.0};
    double relativeP95Advantage{0.0};
    double relativeP99Advantage{0.0};
    double repeatabilityConfidence{0.0};
    std::uint32_t noxIndependentSessions{0};
    std::uint32_t referenceIndependentSessions{0};
    bool absoluteQualificationEvidenceAvailable{false};
    bool statisticallyComparable{false};
    bool pairedComparisonAvailable{false};
    PairedReferenceWinner pairedWinner{
        PairedReferenceWinner::INDETERMINATE};
    double pairedMeanDifferenceMeters{0.0};
    double pairedConfidenceIntervalLowerMeters{0.0};
    double pairedConfidenceIntervalUpperMeters{0.0};
    double pairedProbabilityNoXBetter{0.0};
};

class ReferenceSystemComparisonEngine final {
public:
    struct Config {
        double p95TieFraction{0.05};
        double p99TieFraction{0.08};
        double maximumDropoutDifference{0.02};
        double maximumLagDifferenceMillis{2.0};
        std::uint32_t minimumIndependentSessions{3};
        std::uint64_t minimumMatchedSamples{500};
        double maximumReferenceUncertaintyMeters{0.005};
    };

    ReferenceSystemComparisonEngine() noexcept;
    explicit ReferenceSystemComparisonEngine(Config config) noexcept;

    [[nodiscard]] ReferenceSystemComparison compare(
        const MotionAccuracyReport& nox,
        const MotionAccuracyReport& reference,
        std::uint32_t noxIndependentSessions,
        std::uint32_t referenceIndependentSessions,
        std::uint64_t matchedSamplesPerSystem,
        double referenceUncertaintyMeters) const noexcept;

    [[nodiscard]] ReferenceSystemComparison comparePaired(
        const MotionAccuracyReport& nox,
        const MotionAccuracyReport& reference,
        std::uint32_t noxIndependentSessions,
        std::uint32_t referenceIndependentSessions,
        std::uint64_t matchedSamplesPerSystem,
        double referenceUncertaintyMeters,
        const std::vector<PairedReferenceSample>& pairedSamples,
        std::uint64_t bootstrapSeed) const noexcept;

private:
    Config config_{};
};

} // namespace nox::native::validation

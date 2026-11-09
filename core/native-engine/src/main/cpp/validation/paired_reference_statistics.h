#pragma once

#include <cstdint>
#include <vector>

namespace nox::native::validation {

enum class PairedReferenceWinner {
    NOX,
    REFERENCE,
    TIE,
    INDETERMINATE,
};

struct PairedReferenceSample {
    double noxErrorMeters{0.0};
    double referenceErrorMeters{0.0};
};

struct PairedReferenceStatistics {
    PairedReferenceWinner winner{
        PairedReferenceWinner::INDETERMINATE};
    std::uint64_t sampleCount{0};
    double meanPairedDifferenceMeters{0.0};
    double medianPairedDifferenceMeters{0.0};
    double confidenceIntervalLowerMeters{0.0};
    double confidenceIntervalUpperMeters{0.0};
    double p95NoXErrorMeters{0.0};
    double p95ReferenceErrorMeters{0.0};
    double p99NoXErrorMeters{0.0};
    double p99ReferenceErrorMeters{0.0};
    double probabilityNoXBetter{0.0};
    double repeatabilityConfidence{0.0};
};

class PairedReferenceStatisticsEngine final {
public:
    struct Config {
        std::uint32_t bootstrapIterations{2048};
        double tieFraction{0.05};
        std::uint64_t minimumSamples{500};
        double minimumConfidenceForDecision{0.95};
    };

    PairedReferenceStatisticsEngine() noexcept;
    explicit PairedReferenceStatisticsEngine(
        Config config) noexcept;

    [[nodiscard]] PairedReferenceStatistics compare(
        const std::vector<PairedReferenceSample>& samples,
        std::uint64_t seed) const noexcept;

private:
    Config config_{};
};

} // namespace nox::native::validation

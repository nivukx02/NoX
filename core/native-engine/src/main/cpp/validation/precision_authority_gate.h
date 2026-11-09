#pragma once

#include <string>

#include "validation/metrology_precision_qualification.h"

namespace nox::native::validation {

enum class PrecisionAuthorityState {
    blocked = 0,
    evidenceOnly = 1,
    qualified = 2,
};

struct PrecisionAuthorityInput {
    QualificationReport metrology{};
    double calibrationConfidence{0.0};
    double synchronizationConfidence{0.0};
    double reconstructionUncertaintyP95Meters{1.0};
    double reconstructionConfidence{0.0};
    double global4DConsistency{0.0};
    double rigStability{0.0};
    bool referenceBenchmarkArmed{false};
    bool runtimeHealthy{true};
    bool thermalHealthy{true};
    bool memoryHealthy{true};
};

struct PrecisionAuthorityReport {
    PrecisionAuthorityState state{
        PrecisionAuthorityState::blocked};
    double evidenceScore{0.0};
    double bottleneckScore{0.0};
    bool measuredMetrologyQualified{false};
    bool calibrationQualified{false};
    bool synchronizationQualified{false};
    bool reconstructionQualified{false};
    bool rigQualified{false};
    std::string reason;
};

class PrecisionAuthorityGate final {
public:
    [[nodiscard]] PrecisionAuthorityReport evaluate(
        const PrecisionAuthorityInput&) const noexcept;

private:
    static double clamp01(
        double) noexcept;
};

} // namespace nox::native::validation

#pragma once

#include <string>
#include <vector>

namespace nox::native::validation {

struct PrecisionCaptureAdmissionInput {
    double requestedP95Millimeters{1.0};
    double measuredP95Millimeters{0.0};
    double measuredMaxMillimeters{0.0};
    bool metrologyAvailable{false};
    bool metrologyTraceable{false};

    double calibrationConfidence{0.0};
    double synchronizationConfidence{0.0};
    double viewDiversity{0.0};
    double crossViewAgreement{0.0};
    double metricScaleConfidence{0.0};
    double reconstructionUncertaintyMeters{1.0};
    double temporalStability{0.0};

    bool calibrationStable{false};
    bool rigStable{false};
    bool geometryConditioned{false};

    double thermalHeadroom{0.0};
    double memoryHeadroom{0.0};
};

struct PrecisionCaptureAdmissionResult {
    bool admitted{false};
    bool referenceGradeEligible{false};
    bool authoritative{false};
    double score{0.0};
    double precisionMarginMillimeters{0.0};
    std::vector<std::string> blockers;
    std::vector<std::string> warnings;
};

class PrecisionCaptureAdmissionGate final {
public:
    [[nodiscard]] PrecisionCaptureAdmissionResult evaluate(
        const PrecisionCaptureAdmissionInput&) const noexcept;
};

} // namespace nox::native::validation

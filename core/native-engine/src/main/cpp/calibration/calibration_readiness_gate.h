#pragma once
#include <string>
#include <vector>
#include "calibration/calibration_health.h"
#include "calibration/rig_observability_analyzer.h"

namespace nox::native::calibration {

struct CalibrationDriftMetrics {
    bool authoritative{false};
    bool recalibrationRequired{false};
    bool recalibrationRecommended{false};
    double stabilityScore{0.0};
    double residualSlopePxPerMinute{0.0};
    double temperatureResidualSlopePxPerCelsius{0.0};
    double residualTemperatureCorrelation{0.0};
    double metricScaleDriftPerMinute{0.0};
    double temperatureDeltaCelsius{0.0};
};

struct CalibrationReadinessInput {
    CalibrationHealthResult calibration{};
    RigObservabilityResult observability{};
    double synchronizationConfidence{0.0};
    double reprojectionRmsPx{0.0};
    double imageCoverage{0.0};
    double depthCoverage{0.0};
    double observationCoverage{0.0};
    bool metricScaleValid{false};
    double metricScaleConfidence{0.0};
    CalibrationDriftMetrics drift{};
};

struct CalibrationReadinessResult {
    bool ready{false};
    double confidence{0.0};
    std::vector<std::string> blockers;
    std::vector<std::string> warnings;
};

class CalibrationReadinessGate final {
public:
    CalibrationReadinessResult evaluate(
        const CalibrationReadinessInput&) const;
};

} // namespace nox::native::calibration

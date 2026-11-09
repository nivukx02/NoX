#include <cassert>
#include <cmath>

#include "calibration/calibration_drift_monitor.h"
#include "calibration/calibration_readiness_gate.h"

int main() {
    using namespace nox::native::calibration;

    CalibrationDriftMonitor monitor;

    for (std::uint64_t frame = 1; frame <= 20; ++frame) {
        const double minutes =
            static_cast<double>(frame - 1U) / 60.0;
        monitor.observe({
            frame * 1'000'000'000ull,
            0.8 + minutes * 0.18,
            0.94 - minutes * 0.02,
            0.72,
            0.61,
            1.0 + minutes * 0.004,
            0.92,
            25.0 + minutes * 0.8,
        });
    }

    const auto result=monitor.evaluate();
    assert(result.authoritative);
    assert(result.sampleCount==20u);
    assert(result.residualSlopePxPerMinute>0.15);
    assert(std::abs(result.residualTemperatureSlopePxPerCelsius) >= 0.0);
    assert(result.metricScaleDriftPerMinute>0.003);
    assert(result.recalibrationRecommended);
    assert(result.recalibrationRequired);
    assert(!result.stable);

    CalibrationReadinessGate gate;
    CalibrationReadinessInput input;
    input.calibration.calibrated=true;
    input.calibration.validity=CalibrationValidity::valid;
    input.calibration.confidence=0.95;
    input.observability.observable=true;
    input.observability.score=0.95;
    input.observability.connectedComponents=1;
    input.synchronizationConfidence=0.95;
    input.reprojectionRmsPx=0.9;
    input.imageCoverage=0.80;
    input.depthCoverage=0.70;
    input.observationCoverage=0.95;
    input.metricScaleValid=true;
    input.metricScaleConfidence=0.90;
    input.drift.authoritative=true;
    input.drift.recalibrationRequired=true;
    input.drift.recalibrationRecommended=true;
    input.drift.stabilityScore=result.stabilityScore;
    const auto gated=gate.evaluate(input);
    assert(!gated.ready);
    assert(!gated.blockers.empty());

    CalibrationDriftMonitor thermalMonitor;
    for (std::uint64_t frame = 1; frame <= 20; ++frame) {
        const double temperature =
            25.0 +
            static_cast<double>(frame - 1U) * 0.4;
        thermalMonitor.observe({
            frame * 1'000'000'000ull,
            0.30 +
                (temperature - 25.0) * 0.12,
            0.96,
            0.80,
            0.70,
            1.0,
            0.95,
            temperature,
        });
    }

    const auto thermalResult =
        thermalMonitor.evaluate();

    assert(
        std::abs(
            thermalResult
                .temperatureResidualSlopePxPerCelsius) >
        0.05);
    assert(thermalResult.recalibrationRecommended);

    monitor.reset();
    assert(monitor.size()==0u);
    return 0;
}

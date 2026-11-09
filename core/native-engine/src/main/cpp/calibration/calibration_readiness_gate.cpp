#include "calibration/calibration_readiness_gate.h"

#include <algorithm>
#include <cmath>

namespace nox::native::calibration {

CalibrationReadinessResult CalibrationReadinessGate::evaluate(
    const CalibrationReadinessInput& input) const {
    CalibrationReadinessResult output;

    const bool calibrationSatisfied =
        input.calibration.validity ==
            CalibrationValidity::valid &&
        (
            input.calibration.calibrated ||
            input.calibration.confidence >= 0.90
        );
    if(!calibrationSatisfied){
        output.blockers.emplace_back("calibration-not-valid");
    }
    const bool observabilitySatisfied =
        input.observability.observable ||
        (
            input.observability.connectedComponents <= 1U &&
            input.observability.baselineCoverage >= 0.55 &&
            input.observability.edgeConfidence >= 0.75 &&
            input.observability.p95EdgeUncertainty <= 1.0
        );
    if(!observabilitySatisfied){
        output.blockers.emplace_back("rig-not-observable");
    }
    if(input.synchronizationConfidence<0.60){
        output.blockers.emplace_back("synchronization-below-threshold");
    }
    if(!std::isfinite(input.reprojectionRmsPx)||
       input.reprojectionRmsPx>3.5){
        output.blockers.emplace_back("reprojection-error-too-high");
    }
    if(input.imageCoverage<0.25){
        output.blockers.emplace_back("insufficient-image-coverage");
    }
    if(input.depthCoverage<0.10){
        output.blockers.emplace_back("insufficient-depth-coverage");
    }
    if(input.observationCoverage<0.75){
        output.warnings.emplace_back("observation-coverage-degraded");
    }
    if(!input.metricScaleValid||input.metricScaleConfidence<0.55){
        output.warnings.emplace_back("metric-scale-not-authoritative");
    }
    if(input.observability.connectedComponents>1){
        output.blockers.emplace_back("camera-graph-disconnected");
    }
    if(input.calibration.confidence<0.75){
        output.warnings.emplace_back("calibration-confidence-degraded");
    }
    if(input.drift.authoritative){
        if(
            std::abs(
                input.drift.temperatureResidualSlopePxPerCelsius) >
            0.08
        ){
            output.warnings.emplace_back(
                "calibration-temperature-correlated-residual-drift");
        }
        if(
            std::abs(input.drift.temperatureResidualSlopePxPerCelsius) >
                0.16 &&
            std::abs(input.drift.residualTemperatureCorrelation) >= 0.65
        ){
            output.blockers.emplace_back(
                "calibration-temperature-correlated-drift-too-high");
        }
        if(input.drift.recalibrationRequired){
            output.blockers.emplace_back("calibration-drift-requires-recalibration");
        } else if(input.drift.recalibrationRecommended){
            output.warnings.emplace_back("calibration-drift-recalibration-recommended");
        }
        if(input.drift.temperatureDeltaCelsius>8.0){
            output.warnings.emplace_back("calibration-temperature-shift-observed");
        }
    }

    const double calibration=std::clamp(input.calibration.confidence,0.0,1.0);
    const double geometry =
        input.observability.score > 0.0
            ? std::clamp(input.observability.score,0.0,1.0)
            : std::clamp(
                0.35*input.observability.baselineCoverage+
                0.20*input.observability.angularDiversity+
                0.20*input.observability.edgeConfidence+
                0.15*input.observability.leverageDiversity+
                0.10*input.observability.connectivity,
                0.0,
                1.0);
    const double sync=std::clamp(input.synchronizationConfidence,0.0,1.0);
    const double reprojection=std::isfinite(input.reprojectionRmsPx)
        ? std::exp(-std::max(0.0,input.reprojectionRmsPx)/3.5)
        : 0.0;
    const double thermalAttribution =
        input.drift.authoritative
            ? std::exp(
                -std::max(
                    0.0,
                    std::abs(
                        input.drift.temperatureResidualSlopePxPerCelsius) /
                    0.08))
            : 1.0;
    const double drift = input.drift.authoritative
        ? std::clamp(
            input.drift.stabilityScore *
                (0.82 + 0.18 * thermalAttribution),
            0.0,
            1.0)
        : 1.0;
    const double imageCoverage =
        std::clamp(
            input.imageCoverage,
            0.0,
            1.0);
    const double depthCoverage =
        std::clamp(
            input.depthCoverage,
            0.0,
            1.0);
    const double observationCoverage =
        std::clamp(
            input.observationCoverage,
            0.0,
            1.0);
    const double coverage =
        observationCoverage > 0.0
            ? 0.40 * imageCoverage +
                0.35 * depthCoverage +
                0.25 * observationCoverage
            : 0.65 * imageCoverage +
                0.35 * depthCoverage;
    output.confidence=std::pow(
        std::max(
            1e-9,
            calibration*
            geometry*
            sync*
            reprojection*
            std::max(coverage,1e-6)*
            std::max(drift,1e-6)),
        0.20);
    output.ready=output.blockers.empty()&&output.confidence>=0.60;
    return output;
}

} // namespace nox::native::calibration

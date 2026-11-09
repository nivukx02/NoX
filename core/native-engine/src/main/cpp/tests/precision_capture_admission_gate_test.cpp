#include <cassert>

#include "validation/precision_capture_admission_gate.h"

int main() {
    using namespace nox::native::validation;

    const auto& gate =
        PrecisionCaptureAdmissionGate{};

    const auto missingEvidence =
        gate.evaluate({
            .requestedP95Millimeters = 1.0,
            .measuredP95Millimeters = 0.6,
            .measuredMaxMillimeters = 1.2,
            .metrologyAvailable = false,
            .metrologyTraceable = false,
            .calibrationConfidence = 0.97,
            .synchronizationConfidence = 0.96,
            .viewDiversity = 0.88,
            .crossViewAgreement = 0.90,
            .metricScaleConfidence = 0.95,
            .reconstructionUncertaintyMeters = 0.0004,
            .temporalStability = 0.95,
            .calibrationStable = true,
            .rigStable = true,
            .geometryConditioned = true,
            .thermalHeadroom = 0.95,
            .memoryHeadroom = 0.95,
        });

    assert(!missingEvidence.referenceGradeEligible);
    assert(!missingEvidence.authoritative);
    assert(
        !missingEvidence.blockers.empty());

    const auto qualified =
        gate.evaluate({
            .requestedP95Millimeters = 1.0,
            .measuredP95Millimeters = 0.72,
            .measuredMaxMillimeters = 1.40,
            .metrologyAvailable = true,
            .metrologyTraceable = true,
            .calibrationConfidence = 0.98,
            .synchronizationConfidence = 0.97,
            .viewDiversity = 0.88,
            .crossViewAgreement = 0.90,
            .metricScaleConfidence = 0.96,
            .reconstructionUncertaintyMeters = 0.00045,
            .temporalStability = 0.95,
            .calibrationStable = true,
            .rigStable = true,
            .geometryConditioned = true,
            .thermalHeadroom = 0.92,
            .memoryHeadroom = 0.94,
        });

    assert(qualified.referenceGradeEligible);
    assert(qualified.authoritative);
    assert(qualified.admitted);
    assert(qualified.score > 0.85);
    assert(qualified.precisionMarginMillimeters > 0.0);

    const auto weakGeometry =
        gate.evaluate({
            .requestedP95Millimeters = 1.0,
            .measuredP95Millimeters = 0.80,
            .measuredMaxMillimeters = 1.50,
            .metrologyAvailable = true,
            .metrologyTraceable = true,
            .calibrationConfidence = 0.96,
            .synchronizationConfidence = 0.96,
            .viewDiversity = 0.30,
            .crossViewAgreement = 0.40,
            .metricScaleConfidence = 0.96,
            .reconstructionUncertaintyMeters = 0.00050,
            .temporalStability = 0.92,
            .calibrationStable = true,
            .rigStable = true,
            .geometryConditioned = false,
            .thermalHeadroom = 0.90,
            .memoryHeadroom = 0.90,
        });

    assert(!weakGeometry.referenceGradeEligible);
    assert(!weakGeometry.authoritative);
    assert(!weakGeometry.blockers.empty());

    return 0;
}

#include <cassert>

#include "validation/precision_authority_gate.h"

int main() {
    using namespace nox::native::validation;

    QualificationReport qualified;
    qualified.state = QualificationState::qualified;
    qualified.validSamples = 120;
    qualified.p95ErrorMeters = 0.00072;
    qualified.maxErrorMeters = 0.00165;
    qualified.referenceUncertaintyP95Meters = 0.00031;
    qualified.confidence = 0.94;

    PrecisionAuthorityInput input;
    input.metrology = qualified;
    input.calibrationConfidence = 0.96;
    input.synchronizationConfidence = 0.95;
    input.reconstructionUncertaintyP95Meters = 0.00082;
    input.reconstructionConfidence = 0.94;
    input.global4DConsistency = 0.93;
    input.rigStability = 0.94;
    input.referenceBenchmarkArmed = true;

    const auto report =
        PrecisionAuthorityGate{}.evaluate(input);

    assert(
        report.state ==
        PrecisionAuthorityState::qualified);
    assert(report.measuredMetrologyQualified);
    assert(report.calibrationQualified);
    assert(report.synchronizationQualified);
    assert(report.reconstructionQualified);
    assert(report.rigQualified);
    assert(report.evidenceScore > 0.80);

    input.metrology.p95ErrorMeters = 0.0014;
    const auto unqualified =
        PrecisionAuthorityGate{}.evaluate(input);
    assert(
        unqualified.state ==
        PrecisionAuthorityState::evidenceOnly);

    input.metrology.p95ErrorMeters = 0.0007;
    input.referenceBenchmarkArmed = false;
    const auto notArmed =
        PrecisionAuthorityGate{}.evaluate(input);
    assert(
        notArmed.state ==
        PrecisionAuthorityState::evidenceOnly);

    input.referenceBenchmarkArmed = true;
    input.calibrationConfidence = 0.80;
    const auto badCalibration =
        PrecisionAuthorityGate{}.evaluate(input);
    assert(
        badCalibration.state ==
        PrecisionAuthorityState::blocked);

    return 0;
}

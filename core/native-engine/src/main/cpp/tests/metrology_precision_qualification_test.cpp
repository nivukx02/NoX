#include "validation/metrology_precision_qualification.h"
#include <cassert>

int main() {
    using namespace nox::native::validation;

    QualificationInput input;
    input.minimumSamples = 5U;
    input.minimumIndependentSessions = 2U;
    input.independentSessions = 3U;
    input.targetP95Meters = 0.001;
    input.targetMaxMeters = 0.002;
    input.maximumReferenceUncertaintyMeters = 0.0005;
    input.maximumCalibrationUncertaintyMeters = 0.0003;
    input.maximumSynchronizationUncertaintyMeters = 0.0003;
    input.maximumTemporalUncertaintyMeters = 0.0003;
    input.minimumConditioningScore = 0.85;
    input.calibrationUncertaintyMeters = 0.0001;
    input.synchronizationUncertaintyMeters = 0.0001;
    input.temporalUncertaintyMeters = 0.0001;
    input.systematicUncertaintyMeters = 0.0001;
    input.conditioningScore = 0.97;

    for (int i = 0; i < 5; ++i) {
        input.samples.push_back({
            0.00040 + 0.00002 * static_cast<double>(i),
            0.00010,
            true,
        });
    }

    const auto qualified =
        MetrologyPrecisionQualification{}.evaluate(input);

    assert(qualified.state == QualificationState::qualified);
    assert(qualified.p95ErrorMeters < 0.001);
    assert(qualified.referenceUncertaintyP95Meters < 0.0005);
    assert(qualified.combinedP95UncertaintyMeters > qualified.p95ErrorMeters);

    input.calibrationUncertaintyMeters = 0.0010;
    const auto calibrationRejected =
        MetrologyPrecisionQualification{}.evaluate(input);

    assert(calibrationRejected.state == QualificationState::indeterminate);
    assert(calibrationRejected.reason == "calibration-uncertainty-too-large");

    input.calibrationUncertaintyMeters = 0.0001;
    input.conditioningScore = 0.40;
    const auto geometryRejected =
        MetrologyPrecisionQualification{}.evaluate(input);

    assert(geometryRejected.state == QualificationState::indeterminate);
    assert(geometryRejected.reason == "geometric-conditioning-below-threshold");

    return 0;
}

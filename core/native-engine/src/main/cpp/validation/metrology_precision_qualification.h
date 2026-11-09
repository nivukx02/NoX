#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nox::native::validation {

enum class QualificationState : std::uint8_t {
    notQualified = 0,
    qualified = 1,
    indeterminate = 2,
};

struct ReferenceSample {
    double measuredErrorMeters{0.0};
    double referenceUncertaintyMeters{0.0};
    bool valid{false};
};

struct QualificationInput {
    double targetP95Meters{0.001};
    double targetMaxMeters{0.002};
    double maximumReferenceUncertaintyMeters{0.0005};
    double maximumOutlierFraction{0.02};
    double maximumCalibrationUncertaintyMeters{0.00075};
    double maximumSynchronizationUncertaintyMeters{0.00075};
    double maximumTemporalUncertaintyMeters{0.00050};
    double minimumConditioningScore{0.85};
    double calibrationUncertaintyMeters{0.0};
    double synchronizationUncertaintyMeters{0.0};
    double temporalUncertaintyMeters{0.0};
    double conditioningScore{1.0};
    double systematicUncertaintyMeters{0.0};
    std::size_t minimumSamples{30};
    std::size_t minimumIndependentSessions{3};
    std::size_t independentSessions{0};
    std::vector<ReferenceSample> samples;
};

struct QualificationReport {
    QualificationState state{QualificationState::notQualified};
    std::size_t validSamples{0};
    double meanErrorMeters{0.0};
    double p95ErrorMeters{0.0};
    double maxErrorMeters{0.0};
    double referenceUncertaintyP95Meters{0.0};
    double combinedP95UncertaintyMeters{0.0};
    double systematicUncertaintyMeters{0.0};
    double outlierFraction{1.0};
    double sampleConfidence{0.0};
    double referenceConfidence{0.0};
    double conditioningConfidence{0.0};
    double confidence{0.0};
    std::string reason;
};

class MetrologyPrecisionQualification final {
public:
    [[nodiscard]] QualificationReport evaluate(const QualificationInput& input) const;

private:
    static double percentile(std::vector<double> values, double quantile) noexcept;
    static double clamp01(double value) noexcept;
};

} // namespace nox::native::validation

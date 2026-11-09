#pragma once

#include <cstdint>
#include <vector>

namespace nox::native::validation {

struct GroundTruthSample {
    std::uint64_t timestampNanos{0};
    std::uint32_t jointId{0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

struct EstimateSample {
    std::uint64_t timestampNanos{0};
    std::uint32_t jointId{0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

struct JointAccuracy {
    std::uint32_t jointId{0};
    std::uint64_t matchedSamples{0};
    double meanErrorMeters{0.0};
    double p95ErrorMeters{0.0};
    double maxErrorMeters{0.0};
};

struct MotionAccuracyReport {
    std::uint64_t groundTruthSamples{0};
    std::uint64_t estimateSamples{0};
    std::uint64_t matchedSamples{0};
    std::uint64_t unmatchedGroundTruthSamples{0};
    double dropoutRate{1.0};
    double meanAbsoluteErrorMeters{0.0};
    double rmseMeters{0.0};
    double p50ErrorMeters{0.0};
    double p95ErrorMeters{0.0};
    double p99ErrorMeters{0.0};
    double maxErrorMeters{0.0};
    double temporalOffsetMedianMillis{0.0};
    double temporalOffsetP95Millis{0.0};
    double jitterRmsMeters{0.0};
    double scaleRatioMedian{1.0};
    double scaleDriftPerMinute{0.0};
    std::vector<JointAccuracy> joints;

    [[nodiscard]] bool passesPrecisionBudget(
        double p95Millimeters,
        double jitterMillimeters,
        double maximumDropoutRatio,
        double maximumLagMillis) const noexcept;
};

class MotionAccuracyEvaluator final {
public:
    struct Config {
        std::uint64_t maxMatchDeltaNanos{20'000'000ull};
        std::uint64_t maximumSupportedLagNanos{100'000'000ull};
        std::uint64_t maximumInterpolationGapNanos{40'000'000ull};
    };

    MotionAccuracyEvaluator() noexcept;
    explicit MotionAccuracyEvaluator(
        Config config) noexcept;

    [[nodiscard]] MotionAccuracyReport evaluate(
        const std::vector<GroundTruthSample>& groundTruth,
        const std::vector<EstimateSample>& estimates) const;

    [[nodiscard]] MotionAccuracyReport evaluateInterpolated(
        const std::vector<GroundTruthSample>& groundTruth,
        const std::vector<EstimateSample>& estimates) const;

private:
    Config config_{};

    static double distance(
        const GroundTruthSample&,
        const EstimateSample&) noexcept;
    static double percentile(
        std::vector<double>,
        double) noexcept;
};

} // namespace nox::native::validation

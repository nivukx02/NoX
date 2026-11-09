#pragma once

#include <cstdint>

namespace nox::native::tracking {

enum class AdaptiveMotionRegime : std::uint8_t {
    staticScene = 0,
    microMotion = 1,
    normal = 2,
    fast = 3,
    extreme = 4,
    discontinuous = 5,
};

struct AdaptiveMotionRegimeInput {
    double velocityMagnitudePixelsPerSecond{0.0};
    double accelerationMagnitudePixelsPerSecond2{0.0};
    double jerkMagnitudePixelsPerSecond3{0.0};
    double innovationPixels{0.0};
    double confidence{0.0};
    double temporalGapSeconds{0.0};
    double rollingShutterSkewSeconds{0.0};
    double directionConsistency{1.0};
    double microMotionRetention{1.0};
};

struct AdaptiveMotionRegimeDecision {
    AdaptiveMotionRegime regime{AdaptiveMotionRegime::staticScene};
    double dynamicStress{0.0};
    double continuityScore{1.0};
    double predictionTrustScale{1.0};
    double processNoiseScale{1.0};
    double innovationGateScale{1.0};
    double scoreThresholdScale{1.0};
    double microMotionPreservation{1.0};
    double motionEnergyScale{1.0};
    bool shouldPredict{true};
    bool shouldReject{false};
};

class AdaptiveMotionRegimeModel final {
public:
    struct Config {
        double staticVelocityPixelsPerSecond{3.0};
        double microVelocityPixelsPerSecond{18.0};
        double fastVelocityPixelsPerSecond{240.0};
        double extremeVelocityPixelsPerSecond{900.0};
        double staticAccelerationPixelsPerSecond2{80.0};
        double fastAccelerationPixelsPerSecond2{1400.0};
        double extremeAccelerationPixelsPerSecond2{6000.0};
        double discontinuityInnovationPixels{80.0};
        double extremeInnovationPixels{140.0};
        double maximumTemporalGapSeconds{0.25};
        double maximumRollingShutterSeconds{0.020};
    };

    AdaptiveMotionRegimeModel() noexcept;
    explicit AdaptiveMotionRegimeModel(Config config) noexcept;

    [[nodiscard]] AdaptiveMotionRegimeDecision evaluate(
        const AdaptiveMotionRegimeInput& input) const noexcept;

private:
    Config config_{};

    static double clamp01(double value) noexcept;
    static double safeRatio(double numerator, double denominator) noexcept;
};

} // namespace nox::native::tracking

#pragma once

#include <cstddef>
#include <cstdint>
#include "tracking/adaptive_motion_regime.h"
#include <unordered_map>

#include "core/observation/motion_observation.h"

namespace nox::native::tracking {

struct NativeObservationQuality {
    std::uint32_t cameraId{0};
    std::uint64_t frameId{0};
    std::uint32_t inputKeypoints{0};
    std::uint32_t finiteKeypoints{0};
    std::uint32_t confidentKeypoints{0};
    double meanConfidence{0.0};
    double confidenceCoverage{0.0};
    double motionEnergyPixels{0.0};
    double motionContinuity{1.0};
    double temporalGapPenalty{0.0};
    double rollingShutterPenalty{0.0};
    double exposureMotionBlurRisk{0.0};
    double meanVelocityPixelsPerSecond{0.0};
    double meanAccelerationPixelsPerSecond2{0.0};
    double meanJerkPixelsPerSecond3{0.0};
    double p95InnovationPixels{0.0};
    double motionDirectionConsistency{1.0};
    double microMotionRetention{1.0};
    double subpixelMotionRetention{1.0};
    double dynamicStability{1.0};
    double score{0.0};
    bool accepted{false};
};

class NativeObservationProcessor final {
public:
    struct Config {
        double minimumKeypointConfidence{0.08};
        double confidenceCoverageTarget{0.35};
        double maximumMotionEnergyPixels{180.0};
        double maximumGapSeconds{0.25};
        double maximumPredictionAccelerationPixelsPerSecond2{6000.0};
        double maximumPredictionJerkPixelsPerSecond3{30000.0};
        double rollingShutterReferenceSeconds{0.020};
        double exposureMotionReferencePixels{1.50};
        double subpixelNoiseFloorPixels{0.12};
        double minimumAcceptedScore{0.35};
        double stateRetentionSeconds{2.0};
        std::size_t maximumTrackedKeypointsPerCamera{1024};
        std::size_t maximumCameraStates{128};
        std::uint32_t pruneEveryFrames{30};
    };

    NativeObservationProcessor() noexcept;
    explicit NativeObservationProcessor(Config config) noexcept;

    [[nodiscard]] NativeObservationQuality process(
        const MotionObservation& observation) noexcept;

    void reset() noexcept;
    void resetCamera(std::uint32_t cameraId) noexcept;

private:
    struct KeypointState {
        float x{0.0f};
        float y{0.0f};
        double velocityX{0.0};
        double velocityY{0.0};
        double accelerationX{0.0};
        double accelerationY{0.0};
        double jerkX{0.0};
        double jerkY{0.0};
        double confidence{0.0};
        std::uint64_t timestampNanos{0};
        bool initialized{false};
    };

    Config config_{};
    AdaptiveMotionRegimeModel regimeModel_{};
    std::uint64_t processedFrames_{0};
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, KeypointState>>
        states_;
    std::unordered_map<std::uint32_t, double>
        expectedIntervalSeconds_;

    static double clamp01(double value) noexcept;
    static bool finite(float value) noexcept;
    void pruneState(std::uint64_t timestampNanos) noexcept;
};

} // namespace nox::native::tracking

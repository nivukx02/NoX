#pragma once

#include <cstdint>
#include <vector>

#include "synchronization/motion_temporal_alignment.h"

namespace nox::native::sync {

struct SynchronizationOptions {
    std::uint32_t minimumSamples{3U};
    double maximumResidualP95Nanos{12'000'000.0};
    double maximumOutlierRatio{0.20};
    double minimumClockSupportRatio{0.60};
    double minimumTemporalConfidence{0.50};
    double minimumSubframeConfidence{0.0};
    bool requireSubframeEvidence{false};

    static SynchronizationOptions highPrecision() noexcept {
        SynchronizationOptions options;
        options.minimumSamples = 8U;
        options.maximumResidualP95Nanos = 4'000'000.0;
        options.maximumOutlierRatio = 0.05;
        options.minimumClockSupportRatio = 0.90;
        options.minimumTemporalConfidence = 0.85;
        options.minimumSubframeConfidence = 0.65;
        options.requireSubframeEvidence = true;
        return options;
    }
};

struct SynchronizationResult {
    double offset_nanos{0.0};
    double drift_ppm{0.0};
    double jitter_nanos{0.0};
    double latency_nanos{0.0};
    double subframe_delta_nanos{0.0};
    double subframe_confidence{0.0};
    double subframe_sharpness{0.0};
    double temporal_confidence{0.0};
    double residual_p95_nanos{0.0};
    double residualMaxNanos{0.0};
    double outlier_ratio{0.0};
    double clockSupportRatio{0.0};
    double offsetUncertaintyNanos{0.0};
    double driftUncertaintyPpm{0.0};
    std::uint32_t clockIterations{0};
    std::uint32_t sampleCount{0};
    bool monotonicInput{true};
    bool valid{false};
    double motionOffsetNanos{0.0};
    double motionConfidence{0.0};
    double fusedOffsetNanos{0.0};
};

class SynchronizationEstimator final {
public:
    [[nodiscard]] SynchronizationResult estimate(
        const std::vector<std::uint64_t>& reference_nanos,
        const std::vector<std::uint64_t>& remote_nanos,
        const std::vector<double>& reference_events = {},
        const std::vector<double>& remote_events = {},
        double event_sample_interval_nanos = 1.0,
        const SynchronizationOptions& options = {}) const;

    [[nodiscard]] SynchronizationResult estimateWithMotionSignals(
        const std::vector<std::uint64_t>& reference_nanos,
        const std::vector<std::uint64_t>& remote_nanos,
        const std::vector<MotionTemporalSample>& reference_motion,
        const std::vector<MotionTemporalSample>& remote_motion,
        double motion_sample_interval_nanos,
        const SynchronizationOptions& options = {}) const;
};

} // namespace nox::native::sync

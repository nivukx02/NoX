#include "synchronization/synchronization_estimator.h"

#include "synchronization/clock_model.h"
#include "synchronization/event_correlation.h"
#include "synchronization/subframe_alignment.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace nox::native::sync {
namespace {

bool monotonic(
    const std::vector<std::uint64_t>& values) {
    for (
        std::size_t index = 1U;
        index < values.size();
        ++index) {
        if (
            values[index] <=
            values[index - 1U]) {
            return false;
        }
    }

    return true;
}

} // namespace

SynchronizationResult
SynchronizationEstimator::estimate(
    const std::vector<std::uint64_t>& reference,
    const std::vector<std::uint64_t>& remote,
    const std::vector<double>& referenceEvents,
    const std::vector<double>& remoteEvents,
    double eventSampleIntervalNanos,
    const SynchronizationOptions& options) const {
    SynchronizationResult output;

    const bool referenceMonotonic =
        monotonic(
            reference);
    const bool remoteMonotonic =
        monotonic(
            remote);

    output.monotonicInput =
        referenceMonotonic &&
        remoteMonotonic;

    if (
        reference.size() !=
            remote.size() ||
        reference.size() <
            std::max<std::size_t>(
                3U,
                options.minimumSamples) ||
        !output.monotonicInput) {
        return output;
    }

    output.sampleCount =
        static_cast<std::uint32_t>(
            std::min<std::size_t>(
                reference.size(),
                std::numeric_limits<
                    std::uint32_t>::max()));

    const auto clock =
        ClockModel{}.fit(
            reference,
            remote);

    if (!clock.valid) {
        return output;
    }

    output.offset_nanos =
        clock.offset_nanos;
    output.drift_ppm =
        clock.drift_ppm;
    output.jitter_nanos =
        clock.jitter_rms_nanos;
    output.latency_nanos =
        clock.offset_nanos;

    output.residual_p95_nanos =
        clock.residual_p95_nanos;
    output.residualMaxNanos =
        clock.residual_max_nanos;
    output.outlier_ratio =
        clock.outlier_ratio;
    output.clockSupportRatio =
        clock.support_ratio;
    output.offsetUncertaintyNanos =
        clock.offset_uncertainty_nanos;
    output.driftUncertaintyPpm =
        clock.slope_uncertainty_ppm;
    output.clockIterations =
        clock.iterations;

    output.temporal_confidence =
        clock.confidence;

    if (
        referenceEvents.size() >= 5U &&
        referenceEvents.size() ==
            remoteEvents.size() &&
        std::isfinite(
            eventSampleIntervalNanos) &&
        eventSampleIntervalNanos >
            0.0) {
        const auto correlation =
            EventCorrelation{}.estimate(
                referenceEvents,
                remoteEvents);

        if (
            correlation.valid) {
            output.subframe_delta_nanos =
                SubframeAlignment::samplesToNanoseconds(
                    correlation.offset_samples,
                    eventSampleIntervalNanos);

            output.subframe_confidence =
                correlation.confidence;
            output.subframe_sharpness =
                correlation.sharpness;

            output.temporal_confidence =
                0.72 *
                    output.temporal_confidence +
                0.28 *
                    std::clamp(
                        correlation.confidence,
                        0.0,
                        1.0);
        }
    }

    const double residualScore =
        std::exp(
            -output.residual_p95_nanos /
            2'500'000.0);

    const double jitterScore =
        std::exp(
            -output.jitter_nanos /
            1'500'000.0);

    const double outlierScore =
        std::exp(
            -output.outlier_ratio *
            3.0);

    output.temporal_confidence =
        std::clamp(
            output.temporal_confidence *
                residualScore *
                jitterScore *
                outlierScore,
            0.0,
            1.0);

    const bool subframeEvidenceAvailable =
        output.subframe_confidence > 0.0;

    const bool robustClockEvidence =
        output.clockSupportRatio >= 0.90 &&
        output.outlier_ratio <=
            std::clamp(
                options.maximumOutlierRatio,
                0.0,
                1.0) &&
        output.residual_p95_nanos <=
            std::max(
                options.maximumResidualP95Nanos,
                1.0) &&
        !options.requireSubframeEvidence;

    output.valid =
        (
            output.temporal_confidence >=
                std::clamp(
                    options.minimumTemporalConfidence,
                    0.0,
                    1.0) ||
            robustClockEvidence
        ) &&
        output.clockSupportRatio >=
            std::clamp(
                options.minimumClockSupportRatio,
                0.0,
                1.0) &&
        output.outlier_ratio <=
            std::clamp(
                options.maximumOutlierRatio,
                0.0,
                1.0) &&
        output.residual_p95_nanos <=
            std::max(
                options.maximumResidualP95Nanos,
                1.0) &&
        (
            !options.requireSubframeEvidence ||
            (
                subframeEvidenceAvailable &&
                output.subframe_confidence >=
                    std::clamp(
                        options.minimumSubframeConfidence,
                        0.0,
                        1.0)
            )
        );

    return output;
}


SynchronizationResult
SynchronizationEstimator::estimateWithMotionSignals(
    const std::vector<std::uint64_t>& reference,
    const std::vector<std::uint64_t>& remote,
    const std::vector<MotionTemporalSample>& referenceMotion,
    const std::vector<MotionTemporalSample>& remoteMotion,
    double motionSampleIntervalNanos,
    const SynchronizationOptions& options) const {
    SynchronizationResult output =
        estimate(
            reference,
            remote,
            {},
            {},
            1.0,
            options);

    const auto motion =
        MotionTemporalAlignment{}.estimate(
            referenceMotion,
            remoteMotion,
            motionSampleIntervalNanos);

    output.motionOffsetNanos =
        motion.offsetNanos;
    output.motionConfidence =
        motion.confidence;

    if (
        !motion.valid) {
        output.fusedOffsetNanos =
            output.offset_nanos;
        return output;
    }

    const double clockWeight =
        std::clamp(
            output.temporal_confidence,
            0.0,
            1.0);

    const double motionWeight =
        std::clamp(
            motion.confidence,
            0.0,
            1.0);

    const double totalWeight =
        clockWeight +
        motionWeight;

    if (
        totalWeight <=
        1e-9) {
        output.fusedOffsetNanos =
            output.offset_nanos;
        return output;
    }

    output.fusedOffsetNanos =
        (
            output.offset_nanos *
                clockWeight +
            motion.offsetNanos *
                motionWeight
        ) /
        totalWeight;

    const double offsetDisagreement =
        std::abs(
            output.offset_nanos -
            motion.offsetNanos);

    const double agreementConfidence =
        std::exp(
            -offsetDisagreement /
            std::max(
                motionSampleIntervalNanos,
                1.0));

    output.temporal_confidence =
        std::clamp(
            output.temporal_confidence *
                (
                    0.70 +
                    0.30 *
                        agreementConfidence
                ) *
                (
                    0.65 +
                    0.35 *
                        motion.confidence
                ),
            0.0,
            1.0);

    output.latency_nanos =
        output.fusedOffsetNanos;

    if (
        offsetDisagreement >
            std::max(
                2.0 *
                    motionSampleIntervalNanos,
                10'000'000.0)) {
        output.valid = false;
    }

    return output;
}

} // namespace nox::native::sync

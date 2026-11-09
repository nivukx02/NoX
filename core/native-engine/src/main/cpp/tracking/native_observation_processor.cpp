#include "tracking/native_observation_processor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <unordered_set>

namespace nox::native::tracking {

namespace {

constexpr std::uint32_t kFaceLandmarkIdBase = 1000u;
constexpr std::uint32_t kEyeLandmarkIdBase = 2000u;
constexpr std::uint32_t kHandLandmarkIdBase = 3000u;

double directionConsistency(
    double ax,
    double ay,
    double bx,
    double by) noexcept {
    const double aNorm =
        std::hypot(ax, ay);
    const double bNorm =
        std::hypot(bx, by);

    if (
        !std::isfinite(aNorm) ||
        !std::isfinite(bNorm) ||
        aNorm < 1e-9 ||
        bNorm < 1e-9) {
        return 1.0;
    }

    const double cosine =
        std::clamp(
            (ax * bx + ay * by) /
                (aNorm * bNorm),
            -1.0,
            1.0);

    return std::clamp(
        0.5 + 0.5 * cosine,
        0.0,
        1.0);
}

double percentile95(
    std::array<double, kMaxObservationKeypoints>& values,
    std::size_t count) noexcept {
    if (count == 0U) return 0.0;

    count = std::min(
        count,
        kMaxObservationKeypoints);

    std::sort(
        values.begin(),
        values.begin() + static_cast<std::ptrdiff_t>(count));

    const double index =
        0.95 *
        static_cast<double>(
            count - 1U);

    const auto lower =
        static_cast<std::size_t>(
            std::floor(index));
    const auto upper =
        std::min(
            count - 1U,
            lower + 1U);

    const double alpha =
        index -
        static_cast<double>(
            lower);

    return values[lower] +
        (
            values[upper] -
            values[lower]
        ) *
        alpha;
}

std::uint32_t stableLandmarkSlot(
    std::uint32_t landmarkId) noexcept {
    if (landmarkId < 33u) {
        return landmarkId;
    }

    if (
        landmarkId >= kFaceLandmarkIdBase &&
        landmarkId < kFaceLandmarkIdBase + 49u) {
        return 33u + (landmarkId - kFaceLandmarkIdBase);
    }

    if (
        landmarkId >= kEyeLandmarkIdBase &&
        landmarkId < kEyeLandmarkIdBase + 4u) {
        return 82u + (landmarkId - kEyeLandmarkIdBase);
    }

    if (landmarkId >= kHandLandmarkIdBase) {
        const std::uint32_t relative =
            landmarkId - kHandLandmarkIdBase;
        const std::uint32_t handSlot =
            relative / 32u;
        const std::uint32_t landmarkSlot =
            relative % 32u;
        if (
            handSlot < 2u &&
            landmarkSlot < 21u) {
            return 86u +
                handSlot * 21u +
                landmarkSlot;
        }
    }

    return landmarkId % kMaxObservationKeypoints;
}

std::uint64_t effectiveDynamicsTimestampNanos(
    const MotionObservation& observation) noexcept {
    if (
        observation.exposure_start_nanos > 0U &&
        observation.exposure_duration_nanos > 0U &&
        observation.exposure_start_nanos <=
            std::numeric_limits<std::uint64_t>::max() -
                observation.exposure_duration_nanos
    ) {
        return observation.exposure_start_nanos +
            observation.exposure_duration_nanos / 2U;
    }
    return observation.monotonic_timestamp_nanos;
}

} // namespace

NativeObservationProcessor::NativeObservationProcessor() noexcept
    : NativeObservationProcessor(Config{}) {}

NativeObservationProcessor::NativeObservationProcessor(
    Config config) noexcept
    : config_(config) {
    config_.minimumKeypointConfidence =
        clamp01(config_.minimumKeypointConfidence);
    config_.confidenceCoverageTarget =
        std::clamp(
            config_.confidenceCoverageTarget,
            0.01,
            1.0);
    config_.maximumMotionEnergyPixels =
        std::max(
            config_.maximumMotionEnergyPixels,
            1.0);
    config_.maximumGapSeconds =
        std::clamp(
            config_.maximumGapSeconds,
            0.010,
            2.0);
    config_.rollingShutterReferenceSeconds =
        std::clamp(
            config_.rollingShutterReferenceSeconds,
            0.001,
            0.25);
    config_.exposureMotionReferencePixels =
        std::clamp(
            config_.exposureMotionReferencePixels,
            0.10,
            32.0);
    config_.minimumAcceptedScore =
        std::clamp(
            config_.minimumAcceptedScore,
            0.0,
            1.0);
    config_.stateRetentionSeconds =
        std::clamp(config_.stateRetentionSeconds, 0.10, 10.0);
    config_.maximumTrackedKeypointsPerCamera =
        std::max<std::size_t>(
            config_.maximumTrackedKeypointsPerCamera,
            64U);
    config_.maximumCameraStates =
        std::max<std::size_t>(
            config_.maximumCameraStates,
            4U);
    config_.pruneEveryFrames =
        std::max<std::uint32_t>(
            config_.pruneEveryFrames,
            1U);
}

NativeObservationQuality NativeObservationProcessor::process(
    const MotionObservation& observation) noexcept {
    try {
        ++processedFrames_;
        const auto dynamicsTimestampNanos =
            effectiveDynamicsTimestampNanos(observation);
        if (processedFrames_ % config_.pruneEveryFrames == 0U) {
            pruneState(dynamicsTimestampNanos);
        }
        NativeObservationQuality output{};
        output.cameraId = observation.camera_id;
        output.frameId = observation.frame_id;
        output.inputKeypoints = observation.keypoint_count;

        if (
            observation.keypoint_count == 0 ||
            observation.keypoint_count > kMaxObservationKeypoints
        ) {
            output.accepted = false;
            return output;
        }

        auto cameraIterator =
            states_.find(observation.camera_id);
        if (
            cameraIterator == states_.end() &&
            states_.size() >= config_.maximumCameraStates
        ) {
            auto evictionCandidate = states_.end();
            std::uint64_t oldestTimestamp =
                std::numeric_limits<std::uint64_t>::max();

            for (auto iterator = states_.begin();
                 iterator != states_.end();
                 ++iterator) {
                std::uint64_t latestTimestamp = 0U;
                for (const auto& keypoint : iterator->second) {
                    latestTimestamp =
                        std::max(
                            latestTimestamp,
                            keypoint.second.timestampNanos);
                }

                if (
                    latestTimestamp < oldestTimestamp ||
                    (
                        latestTimestamp == oldestTimestamp &&
                        (
                            evictionCandidate == states_.end() ||
                            iterator->first < evictionCandidate->first
                        )
                    )
                ) {
                    oldestTimestamp = latestTimestamp;
                    evictionCandidate = iterator;
                }
            }

            if (evictionCandidate != states_.end()) {
                expectedIntervalSeconds_.erase(
                    evictionCandidate->first);
                states_.erase(evictionCandidate);
            }
        }

        /*
         * Enforce the camera-state ceiling before operator[] can allocate a new map. Periodic
         * pruning remains responsible for fine-grained stale keypoint cleanup, but admission-time
         * eviction prevents bursts of unique network cameras from temporarily expanding native
         * state without bound between prune intervals.
         */
        if (states_.find(observation.camera_id) == states_.end() &&
            states_.size() >= config_.maximumCameraStates) {
            auto oldestCamera =
                states_.end();
            std::uint64_t oldestTimestamp =
                std::numeric_limits<std::uint64_t>::max();

            for (auto it = states_.begin();
                 it != states_.end();
                 ++it) {
                std::uint64_t latestTimestamp = 0U;
                for (const auto& keypoint : it->second) {
                    latestTimestamp =
                        std::max(
                            latestTimestamp,
                            keypoint.second.timestampNanos);
                }

                if (
                    oldestCamera == states_.end() ||
                    latestTimestamp < oldestTimestamp ||
                    (
                        latestTimestamp == oldestTimestamp &&
                        it->first < oldestCamera->first
                    )
                ) {
                    oldestTimestamp = latestTimestamp;
                    oldestCamera = it;
                }
            }

            if (oldestCamera != states_.end()) {
                expectedIntervalSeconds_.erase(
                    oldestCamera->first);
                states_.erase(oldestCamera);
            }
        }

        auto& cameraStates = states_[observation.camera_id];

        double confidenceSum = 0.0;
        std::uint32_t finiteCount = 0;
        std::uint32_t confidentCount = 0;
        double motionSum = 0.0;
        double velocitySum = 0.0;
        double accelerationSum = 0.0;
        double jerkSum = 0.0;
        double directionSum = 0.0;
        double microMotionSum = 0.0;
        double subpixelMotionSum = 0.0;
        double stabilitySum = 0.0;
        std::array<double, kMaxObservationKeypoints> innovations{};
        std::size_t innovationCount = 0U;
        std::uint32_t motionPairs = 0;
        double continuitySum = 0.0;
        std::uint32_t continuityPairs = 0;
        double maximumGapSeconds = 0.0;
        double observedFrameDtSeconds = 0.0;

        auto expectedIntervalIterator =
            expectedIntervalSeconds_.find(
                observation.camera_id);

        double expectedIntervalSeconds =
            expectedIntervalIterator ==
                    expectedIntervalSeconds_.end()
                ? (1.0 / 60.0)
                : expectedIntervalIterator->second;

        if (observation.frame_interval_nanos > 0ULL) {
            const double packetIntervalSeconds =
                static_cast<double>(
                    observation.frame_interval_nanos) /
                1e9;
            if (
                std::isfinite(packetIntervalSeconds) &&
                packetIntervalSeconds > 0.0
            ) {
                expectedIntervalSeconds =
                    packetIntervalSeconds;
            }
        }

        expectedIntervalSeconds =
            std::clamp(
                expectedIntervalSeconds,
                1.0 / 240.0,
                1.0 / 15.0);

        for (std::uint32_t index = 0;
             index < observation.keypoint_count;
             ++index) {
            const auto& keypoint = observation.keypoints[index];
            const auto stateSlot =
                stableLandmarkSlot(
                    observation.keypoint_ids[index]);
            if (!finite(keypoint.x) || !finite(keypoint.y)) {
                continue;
            }

            ++finiteCount;

            const double confidence =
                clamp01(static_cast<double>(keypoint.confidence));
            confidenceSum += confidence;

            if (confidence >= config_.minimumKeypointConfidence) {
                ++confidentCount;
            }

            auto& previous = cameraStates[stateSlot];
            if (previous.initialized &&
                dynamicsTimestampNanos >
                    previous.timestampNanos) {
                const double dt =
                    static_cast<double>(
                        dynamicsTimestampNanos -
                        previous.timestampNanos) /
                    1e9;

                if (dt > 0.0 && std::isfinite(dt)) {
                    const double dx =
                        static_cast<double>(keypoint.x) -
                        static_cast<double>(previous.x);
                    const double dy =
                        static_cast<double>(keypoint.y) -
                        static_cast<double>(previous.y);
                    const double distance =
                        std::sqrt(dx * dx + dy * dy);

                    if (std::isfinite(distance)) {
                        const double weight =
                            std::min(
                                confidence,
                                previous.confidence);

                        const double velocityX = dx / dt;
                        const double velocityY = dy / dt;

                        const double accelerationX =
                            (velocityX - previous.velocityX) / dt;
                        const double accelerationY =
                            (velocityY - previous.velocityY) / dt;

                        const double jerkX =
                            (accelerationX - previous.accelerationX) / dt;
                        const double jerkY =
                            (accelerationY - previous.accelerationY) / dt;

                        const double velocityMagnitude =
                            std::hypot(velocityX, velocityY);
                        const double accelerationMagnitude =
                            std::hypot(
                                accelerationX,
                                accelerationY);
                        const double jerkMagnitude =
                            std::hypot(jerkX, jerkY);

                        const double expectedForPrediction =
                            std::max(
                                expectedIntervalSeconds,
                                1e-6);

                        const double gapRatio =
                            std::max(
                                0.0,
                                dt /
                                    expectedForPrediction -
                                    1.0);

                        const double higherOrderTrust =
                            std::exp(
                                -std::min(
                                    3.0,
                                    gapRatio * 0.90));

                        const double accelerationTrust =
                            higherOrderTrust;

                        const double jerkTrust =
                            higherOrderTrust *
                            (
                                1.0 -
                                    std::min(
                                        0.70,
                                        gapRatio * 0.35)
                            );

                        const double boundedAccelerationX =
                            std::clamp(
                                previous.accelerationX,
                                -config_.maximumPredictionAccelerationPixelsPerSecond2,
                                config_.maximumPredictionAccelerationPixelsPerSecond2);
                        const double boundedAccelerationY =
                            std::clamp(
                                previous.accelerationY,
                                -config_.maximumPredictionAccelerationPixelsPerSecond2,
                                config_.maximumPredictionAccelerationPixelsPerSecond2);

                        const double boundedJerkX =
                            std::clamp(
                                previous.jerkX,
                                -config_.maximumPredictionJerkPixelsPerSecond3,
                                config_.maximumPredictionJerkPixelsPerSecond3);
                        const double boundedJerkY =
                            std::clamp(
                                previous.jerkY,
                                -config_.maximumPredictionJerkPixelsPerSecond3,
                                config_.maximumPredictionJerkPixelsPerSecond3);

                        const double rawDisplacementX =
                            previous.velocityX * dt +
                            0.5 *
                                boundedAccelerationX *
                                accelerationTrust *
                                dt *
                                dt +
                            (
                                1.0 / 6.0
                            ) *
                                boundedJerkX *
                                jerkTrust *
                                dt *
                                dt *
                                dt;

                        const double rawDisplacementY =
                            previous.velocityY * dt +
                            0.5 *
                                boundedAccelerationY *
                                accelerationTrust *
                                dt *
                                dt +
                            (
                                1.0 / 6.0
                            ) *
                                boundedJerkY *
                                jerkTrust *
                                dt *
                                dt *
                                dt;

                        const double displacementMagnitude =
                            std::hypot(
                                rawDisplacementX,
                                rawDisplacementY);

                        const double maximumPredictedDisplacement =
                            std::max(
                                4.0,
                                config_.maximumMotionEnergyPixels *
                                    std::max(
                                        0.25,
                                        std::min(
                                            2.50,
                                            dt /
                                                expectedForPrediction)));

                        const double displacementScale =
                            displacementMagnitude >
                                    maximumPredictedDisplacement
                                ? maximumPredictedDisplacement /
                                    displacementMagnitude
                                : 1.0;

                        const double predictedX =
                            static_cast<double>(previous.x) +
                            rawDisplacementX *
                                displacementScale;
                        const double predictedY =
                            static_cast<double>(previous.y) +
                            rawDisplacementY *
                                displacementScale;
                        const double innovation =
                            std::hypot(
                                static_cast<double>(keypoint.x) -
                                    predictedX,
                                static_cast<double>(keypoint.y) -
                                    predictedY);

                        const double innovationScale =
                            std::max(
                                2.0,
                                config_.maximumMotionEnergyPixels *
                                    0.08);

                        const double innovationWeight =
                            1.0 /
                                (
                                    1.0 +
                                    (
                                        innovation /
                                            innovationScale
                                    ) *
                                    (
                                        innovation /
                                            innovationScale
                                    )
                                );

                        const double direction =
                            directionConsistency(
                                velocityX,
                                velocityY,
                                previous.velocityX,
                                previous.velocityY);

                        const double expected =
                            expectedIntervalSeconds;
                        const double dtDeviation =
                            std::abs(dt - expected) /
                            std::max(
                                expected,
                                1e-6);
                        const double temporalGapSeconds =
                            std::max(
                                0.0,
                                dt - expected);

                        if (
                            observedFrameDtSeconds <= 0.0 &&
                            dt >= 1.0 / 240.0 &&
                            dt <= 1.0 / 15.0) {
                            observedFrameDtSeconds = dt;
                        }
                        const double motionNoiseFloor =
                            0.18 +
                                (
                                    1.0 -
                                        weight
                                ) *
                                0.72;

                        const double microRetention =
                            (
                                distance /
                                    (
                                        distance +
                                            motionNoiseFloor
                                    )
                                ).coerceAtLeast(0.0);

                        const double subpixelSignal =
                            clamp01(
                                1.0 -
                                    std::exp(
                                        -std::max(
                                            0.0,
                                            distance -
                                                config_.subpixelNoiseFloorPixels,
                                        ) /
                                        std::max(
                                            config_.subpixelNoiseFloorPixels,
                                            1e-6,
                                        )));

                        const auto regime =
                            regimeModel_.evaluate(
                                {
                                    velocityMagnitude,
                                    accelerationMagnitude,
                                    jerkMagnitude,
                                    innovation,
                                    weight,
                                    temporalGapSeconds,
                                    static_cast<double>(
                                        observation.rolling_shutter_skew_nanos) /
                                        1e9,
                                    direction,
                                    microRetention,
                                });

                        const double continuity =
                            std::exp(
                                -std::min(
                                    8.0,
                                    dtDeviation * 2.0)) *
                            regime.continuityScore;

                        motionSum +=
                            distance *
                            (0.25 + 0.75 * weight) *
                            regime.motionEnergyScale;
                        velocitySum +=
                            velocityMagnitude *
                            weight;
                        accelerationSum +=
                            accelerationMagnitude *
                            weight;
                        jerkSum +=
                            jerkMagnitude *
                            weight;
                        directionSum +=
                            direction *
                            weight;
                        microMotionSum +=
                            microRetention *
                            regime.microMotionPreservation *
                            weight;
                        subpixelMotionSum +=
                            subpixelSignal *
                            (0.55 + 0.45 * weight);
                        stabilitySum +=
                            continuity *
                            direction *
                            regime.continuityScore *
                            weight *
                            innovationWeight;
                        if (innovationCount < innovations.size()) {
                            innovations[innovationCount++] = innovation;
                        }
                        ++motionPairs;

                        maximumGapSeconds =
                            std::max(
                                maximumGapSeconds,
                                std::max(
                                    0.0,
                                    dt - expected));

                        continuitySum += continuity;
                        ++continuityPairs;

                        const double derivativeBlend =
                            std::clamp(
                                0.22 +
                                    0.58 * weight +
                                    0.20 * innovationWeight,
                                0.16,
                                0.96);

                        previous.velocityX =
                            previous.velocityX *
                                (1.0 - derivativeBlend) +
                            std::clamp(
                                velocityX,
                                -50'000.0,
                                50'000.0) *
                                derivativeBlend;
                        previous.velocityY =
                            previous.velocityY *
                                (1.0 - derivativeBlend) +
                            std::clamp(
                                velocityY,
                                -50'000.0,
                                50'000.0) *
                                derivativeBlend;

                        const double accelerationBlend =
                            std::clamp(
                                derivativeBlend * 0.82,
                                0.12,
                                0.90);

                        previous.accelerationX =
                            previous.accelerationX *
                                (1.0 - accelerationBlend) +
                            std::clamp(
                                accelerationX,
                                -500'000.0,
                                500'000.0) *
                                accelerationBlend;
                        previous.accelerationY =
                            previous.accelerationY *
                                (1.0 - accelerationBlend) +
                            std::clamp(
                                accelerationY,
                                -500'000.0,
                                500'000.0) *
                                accelerationBlend;

                        const double jerkBlend =
                            std::clamp(
                                accelerationBlend * 0.68,
                                0.08,
                                0.78);

                        previous.jerkX =
                            previous.jerkX *
                                (1.0 - jerkBlend) +
                            std::clamp(
                                jerkX,
                                -5'000'000.0,
                                5'000'000.0) *
                                jerkBlend;
                        previous.jerkY =
                            previous.jerkY *
                                (1.0 - jerkBlend) +
                            std::clamp(
                                jerkY,
                                -5'000'000.0,
                                5'000'000.0) *
                                jerkBlend;
                    }
                }
            }

            previous.x = keypoint.x;
            previous.y = keypoint.y;
            previous.confidence = confidence;
            previous.timestampNanos =
                dynamicsTimestampNanos;
            previous.initialized = true;
        }

        if (
            observedFrameDtSeconds > 0.0 &&
            std::isfinite(observedFrameDtSeconds)
        ) {
            expectedIntervalSeconds =
                expectedIntervalSeconds * 0.92 +
                    observedFrameDtSeconds * 0.08;

            expectedIntervalSeconds =
                std::clamp(
                    expectedIntervalSeconds,
                    1.0 / 240.0,
                    1.0 / 15.0);

            expectedIntervalSeconds_[
                observation.camera_id] =
                expectedIntervalSeconds;
        }

        output.finiteKeypoints = finiteCount;
        output.confidentKeypoints = confidentCount;
        output.meanConfidence =
            finiteCount == 0
                ? 0.0
                : clamp01(
                    confidenceSum /
                    static_cast<double>(finiteCount));

        output.confidenceCoverage =
            observation.keypoint_count == 0
                ? 0.0
                : clamp01(
                    static_cast<double>(confidentCount) /
                    static_cast<double>(
                        observation.keypoint_count));

        const double weightedCount =
            static_cast<double>(motionPairs);
        output.motionEnergyPixels =
            motionPairs == 0
                ? 0.0
                : motionSum / weightedCount;

        output.meanVelocityPixelsPerSecond =
            motionPairs == 0
                ? 0.0
                : velocitySum / weightedCount;
        output.meanAccelerationPixelsPerSecond2 =
            motionPairs == 0
                ? 0.0
                : accelerationSum / weightedCount;
        output.meanJerkPixelsPerSecond3 =
            motionPairs == 0
                ? 0.0
                : jerkSum / weightedCount;
        output.p95InnovationPixels =
            percentile95(innovations, innovationCount);
        output.motionDirectionConsistency =
            motionPairs == 0
                ? 1.0
                : clamp01(directionSum / weightedCount);
        output.microMotionRetention =
            motionPairs == 0
                ? 1.0
                : clamp01(microMotionSum / weightedCount);
        output.subpixelMotionRetention =
            motionPairs == 0
                ? 1.0
                : clamp01(subpixelMotionSum / weightedCount);
        output.dynamicStability =
            motionPairs == 0
                ? 1.0
                : clamp01(stabilitySum / weightedCount);

        output.motionContinuity =
            continuityPairs == 0
                ? 1.0
                : clamp01(
                    continuitySum /
                    static_cast<double>(continuityPairs));

        output.temporalGapPenalty =
            clamp01(
                maximumGapSeconds /
                config_.maximumGapSeconds);

        output.rollingShutterPenalty =
            clamp01(
                static_cast<double>(
                    observation.rolling_shutter_skew_nanos) /
                (config_.rollingShutterReferenceSeconds * 1e9));

        const double effectiveFrameSeconds =
            observedFrameDtSeconds > 0.0
                ? observedFrameDtSeconds
                : expectedIntervalSeconds;

        /*
         * Exposure blur is evaluated in image-space motion, not as a generic exposure penalty.
         * A short exposure during slow motion should remain almost neutral, while the same shutter
         * during fast motion should lose authority. When exposure metadata is unavailable we keep
         * this channel neutral rather than inventing sensor timing.
         */
        const double exposureSeconds =
            static_cast<double>(
                observation.exposure_duration_nanos) /
            1e9;
        const bool exposureMetadataValid =
            std::isfinite(exposureSeconds) &&
            exposureSeconds > 0.0 &&
            exposureSeconds <=
                std::max(
                    effectiveFrameSeconds * 2.0,
                    1.0 / 15.0);
        if (
            exposureMetadataValid &&
            std::isfinite(output.meanVelocityPixelsPerSecond)
        ) {
            const double exposureMotionPixels =
                std::abs(
                    output.meanVelocityPixelsPerSecond) *
                exposureSeconds;
            output.exposureMotionBlurRisk =
                clamp01(
                    exposureMotionPixels /
                    config_.exposureMotionReferencePixels);
        } else {
            output.exposureMotionBlurRisk = 0.0;
        }

        const double coverageScore =
            clamp01(
                output.confidenceCoverage /
                config_.confidenceCoverageTarget);

        const double motionScore =
            std::exp(
                -output.motionEnergyPixels /
                config_.maximumMotionEnergyPixels);

        const double temporalScore =
            clamp01(
                output.motionContinuity *
                (1.0 - output.temporalGapPenalty));

        const double shutterScore =
            1.0 - output.rollingShutterPenalty;

        const double exposureBlurScore =
            1.0 - output.exposureMotionBlurRisk;

        const double innovationScore =
            std::exp(
                -output.p95InnovationPixels /
                std::max(
                    config_.maximumMotionEnergyPixels * 0.18,
                    1.0));
        const double dynamicsScore =
            clamp01(
                output.motionDirectionConsistency * 0.38 +
                output.dynamicStability * 0.32 +
                output.microMotionRetention * 0.30);
        const double jerkGuard =
            std::exp(
                -output.meanJerkPixelsPerSecond3 /
                std::max(
                    config_.maximumMotionEnergyPixels * 18.0,
                    1.0));

        const double regimeProtection =
            std::clamp(
                output.motionDirectionConsistency * 0.45 +
                output.dynamicStability * 0.35 +
                output.microMotionRetention * 0.20,
                0.0,
                1.0);

        output.score = clamp01(
            output.meanConfidence * 0.20 +
            coverageScore * 0.15 +
            motionScore * 0.08 +
            temporalScore * 0.17 +
            shutterScore * 0.05 +
            exposureBlurScore * 0.05 +
            innovationScore * 0.10 +
            dynamicsScore * 0.08 +
            regimeProtection * 0.06 +
            jerkGuard * 0.03 +
            output.subpixelMotionRetention * 0.03);

        output.accepted =
            output.finiteKeypoints >= 1 &&
            output.confidenceCoverage >=
                config_.confidenceCoverageTarget * 0.50 &&
            output.score >= config_.minimumAcceptedScore;

        return output;
    } catch (...) {
        NativeObservationQuality faulted{};
        faulted.cameraId = observation.camera_id;
        faulted.frameId = observation.frame_id;
        faulted.inputKeypoints = observation.keypoint_count;
        faulted.accepted = false;
        faulted.score = 0.0;
        return faulted;
    }
}

void NativeObservationProcessor::reset() noexcept {
    states_.clear();
    expectedIntervalSeconds_.clear();
}

void NativeObservationProcessor::resetCamera(
    std::uint32_t cameraId) noexcept {
    states_.erase(cameraId);
    expectedIntervalSeconds_.erase(cameraId);
}

double NativeObservationProcessor::clamp01(double value) noexcept {
    if (!std::isfinite(value)) {
        return 0.0;
    }
    return std::clamp(value, 0.0, 1.0);
}

bool NativeObservationProcessor::finite(float value) noexcept {
    return std::isfinite(static_cast<double>(value));
}


void NativeObservationProcessor::pruneState(
    std::uint64_t timestampNanos) noexcept {
    const std::uint64_t retentionNanos =
        static_cast<std::uint64_t>(
            config_.stateRetentionSeconds * 1e9);

    for (auto cameraIt = states_.begin();
         cameraIt != states_.end();) {
        auto& cameraStates = cameraIt->second;

        for (auto keypointIt = cameraStates.begin();
             keypointIt != cameraStates.end();) {
            const auto timestamp =
                keypointIt->second.timestampNanos;
            const bool stale =
                timestamp > 0U &&
                timestamp < timestampNanos &&
                timestampNanos - timestamp > retentionNanos;

            if (stale) {
                keypointIt =
                    cameraStates.erase(keypointIt);
            } else {
                ++keypointIt;
            }
        }

        if (
            cameraStates.size() >
            config_.maximumTrackedKeypointsPerCamera) {
            std::vector<
                std::pair<
                    std::uint32_t,
                    KeypointState
                >
            > entries;
            entries.reserve(cameraStates.size());

            for (const auto& entry : cameraStates) {
                entries.push_back(entry);
            }

            std::sort(
                entries.begin(),
                entries.end(),
                [](const auto& lhs, const auto& rhs) {
                    if (
                        lhs.second.confidence !=
                        rhs.second.confidence) {
                        return lhs.second.confidence >
                            rhs.second.confidence;
                    }
                    return lhs.second.timestampNanos >
                        rhs.second.timestampNanos;
                });

            cameraStates.clear();

            const auto keep =
                std::min(
                    config_.maximumTrackedKeypointsPerCamera,
                    entries.size());

            for (std::size_t index = 0;
                 index < keep;
                 ++index) {
                cameraStates.emplace(
                    entries[index].first,
                    entries[index].second);
            }
        }

        if (cameraStates.empty()) {
            expectedIntervalSeconds_.erase(
                cameraIt->first);
            cameraIt = states_.erase(cameraIt);
        } else {
            ++cameraIt;
        }
    }

    if (
        states_.size() <=
        config_.maximumCameraStates) {
        return;
    }

    std::vector<
        std::pair<
            std::uint32_t,
            std::uint64_t
        >
    > cameras;
    cameras.reserve(states_.size());

    for (const auto& camera : states_) {
        std::uint64_t latest = 0U;
        for (const auto& keypoint : camera.second) {
            latest =
                std::max(
                    latest,
                    keypoint.second.timestampNanos);
        }
        cameras.emplace_back(
            camera.first,
            latest);
    }

    std::sort(
        cameras.begin(),
        cameras.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.second > rhs.second;
        });

    for (
        std::size_t index =
            config_.maximumCameraStates;
        index < cameras.size();
        ++index) {
        const auto cameraId =
            cameras[index].first;
        states_.erase(cameraId);
        expectedIntervalSeconds_.erase(
            cameraId);
    }
}

} // namespace nox::native::tracking

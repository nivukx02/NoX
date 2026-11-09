#include <cassert>
#include <cmath>

#include "tracking/native_observation_processor.h"

namespace {

nox::native::MotionObservation observation(
    std::uint32_t camera,
    std::uint64_t frame,
    std::uint64_t timestamp,
    float offset) {
    nox::native::MotionObservation value{};
    value.camera_id = camera;
    value.frame_id = frame;
    value.monotonic_timestamp_nanos = timestamp;
    value.sensor_timestamp_nanos = timestamp;
    value.keypoint_count = 4;
    for (std::uint32_t i = 0; i < value.keypoint_count; ++i) {
        value.keypoint_ids[i] = i;
        value.keypoints[i].x =
            offset + static_cast<float>(i);
        value.keypoints[i].y =
            offset * 0.5f + static_cast<float>(i);
        value.keypoints[i].confidence = 0.95f;
    }
    return value;
}

}

int main() {
    using nox::native::tracking::NativeObservationProcessor;

    NativeObservationProcessor processor;

    const auto first =
        processor.process(observation(1,1,1'000'000'000ull,0.0f));
    assert(first.accepted);
    assert(first.confidentKeypoints == 4u);
    assert(first.confidenceCoverage > 0.9);

    const auto second =
        processor.process(observation(1,2,1'016'666'667ull,2.0f));
    auto semanticFirst =
        observation(2, 10, 2'000'000'000ull, 0.0f);
    semanticFirst.keypoint_ids[0] = 0;
    semanticFirst.keypoint_ids[1] = 1;
    semanticFirst.keypoint_ids[2] = 2;
    semanticFirst.keypoint_ids[3] = 3;
    semanticFirst.keypoints[0].x = 10.0f;
    semanticFirst.keypoints[1].x = 20.0f;
    semanticFirst.keypoints[2].x = 30.0f;
    semanticFirst.keypoints[3].x = 40.0f;

    const auto semanticBaseline =
        processor.process(semanticFirst);
    assert(semanticBaseline.accepted);

    auto semanticReordered = semanticFirst;
    semanticReordered.frame_id = 11;
    semanticReordered.monotonic_timestamp_nanos =
        2'016'666'667ull;

    const auto move =
        semanticReordered.keypoints;

    semanticReordered.keypoint_ids[0] = 2;
    semanticReordered.keypoint_ids[1] = 0;
    semanticReordered.keypoint_ids[2] = 3;
    semanticReordered.keypoint_ids[3] = 1;

    semanticReordered.keypoints[0].x =
        move[2].x + 4.0f;
    semanticReordered.keypoints[1].x =
        move[0].x + 1.0f;
    semanticReordered.keypoints[2].x =
        move[3].x + 3.0f;
    semanticReordered.keypoints[3].x =
        move[1].x + 2.0f;

    const auto semanticResult =
        processor.process(semanticReordered);
    assert(semanticResult.accepted);
    assert(
        semanticResult.motionEnergyPixels > 2.45 &&
        semanticResult.motionEnergyPixels < 2.55);

    auto highRate = observation(
        9,
        1,
        4'000'000'000ull,
        0.0f);
    highRate.frame_interval_nanos =
        8'333'333ull;
    highRate.monotonic_timestamp_nanos =
        4'000'000'000ull;

    auto highRateSecond = highRate;
    highRateSecond.frame_id = 2;
    highRateSecond.sensor_timestamp_nanos =
        4'008'333'333ull;
    highRateSecond.monotonic_timestamp_nanos =
        4'008'333'333ull;
    highRateSecond.frame_interval_nanos =
        8'333'333ull;
    highRateSecond.keypoints[0].x += 1.0f;

    NativeObservationProcessor highRateProcessor;
    static_cast<void>(
        highRateProcessor.process(highRate));
    const auto highRateResult =
        highRateProcessor.process(highRateSecond);
    assert(
        highRateResult.temporalGapPenalty <
        0.20);

    assert(second.accepted);
    assert(std::isfinite(second.motionEnergyPixels));
    assert(second.motionEnergyPixels > 0.0);
    assert(std::isfinite(second.meanVelocityPixelsPerSecond));
    assert(std::isfinite(second.meanAccelerationPixelsPerSecond2));
    assert(std::isfinite(second.meanJerkPixelsPerSecond3));
    assert(second.meanAccelerationPixelsPerSecond2 >= 0.0);
    assert(second.meanJerkPixelsPerSecond3 >= 0.0);
    assert(std::isfinite(second.p95InnovationPixels));
    assert(second.motionDirectionConsistency > 0.45);
    assert(second.dynamicStability > 0.45);
    assert(std::isfinite(second.subpixelMotionRetention));
    assert(second.subpixelMotionRetention >= 0.0);
    assert(second.subpixelMotionRetention <= 1.0);

    {
        NativeObservationProcessor timingProcessor;
        auto firstExposure =
            observation(
                7,
                1,
                1'000'000'000ull,
                0.0f);
        auto secondExposure =
            observation(
                7,
                2,
                1'016'666'667ull,
                5.0f);

        firstExposure.exposure_start_nanos =
            990'000'000ull;
        firstExposure.exposure_duration_nanos =
            20'000'000ull;
        secondExposure.exposure_start_nanos =
            1'010'000'000ull;
        secondExposure.exposure_duration_nanos =
            20'000'000ull;

        static_cast<void>(
            timingProcessor.process(firstExposure));
        const auto midpointTimed =
            timingProcessor.process(secondExposure);

        assert(
            midpointTimed.meanVelocityPixelsPerSecond > 180.0 &&
            midpointTimed.meanVelocityPixelsPerSecond < 320.0);
    }

    /*
     * The same image-space velocity with a longer exposure must lose capture authority.
     * This regression exercises the production optical path instead of a synthetic Python model.
     */
    MotionObservation fastExposure =
        observation(1, 2'000, 1'033'333'334ull, 4.0f);
    assert(fastExposure.keypoint_count == second.inputKeypoints);
    fastExposure.exposure_duration_nanos = 1'000'000ull;
    const autoExposureResult = processor.process(fastExposure);

    MotionObservation longExposure =
        observation(1, 2'001, 1'050'000'001ull, 6.0f);
    longExposure.exposure_duration_nanos = 12'000'000ull;
    const longExposureResult = processor.process(longExposure);

    assert(std::isfinite(autoExposureResult.exposureMotionBlurRisk));
    assert(std::isfinite(longExposureResult.exposureMotionBlurRisk));
    assert(
        longExposureResult.exposureMotionBlurRisk >=
        autoExposureResult.exposureMotionBlurRisk);
    assert(longExposureResult.score <= autoExposureResult.score + 1e-9);

    auto broken = observation(1,3,1'033'333'334ull,4.0f);
    broken.keypoints[2].x = NAN;
    broken.keypoints[2].y = NAN;
    broken.keypoints[3].confidence = 0.01f;

    const auto degraded = processor.process(broken);
    assert(degraded.finiteKeypoints == 3u);
    assert(degraded.confidentKeypoints == 3u);
    assert(degraded.score < second.score);

    const auto gap =
        processor.process(observation(1,4,1'450'000'001ull,8.0f));
    assert(gap.temporalGapPenalty > 0.8);
    assert(gap.score < second.score);
    assert(gap.temporalGapPenalty > second.temporalGapPenalty);

    processor.resetCamera(1);

    /*
     * Stress the bounded camera-state admission path with more unique camera IDs than the
     * configured state budget. Every input remains a real native MotionObservation.
     */
    NativeObservationProcessor::Config boundedConfig{};
    boundedConfig.maximumCameraStates = 4U;
    boundedConfig.pruneEveryFrames = 1U;
    NativeObservationProcessor boundedProcessor(boundedConfig);

    for (std::uint32_t cameraId = 0U; cameraId < 24U; ++cameraId) {
        const auto boundedResult =
            boundedProcessor.process(
                observation(
                    cameraId,
                    10U + cameraId,
                    3'000'000'000ull + cameraId * 16'666'667ull,
                    1.0f));
        assert(boundedResult.finiteKeypoints == 4U);
    }

    const auto reacquired =
        processor.process(observation(1,5,1'466'666'668ull,8.0f));
    assert(reacquired.motionContinuity >= 0.99);

    return 0;
}

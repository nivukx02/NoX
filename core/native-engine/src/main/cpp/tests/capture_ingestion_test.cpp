#include <cassert>
#include <limits>

#include "capture/native_capture_ingestor.h"

using nox::native::MotionObservation;
using nox::native::NativeCaptureIngestor;

static MotionObservation sample(std::uint64_t frame, std::uint64_t timestamp) {
    MotionObservation o{};
    o.camera_id = 1;
    o.frame_id = frame;
    o.sensor_timestamp_nanos = timestamp;
    o.monotonic_timestamp_nanos = timestamp + 1'000;
    return o;
}

int main() {
    NativeCaptureIngestor ingestor(4, 1'000'000ull, 1);

    auto first = ingestor.push(sample(1, 10'000'000ull));
    assert(first.accepted);
    assert(first.epoch == 1u);

    auto duplicate = ingestor.push(sample(1, 10'000'000ull));
    assert(!duplicate.accepted);
    assert(duplicate.duplicate);

    auto ordered = ingestor.push(sample(2, 11'000'000ull));
    assert(ordered.accepted);

    auto gap = ingestor.push(sample(3, 14'000'000ull));
    assert(gap.accepted);
    assert(gap.gap);
    assert(gap.gap_nanos == 2'000'000ull);

    auto out_of_order = ingestor.push(sample(2, 13'000'000ull));
    assert(!out_of_order.accepted);
    assert(out_of_order.out_of_order);

    NativeCaptureIngestor saturated(2, 1'000'000ull, 1);
    const auto a = saturated.push(sample(10, 20'000'000ull));
    const auto b = saturated.push(sample(11, 21'000'000ull));
    assert(a.accepted && !a.overflow);
    assert(b.accepted && !b.overflow);

    const auto newest = saturated.push(sample(12, 22'000'000ull));
    assert(newest.accepted);
    assert(newest.overflow);
    assert(newest.queue_depth == 2u);


    MotionObservation malformed = sample(
        20,
        30'000'000ull);
    malformed.keypoint_count = 2;
    malformed.keypoint_ids[0] = 7;
    malformed.keypoint_ids[1] = 8;
    malformed.keypoints[0] = {
        std::numeric_limits<float>::quiet_NaN(),
        2.0F,
        0.8F,
    };
    malformed.keypoints[1] = {
        2.0F,
        3.0F,
        1.4F,
    };

    assert(!malformed.validate());
    const auto sanitized =
        malformed.sanitize();
    assert(sanitized);
    assert(malformed.validate());
    assert(malformed.keypoints[0].confidence == 0.8F);
    assert(malformed.keypoints[1].confidence == 1.0F);


    MotionObservation popped{};
    assert(saturated.pop(popped));
    assert(popped.frame_id == 11u);
    assert(saturated.pop(popped));
    assert(popped.frame_id == 12u);
    assert(!saturated.pop(popped));

    return 0;
}

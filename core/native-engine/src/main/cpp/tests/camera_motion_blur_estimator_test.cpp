#include <cassert>
#include <cmath>

#include "tracking/camera_motion_blur_estimator.h"

int main() {
    using nox::native::tracking::CameraMotionBlurEstimator;

    const auto short_exposure =
        CameraMotionBlurEstimator{}.estimate(
            1'000'000ULL,
            0.5,
        );

    assert(short_exposure.estimated_blur_radians > 0.0);
    assert(short_exposure.penalty < 0.20);
    assert(short_exposure.confidence > 0.90);

    const auto fast_motion =
        CameraMotionBlurEstimator{}.estimate(
            8'000'000ULL,
            8.0,
        );

    assert(fast_motion.estimated_blur_radians >
           short_exposure.estimated_blur_radians);
    assert(fast_motion.penalty > 0.20);
    assert(fast_motion.penalty <= 1.0);

    const auto invalid =
        CameraMotionBlurEstimator{}.estimate(
            0ULL,
            8.0,
        );

    assert(invalid.penalty == 0.0);
    assert(invalid.confidence == 0.0);

    return 0;
}

#include <cassert>
#include <cmath>
#include <vector>

#include "calibration/rig_stability_monitor.h"

int main() {
    using namespace nox::native::calibration;
    using nox::native::math::Quaterniond;
    using nox::native::math::SE3d;

    const auto identity =
        SE3d{
            Quaterniond{1,0,0,0},
            {0,0,0},
        };

    const auto rotated =
        SE3d{
            Quaterniond::fromRotationVector(
                {0.0004,0.0,0.0}),
            {0.00015,0.0,0.0},
        };

    std::vector<RigPoseSample> samples;
    for (
        std::size_t camera = 0;
        camera < 4U;
        ++camera) {
        samples.push_back({
            1'000'000'000ULL,
            camera,
            identity,
            0.98,
        });
        samples.push_back({
            2'000'000'000ULL,
            camera,
            rotated,
            0.97,
        });
        samples.push_back({
            3'000'000'000ULL,
            camera,
            rotated,
            0.98,
        });
    }

    const auto stable =
        RigStabilityMonitor{}.evaluate(
            samples);

    assert(stable.cameraCount == 4U);
    assert(stable.validCameraCount == 4U);
    assert(stable.maximumTranslationDriftMeters < 0.001);
    assert(stable.maximumRotationDriftRadians < 0.0015);
    assert(!stable.unstable);
    assert(stable.confidence > 0.50);

    samples.back().worldToCamera.translation.x = 0.01;

    const auto unstable =
        RigStabilityMonitor{}.evaluate(
            samples);

    assert(unstable.maximumTranslationDriftMeters > 0.005);
    assert(unstable.unstable);

    return 0;
}

#include <cassert>
#include <cmath>

#include "capture/capture_settings_optimizer.h"

int main() {
    using nox::native::capture::CaptureSettingsOptimizer;

    CaptureSettingsOptimizer optimizer{};

    const auto moderate =
        optimizer.optimize(
            0.5,
            0.8,
            0.9);

    assert(moderate.frameRate == 60.0);
    assert(moderate.exposureMs > 0.0);

    const auto highMotion =
        optimizer.optimize(
            1.0,
            0.97,
            0.95);

    assert(highMotion.frameRate == 120.0);
    assert(highMotion.exposureMs <= 2.5);
    assert(highMotion.motionBlurRisk >= 0.0);
    assert(highMotion.motionBlurRisk <= 1.0);

    const auto thermalLimited =
        optimizer.optimize(
            1.0,
            0.97,
            0.20);

    assert(thermalLimited.frameRate <= 60.0);
    assert(thermalLimited.exposureMs > 0.0);

    const auto invalid =
        optimizer.optimize(
            -1.0,
            3.0,
            -5.0);

    assert(std::isfinite(invalid.exposureMs));
    assert(std::isfinite(invalid.gain));
    assert(std::isfinite(invalid.frameRate));
    assert(invalid.exposureMs > 0.0);
    assert(invalid.gain >= 1.0);
    assert(invalid.frameRate >= 30.0);

    return 0;
}

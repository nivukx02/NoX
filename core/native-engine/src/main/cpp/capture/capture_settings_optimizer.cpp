#include "capture/capture_settings_optimizer.h"

#include <algorithm>
#include <cmath>

namespace nox::native::capture {

namespace {

double clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

} // namespace

CaptureSettings CaptureSettingsOptimizer::optimize(
    double luminance,
    double motionScore,
    double thermalHeadroom) const noexcept {
    const double safeLuminance =
        std::max(
            std::isfinite(luminance) ? luminance : 0.05,
            0.05);
    const double safeMotion =
        clamp01(motionScore);
    const double safeThermal =
        clamp01(thermalHeadroom);

    CaptureSettings output{};

    const double baseExposureMs =
        std::clamp(
            12.0 / safeLuminance,
            1.0,
            20.0);

    /*
     * Moderate motion keeps the established 60 FPS policy. Very high motion escalates to 120 FPS
     * and reduces exposure so the camera spends less of each frame integrating moving subjects.
     * Thermal headroom remains an authority: the optimizer will not force an unstable rate.
     */
    if (safeMotion >= 0.92) {
        output.frameRate = 120.0;
        output.exposureMs =
            std::min(
                baseExposureMs,
                2.5);
    } else {
        output.frameRate =
            safeMotion > 0.7
                ? 60.0
                : 30.0;
        output.exposureMs =
            safeMotion > 0.7
                ? std::min(baseExposureMs, 6.0)
                : std::min(baseExposureMs, 20.0);
    }

    if (safeThermal < 0.25) {
        output.frameRate =
            std::min(
                output.frameRate,
                30.0);
        output.exposureMs =
            std::max(
                output.exposureMs,
                2.0);
    } else if (safeThermal < 0.45) {
        output.frameRate =
            std::min(
                output.frameRate,
                60.0);
    }

    output.gain =
        std::clamp(
            0.8 / safeLuminance,
            1.0,
            safeMotion >= 0.92
                ? 12.0
                : 8.0);

    const double frameDurationMs =
        1000.0 /
            std::max(
                output.frameRate,
                1.0);
    const double exposureFraction =
        output.exposureMs /
            frameDurationMs;

    /*
     * This is an acquisition-policy risk indicator, not a measured optical-blur magnitude.
     * It becomes high when a large fraction of the frame interval is spent exposing under motion.
     */
    const double motionBlurRisk =
        clamp01(
            safeMotion *
            (
                exposureFraction /
                    0.25
            ));

    output.motionBlurRisk =
        motionBlurRisk;

    return output;
}

} // namespace nox::native::capture

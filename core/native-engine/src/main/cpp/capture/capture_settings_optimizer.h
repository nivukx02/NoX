#pragma once

namespace nox::native::capture {

struct CaptureSettings {
    double exposureMs{8.3};
    double gain{1.0};
    double frameRate{30.0};
    double motionBlurRisk{0.0};
};

class CaptureSettingsOptimizer final {
public:
    [[nodiscard]] CaptureSettings optimize(
        double luminance,
        double motionScore,
        double thermalHeadroom) const noexcept;
};

} // namespace nox::native::capture

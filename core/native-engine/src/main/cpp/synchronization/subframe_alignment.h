#pragma once

namespace nox::native::sync {

struct SubframePeak {
    double offset_samples{0.0};
    double curvature{0.0};
    double confidence{0.0};
    bool valid{false};
};

class SubframeAlignment final {
public:
    static double refineParabolicPeak(
        double left,
        double center,
        double right) noexcept;

    static double samplesToNanoseconds(
        double samples,
        double sampleIntervalNanoseconds) noexcept;

    static SubframePeak refinePeak(
        double left,
        double center,
        double right) noexcept;
};

} // namespace nox::native::sync

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nox::native::calibration {

struct BundlePoint {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    bool fixed{false};
};

struct BundleObservation {
    std::uint32_t camera{0};
    BundlePoint observed{};
    double weight{1.0};
    std::size_t pointIndex{
        static_cast<std::size_t>(-1)};
};

struct BundleResult {
    double rms{0.0};
    double p95Error{0.0};
    double maxError{0.0};
    double robustCost{0.0};
    double confidence{0.0};
    double spreadMeters{0.0};
    std::uint32_t iterations{0};
    std::size_t acceptedObservations{0};
    std::size_t rejectedObservations{0};
    bool converged{false};
    bool valid{false};
};

class RobustBundleAdjustment final {
public:
    [[nodiscard]] BundleResult solve(
        std::vector<BundlePoint>&,
        const std::vector<BundleObservation>&,
        std::uint32_t maxIterations = 20) const;

private:
    static double distance(
        const BundlePoint&,
        const BundlePoint&) noexcept;

    static double percentile95(
        std::vector<double>) noexcept;

    static double clamp01(
        double) noexcept;
};

} // namespace nox::native::calibration

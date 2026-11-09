#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nox::native::sync {

struct ClockEstimate {
    double offset_nanos{0.0};
    double drift_ppm{0.0};
    double jitter_rms_nanos{0.0};
    double residual_p95_nanos{0.0};
    double residual_max_nanos{0.0};
    double support_ratio{0.0};
    double outlier_ratio{0.0};
    double slope_uncertainty_ppm{0.0};
    double offset_uncertainty_nanos{0.0};
    double confidence{0.0};
    std::uint32_t iterations{0};
    std::size_t sample_count{0};
    bool valid{false};
};

class ClockModel final {
public:
    [[nodiscard]] ClockEstimate fit(
        const std::vector<std::uint64_t>& reference,
        const std::vector<std::uint64_t>& remote) const;
};

} // namespace nox::native::sync

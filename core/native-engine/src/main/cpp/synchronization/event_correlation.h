#pragma once

#include <vector>

namespace nox::native::sync {

struct CorrelationPeak {
    double offset_samples{0.0};
    double coarse_offset_samples{0.0};
    double confidence{0.0};
    double sharpness{0.0};
    double peak_value{0.0};
    double side_lobe_value{0.0};
    bool valid{false};
};

class EventCorrelation final {
public:
    [[nodiscard]] CorrelationPeak estimate(
        const std::vector<double>&,
        const std::vector<double>&) const;
};

} // namespace nox::native::sync

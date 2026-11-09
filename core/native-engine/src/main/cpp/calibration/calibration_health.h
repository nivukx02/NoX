#pragma once

#include "calibration/calibration_solver.h"

namespace nox::native::calibration {

enum class CalibrationValidity : unsigned char {
    valid = 0,
    degraded = 1,
    invalid = 2,
};

struct CalibrationHealthResult {
    CalibrationValidity validity{CalibrationValidity::invalid};
    double confidence{0.0};
    double residual_rms_px{0.0};
    double p95_camera_rms_px{0.0};
    double image_coverage{0.0};
    double depth_coverage{0.0};
    double observation_coverage{0.0};
    double geometry_confidence{0.0};
    double calibration_completeness{0.0};
    double intrinsic_refinement_health{1.0};
    double intrinsic_refinement_fallback_ratio{0.0};
    double translation_refinement_health{1.0};
    bool calibrated{false};
};

class CalibrationHealth final {
public:
    CalibrationHealthResult evaluate(
        const CalibrationResult& result,
        double degradedResidualPx = 3.0,
        double invalidResidualPx = 8.0) const noexcept;
};

} // namespace nox::native::calibration

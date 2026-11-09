#pragma once
#include <cstdint>
#include "eyes/eye_geometry.h"
namespace nox::native::eyes{
enum GazeFailureFlags : std::uint32_t {
    GAZE_FAILURE_NONE = 0u,
    GAZE_FAILURE_INVALID_EYE = 1u << 0u,
    GAZE_FAILURE_LOW_BINOCULAR_CONFIDENCE = 1u << 1u,
    GAZE_FAILURE_IPD_OUT_OF_RANGE = 1u << 2u,
    GAZE_FAILURE_ANGULAR_DISAGREEMENT = 1u << 3u,
    GAZE_FAILURE_RAY_MISS = 1u << 4u,
    GAZE_FAILURE_LOW_PARALLAX = 1u << 5u,
    GAZE_FAILURE_HIGH_UNCERTAINTY = 1u << 6u,
    GAZE_FAILURE_HEAD_MISALIGNMENT = 1u << 7u,
};

struct GazeTemporalOptions {
    double maximumAngularVelocityDegreesPerSecond{720.0};
    double maximumTemporalInnovationDegrees{18.0};
    double innovationScaleDegrees{4.0};
    double minimumTemporalConfidence{0.30};
    double maximumDtSeconds{0.25};
};

struct GazeState{
Vec3 head_relative{0,0,1};
Vec3 world_relative{0,0,1};
Vec3 fixationPoint{0,0,0};
double fixationDistance{0};
double fixation_uncertainty_meters{1};
double binocular_confidence{0};
double temporal_confidence{0};
double angular_disagreement_degrees{180};
double parallax_degrees{0};
double eye_uncertainty{1};
double interpupillary_distance_meters{0};
double ipd_quality{0};
double vergence_confidence{0};
double angular_consistency{0};
double temporal_consistency{0};
double uncertainty_confidence{0};
double parallax_quality{0};
double angular_velocity_degrees_per_second{0};
double temporal_residual_degrees{0};
std::uint32_t failure_flags{GAZE_FAILURE_NONE};
bool accepted{false};
bool consistent{false};
bool strict_submillimeter_qualified{false};
};
class GazeFusion final{
public:
    GazeState update(
        const EyeGeometry&,
        const EyeGeometry&,
        const Vec3& headForward,
        const Vec3& worldUp) const noexcept;

    GazeState updateTemporal(
        const EyeGeometry&,
        const EyeGeometry&,
        const Vec3& headForward,
        const Vec3& worldUp,
        const GazeState& previous,
        double dtSeconds,
        const GazeTemporalOptions& options = {}) const noexcept;
};}
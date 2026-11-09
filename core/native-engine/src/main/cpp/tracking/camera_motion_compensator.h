#pragma once

#include <vector>

#include "math/se3.h"
#include "tracking/joint_track.h"

namespace nox::native::tracking {

struct CameraMotionContext {
    math::SE3d cameraToWorld{};
    Vec3 linearVelocityWorld{};
    Vec3 angularVelocityWorld{};
    Vec3 linearAccelerationWorld{};
    Vec3 angularAccelerationWorld{};
    double captureOffsetSeconds{0.0};
    double readoutDurationSeconds{0.0};
    double normalizedSensorRow{0.5};
    double positionUncertaintyMeters{0.0};
    double angularUncertaintyRadians{0.0};
    double motionConfidence{1.0};
    /** Measured optical blur risk from the acquisition optimizer, 0 = clean, 1 = unusable. */
    double motionBlurRisk{0.0};
};

struct CameraMotionCompensationResult {
    std::vector<JointObservation> observations;
    double meanTranslationCorrectionMeters{0.0};
    double maximumTranslationCorrectionMeters{0.0};
    double angularCorrectionRadians{0.0};
    double confidence{0.0};
    double meanMotionBlurRisk{0.0};
    bool compensated{false};
};

class CameraMotionCompensator final {
public:
    std::vector<JointObservation> toWorld(
        const std::vector<JointObservation>&,
        const math::SE3d&) const;

    [[nodiscard]] CameraMotionCompensationResult toWorldHighEnd(
        const std::vector<JointObservation>&,
        const CameraMotionContext&) const;
};

} // namespace nox::native::tracking

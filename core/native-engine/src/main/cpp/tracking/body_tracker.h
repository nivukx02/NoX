#pragma once
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include "tracking/body_track_state.h"
#include "tracking/camera_motion_blur_estimator.h"
#include "math/se3.h"
namespace nox::native::tracking{
class BodyTracker final{
public:
    struct QualityEvidence {
        double detectorConfidence{1.0};
        double reprojectionRmsPx{0.0};
        double positionUncertaintyMeters{0.0};
        double syncConfidence{1.0};
        double cameraCoverage{1.0};
        double occlusionRatio{0.0};
        double rollingShutterPenalty{0.0};
        double motionBlurPenalty{0.0};
        bool autoEstimateMotionBlur{true};
        std::uint64_t exposureTimeNanos{0};
        double cameraAngularVelocityRadiansPerSecond{0.0};
        double depthConditioning{1.0};
        double calibrationConfidence{1.0};
        double cameraHealth{1.0};
        std::uint32_t supportingViews{1};
        double uncertaintyBudgetMeters{0.08};
    };
    BodyTrackState update(const std::vector<JointObservation>& observations,std::uint64_t timestampNanos,double trustScale=1.0);
    BodyTrackState predict(std::uint64_t timestampNanos,double trustScale=1.0);
    BodyTrackState updateWithQuality(const std::vector<JointObservation>& observations,std::uint64_t timestampNanos,const QualityEvidence& evidence);
    BodyTrackState updateCameraCompensated(const std::vector<JointObservation>& observations,std::uint64_t timestampNanos,const math::SE3d& cameraToWorld,double trustScale=1.0);
private:
    std::unordered_map<std::uint32_t,std::unique_ptr<JointTrack>> tracks_;
    std::uint64_t subject_id_{1};
    std::uint64_t last_timestamp_nanos_{0};
    CameraMotionBlurEstimator motionBlurEstimator_{};
};
} // namespace nox::native::tracking

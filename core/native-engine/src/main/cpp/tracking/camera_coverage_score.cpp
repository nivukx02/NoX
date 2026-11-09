#include "tracking/camera_coverage_score.h"
#include <algorithm>
namespace nox::native::tracking {
CameraCoverageScore CameraCoverageScoreModel::evaluate(const CameraCoverageInput&i)const noexcept{const double g=std::clamp(i.baselineMeters/0.25,0.0,1.0)*std::clamp(i.viewAngleDegrees/35.0,0.0,1.0);const double t=std::clamp(i.syncConfidence,0.0,1.0)*std::clamp(i.calibrationConfidence,0.0,1.0);const double tr=std::clamp(i.trackedRatio,0.0,1.0);return{.45*g+.30*t+.25*tr,g,t,tr,(.45*g+.30*t+.25*tr)>=.45};}
}
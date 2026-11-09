#pragma once
namespace nox::native::tracking {
struct CameraCoverageInput { double trackedRatio{0}; double baselineMeters{0}; double viewAngleDegrees{0}; double syncConfidence{0}; double calibrationConfidence{0}; };
struct CameraCoverageScore { double score{0}; double geometry{0}; double temporal{0}; double tracking{0}; bool usable{false}; };
class CameraCoverageScoreModel final { public: CameraCoverageScore evaluate(const CameraCoverageInput&) const noexcept; };
}
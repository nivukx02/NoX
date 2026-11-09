#pragma once
#include <vector>
#include "kinematics/skeleton_model.h"
namespace nox::native::optimization {
struct ObjectiveBreakdown {
    double bone{0}, jointLimit{0}, contact{0}, temporal{0},
           reprojection{0}, multiView{0}, faceHead{0}, handWrist{0}, total{0};
};
class ObjectiveTerms final {
public:
    ObjectiveBreakdown evaluate(const kinematics::Pose& candidate,
                                const kinematics::Pose& observed,
                                const std::vector<kinematics::Bone>& bones,
                                double contactFloor=0.0) const;
};
}
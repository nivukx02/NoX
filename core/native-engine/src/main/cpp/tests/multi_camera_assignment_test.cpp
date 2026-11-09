#include <cassert>
#include "tracking/multi_camera_assignment.h"

int main(){
    using namespace nox::native::tracking;

    std::vector<GeometricIdentityTrack> tracks={
        GeometricIdentityTrack{1},
        GeometricIdentityTrack{2},
    };

    std::vector<Vec3> predicted={{0,0,0},{2,0,0}};
    std::vector<DetectionCandidate> detections={
        {11,{.05,0,0},.95,0},
        {12,{2.04,0,0},.9,0},
        {13,{4,0,0},.8,0},
    };
    auto basic=MultiCameraAssignment{}.solve(tracks,predicted,detections);
    assert(basic.size()==2);
    assert(basic[0].detectionId==11);
    assert(basic[1].detectionId==12);

    std::vector<Vec3> crossingPredicted={{0.5,0,0},{0.0,0,0}};
    std::vector<DetectionCandidate> crossingDetections={
        {21,{0.0,0,0},.95,0},
        {22,{1.0,0,0},.95,0},
    };
    auto global=MultiCameraAssignment{}.solve(
        tracks,crossingPredicted,crossingDetections);
    assert(global.size()==2);
    assert(global[0].detectionId==22);
    assert(global[1].detectionId==21);
    return 0;
}

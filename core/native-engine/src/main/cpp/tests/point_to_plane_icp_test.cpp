#include <cassert>
#include <cmath>
#include "scene/point_to_plane_icp.h"

int main(){
    using namespace nox::native;
    std::vector<scene::IcpCorrespondence> correspondences;
    for(int i=0;i<32;++i){
        const double x=(i%8)*0.1;
        const double y=(i/8)*0.1;
        correspondences.push_back({
            {x,y,1.0},
            {x+0.02,y-0.01,1.08},
            {0,0,1},
            1.0,
        });
    }
    scene::IcpOptions options;
    options.iterations=40;
    options.robustScale=0.02;
    options.maxCorrespondenceDistance=0.5;
    const auto result=scene::PointToPlaneIcp{}.align(correspondences,{},options);
    assert(result.initialRms>0.05);
    assert(result.finalRms<=result.initialRms);
    assert(result.inliers>20);
    assert(result.confidence>0.5);
    return 0;
}

#include <cassert>
#include <cmath>
#include "eyes/binocular_vergence.h"

int main(){
    using namespace nox::native::eyes;
    EyeGeometry left{
        {-0.03,0,0},
        {-0.03,0,0},
        {0,0,1},
        0.95,
        true,
    };
    EyeGeometry right{
        {0.03,0,0},
        {0.03,0,0},
        {0,0,1},
        0.95,
        true,
    };
    auto parallel=BinocularVergenceSolver{}.solve(left,right);
    assert(!parallel.valid);

    left.gaze={0.03,0,1};
    right.gaze={-0.03,0,1};
    auto fixation=BinocularVergenceSolver{}.solve(left,right);
    assert(fixation.valid);
    assert(fixation.distance>0.5);
    assert(fixation.confidence>0.5);
    assert(std::isfinite(fixation.point.z));
    assert(fixation.convergenceAngleDegrees > 0.0);
    assert(fixation.parallaxDegrees > 0.0);
    assert(fixation.conditioning > 0.02);
    assert(fixation.uncertaintyMeters > 0.0);
    assert(fixation.depthAsymmetry >= 0.0);
    assert(fixation.depthAsymmetry < 0.20);
    assert(fixation.rayAlignmentConfidence >= 0.0);
    assert(fixation.rayAlignmentConfidence <= 1.0);
    assert(fixation.angularResidualDegrees >= 0.0);
    return 0;
}

#include <cassert>
#include <limits>
#include "reconstruction/missing_data_recovery.h"
int main(){
    using namespace nox::native::recovery;
    Track t{{0,0,0},{1,0,0},0.9,0.1};
    t.acceleration = {2.0,0.0,0.0};
    t.jerk = {1.0,0.0,0.0};
    auto shortGap=MissingDataRecovery{}.predict(t,0.05,0.2);
    assert(shortGap.observation.state==nox::native::occlusion::State::Predicted);
    assert(shortGap.track.position.x>0.04);
    assert(shortGap.track.position.x<0.08);
    assert(shortGap.track.position.x<0.08);
    assert(shortGap.track.confidence<0.9);
    assert(shortGap.track.uncertainty>0.1);
    auto longGap=MissingDataRecovery{}.predict(t,0.25,0.2);
    assert(longGap.observation.state==nox::native::occlusion::State::Unsupported);
    assert(longGap.track.confidence==0.0);

    Track invalid = t;
    invalid.velocity.x =
        std::numeric_limits<double>::quiet_NaN();
    auto invalidResult =
        MissingDataRecovery{}.predict(
            invalid,
            0.05,
            0.2);
    assert(
        invalidResult.observation.state ==
        nox::native::occlusion::State::Unsupported);
    assert(invalidResult.track.confidence == 0.0);
}
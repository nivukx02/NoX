#include <cassert>
#include <limits>
#include "replay/frontier_replay_contract.h"
using nox::native::replay::FrontierReplayContract;
using nox::native::replay::ReplayManifest;
using nox::native::replay::ReplayResult;
int main() {
    const FrontierReplayContract contract{};
    const ReplayManifest manifest{8U,12000U,32.0,1.4,0.22,12U,0,17U};
    const ReplayResult result{1234U,0.96,0.94,0.88,5U,true};
    const auto qualified = contract.qualify(manifest,result);
    assert(qualified.manifestValid);
    assert(qualified.outputValid);
    assert(qualified.bounded);
    assert(qualified.qualificationReady);
    assert(qualified.blocker.empty());
    const auto repeat = contract.qualify(manifest,result);
    assert(repeat.digest == qualified.digest);
    const auto invalid = contract.qualify(
        ReplayManifest{129U,1U,1.0,1.0,0.0,0U,0,1U}, result);
    assert(!invalid.qualificationReady);
    const auto drifted = contract.qualify(
        ReplayManifest{8U,12000U,180.0,1.4,0.22,12U,0,17U}, result);
    assert(!drifted.qualificationReady);
    return 0;
}

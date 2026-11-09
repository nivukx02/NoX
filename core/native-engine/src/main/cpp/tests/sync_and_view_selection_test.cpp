#include <cassert>
#include <vector>
#include "synchronization/audio_event_sync.h"
#include "synchronization/timecode_clock.h"
#include "tracking/camera_coverage_score.h"
#include "tracking/adaptive_view_selector.h"
#include "tracking/multi_person_interaction.h"

int main() {
    std::vector<double> reference(64,0.0);
    std::vector<double> remote(64,0.0);
    for(int i=10;i<20;++i){
        reference[i]=1.0;
        remote[i+3]=1.0;
    }
    const auto audio=nox::native::sync::AudioEventSync{}.estimate(reference,remote);
    assert(audio.valid);

    nox::native::sync::Timecode tc{};
    assert(nox::native::sync::TimecodeClock::parse("00:00:01:12",tc));
    assert(tc.seconds==1);

    const auto score=nox::native::tracking::CameraCoverageScoreModel{}.evaluate(
        {0.9,0.4,50.0,0.95,0.95});
    assert(score.usable);

    const auto views=nox::native::tracking::AdaptiveViewSelector{}.select(
        {{1,0.9,0.9,5.0,false},{2,0.4,0.8,30.0,true},{3,0.7,0.3,0.4,true}},2);
    assert(views.selected.size()==2);

    const auto interactions=
        nox::native::tracking::MultiPersonInteraction{}.evaluate({
            {1,0,0,0,0.95,0.2,0,0},
            {2,0.22,0,0,0.95,-0.2,0,0},
            {3,2,0,0,0.95,0,0,0},
        });
    assert(interactions.size()==1);
    assert(interactions[0].approachRate<0.0);
    assert(interactions[0].contactProbability>0.35);
    assert(interactions[0].interacting);
    return 0;
}

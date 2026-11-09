#include <cassert>
#include <vector>
#include "calibration/global_camera_graph_optimizer.h"
#include "triangulation/reprojection_consensus.h"
#include "tracking/confidence_keypoint_sampler.h"
#include "reconstruction/rigid_metric_constraint.h"
#include "synchronization/audio_event_sync.h"
#include "synchronization/timecode_clock.h"
#include "tracking/camera_coverage_score.h"
#include "tracking/adaptive_view_selector.h"
#include "tracking/multi_person_interaction.h"
#include "face/surface_visibility.h"

int main() {
    using namespace nox::native;
    const auto graph=calibration::GlobalCameraGraphOptimizer{}.solve(
        2,{{"C0","C1",{1,0,0},1}});
    assert(graph.poses.size()==2);

    const auto consensus=triangulation::ReprojectionConsensus{}.evaluate({0.5,0.6,0.4,8.0,12.0});
    assert(consensus.confidence>=0.0);

    const auto sampled=tracking::ConfidenceKeypointSampler{}.select(
        {{1,.1,.1,.8},{2,.11,.1,.7},{3,.8,.8,.95},{4,.8,.81,.9}},2);
    assert(sampled.size()==2);

    const auto scale=metric::RigidMetricConstraint{}.solve({{1,2,1},{2,4,1}});
    assert(scale.valid);

    std::vector<double> reference(64,0.0);
    std::vector<double> remote(64,0.0);
    for(int i=10;i<20;++i){reference[i]=1.0;remote[i+3]=1.0;}
    assert(sync::AudioEventSync{}.estimate(reference,remote).valid);

    sync::Timecode tc{};
    assert(sync::TimecodeClock::parse("01:02:03:12",tc));

    const auto coverage=tracking::CameraCoverageScoreModel{}.evaluate({.9,.4,50,.95,.95});
    assert(coverage.usable);

    const auto views=tracking::AdaptiveViewSelector{}.select(
        {{1,.9,.9,5,false},{2,.8,.8,30,true}},1);
    assert(views.selected.size()==1);

    const auto interactions=tracking::MultiPersonInteraction{}.evaluate({
        {1,0,0,0,.9},{2,.1,0,0,.9}
    });
    assert(!interactions.empty()&&interactions.front().interacting);

    const auto surface=face::SurfaceVisibility{}.evaluate(
        {{0,0,0,0,0,1,.8}},0,0,1);
    assert(surface.front().visible);
    return 0;
}

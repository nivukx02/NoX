#include <cassert>
#include <vector>
#include "calibration/global_camera_graph_optimizer.h"
#include "triangulation/reprojection_consensus.h"
#include "tracking/confidence_keypoint_sampler.h"
#include "reconstruction/rigid_metric_constraint.h"

int main() {
    using namespace nox::native;
    const auto graph=calibration::GlobalCameraGraphOptimizer{}.solve(
        3,
        {{"C0","C1",{1,0,0},1},{"C1","C2",{1,0,0},1}});
    assert(graph.poses.size()==3);

    const auto consensus=triangulation::ReprojectionConsensus{}.evaluate({1.0,1.4,8.0});
    assert(consensus.confidence>=0.0);

    const auto sampled=tracking::ConfidenceKeypointSampler{}.select(
        {{1,0.1,0.1,0.9,0},{2,0.8,0.2,0.8,1},{3,0.2,0.8,1.0,2}},2);
    assert(sampled.size()==2);

    const auto temporal=tracking::ConfidenceKeypointSampler{}.selectSpatiotemporal(
        {{10,0.1,0.1,0.99,0},{11,0.11,0.1,0.98,0},{12,0.1,0.1,0.97,1},
         {13,0.8,0.8,0.96,2},{14,0.8,0.8,0.95,3}},4,2,2,4);
    assert(temporal.size()==4);
    assert(temporal[0].id==10);
    assert(temporal[1].id==12);
    assert(temporal[2].id==13);

    const auto metric=metric::RigidMetricConstraint{}.solve({
        {1.0,1.0,1.0},{2.0,2.0,1.0}
    });
    assert(metric.valid);
    assert(metric.scale>0.9&&metric.scale<1.1);
    return 0;
}

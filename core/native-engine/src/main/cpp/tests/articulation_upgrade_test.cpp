#include <cassert>
#include <vector>
#include "tracking/multi_person_interaction.h"
#include "face/surface_visibility.h"
#include "kinematics/skeleton_retargeter.h"
#include "kinematics/foot_lock_solver.h"

int main() {
    const std::vector<nox::native::tracking::SubjectState> subjects={
        {1,0,0,0,0.9},{2,0.1,0,0,0.9}
    };
    const auto interaction=nox::native::tracking::MultiPersonInteraction{}.evaluate(subjects);
    assert(!interaction.empty());
    assert(interaction.front().interacting);

    const std::vector<nox::native::face::SurfaceVertex> vertices={
        {0,0,0,0,0,1,0.9}
    };
    const auto surface=nox::native::face::SurfaceVisibility{}.evaluate(vertices,0,0,1);
    assert(surface.front().visible);

    const auto retarget=nox::native::kinematics::SkeletonRetargeter{}.solve(
        {1,1},{2,2},{{0,0,2},{1,1,2}});
    assert(retarget.valid);
    assert(retarget.weights.size()==2);
    assert(retarget.globalScale>1.9);
    assert(retarget.mappingCoverage==1.0);

    const auto robust=nox::native::kinematics::SkeletonRetargeter{}.solve(
        {1,1,1},
        {2,2,4},
        {{0,0,2},{1,1,2},{2,2,2}});
    assert(robust.mappingCoverage==1.0);
    assert(robust.globalScale>1.9&&robust.globalScale<2.1);

    const auto foot=nox::native::kinematics::FootLockSolver{}.solve(
        0.01,0.20,0.01,0.20);
    assert(foot.leftLocked);
    assert(!foot.rightLocked);

    const auto anchored=
        nox::native::kinematics::FootLockSolver{}.updateAnchored(
            foot,
            {0,0.01,0},
            {1,0.20,0},
            0.01,0.20,0.01,0.20);
    assert(anchored.leftLocked);
    assert(anchored.leftHasAnchor);
    assert(anchored.leftAnchorDrift<1e-9);

    const auto moved=
        nox::native::kinematics::FootLockSolver{}.updateAnchored(
            anchored,
            {0.02,0.01,0},
            {1,0.20,0},
            0.06,0.20,0.01,0.20);
    assert(moved.leftLocked);
    assert(moved.leftAnchorDrift>0.01);

    const auto released=
        nox::native::kinematics::FootLockSolver{}.updateAnchored(
            moved,
            {0.02,0.08,0},
            {1,0.20,0},
            0.12,0.20,0.08,0.20);
    assert(!released.leftLocked);
    assert(!released.leftHasAnchor);
    return 0;
}

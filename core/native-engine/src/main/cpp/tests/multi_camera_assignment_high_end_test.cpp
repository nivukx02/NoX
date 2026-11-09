#include <cassert>
#include <vector>

#include "tracking/multi_camera_assignment.h"

int main() {
    using namespace nox::native::tracking;

    const std::vector<GeometricIdentityTrack> tracks{
        GeometricIdentityTrack{1},
        GeometricIdentityTrack{2},
    };

    const std::vector<Vec3> predicted{
        {0.0,0.0,1.0},
        {1.0,0.0,1.0},
    };

    const std::vector<Vec3> predictedVelocity{
        {0.40,0.00,0.00},
        {-0.35,0.00,0.00},
    };

    const std::vector<DetectionCandidate> detections{
        {
            101,
            {0.015,0.002,1.001},
            0.97,
            1.0 / 120.0,
            {0.39,0.01,0.0},
            0.96,
            0.95,
            0.94,
            0.93,
            1.01,
            0.96,
            0.04,
        },
        {
            102,
            {1.012,-0.003,0.999},
            0.96,
            1.0 / 120.0,
            {-0.34,0.02,0.0},
            0.94,
            0.93,
            0.92,
            0.91,
            0.99,
            0.95,
            0.03,
        },
        {
            103,
            {0.12,0.05,1.05},
            0.89,
            0.12,
            {-0.80,0.60,0.0},
            0.20,
            0.15,
            0.22,
            0.18,
            1.70,
            0.35,
            0.85,
        },
    };

    const auto assignments =
        MultiCameraAssignment{}.solveHighEnd(
            tracks,
            predicted,
            predictedVelocity,
            detections);

    assert(assignments.size() == 2U);
    assert(assignments[0].detectionId == 101U);
    assert(assignments[1].detectionId == 102U);
    assert(assignments[0].accepted);
    assert(assignments[1].accepted);
    assert(assignments[0].confidence > 0.50);
    assert(assignments[1].confidence > 0.50);

    return 0;
}

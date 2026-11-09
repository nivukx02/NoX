#include <cassert>
#include <cmath>
#include <vector>

#include "face/face_surface_alignment.h"

int main() {
    using namespace nox::native::face;

    FaceGeometryState previous;
    previous.timestamp_nanos = 1'000'000'000ULL;
    previous.head_center = {0.0,0.0,1.0};
    previous.head_forward = {0.0,0.0,1.0};
    previous.confidence = 0.90;
    previous.mean_landmark_uncertainty = 0.002;
    previous.conditioning_score = 0.88;
    previous.robust_support_ratio = 0.92;
    previous.metric = true;

    previous.landmarks = {
        {1,{0.00,0.00,1.00},0.95,0.001,0.90,6,true},
        {2,{0.03,0.00,1.00},0.94,0.001,0.91,6,true},
        {3,{0.00,0.03,1.00},0.93,0.001,0.89,6,true},
        {4,{0.02,0.02,1.02},0.92,0.001,0.87,5,true},
    };

    FaceGeometryState current = previous;
    current.timestamp_nanos += 16'000'000ULL;
    current.head_center = {0.002,0.0,1.001};
    current.confidence = 0.92;

    current.landmarks[0].position = {0.002,0.0,1.001};
    current.landmarks[1].position = {0.031,0.001,1.001};
    current.landmarks[2].position = {-0.001,0.031,1.000};
    current.landmarks[3].position = {0.021,0.021,1.021};

    const auto consistency =
        FaceSurfaceAlignment::shapeConsistency(
            current);
    assert(consistency >= 0.0);
    assert(consistency <= 1.0);
    assert(consistency > 0.20);

    const auto stabilized =
        FaceSurfaceAlignment::stabilize(
            previous,
            current,
            0.35);

    assert(stabilized.landmarks.size() == current.landmarks.size());
    assert(stabilized.confidence >= 0.0);
    assert(stabilized.confidence <= 1.0);
    assert(stabilized.conditioning_score >= 0.0);
    assert(stabilized.conditioning_score <= 1.0);
    assert(stabilized.robust_support_ratio >= 0.0);
    assert(stabilized.robust_support_ratio <= 1.0);
    assert(stabilized.metric);
    assert(
        stabilized.landmarks.front().uncertainty >=
        0.0);
    assert(
        stabilized.landmarks.front().innovation_magnitude >=
        0.0);
    assert(
        stabilized.landmarks.front().temporal_confidence >=
        0.0);
    assert(
        stabilized.landmarks.front().temporal_confidence <=
        1.0);
    assert(
        stabilized.micro_expression_retention >=
        0.0);
    assert(
        stabilized.micro_expression_retention <=
        1.0);
    assert(
        stabilized.dynamic_stability >=
        0.0);
    assert(
        stabilized.dynamic_stability <=
        1.0);

    return 0;
}

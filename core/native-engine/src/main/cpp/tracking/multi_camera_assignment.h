#pragma once
#include <cstdint>
#include <vector>
#include "tracking/camera_visibility_graph.h"
#include "tracking/geometric_identity_track.h"

namespace nox::native::tracking {
struct DetectionCandidate {
    std::uint64_t detectionId{0};
    Vec3 position{};
    double confidence{0.0};
    double timestampDelta{0.0};
    Vec3 velocity{};
    double appearanceSimilarity{1.0};
    double faceConsistency{1.0};
    double handConsistency{1.0};
    double contactConsistency{1.0};
    double scaleRatio{1.0};
    double visibility{1.0};
    double occlusionRisk{0.0};
};
struct Assignment {
    std::uint64_t trackId{0};
    std::uint64_t detectionId{0};
    double cost{0.0};
    double confidence{0.0};
    double ambiguityMargin{0.0};
    bool accepted{false};
};
class MultiCameraAssignment final {
public:
    std::vector<Assignment> solve(
        const std::vector<GeometricIdentityTrack>& tracks,
        const std::vector<Vec3>& predicted,
        const std::vector<DetectionCandidate>& detections,
        const CameraVisibilityGraph* visibility = nullptr) const;

    std::vector<Assignment> solveHighEnd(
        const std::vector<GeometricIdentityTrack>& tracks,
        const std::vector<Vec3>& predicted,
        const std::vector<Vec3>& predictedVelocity,
        const std::vector<DetectionCandidate>& detections,
        const CameraVisibilityGraph* visibility = nullptr) const;
};
}

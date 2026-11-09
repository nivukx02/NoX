#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nox::native::triangulation {
struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};
};
struct RayObservation {
    Vec3 origin{};
    Vec3 direction{0.0,0.0,1.0};
    double weight{1.0};
    double uncertaintyScale{1.0};
    std::uint32_t cameraId{0};
    double rollingShutterConfidence{1.0};
    double metricScaleConfidence{1.0};
    double temporalAlignmentConfidence{1.0};
    double crossViewIndependence{1.0};
    double depthConditioningConfidence{1.0};
};
struct TriangulationResult {
    Vec3 point{};
    double reprojectionRms{0.0};
    double conditioning{0.0};
    double confidence{0.0};
    double covariance{1.0};
    std::array<double,9> covarianceMatrix{
        1.0,0.0,0.0,
        0.0,1.0,0.0,
        0.0,0.0,1.0,
    };
    double covarianceTrace{3.0};
    double covarianceP95{3.0};
    double robustInlierRatio{0.0};
    double angularDiversity{0.0};
    double cheiralityConfidence{0.0};
    double minimumAngularSeparationRadians{0.0};
    double meanAngularSeparationRadians{0.0};
    double baselineMeters{0.0};
    double depthConditioning{0.0};
    double metricScaleConfidence{0.0};
    double residualScale{0.0};
    double reprojectionConsensusConfidence{0.0};
    double reprojectionConsensusMedian{0.0};
    double reprojectionConsensusMad{0.0};
    std::uint32_t contributingRays{0};
    std::uint32_t rejectedRays{0};
    std::uint32_t distinctCameraCount{0};
    double viewIndependenceScore{0.0};
    bool cameraIdentityUnknown{true};
    bool valid{false};
};
class WeightedTriangulator final {
public:
    TriangulationResult solve(const std::vector<RayObservation>&) const;
};
}

#pragma once

#include <cstdint>

namespace nox::native::optimization {

enum class QualityPolicy {
    Accept,
    Hold,
    Degrade,
    Reject,
};

enum class QualityBottleneck {
    None,
    Reconstruction,
    Geometry,
    DirectionalUncertainty,
    Temporal,
    Synchronization,
    Calibration,
    Tracking,
    FaceEye,
    RigCoverage,
    RigStability,
    Metrology,
    MetricAuthority,
    EpipolarAuthority,
    Physics,
    Surface,
    Performance,
    Thermal,
    Memory,
};

struct ProductionQualityInput {
    double reconstructionConfidence{1.0};
    double geometricConfidence{1.0};
    double temporalConfidence{1.0};
    double synchronizationConfidence{1.0};
    double calibrationConfidence{1.0};
    double trackingConfidence{1.0};
    double faceEyeConfidence{1.0};
    double faceEyeOutOfDistributionRisk{0.0};
    double faceEyeReliabilityTrend{0.0};
    double global4DScore{1.0};
    double global4DRootMotionConsistency{1.0};
    double global4DSceneConsistency{0.72};
    double global4DDirectionalUncertaintyConsistency{1.0};
    double global4DFaceEyeConsistency{1.0};
    double bodyMetricEvidenceAuthority{1.0};
    double faceMetricEvidenceAuthority{1.0};
    double handMetricEvidenceAuthority{1.0};
    double bodyEpipolarConsistency{1.0};
    double faceEpipolarConsistency{1.0};
    double handEpipolarConsistency{1.0};
    double directionalUncertaintyMajor95Meters{0.0};
    double directionalUncertaintyDepthMeters{0.0};
    double directionalUncertaintyAnisotropy{1.0};
    double directionalUncertaintyConfidence{1.0};
    double directionalUncertaintyDegradedJointCount{0.0};
    bool global4DAccepted{true};
    bool global4DPersistentFailure{false};
    double physicsConsistencyScore{1.0};
    double globalRigTrust{1.0};
    double rigStabilityConfidence{1.0};
    double rigTranslationDriftMeters{0.0};
    double rigRotationDriftRadians{0.0};
    bool rigStabilityRequired{false};
    bool rigUnstable{false};
    double dynamicMetrologyP95Millimeters{0.0};
    double dynamicMetrologyRmseMillimeters{0.0};
    double dynamicMetrologyMaxErrorMillimeters{0.0};
    double dynamicMetrologyConfidence{1.0};
    bool dynamicMetrologyRequired{false};
    double gaussian4DTemporalStability{1.0};
    double gaussian4DTrustedRatio{1.0};
    double gaussian4DNodeConfidence{1.0};
    double gaussian4DBindingConfidence{1.0};
    double gaussian4DP95StrainRatio{0.0};
    double gaussian4DMaximumStrainRatio{0.0};
    bool gaussian4DEvidenceAvailable{false};
    bool dynamicMetrologyValid{true};
    bool faceEyeFusionAccepted{true};
    bool physicsConstraintAccepted{true};
    double surfaceConfidence{1.0};
    double identitySwapRisk{0.0};
    double occlusionRisk{0.0};
    double cameraMotionRisk{0.0};
    double meanUncertaintyMeters{0.0};
    double uncertaintyLimitMeters{0.05};
    double reprojectionErrorPixels{0.0};
    double reprojectionLimitPixels{3.0};
    double thermalHeadroom{1.0};
    double memoryHeadroom{1.0};
    double runtimeQualityScale{1.0};
    double runtimeInferenceScale{1.0};
    double gpuDispatchMilliseconds{0.0};
    double qualityFloor{0.30};
    double acceptThreshold{0.76};
    double holdThreshold{0.52};
    bool hardFailure{false};
};

struct ProductionQualityBudget {
    std::int32_t detectorIntervalFrames{3};
    std::int32_t targetInferenceFps{30};
    std::int32_t denseMeshHz{10};
    std::int32_t eyeTrackingHz{30};
    std::int32_t handTrackingHz{24};
    std::int32_t bundleAdjustmentIterations{8};
    std::int32_t tsdfStride{2};
    std::int32_t meshVertexBudget{150000};
    bool allowDenseSurface{true};
    bool allowReferenceOptimization{false};
};

struct ProductionQualityDecision {
    QualityPolicy policy{QualityPolicy::Hold};
    QualityBottleneck bottleneck{QualityBottleneck::None};
    double rawScore{0.0};
    double filteredScore{0.0};
    double pressure{0.0};
    bool outputAllowed{false};
    bool hardFailure{false};
    ProductionQualityBudget budget{};
};

struct ProductionQualityConfig {
    double geometryWeight{1.20};
    double calibrationWeight{1.20};
    double synchronizationWeight{1.18};
    double temporalWeight{1.05};
    double reconstructionWeight{1.05};
    double trackingWeight{1.00};
    double uncertaintyWeight{1.05};
    double surfaceWeight{0.62};
    double runtimeWeight{0.70};
    double thermalWeight{0.58};
    double memoryWeight{0.50};
    double smoothingGain{0.28};
    double holdHysteresis{0.035};
    double rejectHysteresis{0.065};
    double minimumOutputScore{0.30};
    double qualityFloor{0.30};
    double acceptThreshold{0.76};
    double holdThreshold{0.52};
    double faceEyeWeight{1.05};
    double faceEyeOodWeight{0.90};
    double faceEyeTrendWeight{0.55};
    double global4DWeight{1.08};
    double global4DDirectionalWeight{0.72};
    double global4DFaceEyeWeight{0.64};
    double directionalUncertaintyWeight{0.88};
    double physicsWeight{1.00};
    double globalRigTrustWeight{1.04};
    double rigStabilityWeight{1.10};
    double metrologyWeight{1.08};
    double gaussian4DWeight{1.02};
    double gaussianStrainWeight{1.05};
    double metricAuthorityWeight{0.72};
    double epipolarAuthorityWeight{0.88};
    double maximumGaussianP95StrainRatio{0.35};
    double maximumGaussianMaximumStrainRatio{0.75};
    double minimumGaussianTrustedRatio{0.55};
};

class ProductionQualityGate final {
public:
    explicit ProductionQualityGate(ProductionQualityConfig config = {}) noexcept;

    [[nodiscard]] ProductionQualityDecision evaluate(
        const ProductionQualityInput& input) noexcept;

    void reset() noexcept;

private:
    ProductionQualityConfig config_{};
    double filteredScore_{0.0};
    bool initialized_{false};
    QualityPolicy previousPolicy_{QualityPolicy::Hold};

    static double sanitize(double value) noexcept;
    static double clamp01(double value) noexcept;
    static double inverseError(double value, double limit) noexcept;
    static double faceEyeTrendQuality(double trend) noexcept;
    static double componentTracking(
        double trackingConfidence,
        double identitySwapRisk,
        double occlusionRisk,
        double cameraMotionRisk) noexcept;
    static double weightedGeometricMean(
        const ProductionQualityInput& input,
        const ProductionQualityConfig& config) noexcept;

    QualityPolicy classify(
        double score,
        const ProductionQualityInput& input,
        double pressure) const noexcept;

    static QualityBottleneck bottleneck(
        const ProductionQualityInput& input) noexcept;

    static ProductionQualityBudget budget(
        QualityPolicy policy,
        double score,
        double pressure,
        const ProductionQualityInput& input) noexcept;
};

} // namespace nox::native::optimization

#include "optimization/production_quality_gate.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace nox::native::optimization {

namespace {

struct WeightedValue {
    double value;
    double weight;
};

double positiveOrOne(double value) noexcept {
    return std::max(value, 1e-6);
}

double weightedGeometricMeanTerms(
    const std::initializer_list<WeightedValue>& values) noexcept {
    double totalWeight = 0.0;
    double weightedLog = 0.0;

    for (const auto& item : values) {
        const double weight = std::max(item.weight, 0.0);
        const double value = std::clamp(item.value, 0.015, 1.0);
        totalWeight += weight;
        weightedLog += weight * std::log(value);
    }

    if (totalWeight <= 1e-9) {
        return 0.0;
    }

    return std::clamp(
        std::exp(weightedLog / totalWeight),
        0.0,
        1.0);
}

} // namespace

ProductionQualityGate::ProductionQualityGate(
    ProductionQualityConfig config) noexcept
    : config_(config) {
    config_.geometryWeight = std::max(
        sanitize(config_.geometryWeight),
        0.01);
    config_.calibrationWeight = std::max(
        sanitize(config_.calibrationWeight),
        0.01);
    config_.synchronizationWeight = std::max(
        sanitize(config_.synchronizationWeight),
        0.01);
    config_.temporalWeight = std::max(
        sanitize(config_.temporalWeight),
        0.01);
    config_.reconstructionWeight = std::max(
        sanitize(config_.reconstructionWeight),
        0.01);
    config_.trackingWeight = std::max(
        sanitize(config_.trackingWeight),
        0.01);
    config_.uncertaintyWeight = std::max(
        sanitize(config_.uncertaintyWeight),
        0.01);
    config_.surfaceWeight = std::max(
        sanitize(config_.surfaceWeight),
        0.01);
    config_.runtimeWeight = std::max(
        sanitize(config_.runtimeWeight),
        0.01);
    config_.thermalWeight = std::max(
        sanitize(config_.thermalWeight),
        0.01);
    config_.memoryWeight = std::max(
        sanitize(config_.memoryWeight),
        0.01);

    config_.smoothingGain = std::clamp(
        sanitize(config_.smoothingGain),
        0.05,
        0.95);
    config_.holdHysteresis = std::clamp(
        sanitize(config_.holdHysteresis),
        0.0,
        0.20);
    config_.rejectHysteresis = std::clamp(
        sanitize(config_.rejectHysteresis),
        0.0,
        0.25);
    config_.minimumOutputScore = std::clamp(
        sanitize(config_.minimumOutputScore),
        0.0,
        1.0);
    config_.faceEyeWeight = std::max(
        sanitize(config_.faceEyeWeight),
        0.01);
    config_.faceEyeOodWeight = std::max(
        sanitize(config_.faceEyeOodWeight),
        0.01);
    config_.faceEyeTrendWeight = std::max(
        sanitize(config_.faceEyeTrendWeight),
        0.01);
    config_.global4DWeight = std::max(
        sanitize(config_.global4DWeight),
        0.01);
    config_.global4DDirectionalWeight = std::max(
        sanitize(config_.global4DDirectionalWeight),
        0.01);
    config_.global4DFaceEyeWeight = std::max(
        sanitize(config_.global4DFaceEyeWeight),
        0.01);
    config_.directionalUncertaintyWeight = std::max(
        sanitize(config_.directionalUncertaintyWeight),
        0.01);
    config_.physicsWeight = std::max(
        sanitize(config_.physicsWeight),
        0.01);
    config_.globalRigTrustWeight = std::max(
        sanitize(config_.globalRigTrustWeight),
        0.01);
    config_.rigStabilityWeight = std::max(
        sanitize(config_.rigStabilityWeight),
        0.01);
    config_.metrologyWeight = std::max(
        sanitize(config_.metrologyWeight),
        0.01);
    config_.gaussian4DWeight = std::max(
        sanitize(config_.gaussian4DWeight),
        0.01);
    config_.gaussianStrainWeight = std::max(
        sanitize(config_.gaussianStrainWeight),
        0.01);
    config_.metricAuthorityWeight = std::max(
        sanitize(config_.metricAuthorityWeight),
        0.01);
    config_.epipolarAuthorityWeight = std::max(
        sanitize(config_.epipolarAuthorityWeight),
        0.01);
    config_.maximumGaussianP95StrainRatio = std::max(
        sanitize(config_.maximumGaussianP95StrainRatio),
        0.01);
    config_.maximumGaussianMaximumStrainRatio = std::max(
        sanitize(config_.maximumGaussianMaximumStrainRatio),
        config_.maximumGaussianP95StrainRatio);
    config_.minimumGaussianTrustedRatio = std::clamp(
        sanitize(config_.minimumGaussianTrustedRatio),
        0.0,
        1.0);
}

ProductionQualityDecision ProductionQualityGate::evaluate(
    const ProductionQualityInput& raw) noexcept {
    ProductionQualityInput input = raw;

    input.reconstructionConfidence =
        clamp01(input.reconstructionConfidence);
    input.geometricConfidence =
        clamp01(input.geometricConfidence);
    input.temporalConfidence =
        clamp01(input.temporalConfidence);
    input.synchronizationConfidence =
        clamp01(input.synchronizationConfidence);
    input.calibrationConfidence =
        clamp01(input.calibrationConfidence);
    input.trackingConfidence =
        clamp01(input.trackingConfidence);
    input.faceEyeConfidence =
        clamp01(input.faceEyeConfidence);
    input.faceEyeOutOfDistributionRisk =
        clamp01(input.faceEyeOutOfDistributionRisk);
    input.faceEyeReliabilityTrend =
        std::clamp(
            sanitize(input.faceEyeReliabilityTrend),
            -1.0,
            1.0);
    input.global4DScore =
        clamp01(input.global4DScore);
    input.global4DRootMotionConsistency =
        clamp01(input.global4DRootMotionConsistency);
    input.global4DSceneConsistency =
        clamp01(input.global4DSceneConsistency);
    input.global4DDirectionalUncertaintyConsistency =
        clamp01(input.global4DDirectionalUncertaintyConsistency);
    input.global4DFaceEyeConsistency =
        clamp01(input.global4DFaceEyeConsistency);
    input.bodyMetricEvidenceAuthority =
        clamp01(input.bodyMetricEvidenceAuthority);
    input.faceMetricEvidenceAuthority =
        clamp01(input.faceMetricEvidenceAuthority);
    input.handMetricEvidenceAuthority =
        clamp01(input.handMetricEvidenceAuthority);
    input.bodyEpipolarConsistency =
        clamp01(input.bodyEpipolarConsistency);
    input.faceEpipolarConsistency =
        clamp01(input.faceEpipolarConsistency);
    input.handEpipolarConsistency =
        clamp01(input.handEpipolarConsistency);
    input.directionalUncertaintyMajor95Meters =
        std::max(0.0, sanitize(input.directionalUncertaintyMajor95Meters));
    input.directionalUncertaintyDepthMeters =
        std::max(0.0, sanitize(input.directionalUncertaintyDepthMeters));
    input.directionalUncertaintyAnisotropy =
        std::max(1.0, sanitize(input.directionalUncertaintyAnisotropy));
    input.directionalUncertaintyConfidence =
        clamp01(input.directionalUncertaintyConfidence);
    input.directionalUncertaintyDegradedJointCount =
        std::max(0.0, sanitize(input.directionalUncertaintyDegradedJointCount));
    input.physicsConsistencyScore =
        clamp01(input.physicsConsistencyScore);
    input.globalRigTrust =
        clamp01(input.globalRigTrust);
    input.rigStabilityConfidence =
        clamp01(input.rigStabilityConfidence);
    input.rigTranslationDriftMeters =
        std::max(
            0.0,
            sanitize(
                input.rigTranslationDriftMeters));
    input.rigRotationDriftRadians =
        std::max(
            0.0,
            sanitize(
                input.rigRotationDriftRadians));
    input.dynamicMetrologyP95Millimeters =
        std::max(
            0.0,
            sanitize(
                input.dynamicMetrologyP95Millimeters));
    input.dynamicMetrologyRmseMillimeters =
        std::max(
            0.0,
            sanitize(
                input.dynamicMetrologyRmseMillimeters));
    input.dynamicMetrologyMaxErrorMillimeters =
        std::max(
            0.0,
            sanitize(
                input.dynamicMetrologyMaxErrorMillimeters));
    input.dynamicMetrologyConfidence =
        clamp01(
            input.dynamicMetrologyConfidence);
    input.gaussian4DTemporalStability =
        clamp01(
            input.gaussian4DTemporalStability);
    input.gaussian4DTrustedRatio =
        clamp01(
            input.gaussian4DTrustedRatio);
    input.gaussian4DNodeConfidence =
        clamp01(
            input.gaussian4DNodeConfidence);
    input.gaussian4DBindingConfidence =
        clamp01(
            input.gaussian4DBindingConfidence);
    input.gaussian4DP95StrainRatio =
        std::max(
            0.0,
            sanitize(
                input.gaussian4DP95StrainRatio));
    input.gaussian4DMaximumStrainRatio =
        std::max(
            0.0,
            sanitize(
                input.gaussian4DMaximumStrainRatio));
    input.surfaceConfidence =
        clamp01(input.surfaceConfidence);
    input.identitySwapRisk =
        clamp01(input.identitySwapRisk);
    input.occlusionRisk =
        clamp01(input.occlusionRisk);
    input.cameraMotionRisk =
        clamp01(input.cameraMotionRisk);
    input.thermalHeadroom =
        clamp01(input.thermalHeadroom);
    input.memoryHeadroom =
        clamp01(input.memoryHeadroom);
    input.runtimeQualityScale =
        clamp01(input.runtimeQualityScale);
    input.runtimeInferenceScale =
        clamp01(input.runtimeInferenceScale);

    const double reconstruction =
        input.reconstructionConfidence;

    const double geometry =
        input.geometricConfidence *
        inverseError(
            input.reprojectionErrorPixels,
            input.reprojectionLimitPixels);

    const double uncertainty =
        inverseError(
            input.meanUncertaintyMeters,
            input.uncertaintyLimitMeters);

    const double directionalUncertainty =
        std::clamp(
            std::exp(
                -input.directionalUncertaintyMajor95Meters / 0.42) * 0.28 +
            std::exp(
                -input.directionalUncertaintyDepthMeters / 0.34) * 0.24 +
            std::clamp(
                1.0 -
                    (input.directionalUncertaintyAnisotropy - 1.0) / 12.0,
                0.0,
                1.0) * 0.16 +
            input.directionalUncertaintyConfidence * 0.24 +
            std::exp(
                -input.directionalUncertaintyDegradedJointCount / 12.0) * 0.08,
            0.0,
            1.0);

    const double tracking = componentTracking(
        input.trackingConfidence,
        input.identitySwapRisk,
        input.occlusionRisk,
        input.cameraMotionRisk);

    const double rigStabilityScore =
        std::clamp(
            input.rigStabilityConfidence *
            std::exp(
                -input.rigTranslationDriftMeters /
                    0.0015) *
            std::exp(
                -input.rigRotationDriftRadians /
                    0.0025) *
            (
                input.rigUnstable
                    ? 0.15
                    : 1.0
            ),
            0.0,
            1.0);

    const double metrologyP95Score =
        std::exp(
            -input.dynamicMetrologyP95Millimeters /
            1.5);

    const double metrologyRmseScore =
        std::exp(
            -input.dynamicMetrologyRmseMillimeters /
            1.0);

    const double metrologyMaxScore =
        std::exp(
            -input.dynamicMetrologyMaxErrorMillimeters /
            3.0);

    const double metrologyScore =
        std::clamp(
            input.dynamicMetrologyConfidence *
            (
                metrologyP95Score * 0.45 +
                metrologyRmseScore * 0.30 +
                metrologyMaxScore * 0.25
            ) *
            (
                input.dynamicMetrologyValid
                    ? 1.0
                    : 0.20
            ),
            0.0,
            1.0);

    const double crossModalMetricAuthority =
        std::min({
            input.bodyMetricEvidenceAuthority,
            input.faceMetricEvidenceAuthority,
            input.handMetricEvidenceAuthority,
        });
    const double crossModalEpipolarAuthority =
        std::min({
            input.bodyEpipolarConsistency,
            input.faceEpipolarConsistency,
            input.handEpipolarConsistency,
        });

    const double gaussian4DScore =
        std::clamp(
            input.gaussian4DTemporalStability * 0.34 +
            input.gaussian4DTrustedRatio * 0.28 +
            input.gaussian4DNodeConfidence * 0.18 +
            input.gaussian4DBindingConfidence * 0.20,
            0.0,
            1.0);

    const double gaussianStrainScore =
        std::clamp(
            std::exp(
                -input.gaussian4DP95StrainRatio /
                std::max(
                    config_.maximumGaussianP95StrainRatio,
                    1e-6)
            ) * 0.65 +
            std::exp(
                -input.gaussian4DMaximumStrainRatio /
                std::max(
                    config_.maximumGaussianMaximumStrainRatio,
                    1e-6)
            ) * 0.35,
            0.0,
            1.0);

    const double runtime =
        input.runtimeQualityScale * 0.70 +
        input.runtimeInferenceScale * 0.30;

    const double thermal =
        input.thermalHeadroom;

    const double memory =
        input.memoryHeadroom;

    const double rawScore = weightedGeometricMeanTerms({
        {reconstruction, config_.reconstructionWeight},
        {geometry, config_.geometryWeight},
        {uncertainty, config_.uncertaintyWeight},
        {input.temporalConfidence, config_.temporalWeight},
        {input.synchronizationConfidence, config_.synchronizationWeight},
        {input.calibrationConfidence, config_.calibrationWeight},
        {tracking, config_.trackingWeight},
        {input.faceEyeConfidence, config_.faceEyeWeight},
        {
            1.0 - input.faceEyeOutOfDistributionRisk,
            config_.faceEyeOodWeight,
        },
        {
            faceEyeTrendQuality(
                input.faceEyeReliabilityTrend),
            config_.faceEyeTrendWeight,
        },
        {
            input.global4DScore,
            config_.global4DWeight,
        },
        {
            input.global4DRootMotionConsistency,
            config_.global4DWeight * 0.46,
        },
        {
            input.global4DSceneConsistency,
            config_.global4DWeight * 0.20,
        },
        {
            input.global4DDirectionalUncertaintyConsistency,
            config_.global4DDirectionalWeight,
        },
        {
            input.global4DFaceEyeConsistency,
            config_.global4DFaceEyeWeight,
        },
        {
            crossModalMetricAuthority,
            config_.metricAuthorityWeight,
        },
        {
            crossModalEpipolarAuthority,
            config_.epipolarAuthorityWeight,
        },
        {
            directionalUncertainty,
            config_.directionalUncertaintyWeight,
        },
        {input.physicsConsistencyScore, config_.physicsWeight},
        {input.globalRigTrust, config_.globalRigTrustWeight},
        {rigStabilityScore, config_.rigStabilityWeight},
        {metrologyScore, config_.metrologyWeight},
        {
            gaussian4DScore,
            config_.gaussian4DWeight,
        },
        {
            gaussianStrainScore,
            config_.gaussianStrainWeight,
        },
        {input.surfaceConfidence, config_.surfaceWeight},
        {runtime, config_.runtimeWeight},
        {thermal, config_.thermalWeight},
        {memory, config_.memoryWeight},
    });

    const double thermalPressure =
        std::max(0.0, 0.45 - thermal) / 0.45;
    const double memoryPressure =
        std::max(0.0, 0.40 - memory) / 0.40;
    const double uncertaintyPressure =
        std::max(
            0.0,
            input.meanUncertaintyMeters /
                positiveOrOne(input.uncertaintyLimitMeters) -
                1.0);
    const double trackingPressure =
        std::max(
            input.identitySwapRisk,
            std::max(input.occlusionRisk, input.cameraMotionRisk));
    const double runtimePressure =
        1.0 - runtime;
    const double faceEyePressure =
        1.0 - input.faceEyeConfidence;
    const double faceEyeOodPressure =
        input.faceEyeOutOfDistributionRisk;
    const double physicsPressure =
        1.0 - input.physicsConsistencyScore;
    const double globalRigPressure =
        1.0 - input.globalRigTrust;
    const double rigStabilityPressure =
        1.0 -
        rigStabilityScore;
    const double metrologyPressure =
        1.0 -
        metrologyScore;
    const double gaussian4DPressure =
        input.gaussian4DEvidenceAvailable
            ? clamp01(
                1.0 -
                (
                    gaussian4DScore * 0.55 +
                    gaussianStrainScore * 0.45
                ))
            : 0.0;

    const double global4DPressure =
        clamp01(
            1.0 -
            (
                input.global4DScore * 0.46 +
                input.global4DRootMotionConsistency * 0.20 +
                input.global4DSceneConsistency * 0.10 +
                input.global4DDirectionalUncertaintyConsistency * 0.14 +
                input.global4DFaceEyeConsistency * 0.10
            ));
    const double faceEyeTrendPressure =
        std::max(
            0.0,
            -input.faceEyeReliabilityTrend);
    const double pressure = clamp01(
        thermalPressure * 0.19 +
        memoryPressure * 0.14 +
        uncertaintyPressure * 0.15 +
        trackingPressure * 0.17 +
        runtimePressure * 0.12 +
        faceEyePressure * 0.08 +
        faceEyeOodPressure * 0.065 +
        faceEyeTrendPressure * 0.065 +
        physicsPressure * 0.075 +
        globalRigPressure * 0.055 +
        rigStabilityPressure * 0.07 +
        metrologyPressure * 0.065 +
        gaussian4DPressure * 0.075 +
        global4DPressure * 0.07 +
        (1.0 - input.global4DRootMotionConsistency) * 0.025 +
        (1.0 - input.global4DSceneConsistency) * 0.015 +
        (1.0 - input.synchronizationConfidence) * 0.02 +
        (1.0 - input.calibrationConfidence) * 0.025 +
        (1.0 - crossModalMetricAuthority) * 0.045 +
        (1.0 - crossModalEpipolarAuthority) * 0.055);

    if (!initialized_) {
        filteredScore_ = rawScore;
        initialized_ = true;
    } else {
        const double adaptiveGain =
            std::clamp(
                config_.smoothingGain +
                    std::abs(rawScore - filteredScore_) * 0.35,
                0.05,
                0.82);

        filteredScore_ +=
            (rawScore - filteredScore_) * adaptiveGain;
    }

    filteredScore_ = clamp01(filteredScore_);

    const bool hardFailure =
        input.hardFailure ||
        input.synchronizationConfidence < 0.12 ||
        input.calibrationConfidence < 0.12 ||
        input.identitySwapRisk >= 0.985 ||
        input.faceEyeReliabilityTrend <= -0.78 ||
        !input.faceEyeFusionAccepted ||
        input.faceEyeOutOfDistributionRisk >= 0.985 ||
        !input.global4DAccepted ||
        input.global4DPersistentFailure ||
        input.global4DScore < 0.16 ||
        input.global4DDirectionalUncertaintyConsistency < 0.12 ||
        input.global4DFaceEyeConsistency < 0.12 ||
        crossModalMetricAuthority < 0.16 ||
        crossModalEpipolarAuthority < 0.18 ||
        input.directionalUncertaintyConfidence < 0.10 ||
        input.directionalUncertaintyAnisotropy >= 18.0 ||
        !input.physicsConstraintAccepted ||
        input.physicsConsistencyScore < 0.16 ||
        input.globalRigTrust < 0.12 ||
        (input.rigStabilityRequired &&
            (
                input.rigUnstable ||
                input.rigStabilityConfidence < 0.60
            )) ||
        (input.dynamicMetrologyRequired &&
            (
                !input.dynamicMetrologyValid ||
                input.dynamicMetrologyP95Millimeters > 1.0 ||
                input.dynamicMetrologyMaxErrorMillimeters > 2.0 ||
                input.dynamicMetrologyConfidence < 0.60
            )) ||
        (input.gaussian4DEvidenceAvailable &&
            (
                input.gaussian4DTrustedRatio <
                    config_.minimumGaussianTrustedRatio ||
                input.gaussian4DP95StrainRatio >
                    config_.maximumGaussianP95StrainRatio ||
                input.gaussian4DMaximumStrainRatio >
                    config_.maximumGaussianMaximumStrainRatio ||
                gaussian4DScore < 0.20 ||
                gaussianStrainScore < 0.20
            )
        ) ||
        !std::isfinite(rawScore);

    QualityPolicy policy =
        classify(
            filteredScore_,
            input,
            pressure);

    if (previousPolicy_ == QualityPolicy::Accept &&
        policy == QualityPolicy::Hold &&
        filteredScore_ >=
            0.76 - config_.holdHysteresis) {
        policy = QualityPolicy::Accept;
    }

    if (previousPolicy_ == QualityPolicy::Hold &&
        policy == QualityPolicy::Degrade &&
        filteredScore_ >=
            0.52 - config_.rejectHysteresis) {
        policy = QualityPolicy::Hold;
    }

    if (hardFailure) {
        policy = QualityPolicy::Reject;
    }

    previousPolicy_ = policy;

    return {
        policy,
        bottleneck(input),
        rawScore,
        filteredScore_,
        pressure,
        !hardFailure && policy != QualityPolicy::Reject &&
            filteredScore_ >= config_.minimumOutputScore,
        hardFailure,
        budget(
            policy,
            filteredScore_,
            pressure,
            input),
    };
}

void ProductionQualityGate::reset() noexcept {
    filteredScore_ = 0.0;
    initialized_ = false;
    previousPolicy_ = QualityPolicy::Hold;
}

double ProductionQualityGate::sanitize(double value) noexcept {
    return std::isfinite(value) ? value : 0.0;
}

double ProductionQualityGate::clamp01(double value) noexcept {
    return std::clamp(sanitize(value), 0.0, 1.0);
}

double ProductionQualityGate::faceEyeTrendQuality(
    double trend) noexcept {
    const double bounded =
        std::clamp(
            sanitize(trend),
            -1.0,
            1.0);

    if (bounded >= 0.0) {
        return std::clamp(
            0.88 + bounded * 0.12,
            0.0,
            1.0);
    }

    if (bounded >= -0.08) {
        return std::clamp(
            0.82 +
                (bounded + 0.08) / 0.08 * 0.06,
            0.0,
            1.0);
    }

    if (bounded >= -0.35) {
        return std::clamp(
            0.82 -
                (-bounded - 0.08) / 0.27 * 0.42,
            0.34,
            0.82);
    }

    return std::clamp(
        0.40 -
            (-bounded - 0.35) / 0.65 * 0.32,
        0.08,
        0.40);
}

double ProductionQualityGate::inverseError(
    double value,
    double limit) noexcept {
    if (!std::isfinite(value)) {
        return 0.0;
    }

    if (value <= 0.0) {
        return 1.0;
    }

    const double normalized =
        value / positiveOrOne(limit);

    return std::exp(-0.75 * normalized);
}

double directionalUncertaintyScore(
    double major95Meters,
    double depthMeters,
    double anisotropy,
    double confidence,
    double degradedJointCount) noexcept {
    const double major =
        std::exp(-std::max(0.0, major95Meters) / 0.42);
    const double depth =
        std::exp(-std::max(0.0, depthMeters) / 0.34);
    const double anisotropyScore =
        std::clamp(
            1.0 -
                (std::max(1.0, anisotropy) - 1.0) / 12.0,
            0.0,
            1.0);
    const double degraded =
        std::exp(-std::max(0.0, degradedJointCount) / 12.0);

    return std::clamp(
        major * 0.28 +
            depth * 0.24 +
            anisotropyScore * 0.16 +
            std::clamp(confidence, 0.0, 1.0) * 0.24 +
            degraded * 0.08,
        0.0,
        1.0);
}

double ProductionQualityGate::componentTracking(
    double trackingConfidence,
    double identitySwapRisk,
    double occlusionRisk,
    double cameraMotionRisk) noexcept {
    const double identity =
        1.0 - clamp01(identitySwapRisk);
    const double occlusion =
        1.0 - clamp01(occlusionRisk);
    const double camera =
        1.0 - clamp01(cameraMotionRisk);

    return clamp01(
        clamp01(trackingConfidence) * 0.54 +
        identity * 0.22 +
        occlusion * 0.14 +
        camera * 0.10);
}

double ProductionQualityGate::weightedGeometricMean(
    const ProductionQualityInput& input,
    const ProductionQualityConfig& config) noexcept {
    const double geometry =
        input.geometricConfidence *
        inverseError(
            input.reprojectionErrorPixels,
            input.reprojectionLimitPixels);

    const double tracking =
        componentTracking(
            input.trackingConfidence,
            input.identitySwapRisk,
            input.occlusionRisk,
            input.cameraMotionRisk);

    const double runtime =
        input.runtimeQualityScale * 0.70 +
        input.runtimeInferenceScale * 0.30;

    return weightedGeometricMeanTerms({
        {input.reconstructionConfidence, config.reconstructionWeight},
        {geometry, config.geometryWeight},
        {
            inverseError(
                input.meanUncertaintyMeters,
                input.uncertaintyLimitMeters),
            config.uncertaintyWeight,
        },
        {input.temporalConfidence, config.temporalWeight},
        {input.synchronizationConfidence, config.synchronizationWeight},
        {input.calibrationConfidence, config.calibrationWeight},
        {tracking, config.trackingWeight},
        {input.faceEyeConfidence, config.faceEyeWeight},
        {
            faceEyeTrendQuality(
                input.faceEyeReliabilityTrend),
            config.faceEyeTrendWeight,
        },
        {
            1.0 - input.faceEyeOutOfDistributionRisk,
            config.faceEyeOodWeight,
        },
        {input.physicsConsistencyScore, config.physicsWeight},
        {input.surfaceConfidence, config.surfaceWeight},
        {runtime, config.runtimeWeight},
        {input.thermalHeadroom, config.thermalWeight},
        {input.memoryHeadroom, config.memoryWeight},
    });
}

QualityPolicy ProductionQualityGate::classify(
    double score,
    const ProductionQualityInput& input,
    double pressure) const noexcept {
    if (input.hardFailure) {
        return QualityPolicy::Reject;
    }

    if (input.thermalHeadroom < 0.18 ||
        input.memoryHeadroom < 0.15) {
        return QualityPolicy::Degrade;
    }

    if (
        input.faceEyeReliabilityTrend <= -0.78 ||
        !input.faceEyeFusionAccepted ||
        input.faceEyeOutOfDistributionRisk >= 0.985 ||
        !input.global4DAccepted ||
        input.global4DPersistentFailure ||
        input.global4DScore < 0.16 ||
        !input.physicsConstraintAccepted ||
        input.physicsConsistencyScore < 0.16 ||
        input.globalRigTrust < 0.12) {
        return QualityPolicy::Reject;
    }

    if (input.synchronizationConfidence < 0.35 ||
        input.calibrationConfidence < 0.35) {
        return QualityPolicy::Hold;
    }

    if (pressure >= 0.82 ||
        score < config_.qualityFloor) {
        return QualityPolicy::Reject;
    }

    if (score >= config_.acceptThreshold &&
        pressure < 0.45) {
        return QualityPolicy::Accept;
    }

    if (score >= config_.holdThreshold) {
        return QualityPolicy::Hold;
    }

    return QualityPolicy::Degrade;
}

QualityBottleneck ProductionQualityGate::bottleneck(
    const ProductionQualityInput& input) noexcept {
    struct Candidate {
        QualityBottleneck id;
        double score;
    };

    const std::array<Candidate, 17> candidates{
        Candidate{
            QualityBottleneck::Synchronization,
            input.synchronizationConfidence},
        Candidate{
            QualityBottleneck::Calibration,
            input.calibrationConfidence},
        Candidate{
            QualityBottleneck::Temporal,
            input.temporalConfidence},
        Candidate{
            QualityBottleneck::Tracking,
            componentTracking(
                input.trackingConfidence,
                input.identitySwapRisk,
                input.occlusionRisk,
                input.cameraMotionRisk)},
        Candidate{
            QualityBottleneck::FaceEye,
            input.faceEyeConfidence *
                faceEyeTrendQuality(
                    input.faceEyeReliabilityTrend) *
                (input.faceEyeFusionAccepted ? 1.0 : 0.40)},
        Candidate{
            QualityBottleneck::DirectionalUncertainty,
            directionalUncertaintyScore(
                input.directionalUncertaintyMajor95Meters,
                input.directionalUncertaintyDepthMeters,
                input.directionalUncertaintyAnisotropy,
                input.directionalUncertaintyConfidence,
                input.directionalUncertaintyDegradedJointCount)},
        Candidate{
            QualityBottleneck::Physics,
            input.physicsConsistencyScore *
                (input.physicsConstraintAccepted ? 1.0 : 0.35)},
        Candidate{
            QualityBottleneck::Reconstruction,
            input.global4DScore *
                (input.global4DAccepted ? 1.0 : 0.35) *
                (
                    input.global4DRootMotionConsistency * 0.46 +
                    input.global4DSceneConsistency * 0.10 +
                    input.global4DDirectionalUncertaintyConsistency * 0.14 +
                    input.global4DFaceEyeConsistency * 0.10 +
                    0.20
                )},
        Candidate{
            QualityBottleneck::RigCoverage,
            input.globalRigTrust,
        },
        Candidate{
            QualityBottleneck::RigStability,
            std::clamp(
                input.rigStabilityConfidence *
                    std::exp(
                        -input.rigTranslationDriftMeters /
                            0.0015) *
                    std::exp(
                        -input.rigRotationDriftRadians /
                            0.0025),
                0.0,
                1.0),
        },
        Candidate{
            QualityBottleneck::Metrology,
            std::clamp(
                input.dynamicMetrologyConfidence *
                    std::exp(
                        -input.dynamicMetrologyP95Millimeters /
                            1.5) *
                    (
                        input.dynamicMetrologyValid
                            ? 1.0
                            : 0.20
                    ),
                0.0,
                1.0),
        },
        Candidate{
            QualityBottleneck::Geometry,
            input.geometricConfidence *
                inverseError(
                    input.reprojectionErrorPixels,
                    input.reprojectionLimitPixels)},
        Candidate{
            QualityBottleneck::Reconstruction,
            input.reconstructionConfidence},
        Candidate{
            QualityBottleneck::Surface,
            input.surfaceConfidence},
        Candidate{
            QualityBottleneck::Performance,
            input.runtimeQualityScale},
        Candidate{
            QualityBottleneck::Thermal,
            input.thermalHeadroom},
        Candidate{
            QualityBottleneck::Memory,
            input.memoryHeadroom},
    };

    const auto weakest =
        std::min_element(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& lhs, const Candidate& rhs) {
                return lhs.score < rhs.score;
            });

    if (weakest == candidates.end() ||
        weakest->score >= 0.92) {
        return QualityBottleneck::None;
    }

    return weakest->id;
}

ProductionQualityBudget ProductionQualityGate::budget(
    QualityPolicy policy,
    double score,
    double pressure,
    const ProductionQualityInput& input) noexcept {
    const double policyScale =
        policy == QualityPolicy::Accept ? 1.00 :
        policy == QualityPolicy::Hold ? 0.86 :
        policy == QualityPolicy::Degrade ? 0.64 :
        0.45;

    const double recoveryPressure =
        std::clamp(
            std::max(
                0.0,
                std::max(
                    1.0 -
                        std::min({
                            input.faceEyeConfidence,
                            input.physicsConsistencyScore,
                            input.global4DScore,
                            input.global4DDirectionalUncertaintyConsistency,
                            input.global4DFaceEyeConsistency,
                            input.directionalUncertaintyConfidence,
                            input.rigStabilityConfidence,
                            input.dynamicMetrologyConfidence,
                        }),
                    -input.faceEyeReliabilityTrend)),
            0.0,
            1.0);
    const double scale = std::clamp(
        std::min({
            policyScale,
            input.thermalHeadroom,
            input.memoryHeadroom,
            input.runtimeQualityScale,
        }) *
            (1.0 - recoveryPressure * 0.12),
        0.25,
        1.0);

    ProductionQualityBudget result{};

    result.targetInferenceFps =
        std::clamp(
            static_cast<std::int32_t>(
                60.0 * input.runtimeInferenceScale * scale),
            6,
            120);

    result.detectorIntervalFrames =
        score >= 0.84 ? 2 :
        score >= 0.64 ? 3 :
        score >= 0.48 ? 4 :
        6;

    result.denseMeshHz =
        std::clamp(
            static_cast<std::int32_t>(30.0 * scale),
            2,
            60);

    result.eyeTrackingHz =
        std::clamp(
            static_cast<std::int32_t>(
                90.0 * scale *
                (1.0 + recoveryPressure * 0.52)),
            6,
            120);

    result.handTrackingHz =
        std::clamp(
            static_cast<std::int32_t>(60.0 * scale),
            6,
            90);

    result.bundleAdjustmentIterations =
        std::clamp(
            static_cast<std::int32_t>(
                48.0 * scale *
                    (1.0 + recoveryPressure * 0.28) *
                    (1.0 + (1.0 - input.global4DScore) * 0.18) *
                    (1.0 - pressure * 0.25)),
            2,
            80);

    result.tsdfStride =
        scale >= 0.85 ? 1 :
        scale >= 0.68 ? 2 :
        scale >= 0.50 ? 3 :
        4;

    result.meshVertexBudget =
        std::clamp(
            static_cast<std::int32_t>(
                600000.0 * scale),
            16000,
            1000000);

    result.allowDenseSurface =
        scale >= 0.58;

    result.allowReferenceOptimization =
        policy == QualityPolicy::Accept &&
        score >= 0.90 &&
        pressure < 0.30 &&
        input.synchronizationConfidence >= 0.90 &&
        input.calibrationConfidence >= 0.90 &&
        input.globalRigTrust >= 0.90 &&
        input.rigStabilityConfidence >= 0.92 &&
        !input.rigUnstable &&
        input.dynamicMetrologyRequired &&
        input.dynamicMetrologyValid &&
        input.dynamicMetrologyP95Millimeters <= 1.0 &&
        input.dynamicMetrologyMaxErrorMillimeters <= 2.0 &&
        input.dynamicMetrologyConfidence >= 0.90 &&
        input.global4DDirectionalUncertaintyConsistency >= 0.82 &&
        input.global4DFaceEyeConsistency >= 0.82 &&
        input.directionalUncertaintyConfidence >= 0.72 &&
        input.directionalUncertaintyAnisotropy < 6.0;

    return result;
}

} // namespace nox::native::optimization

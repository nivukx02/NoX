package com.nivukx.nox.core.ai

import com.nivukx.nox.core.calibration.UltraMultiCameraCalibrationAuthorityEngine
import com.nivukx.nox.core.model.HighEndMotionCaptureSettings
import com.nivukx.nox.core.reconstruction.ProbabilisticDynamicSurfaceFusionEngine

/**
 * Converts the operator-facing high-end settings into bounded engine configurations.
 *
 * Centralizing this mapping prevents UI settings from drifting away from the actual runtime
 * constraints and makes profile changes auditable.
 */
object FrontierHighEndConfigurationAdapter {

    data class ConfigurationBundle(
        val worldSpace: WorldSpaceMotionProposalEngine.Config,
        val calibration: UltraMultiCameraCalibrationAuthorityEngine.Config,
        val surface: ProbabilisticDynamicSurfaceFusionEngine.Config,
        val federation: FederationLimits,
    )

    data class FederationLimits(
        val maximumCameras: Int,
        val authorityCameras: Int,
        val witnessCameras: Int,
        val maximumActors: Int = 64,
        val targetActors: Int = 50,
    )

    fun from(
        settings: HighEndMotionCaptureSettings,
    ): ConfigurationBundle =
        ConfigurationBundle(
            worldSpace =
                WorldSpaceMotionProposalEngine.Config(
                    maxObservations =
                        settings.frontierAuthorityCameraBudget
                            .coerceIn(2, 48),
                    minimumGeometricConfidence =
                        (
                            0.06 +
                                settings.captureIntegrityStrictness * 0.14
                            )
                            .coerceIn(0.06, 0.30),
                    maximumRayPairDistanceMeters =
                        settings.reconstructionUncertaintyMeters
                            .times(25.0)
                            .coerceIn(0.25, 2.0),
                    maximumLearnedDisplacementMeters = 0.75,
                    learnedContributionWeight =
                        (
                            0.05 +
                                settings.temporalContinuityPriority * 0.15
                            )
                            .coerceIn(0.05, 0.20),
                    minimumViewDiversity =
                        settings.minimumViewDiversity.coerceIn(0.02, 1.0),
                    robustIterations =
                        settings.frontierWorldSpaceRobustIterations,
                    huberResidualMeters =
                        settings.frontierWorldSpaceHuberResidualMillimeters /
                            1000.0,
                    maximumConditionEstimate =
                        settings.frontierWorldSpaceMaximumConditionEstimate,
                ),
            calibration =
                UltraMultiCameraCalibrationAuthorityEngine.Config(
                    maximumCameras =
                        settings.frontierMaximumCameraFederation,
                    maximumAuthoritativeCameras =
                        settings.frontierAuthorityCameraBudget,
                    minimumEdgeConfidence =
                        settings.minimumCalibrationConfidence
                            .times(0.48)
                            .coerceIn(0.20, 0.80),
                    minimumLoopConfidence =
                        settings.minimumCrossViewAgreement
                            .times(0.48)
                            .coerceIn(0.20, 0.80),
                    maximumTranslationSigmaMeters =
                        settings.ultraPrecisionTargetP95ErrorMillimeters
                            .coerceIn(0.05, 5.0)
                            / 1000.0,
                    maximumRotationSigmaRadians = 0.0025,
                    maximumTimingResidualMillis =
                        settings.maximumSynchronizationResidualP95Ms,
                    maximumReprojectionErrorPixels =
                        settings.reprojectionGatePx,
                    minimumParallaxDegrees =
                        settings.frontierMinimumObservedParallaxDegrees,
                    preferredParallaxDegrees =
                        maxOf(
                            settings.frontierMinimumObservedParallaxDegrees,
                            settings.referenceGradeMinimumParallaxDegrees,
                        ),
                    maximumCycleResidualMeters =
                        settings.frontierCalibrationMaximumCycleResidualMillimeters /
                            1000.0,
                ),
            surface =
                ProbabilisticDynamicSurfaceFusionEngine.Config(
                    cellSizeMeters =
                        settings.reconstructionUncertaintyMeters
                            .div(2.0)
                            .coerceIn(0.005, 0.08),
                    maximumPatches =
                        settings.frontierSurfacePatchCapacity,
                    maximumHistoryFrames =
                        settings.frontierSurfaceHistoryFrames,
                    minimumConfidence =
                        (
                            0.05 +
                                settings.denseLandmarkCoverageStrictness * 0.08
                            )
                            .coerceIn(0.05, 0.14),
                    robustScaleMeters =
                        settings.frontierWorldSpaceHuberResidualMillimeters
                            .div(1000.0)
                            .coerceIn(0.005, 0.10),
                    maximumUncertaintyMeters =
                        settings.reconstructionUncertaintyMeters
                            .coerceIn(0.001, 1.0),
                    temporalHalfLifeSeconds =
                        (
                            settings.temporalFusionConfidenceHalfLifeMillis /
                                1000.0
                        )
                            .coerceIn(0.05, 1.0),
                    velocityDecay =
                        settings.temporalFusionHardOutlierGate
                            .div(8.0)
                            .coerceIn(0.70, 0.98),
                    topologyDecay = 0.985,
                    maximumVelocityMetersPerSecond = 15.0,
                ),
            federation =
                FederationLimits(
                    maximumCameras =
                        settings.frontierMaximumCameraFederation,
                    authorityCameras =
                        settings.frontierAuthorityCameraBudget,
                    witnessCameras =
                        settings.frontierWitnessCameraBudget,
                ),
        )
}

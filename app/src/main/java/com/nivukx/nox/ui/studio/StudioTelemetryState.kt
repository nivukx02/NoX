package com.nivukx.nox.ui.studio

data class StudioTelemetryState(
    val camerasConnected: Int?,
    val camerasExpected: Int?,
    val synchronizationConfidence: Double?,
    val calibrationReady: Boolean?,
    val trackingConfidence: Double?,
    val reconstructionConfidence: Double?,
    val droppedObservations: Long?,
    val thermalConstrained: Boolean?,
    val exportReady: Boolean?,
    val concurrentRigAdmissionState: String? = null,
    val concurrentRigAdmissionScore: Double? = null,
    val concurrentRigSelectedCameraIds: List<String>? = null,
    val concurrentRigBlockers: List<String>? = null,
    val concurrentRigWarnings: List<String>? = null,
    val thermalExtrinsicDriftState: String? = null,
    val thermalExtrinsicDriftConfidence: Double? = null,
    val thermalExtrinsicTranslationDriftMm: Double? = null,
    val thermalExtrinsicRotationDriftDegrees: Double? = null,
    val thermalExtrinsicTemperatureDeltaCelsius: Double? = null,
    val thermalExtrinsicHardFailure: Boolean? = null,
    val thermalExtrinsicWarnings: List<String>? = null,
    val spatialDetailMeanPriority: Double? = null,
    val spatialDetailBudgetUtilization: Double? = null,
    val spatialDetailUltraRatio: Double? = null,
    val spatialDetailFineOrHigherRatio: Double? = null,
    val spatialDetailHold: Boolean? = null,
    val cameraSessionVerification: String? = null,
    val cameraSessionSupported: Boolean? = null,
    val highSpeedSessionVerification: String? = null,
    val highSpeedSessionSupported: Boolean? = null,
    val highSpeedSessionActive: Boolean = false,
    val highSpeedSelectedFps: Int? = null,
    val highSpeedSessionReason: String? = null,
    val hardwareReadoutTimestampSupported: Boolean = false,
    val latestReadoutTimestampNanos: Long? = null,
    val hardwareReadoutTimingState: String? = null,
    val hardwareReadoutTimingConfidence: Double = 0.0,
    val hardwareReadoutP95JitterMillis: Double = Double.POSITIVE_INFINITY,
    val hardwareReadoutMedianDelayMillis: Double = Double.NaN,
    val faceEyeMetricState: String? = null,
    val faceEyeMetricConfidence: Double? = null,
    val faceEyeMetricEyeUncertaintyMm: Double? = null,
    val faceEyeMetricWarnings: List<String>? = null,
    val faceEyeMetricRecommendations: List<String>? = null,
    val multiPersonIdentityContinuity: Double = 0.0,
    val multiPersonIdentityReidentifications: Int = 0,
    val multiPersonIdentityPredictions: Int = 0,
    val multiPersonObservedCount: Int = 0,
    val multiPersonAdmittedCount: Int = 0,
    val multiPersonHeldCount: Int = 0,
    val multiPersonStarvationRisk: Double = 0.0,
    val multiPersonFairnessBoostedCount: Int = 0,
    val multiPerson3DReconstructedCount: Int = 0,
    val multiPerson3DDetailedCount: Int = 0,
    val multiPerson3DPredictedCount: Int = 0,
    val multiPerson3DPredictionMaxAgeMillis: Long = 0L,
    val nativeEngineHealthy: Boolean?,
    val queueDepth: Int? = null,
    val queueCapacity: Int? = null,
    val reconstructionUncertaintyP95Meters: Double? = null,
    val motionP95ErrorMillimeters: Double? = null,
    val motionTemporalMemoryScore: Double? = null,
    val motionP99ErrorMillimeters: Double? = null,
    val motionJitterMillimeters: Double? = null,
    val temporalLagMilliseconds: Double? = null,
    val scaleDriftPercentPerMinute: Double? = null,
    val calibrationImageCoverage: Double? = null,
    val calibrationDepthCoverage: Double? = null,
    val selectedViewDiversity: Double? = null,
    val nativeQueueAgeMicroseconds: Long? = null,
    val nativeProcessingMicroseconds: Long? = null,
    val nativeMotionVelocityPixelsPerSecond: Double? = null,
    val nativeMotionAccelerationPixelsPerSecond2: Double? = null,
    val nativeMotionJerkPixelsPerSecond3: Double? = null,
    val nativeMotionInnovationP95Pixels: Double? = null,
    val nativeMotionDirectionConsistency: Double? = null,
    val nativeMotionMicroMotionRetention: Double? = null,
    val nativeMotionDynamicStability: Double? = null,
    val nativeExposureMotionBlurRisk: Double? = null,
    val nativeSubpixelMotionRetention: Double? = null,
    val captureOverflowCount: Long? = null,
    val nativeDropRatio: Double? = null,
    val cameraZeroCopyReady: Boolean? = null,
    val cameraHostCopyCount: Int? = null,
    val interleavedEffectiveInputRateHz: Double? = null,
    val interleavedTemporalConsistency: Double? = null,
    val interleavedSpatialAgreement: Double? = null,
    val interleavedSupportingCameras: Int? = null,
    val interleavedAuthoritative: Boolean? = null,
    val tsdfMemoryUtilization: Double? = null,
    val tsdfRejectedVoxelAllocations: Long? = null,
    val crossViewAgreement: Double? = null,
    val angularDiversity: Double? = null,
    val calibrationDriftStabilityScore: Double? = null,
    val calibrationRecalibrationRequired: Boolean? = null,
    val calibrationResidualSlopePxPerMinute: Double? = null,
    val productionQualityBand: String? = null,
    val productionQualityScore: Double? = null,
    val captureAuthorityScore: Double? = null,
    val captureAuthorityBand: String? = null,
    val captureAuthorityHardBlocked: Boolean? = null,
    val captureAuthorityPrecisionQualified: Boolean? = null,
    val captureAuthorityWarnings: List<String>? = null,
    val opticalBlurRisk: Double? = null,
    val productionQualityContractSatisfied: Boolean? = null,
    val cameraObservabilityScore: Double? = null,
    val selectedCameraCount: Int? = null,
    val referenceBenchmarkStatus: String? = null,
    val modeledPrecisionP95Millimeters: Double? = null,
    val modeledPrecisionP99Millimeters: Double? = null,
    val precisionGrade: String? = null,
    val precisionFeasibilityScore: Double? = null,
    val referenceGradeFeasible: Boolean? = null,
    val requiredBaselineMetersFor2mm: Double? = null,
    val recommendedBaselineMeters: Double? = null,
    val requiredFocalLengthPixelsFor2mm: Double? = null,
    val recommendedFocalLengthPixels: Double? = null,
    val recommendedMinimumParallaxDegrees: Double? = null,
    val resourceLeakRisk: Double? = null,
    val staleResourceCount: Int? = null,
    val residentResourceBytes: Long? = null,
    val activeResourceCount: Int? = null,
    val synchronizationResidualP95Milliseconds: Double? = null,
    val synchronizationOutlierRatio: Double? = null,
    val surfaceTemporalStability: Double? = null,
    val surfaceTopologyContinuity: Double? = null,
    val surfaceStrainP95Ratio: Double? = null,
    val surfaceNormalP95Radians: Double? = null,
    val surfaceCurvatureP95Delta: Double? = null,
    val gpuLeaseActiveCount: Int? = null,
    val gpuLeaseInFlightCount: Int? = null,
    val gpuLeaseWatchdogAborts: Long? = null,
    val gpuLeaseStaleGenerationReleases: Long? = null,
    val memoryLeaseActiveCount: Int? = null,
    val memoryLeaseStaleCount: Int? = null,
    val memoryLeaseInvalidReleases: Long? = null,
    val memoryLeaseDoubleReleases: Long? = null,
    val memoryLeaseForcedReclaims: Long? = null,
    val memoryLeaseBytesActive: Long? = null,
    val memoryLeaseOldestAgeMilliseconds: Long? = null,
    val captureStageCoreMotionProtected: Boolean? = null,
    val captureStageFidelityReduced: Boolean? = null,
    val captureStageDeferredOptionalCount: Int? = null,
    val captureStagePredictedFrameTimeMilliseconds: Double? = null,
    val denseLandmarkCoverage: Double? = null,
    val denseLandmarkVisibleCoverage: Double? = null,
    val denseLandmarkUncertaintyP95: Double? = null,
    val denseLandmarkCrossViewAgreement: Double? = null,
    val denseLandmarkCrossViewEvidenceAvailable: Boolean? = null,
    val denseLandmarkTemporalStability: Double? = null,
    val denseGeometricConfidence: Double? = null,
    val denseViewDiversity: Double? = null,
    val denseMotionBoundaryConfidence: Double? = null,
    val denseAppearanceConsistency: Double? = null,
    val denseDepthConfidence: Double? = null,
    val denseTemporalInnovationP95: Double? = null,
    val denseLandmarkContactCoverage: Double? = null,
    val denseLandmarkPredictionBridgeRatio: Double? = null,
    val denseLandmarkRejectedObservations: Long? = null,
    val denseLandmarkStaleObservations: Long? = null,
    val denseLandmarkNonFiniteObservations: Long? = null,
    val denseLandmarkHardFailure: Boolean? = null,
    val denseLandmarkContractScore: Double? = null,
    val denseLandmarkContractState: String? = null,
    val denseLandmarkContractAuthoritative: Boolean? = null,
    val denseLandmarkContractFailure: String? = null,
    val geometricInformationConditionScore: Double? = null,
    val geometricInformationP95MajorMeters: Double? = null,
    val geometricInformationP95DepthMeters: Double? = null,
    val denseSurfaceTrustedPointRatio: Double? = null,
    val denseSurfaceUncertaintyP95Meters: Double? = null,
    val denseSurfaceTopologyStability: Double? = null,
    val denseSurfaceViewDiversity: Double? = null,
    val denseSurfaceProbabilisticEvidenceScore: Double? = null,
    val denseSurfaceMeanReprojectionErrorPx: Double? = null,
    val denseSurfaceMeanAppearanceConsistency: Double? = null,
    val denseSurfaceMeanSilhouetteConsistency: Double? = null,
    val denseSurfaceMeanMotionBoundaryConfidence: Double? = null,
    val denseSurfaceMeanLightingStability: Double? = null,
    val denseSurfaceMeanDynamicProbability: Double? = null,
    val denseSurfaceMeanTemporalStability: Double? = null,
    val denseSurfaceTrustedGaussianRatio: Double? = null,
    val denseSurfaceAdaptiveViewSelectionScore: Double? = null,
    val denseSurfaceAdaptiveViewDiversityScore: Double? = null,
    val denseSurfaceAnisotropicWeakAxisConstraint: Double? = null,
    val denseSurfaceAnisotropicViewConditionScore: Double? = null,
    val denseSurfaceExcludedWeakViewCount: Int? = null,
    val denseSurfaceHardFailure: Boolean? = null,
    val denseIdentityGateQuality: Double? = null,
    val denseIdentityGateRejectedCount: Int? = null,
    val denseIdentityGateEvaluatedPoints: Int? = null,
    val denseIdentityGateWarnings: List<String>? = null,
    val identityExchangeState: String? = null,
    val identityExchangeRisk: Double? = null,
    val identityExchangeConfidence: Double? = null,
    val identityExchangeEvaluatedSubjects: Int? = null,
    val identityExchangeHeldSubjects: Int? = null,
    val identityExchangeWarnings: List<String>? = null,
    val correlatedEvidenceScore: Double? = null,
    val correlatedEvidenceCoverage: Double? = null,
    val correlatedEvidenceFailureRisk: Double? = null,
    val correlatedEvidenceCommonModeRisk: Double? = null,
    val correlatedEvidenceRedundantSignalCount: Int? = null,
    val correlatedEvidenceWarnings: List<String>? = null,
    val denseContactConfidence: Double? = null,
    val denseContactProtectedLandmarkCount: Int? = null,
    val denseContactWarnings: List<String>? = null,
    val global4DExternalContactConsistency: Double? = null,
    val global4DGaussianConsistency: Double? = null,
    val global4DGaussianStrainConsistency: Double? = null,
    val global4DGaussianEvidenceAvailable: Boolean? = null,
    val denseMeanJerkMetersPerSecond3: Double? = null,
    val denseMotionStability: Double? = null,
    val denseKinematicInnovationMillimeters: Double? = null,
    val denseSubMillimeterFeasibilityScore: Double? = null,
    val denseFirstOrderDepthErrorMillimeters: Double? = null,
    val densePrecisionQualified: Boolean? = null,
    val cameraContributionMinimumPairwiseDiversity: Double? = null,
    val cameraContributionViewStarvationRisk: Double? = null,
    val infraredReferenceOutcome: String? = null,
    val infraredReferenceQualified: Boolean? = null,
    val infraredReferenceConfidence: Double? = null,
    val infraredReferenceP95DeltaMillimeters: Double? = null,
    val infraredReferenceP99DeltaMillimeters: Double? = null,
    val infraredReferenceUncertaintyMillimeters: Double? = null,
    val cameraOpticalBlurRisk: Double? = null,
) {    val correlatedEvidenceState: String
        get() {
            val failure = correlatedEvidenceFailureRisk
            val coverage = correlatedEvidenceCoverage
            return when {
                failure != null && failure >= 0.85 -> "REJECTED"
                failure != null && failure >= 0.55 -> "DEGRADED"
                coverage != null && coverage < 0.50 -> "DEGRADED"
                failure != null || coverage != null -> "HEALTHY"
                else -> "UNKNOWN"
            }
        }

    val cameraHealth: Double?
        get() {
            val connected = camerasConnected ?: return null
            val expected = camerasExpected ?: return null
            if (expected <= 0) return null
            return (connected.toDouble() / expected).coerceIn(0.0, 1.0)
        }

    val queuePressure: Double?
        get() {
            val depth = queueDepth ?: return null
            val capacity = queueCapacity ?: return null
            if (capacity <= 0) return null
            return (depth.toDouble() / capacity.toDouble()).coerceIn(0.0, 1.0)
        }

    val precisionEvidenceScore: Double?
        get() {
            if (denseLandmarkHardFailure == true) return 0.0
            if (denseLandmarkContractState == "BLOCKED") return 0.0

            if (cameraSessionSupported == false) return 0.0
            if (faceEyeMetricState == "HOLD") return 0.0

        val values = listOfNotNull(
                motionP95ErrorMillimeters?.let {
                    (1.0 - it / 10.0).coerceIn(0.0,1.0)
                },
                motionJitterMillimeters?.let {
                    (1.0 - it / 4.0).coerceIn(0.0,1.0)
                },
                temporalLagMilliseconds?.let {
                    (1.0 - it / 50.0).coerceIn(0.0,1.0)
                },
                calibrationImageCoverage,
                calibrationDepthCoverage,
                selectedViewDiversity,
                crossViewAgreement,
                angularDiversity,
                calibrationDriftStabilityScore,
                scaleDriftPercentPerMinute?.let {
                    (1.0 - it / 0.25).coerceIn(0.0, 1.0)
                },
                precisionFeasibilityScore,
                modeledPrecisionP95Millimeters?.let {
                    (1.0 - it / 2.0).coerceIn(0.0, 1.0)
                },
                synchronizationResidualP95Milliseconds?.let {
                    (1.0 - it / 12.0).coerceIn(0.0, 1.0)
                },
                synchronizationOutlierRatio?.let {
                    (1.0 - it / 0.20).coerceIn(0.0, 1.0)
                },
                surfaceTemporalStability,
                surfaceTopologyContinuity,
                surfaceStrainP95Ratio?.let {
                    (1.0 - it / 0.35).coerceIn(0.0, 1.0)
                },
                denseLandmarkCoverage,
                denseLandmarkVisibleCoverage,
                denseLandmarkUncertaintyP95?.let {
                    (1.0 - it).coerceIn(0.0, 1.0)
                },
                denseSurfaceAdaptiveViewSelectionScore,
                denseSurfaceAdaptiveViewDiversityScore,
                denseSurfaceMeanAppearanceConsistency,
                denseSurfaceMeanSilhouetteConsistency,
                denseSurfaceMeanMotionBoundaryConfidence,
                denseSurfaceMeanLightingStability,
                denseSurfaceTrustedGaussianRatio,
                denseSurfaceMeanTemporalStability,
                denseSurfaceMeanDynamicProbability?.let {
                    (1.0 - it).coerceIn(0.0, 1.0)
                },
                denseLandmarkCrossViewAgreement
                    ?.takeIf {
                        denseLandmarkCrossViewEvidenceAvailable == true
                    },
                denseLandmarkTemporalStability,
                denseLandmarkContactCoverage,
                nativeMotionDirectionConsistency,
                nativeMotionMicroMotionRetention,
                nativeMotionDynamicStability,
                multiPerson3DPredictedCount
                    .toDouble()
                    .let {
                        1.0 -
                            (
                                it /
                                    multiPersonAdmittedCount
                                        .coerceAtLeast(1)
                                        .toDouble()
                            ).coerceIn(0.0, 1.0)
                    },
                hardwareReadoutTimingConfidence.takeIf {
                    hardwareReadoutTimestampSupported
                },
                faceEyeMetricConfidence,
                denseLandmarkPredictionBridgeRatio,
                denseMeanJerkMetersPerSecond3?.let {
                    (1.0 - it / 1200.0).coerceIn(0.0, 1.0)
                },
                denseMotionStability,
                denseKinematicInnovationMillimeters?.let {
                    (1.0 - it / 4.0).coerceIn(0.0, 1.0)
                },
                denseSubMillimeterFeasibilityScore,
                denseFirstOrderDepthErrorMillimeters?.let {
                    (1.0 - it / 1.0).coerceIn(0.0, 1.0)
                },
                denseSurfaceAdaptiveViewSelectionScore,
                denseSurfaceAdaptiveViewDiversityScore,
                denseSurfaceMeanAppearanceConsistency,
                denseSurfaceMeanSilhouetteConsistency,
                denseSurfaceMeanMotionBoundaryConfidence,
                denseSurfaceMeanLightingStability,
                denseIdentityGateQuality,
                faceEyeMetricConfidence,
                denseContactConfidence,
                denseContactProtectedLandmarkCount?.let {
                    if (it > 0) denseContactConfidence ?: 0.0 else 1.0
                },
                denseIdentityGateRejectedCount?.let {
                    (1.0 - (it / 16.0).coerceIn(0.0, 1.0))
                },
                denseIdentityGateEvaluatedPoints?.let {
                    if (it > 0) denseIdentityGateQuality ?: 0.0 else 1.0
                },
                denseSurfaceExcludedWeakViewCount?.let {
                    (1.0 - (it / 16.0).coerceIn(0.0, 1.0))
                },
                denseLandmarkRejectedObservations?.let {
                    (1.0 - (it / 250.0).coerceIn(0.0, 1.0))
                },
                denseLandmarkStaleObservations?.let {
                    (1.0 - (it / 250.0).coerceIn(0.0, 1.0))
                },
                denseLandmarkNonFiniteObservations?.let {
                    if (it == 0L) 1.0 else 0.0
                },
            )
            return values.takeIf { it.isNotEmpty() }?.average()
        }

    val systemConfidence: Double?
        get() {
            if (denseLandmarkHardFailure == true) return 0.0

            val values = listOfNotNull(
                synchronizationConfidence,
                trackingConfidence,
                reconstructionConfidence,
                denseLandmarkCoverage,
                denseLandmarkVisibleCoverage,
                denseLandmarkTemporalStability,
                denseMotionStability,
                denseSubMillimeterFeasibilityScore,
                denseFirstOrderDepthErrorMillimeters?.let {
                    (1.0 - it / 1.0).coerceIn(0.0, 1.0)
                },
                densePrecisionQualified?.let {
                    if (it) 1.0 else 0.0
                },
                denseIdentityGateQuality,
                denseContactConfidence?.takeIf {
                    denseContactProtectedLandmarkCount?.let { count ->
                        count > 0
                    } == true
                },
                denseSurfaceTrustedGaussianRatio,
                denseSurfaceMeanTemporalStability,
                denseSurfaceMeanDynamicProbability?.let {
                    (1.0 - it).coerceIn(0.0, 1.0)
                },
                global4DExternalContactConsistency,
                denseLandmarkCrossViewAgreement?.takeIf {
                    denseLandmarkCrossViewEvidenceAvailable == true
                },
                denseLandmarkUncertaintyP95?.let {
                    (1.0 - it).coerceIn(0.0, 1.0)
                },
            ).map { it.coerceIn(0.0, 1.0) }

            return values
                .takeIf { it.isNotEmpty() }
                ?.average()
        }
}

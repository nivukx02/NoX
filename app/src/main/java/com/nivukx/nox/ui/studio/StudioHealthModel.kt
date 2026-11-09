package com.nivukx.nox.ui.studio

/**
 * Truth-preserving operator health semantics.
 *
 * UNKNOWN means the UI does not have enough evidence to make a claim.
 * BLOCKED means a known hard prerequisite is failed.
 * DEGRADED means the runtime is operating, but measurable quality or capacity is reduced.
 * READY means the required operator-critical signals are known and healthy.
 */
enum class StudioHealthBand {
    UNKNOWN,
    BLOCKED,
    DEGRADED,
    READY,
}

data class StudioHealthSummary(
    val band: StudioHealthBand,
    val blockers: List<String>,
    val warnings: List<String>,
    val knownSignalCount: Int,
    val totalSignalCount: Int,
) {
    val isOperationallyReady: Boolean
        get() = band == StudioHealthBand.READY
}

object StudioHealthModel {
    private const val QUALITY_READY_THRESHOLD = 0.75
    private const val QUALITY_DEGRADED_THRESHOLD = 0.50
    private const val QUEUE_PRESSURE_DEGRADED_THRESHOLD = 0.85

    fun summarize(telemetry: StudioTelemetryState): StudioHealthSummary {
        val blockers = buildList {
            if (telemetry.nativeEngineHealthy == false) {
                add("Native runtime is unhealthy.")
            }
            if (telemetry.calibrationReady == false) {
                add("Multi-camera calibration is not ready.")
            }
            if (telemetry.calibrationRecalibrationRequired == true) {
                add("Measured calibration drift requires recalibration.")
            }
        }

        val warnings = buildList {
            if (telemetry.nativeEngineHealthy == null) {
                add("Native runtime status is unknown.")
            }
            if (telemetry.camerasConnected == null ||
                telemetry.camerasExpected == null
            ) {
                add("Camera connectivity evidence is incomplete.")
            }
            if (telemetry.synchronizationConfidence == null) {
                add("Synchronization confidence is unknown.")
            } else if (telemetry.synchronizationConfidence < QUALITY_READY_THRESHOLD) {
                add("Synchronization evidence is degraded.")
            }
            if (telemetry.trackingConfidence == null) {
                add("Subject tracking confidence is unknown.")
            } else if (telemetry.trackingConfidence < QUALITY_READY_THRESHOLD) {
                add("Subject tracking evidence is degraded.")
            }
            if (telemetry.reconstructionConfidence == null) {
                add("3D reconstruction confidence is unknown.")
            } else if (telemetry.reconstructionConfidence < QUALITY_READY_THRESHOLD) {
                add("3D reconstruction evidence is degraded.")
            }
            if (telemetry.thermalConstrained == true) {
                add("Thermal policy is constraining the runtime.")
            }
            val queuePressure = telemetry.queuePressure
            if (queuePressure != null &&
                queuePressure >= QUEUE_PRESSURE_DEGRADED_THRESHOLD
            ) {
                add("Native work queue is under sustained pressure.")
            }
            if (telemetry.exportReady != true) {
                add("Canonical export is not currently ready.")
            }
            if (telemetry.reconstructionUncertaintyP95Meters != null &&
                telemetry.reconstructionUncertaintyP95Meters > 0.05
            ) {
                add("3D uncertainty P95 exceeds the operator caution band.")
            }
            if (
                telemetry.motionP95ErrorMillimeters != null &&
                telemetry.motionP95ErrorMillimeters > 5.0
            ) {
                add("Measured motion P95 error exceeds the precision caution band.")
            }
            if (
                telemetry.motionTemporalMemoryScore != null &&
                telemetry.motionTemporalMemoryScore < 0.35
            ) {
                add("Temporal motion memory is losing continuity evidence.")
            }
            if (
                telemetry.motionTemporalMemoryScore != null &&
                telemetry.motionTemporalMemoryScore < 0.18
            ) {
                add("Temporal motion memory is critically weak for authoritative prediction.")
            }
            if (
                telemetry.scaleDriftPercentPerMinute != null &&
                telemetry.scaleDriftPercentPerMinute > 0.25
            ) {
                add("Metric scale drift exceeds the precision stability contract.")
            } else if (
                telemetry.scaleDriftPercentPerMinute != null &&
                telemetry.scaleDriftPercentPerMinute > 0.15
            ) {
                add("Metric scale drift is approaching the precision stability limit.")
            }
            if (
                telemetry.modeledPrecisionP95Millimeters != null &&
                telemetry.modeledPrecisionP95Millimeters > 2.0
            ) {
                add("Modeled precision P95 exceeds the production caution envelope.")
            }
            if (
                telemetry.precisionFeasibilityScore != null &&
                telemetry.precisionFeasibilityScore < QUALITY_DEGRADED_THRESHOLD
            ) {
                add("Precision feasibility evidence is degraded.")
            }
            if (
                telemetry.temporalLagMilliseconds != null &&
                telemetry.temporalLagMilliseconds > 25.0
            ) {
                add("Measured temporal lag exceeds the operator caution band.")
            }
            if (
                telemetry.nativeQueueAgeMicroseconds != null &&
                telemetry.nativeQueueAgeMicroseconds > 25_000L
            ) {
                add("Native queue residence exceeds the freshness caution band.")
            }
            if (
                telemetry.nativeProcessingMicroseconds != null &&
                telemetry.nativeProcessingMicroseconds > 12_000L
            ) {
                add("Native worker processing exceeds the frame-time caution band.")
            }
            if (
                telemetry.captureOverflowCount != null &&
                telemetry.captureOverflowCount > 0L
            ) {
                add("Capture queue overflow has discarded stale frames.")
            }
            if (
                telemetry.nativeDropRatio != null &&
                telemetry.nativeDropRatio > 0.02
            ) {
                add("Measured observation loss exceeds the 2% caution threshold.")
            }
            if (
                telemetry.tsdfMemoryUtilization != null &&
                telemetry.tsdfMemoryUtilization >= 0.90
            ) {
                add("3D reconstruction memory is near its resident ceiling.")
            }
            if (
                telemetry.calibrationImageCoverage != null &&
                telemetry.calibrationImageCoverage < 0.25
            ) {
                add("Calibration image coverage is below the reference envelope.")
            }
            if (
                telemetry.calibrationDepthCoverage != null &&
                telemetry.calibrationDepthCoverage < 0.10
            ) {
                add("Calibration depth coverage is below the reference envelope.")
            }
            if (
                telemetry.selectedViewDiversity != null &&
                telemetry.selectedViewDiversity < 0.40
            ) {
                add("Selected camera geometry is information-limited.")
            }
            if (
                telemetry.cameraContributionMinimumPairwiseDiversity != null &&
                telemetry.cameraContributionMinimumPairwiseDiversity < 0.20
            ) {
                add("Selected camera views have weak geometric diversity.")
            }
            if (
                telemetry.cameraContributionViewStarvationRisk != null &&
                telemetry.cameraContributionViewStarvationRisk > 0.72
            ) {
                add("Multi-view reconstruction is becoming dependent on too few strong cameras.")
            }
            if (
                telemetry.correlatedEvidenceState == "REJECTED"
            ) {
                add("Correlated evidence is rejecting the current multi-view state.")
            } else if (
                telemetry.correlatedEvidenceState == "DEGRADED"
            ) {
                add("Correlated evidence is degraded across dependent quality families.")
            }
            if (
                telemetry.correlatedEvidenceCommonModeRisk != null &&
                telemetry.correlatedEvidenceCommonModeRisk > 0.45
            ) {
                add("Common-mode evidence risk is high; quality signals may share the same failure source.")
            }
            if (
                telemetry.identityExchangeState == "HOLD" ||
                telemetry.identityExchangeState == "HIGH_RISK"
            ) {
                add("Multi-person identity continuity is at exchange risk; hold authoritative reconstruction.")
            } else if (
                telemetry.identityExchangeState == "GUARDED"
            ) {
                add("Multi-person identity continuity is guarded by occlusion and interaction evidence.")
            }
            if (
                telemetry.identityExchangeHeldSubjects != null &&
                telemetry.identityExchangeHeldSubjects > 0
            ) {
                add("One or more subjects require identity hold protection.")
            }
            if (telemetry.crossViewAgreement != null &&
                telemetry.crossViewAgreement < QUALITY_DEGRADED_THRESHOLD
            ) {
                add("Cross-view agreement is degraded.")
            }
            if (
                telemetry.geometricInformationConditionScore != null &&
                telemetry.geometricInformationConditionScore < 0.12
            ) {
                add("3D ray-space information is ill-conditioned.")
            }
            if (
                telemetry.geometricInformationP95MajorMeters != null &&
                telemetry.geometricInformationP95MajorMeters > 0.02
            ) {
                add("3D information-matrix major-axis uncertainty exceeds the 20 mm caution band.")
            }
            if (
                telemetry.geometricInformationP95DepthMeters != null &&
                telemetry.geometricInformationP95DepthMeters > 0.05
            ) {
                add("3D depth-axis uncertainty exceeds the 50 mm caution band.")
            }
            if (telemetry.angularDiversity != null &&
                telemetry.angularDiversity < 0.40
            ) {
                add("Camera angular diversity is information-limited.")
            }
            if (telemetry.synchronizationResidualP95Milliseconds != null &&
                telemetry.synchronizationResidualP95Milliseconds > 12.0
            ) {
                add("Synchronization residual P95 exceeds the temporal contract.")
            }
            if (telemetry.synchronizationOutlierRatio != null &&
                telemetry.synchronizationOutlierRatio > 0.20
            ) {
                add("Synchronization outlier ratio exceeds the temporal contract.")
            }
            if (telemetry.surfaceTemporalStability != null &&
                telemetry.surfaceTemporalStability < QUALITY_DEGRADED_THRESHOLD
            ) {
                add("Dense surface temporal stability is degraded.")
            }
            if (telemetry.surfaceTopologyContinuity != null &&
                telemetry.surfaceTopologyContinuity < QUALITY_DEGRADED_THRESHOLD
            ) {
                add("Dense surface topology continuity is degraded.")
            }
            if (telemetry.surfaceStrainP95Ratio != null &&
                telemetry.surfaceStrainP95Ratio > 0.35
            ) {
                add("Dense surface strain P95 exceeds the geometry contract.")
            }
            if (telemetry.denseSurfaceMeanAppearanceConsistency != null &&
                telemetry.denseSurfaceMeanAppearanceConsistency < 0.45
            ) {
                add("Dense surface appearance consistency is below the trust band.")
            }
            if (telemetry.denseSurfaceMeanSilhouetteConsistency != null &&
                telemetry.denseSurfaceMeanSilhouetteConsistency < 0.45
            ) {
                add("Dense surface silhouette consistency is below the trust band.")
            }
            if (telemetry.denseSurfaceMeanMotionBoundaryConfidence != null &&
                telemetry.denseSurfaceMeanMotionBoundaryConfidence < 0.45
            ) {
                add("Dense surface motion-boundary confidence is degraded.")
            }
            if (telemetry.denseSurfaceMeanLightingStability != null &&
                telemetry.denseSurfaceMeanLightingStability < 0.50
            ) {
                add("Dense surface lighting stability is degraded.")
            }
            if (telemetry.gpuLeaseWatchdogAborts != null &&
                telemetry.gpuLeaseWatchdogAborts > 0L
            ) {
                add("GPU buffer watchdog has aborted one or more stale leases.")
            }
            if (telemetry.gpuLeaseStaleGenerationReleases != null &&
                telemetry.gpuLeaseStaleGenerationReleases > 0L
            ) {
                add("Stale GPU buffer generations were safely rejected.")
            }
            if (telemetry.captureStageCoreMotionProtected == false) {
                add("Core motion workload protection is not currently guaranteed.")
            }
            if (telemetry.captureStageFidelityReduced == true) {
                add("Optional fidelity stages are being reduced to protect capture continuity.")
            }
            if (telemetry.captureStageDeferredOptionalCount != null &&
                telemetry.captureStageDeferredOptionalCount > 0
            ) {
                add("Optional capture stages are deferred under runtime pressure.")
            }
            if (telemetry.captureStagePredictedFrameTimeMilliseconds != null &&
                telemetry.captureStagePredictedFrameTimeMilliseconds > 33.34
            ) {
                add("Predicted frame time exceeds the hard capture budget.")
            }
            if (telemetry.calibrationRecalibrationRequired == true) {
                add("Calibration drift requires recalibration before authoritative capture.")
            } else if (
                telemetry.calibrationDriftStabilityScore != null &&
                telemetry.calibrationDriftStabilityScore < QUALITY_READY_THRESHOLD
            ) {
                add("Calibration drift evidence is below the operational target.")
            }
            if (
                telemetry.denseLandmarkCoverage != null &&
                telemetry.denseLandmarkCoverage < QUALITY_DEGRADED_THRESHOLD
            ) {
                add("Dense landmark coverage is degraded.")
            }
            if (
                telemetry.denseLandmarkVisibleCoverage != null &&
                telemetry.denseLandmarkVisibleCoverage < QUALITY_DEGRADED_THRESHOLD
            ) {
                add("Visible dense landmark coverage is degraded.")
            }
            if (
                telemetry.denseLandmarkUncertaintyP95 != null &&
                telemetry.denseLandmarkUncertaintyP95 > 0.35
            ) {
                add("Dense landmark uncertainty P95 is above the caution band.")
            }
            if (
                telemetry.denseLandmarkCrossViewEvidenceAvailable == true &&
                telemetry.denseLandmarkCrossViewAgreement != null &&
                telemetry.denseLandmarkCrossViewAgreement < QUALITY_READY_THRESHOLD
            ) {
                add("Dense landmark cross-view agreement is below the operational target.")
            }
            if (
                telemetry.denseLandmarkTemporalStability != null &&
                telemetry.denseLandmarkTemporalStability < QUALITY_DEGRADED_THRESHOLD
            ) {
                add("Dense landmark temporal stability is degraded.")
            }
            if (
                telemetry.denseLandmarkRejectedObservations != null &&
                telemetry.denseLandmarkRejectedObservations > 0L
            ) {
                add("Dense landmark fusion has rejected one or more observations.")
            }
            if (
                telemetry.denseLandmarkStaleObservations != null &&
                telemetry.denseLandmarkStaleObservations > 0L
            ) {
                add("Stale dense landmark evidence has been rejected.")
            }
            if (
                telemetry.denseLandmarkNonFiniteObservations != null &&
                telemetry.denseLandmarkNonFiniteObservations > 0L
            ) {
                add("Non-finite dense landmark evidence has been rejected.")
            }
            if (telemetry.denseLandmarkHardFailure == true) {
                add("Current dense landmark evidence is hard-blocked.")
            }
            if (
                telemetry.denseIdentityGateQuality != null &&
                telemetry.denseIdentityGateQuality < QUALITY_DEGRADED_THRESHOLD
            ) {
                add("Dense cross-view landmark identity consistency is degraded.")
            }
            if (
                telemetry.denseIdentityGateRejectedCount != null &&
                telemetry.denseIdentityGateRejectedCount > 0
            ) {
                add("Cross-view identity gating rejected one or more dense observations.")
            }
            if (
                telemetry.denseContactProtectedLandmarkCount != null &&
                telemetry.denseContactProtectedLandmarkCount > 0 &&
                telemetry.denseContactConfidence != null &&
                telemetry.denseContactConfidence < 0.45
            ) {
                add("Contact-aware dense evidence is below the interaction confidence band.")
            }
            if (telemetry.denseContactWarnings?.isNotEmpty() == true) {
                addAll(
                    telemetry.denseContactWarnings
                        .take(2)
                        .map { "Dense contact: " + it },
                )
            }
            if (
                telemetry.denseSurfaceTrustedPointRatio != null &&
                telemetry.denseSurfaceTrustedPointRatio < QUALITY_DEGRADED_THRESHOLD
            ) {
                add("Dense surface trusted-point ratio is degraded.")
            }
            if (
                telemetry.denseSurfaceUncertaintyP95Meters != null &&
                telemetry.denseSurfaceUncertaintyP95Meters.isFinite() &&
                telemetry.denseSurfaceUncertaintyP95Meters > 0.20
            ) {
                add("Dense surface uncertainty P95 exceeds the operational band.")
            }
            if (
                telemetry.denseSurfaceTopologyStability != null &&
                telemetry.denseSurfaceTopologyStability < QUALITY_DEGRADED_THRESHOLD
            ) {
                add("Dense surface topology stability is degraded.")
            }
            if (
                telemetry.denseSurfaceViewDiversity != null &&
                telemetry.denseSurfaceViewDiversity < 0.50
            ) {
                add("Dense surface view diversity is limited.")
            }
            if (
                telemetry.denseSurfaceAnisotropicWeakAxisConstraint != null &&
                telemetry.denseSurfaceAnisotropicWeakAxisConstraint < 0.35
            ) {
                add("Dense reconstruction has a weakly constrained 3D information axis.")
            }
            if (
                telemetry.denseSurfaceAnisotropicViewConditionScore != null &&
                telemetry.denseSurfaceAnisotropicViewConditionScore < 0.12
            ) {
                add("Dense camera geometry is near a rank-deficient information configuration.")
            }
            if (
                telemetry.denseSurfaceProbabilisticEvidenceScore != null &&
                telemetry.denseSurfaceProbabilisticEvidenceScore < 0.45
            ) {
                add("2D probability mass is weakly aligned with the current 3D reprojection.")
            }
            if (
                telemetry.denseSurfaceAdaptiveViewSelectionScore != null &&
                telemetry.denseSurfaceAdaptiveViewSelectionScore < 0.50
            ) {
                add("Adaptive dense-surface view selection confidence is degraded.")
            }
            if (
                telemetry.denseSurfaceExcludedWeakViewCount != null &&
                telemetry.denseSurfaceExcludedWeakViewCount > 0
            ) {
                add("Adaptive view selection excluded weak or redundant camera evidence.")
            }
            if (telemetry.denseSurfaceHardFailure == true) {
                add("Current dense surface reconstruction is hard-blocked.")
            }
            if (
                telemetry.denseSubMillimeterFeasibilityScore != null &&
                telemetry.denseSubMillimeterFeasibilityScore < QUALITY_DEGRADED_THRESHOLD
            ) {
                add("Dense first-order sub-millimetre feasibility is below the production caution band.")
            }
            if (
                telemetry.denseFirstOrderDepthErrorMillimeters != null &&
                telemetry.denseFirstOrderDepthErrorMillimeters.isFinite() &&
                telemetry.denseFirstOrderDepthErrorMillimeters > 1.0
            ) {
                add("Dense first-order depth error exceeds the one-millimetre target.")
            }
            if (
                telemetry.densePrecisionQualified == false &&
                telemetry.denseSubMillimeterFeasibilityScore != null
            ) {
                add("Dense geometry is not currently precision-qualified.")
            }
            when (telemetry.infraredReferenceOutcome) {
                "NOX_LOWER_ERROR", "REFERENCE_BETTER" ->
                    if (telemetry.infraredReferenceQualified == true) {
                        add("Matched active-optical reference shows lower measured error for NoX.")
                    } else {
                        add("Active-optical comparison reports lower measured error, but qualification evidence is incomplete.")
                    }
                "STATISTICAL_PARITY", "REFERENCE_PARITY" ->
                    add("NoX and the active-optical reference are within the declared statistical tie band.")
                "REFERENCE_LOWER_ERROR", "REFERENCE_WORSE" ->
                    add("Matched active-optical reference shows lower measured error than NoX.")
                "INCONCLUSIVE" -> add("Active-optical comparison is inconclusive.")
                null -> Unit
                else -> add("Active-optical comparison state: " + telemetry.infraredReferenceOutcome + ".")
            }
            if (
                telemetry.denseMotionStability != null &&
                telemetry.denseMotionStability < QUALITY_DEGRADED_THRESHOLD
            ) {
                add("Dense temporal motion stability is degraded.")
            }
            if (
                telemetry.denseKinematicInnovationMillimeters != null &&
                telemetry.denseKinematicInnovationMillimeters > 4.0
            ) {
                add("Dense kinematic innovation exceeds the preferred motion envelope.")
            }
        }

        val signalPresence = listOf(
            telemetry.camerasConnected != null,
            telemetry.camerasExpected != null,
            telemetry.synchronizationConfidence != null,
            telemetry.calibrationReady != null,
            telemetry.trackingConfidence != null,
            telemetry.reconstructionConfidence != null,
            telemetry.synchronizationResidualP95Milliseconds != null,
            telemetry.synchronizationOutlierRatio != null,
            telemetry.surfaceTemporalStability != null,
            telemetry.surfaceTopologyContinuity != null,
            telemetry.geometricInformationConditionScore != null,
            telemetry.geometricInformationP95MajorMeters != null,
            telemetry.geometricInformationP95DepthMeters != null,
            telemetry.gpuLeaseActiveCount != null,
            telemetry.captureStageCoreMotionProtected != null,
            telemetry.captureStageFidelityReduced != null,
            telemetry.captureStageDeferredOptionalCount != null,
            telemetry.captureStagePredictedFrameTimeMilliseconds != null,
            telemetry.denseLandmarkCoverage != null,
            telemetry.denseLandmarkVisibleCoverage != null,
            telemetry.denseLandmarkUncertaintyP95 != null,
            telemetry.denseLandmarkCrossViewAgreement != null,
            telemetry.denseLandmarkCrossViewEvidenceAvailable != null,
            telemetry.denseLandmarkTemporalStability != null,
            telemetry.denseLandmarkContactCoverage != null,
            telemetry.denseLandmarkPredictionBridgeRatio != null,
            telemetry.denseIdentityGateQuality != null,
            telemetry.denseIdentityGateRejectedCount != null,
            telemetry.denseIdentityGateEvaluatedPoints != null,
            telemetry.denseContactConfidence != null,
            telemetry.denseContactProtectedLandmarkCount != null,
            telemetry.denseContactWarnings != null,
            telemetry.global4DExternalContactConsistency != null,
            telemetry.denseLandmarkRejectedObservations != null,
            telemetry.denseLandmarkStaleObservations != null,
            telemetry.denseLandmarkNonFiniteObservations != null,
            telemetry.denseLandmarkHardFailure != null,
            telemetry.infraredReferenceOutcome != null,
            telemetry.infraredReferenceQualified != null,
            telemetry.denseSurfaceTrustedPointRatio != null,
            telemetry.denseSurfaceUncertaintyP95Meters != null,
            telemetry.denseSurfaceTopologyStability != null,
            telemetry.denseSurfaceViewDiversity != null,
            telemetry.denseSurfaceAdaptiveViewSelectionScore != null,
            telemetry.denseSurfaceAdaptiveViewDiversityScore != null,
            telemetry.denseSurfaceExcludedWeakViewCount != null,
            telemetry.denseSurfaceMeanReprojectionErrorPx != null,
            telemetry.denseSurfaceHardFailure != null,
            telemetry.denseMeanJerkMetersPerSecond3 != null,
            telemetry.denseMotionStability != null,
            telemetry.denseKinematicInnovationMillimeters != null,
            telemetry.denseSubMillimeterFeasibilityScore != null,
            telemetry.denseFirstOrderDepthErrorMillimeters != null,
            telemetry.droppedObservations != null,
            telemetry.thermalConstrained != null,
            telemetry.exportReady != null,
            telemetry.nativeEngineHealthy != null,
            telemetry.resourceLeakRisk != null,
            telemetry.staleResourceCount != null,
            telemetry.residentResourceBytes != null,
            telemetry.activeResourceCount != null,
        )

        val knownSignalCount =
            signalPresence.count { it }

        val requiredUnknown = telemetry.nativeEngineHealthy == null ||
            telemetry.calibrationReady == null ||
            telemetry.synchronizationConfidence == null ||
            telemetry.trackingConfidence == null ||
            telemetry.reconstructionConfidence == null

        val hardResourceFailure =
            telemetry.resourceLeakRisk != null &&
                telemetry.resourceLeakRisk >= 0.98

        val hardStageFailure =
            telemetry.captureStageCoreMotionProtected == false ||
                (
                    telemetry.captureStagePredictedFrameTimeMilliseconds != null &&
                        telemetry.captureStagePredictedFrameTimeMilliseconds > 50.0
                    )

        val hardPrecisionFailure =
            telemetry.precisionGrade == "BLOCKED" ||
                telemetry.infraredReferenceOutcome == "REFERENCE_WORSE" ||
                telemetry.infraredReferenceOutcome == "REFERENCE_LOWER_ERROR" ||
                (
                    telemetry.productionQualityBand == "BLOCKED" &&
                        telemetry.productionQualityContractSatisfied == false
                    )

        val precisionDegraded =
            telemetry.productionQualityContractSatisfied == false ||
                (
                    telemetry.precisionFeasibilityScore != null &&
                        telemetry.precisionFeasibilityScore < QUALITY_READY_THRESHOLD
                    )

        val hardDenseFailure =
            telemetry.denseLandmarkHardFailure == true ||
                telemetry.denseSurfaceHardFailure == true ||
                (telemetry.denseIdentityGateQuality ?: 1.0) < 0.35 ||
                (telemetry.denseLandmarkUncertaintyP95 ?: 0.0) >= 0.95 ||
                (telemetry.denseSurfaceUncertaintyP95Meters ?: 0.0) >= 0.80 ||
                (telemetry.denseSurfaceTrustedPointRatio ?: 1.0) < 0.10 ||
                (telemetry.denseSurfaceAnisotropicViewConditionScore ?: 1.0) < 0.05 ||
                (telemetry.denseSurfaceProbabilisticEvidenceScore ?: 1.0) < 0.15

        val hardQualityFailure =
            hardDenseFailure ||
            (telemetry.synchronizationConfidence ?: 1.0) < QUALITY_DEGRADED_THRESHOLD ||
            (telemetry.synchronizationResidualP95Milliseconds ?: 0.0) > 12.0 ||
            (telemetry.synchronizationOutlierRatio ?: 0.0) > 0.20 ||
            (telemetry.trackingConfidence ?: 1.0) < QUALITY_DEGRADED_THRESHOLD ||
            (telemetry.motionTemporalMemoryScore ?: 1.0) < 0.12 ||
            (telemetry.reconstructionConfidence ?: 1.0) < QUALITY_DEGRADED_THRESHOLD ||
            (telemetry.surfaceStrainP95Ratio ?: 0.0) > 0.35 ||
            (telemetry.geometricInformationConditionScore ?: 1.0) < 0.04 ||
            (telemetry.scaleDriftPercentPerMinute ?: 0.0) > 0.50 ||
            (telemetry.precisionFeasibilityScore ?: 1.0) < 0.25 ||
            (telemetry.denseLandmarkUncertaintyP95 ?: 0.0) > 0.80 ||
            (
                telemetry.denseLandmarkCrossViewEvidenceAvailable == true &&
                    telemetry.denseLandmarkCrossViewAgreement != null &&
                    telemetry.denseLandmarkCrossViewAgreement < 0.35
                ) ||
            (telemetry.denseLandmarkCoverage != null &&
                telemetry.denseLandmarkCoverage < 0.20) ||
            telemetry.denseLandmarkHardFailure == true ||
            telemetry.denseSurfaceHardFailure == true ||
            (telemetry.denseSurfaceTrustedPointRatio ?: 1.0) < 0.10 ||
            (telemetry.denseSurfaceUncertaintyP95Meters ?: 0.0) > 0.80 ||
            telemetry.correlatedEvidenceState == "REJECTED" ||
            (telemetry.correlatedEvidenceFailureRisk ?: 0.0) > 0.78 ||
            telemetry.identityExchangeState == "HOLD"

        val operationalQualityFailure =
            (telemetry.synchronizationConfidence ?: 1.0) < QUALITY_READY_THRESHOLD ||
            (telemetry.trackingConfidence ?: 1.0) < QUALITY_READY_THRESHOLD ||
            (telemetry.motionTemporalMemoryScore != null &&
                telemetry.motionTemporalMemoryScore < 0.35) ||
            (telemetry.reconstructionConfidence ?: 1.0) < QUALITY_READY_THRESHOLD ||
            (telemetry.scaleDriftPercentPerMinute != null &&
                telemetry.scaleDriftPercentPerMinute > 0.15) ||
            (telemetry.precisionFeasibilityScore != null &&
                telemetry.precisionFeasibilityScore < QUALITY_READY_THRESHOLD) ||
            (telemetry.surfaceTemporalStability != null &&
                telemetry.surfaceTemporalStability < QUALITY_READY_THRESHOLD) ||
            (telemetry.surfaceTopologyContinuity != null &&
                telemetry.surfaceTopologyContinuity < QUALITY_READY_THRESHOLD) ||
            (telemetry.denseSurfaceTopologyStability != null &&
                telemetry.denseSurfaceTopologyStability < QUALITY_READY_THRESHOLD) ||
            (telemetry.denseSurfaceViewDiversity != null &&
                telemetry.denseSurfaceViewDiversity < QUALITY_READY_THRESHOLD) ||
            (telemetry.denseSurfaceAnisotropicWeakAxisConstraint != null &&
                telemetry.denseSurfaceAnisotropicWeakAxisConstraint < 0.35) ||
            (telemetry.denseSurfaceAnisotropicViewConditionScore != null &&
                telemetry.denseSurfaceAnisotropicViewConditionScore < 0.12) ||
            (telemetry.denseSurfaceProbabilisticEvidenceScore != null &&
                telemetry.denseSurfaceProbabilisticEvidenceScore < 0.45) ||
            (telemetry.denseSubMillimeterFeasibilityScore != null &&
                telemetry.denseSubMillimeterFeasibilityScore < QUALITY_READY_THRESHOLD) ||
            (telemetry.denseFirstOrderDepthErrorMillimeters != null &&
                telemetry.denseFirstOrderDepthErrorMillimeters > 1.0) ||
            (telemetry.denseMotionStability != null &&
                telemetry.denseMotionStability < QUALITY_READY_THRESHOLD) ||
            telemetry.thermalConstrained == true ||
            (telemetry.cameraHealth?.let { it < 1.0 } == true)

        val band = when {
            blockers.isNotEmpty() -> StudioHealthBand.BLOCKED
            requiredUnknown -> StudioHealthBand.UNKNOWN
            hardResourceFailure -> StudioHealthBand.BLOCKED
            hardStageFailure -> StudioHealthBand.BLOCKED
            hardQualityFailure -> StudioHealthBand.BLOCKED
            hardPrecisionFailure -> StudioHealthBand.BLOCKED
            operationalQualityFailure || precisionDegraded ->
                StudioHealthBand.DEGRADED
            else -> StudioHealthBand.READY
        }

        return StudioHealthSummary(
            band = band,
            blockers = blockers,
            warnings = warnings,
            knownSignalCount = knownSignalCount,
            totalSignalCount = signalPresence.size,
        )
    }
}

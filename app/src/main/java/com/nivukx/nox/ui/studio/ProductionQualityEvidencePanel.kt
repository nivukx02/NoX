package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Production evidence surface for the Motion Studio operator.
 *
 * It intentionally shows measured signals and uncertainty instead of decorative "AI score"
 * numbers. The panel stays compact enough for the live operator flow and uses large targets
 * wherever interaction is introduced.
 */
@Composable
fun ProductionQualityEvidencePanel(
    telemetry: StudioTelemetryState,
    modifier: Modifier = Modifier,
) {
    LiquidGlassSurface(
        clear = false,
        modifier = modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column {
                    Text(
                        text = "PRODUCTION EVIDENCE",
                        style = MaterialTheme.typography.titleSmall,
                    )
                    Text(
                        text = evidenceSubtitle(telemetry),
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Text(
                    text = readinessLabel(telemetry),
                    style = MaterialTheme.typography.labelLarge,
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(18.dp),
            ) {
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    EvidenceLine(
                        label = "P95 error",
                        value = telemetry.motionP95ErrorMillimeters?.let {
                            String.format("%.2f mm", it)
                        },
                    )
                    EvidenceLine(
                        label = "P99 error",
                        value = telemetry.motionP99ErrorMillimeters?.let {
                            String.format("%.2f mm", it)
                        },
                    )
                    EvidenceLine(
                        label = "Lag",
                        value = telemetry.temporalLagMilliseconds?.let {
                            String.format("%.1f ms", it)
                        },
                    )
                    EvidenceLine(
                        label = "Dropout",
                        value = telemetry.droppedObservations?.toString(),
                    )
                    EvidenceLine(
                        label = "View diversity",
                        value = telemetry.selectedViewDiversity?.let {
                            String.format("%.0f%%", it * 100.0)
                        },
                    )
                }

                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    EvidenceLine(
                        label = "Sync",
                        value = telemetry.synchronizationConfidence?.percent(),
                    )
                    EvidenceLine(
                        label = "Sync P95",
                        value = telemetry.synchronizationResidualP95Milliseconds?.let {
                            String.format("%.2f ms", it)
                        },
                    )
                    EvidenceLine(
                        label = "Sync outliers",
                        value = telemetry.synchronizationOutlierRatio?.let {
                            String.format("%.1f%%", it * 100.0)
                        },
                    )
                    EvidenceLine(
                        label = "Calibration",
                        value = telemetry.calibrationReady?.let {
                            if (it) "READY" else "REQUIRED"
                        },
                    )
                    EvidenceLine(
                        label = "Cross-view",
                        value = telemetry.crossViewAgreement?.let {
                            String.format("%.0f%%", it * 100.0)
                        },
                    )
                    EvidenceLine(
                        label = "3D information condition",
                        value = telemetry.geometricInformationConditionScore?.let {
                            String.format("%.3f", it)
                        },
                    )
                    EvidenceLine(
                        label = "3D sigma major P95",
                        value = telemetry.geometricInformationP95MajorMeters?.let {
                            String.format("%.2f mm", it * 1000.0)
                        },
                    )
                    EvidenceLine(
                        label = "3D sigma depth P95",
                        value = telemetry.geometricInformationP95DepthMeters?.let {
                            String.format("%.2f mm", it * 1000.0)
                        },
                    )
                    EvidenceLine(
                        label = "Queue age",
                        value = telemetry.nativeQueueAgeMicroseconds?.let {
                            String.format("%.1f ms", it / 1000.0)
                        },
                    )
                    EvidenceLine(
                        label = "TSDF memory",
                        value = telemetry.tsdfMemoryUtilization?.let {
                            String.format("%.0f%%", it * 100.0)
                        },
                    )
                    EvidenceLine(
                        label = "Surface stability",
                        value = telemetry.surfaceTemporalStability?.percent(),
                    )
                    EvidenceLine(
                        label = "Topology continuity",
                        value = telemetry.surfaceTopologyContinuity?.percent(),
                    )
                    EvidenceLine(
                        label = "Surface strain P95",
                        value = telemetry.surfaceStrainP95Ratio?.let {
                            String.format("%.2f", it)
                        },
                    )
                    EvidenceLine(
                        label = "Appearance",
                        value = telemetry.denseSurfaceMeanAppearanceConsistency?.percent(),
                    )
                    EvidenceLine(
                        label = "Silhouette",
                        value = telemetry.denseSurfaceMeanSilhouetteConsistency?.percent(),
                    )
                    EvidenceLine(
                        label = "Motion boundary",
                        value = telemetry.denseSurfaceMeanMotionBoundaryConfidence?.percent(),
                    )
                    EvidenceLine(
                        label = "Lighting stability",
                        value = telemetry.denseSurfaceMeanLightingStability?.percent(),
                    )
                    EvidenceLine(
                        label = "Identity gate",
                        value = telemetry.denseIdentityGateQuality?.let {
                            String.format("%.0f%%", it * 100.0)
                        },
                    )
                    EvidenceLine(
                        label = "Actors observed",
                        value = telemetry.multiPersonObservedCount.toString(),
                    )
                    EvidenceLine(
                        label = "Actors admitted",
                        value = telemetry.multiPersonAdmittedCount.toString(),
                    )
                    EvidenceLine(
                        label = "Actors held",
                        value = telemetry.multiPersonHeldCount.toString(),
                    )
                    EvidenceLine(
                        label = "Fairness pressure",
                        value = String.format(
                            "%.0f%%",
                            telemetry.multiPersonStarvationRisk.coerceIn(0.0, 1.0) * 100.0,
                        ),
                    )
                    EvidenceLine(
                        label = "Fairness boosted",
                        value = telemetry.multiPersonFairnessBoostedCount.toString(),
                    )
                    EvidenceLine(
                        label = "3D actors reconstructed",
                        value =
                            telemetry.multiPerson3DReconstructedCount.toString(),
                    )
                    EvidenceLine(
                        label = "3D actors detailed",
                        value =
                            telemetry.multiPerson3DDetailedCount.toString(),
                    )
                    EvidenceLine(
                        label = "3D actors predicted",
                        value =
                            telemetry.multiPerson3DPredictedCount.toString(),
                    )
                    EvidenceLine(
                        label = "Prediction max age",
                        value =
                            telemetry.multiPerson3DPredictionMaxAgeMillis
                                .takeIf { it > 0L }
                                ?.let { "$it ms" }
                                ?: "NONE",
                    )
                    EvidenceLine(
                        label = "Identity rejects",
                        value = telemetry.denseIdentityGateRejectedCount?.toString(),
                    )
                    EvidenceLine(
                        label = "Identity points",
                        value = telemetry.denseIdentityGateEvaluatedPoints?.toString(),
                    )
                    EvidenceLine(
                        label = "Contact evidence",
                        value = telemetry.denseContactConfidence?.let {
                            String.format("%.0f%%", it * 100.0)
                        },
                    )
                    EvidenceLine(
                        label = "Contact protected",
                        value = telemetry.denseContactProtectedLandmarkCount?.toString(),
                    )
                    EvidenceLine(
                        label = "Global 4D contact",
                        value = telemetry.global4DExternalContactConsistency?.let {
                            String.format("%.0f%%", it * 100.0)
                        },
                    )
                    EvidenceLine(
                        label = "GPU leases",
                        value = telemetry.gpuLeaseActiveCount?.toString(),
                    )
                    EvidenceLine(
                        label = "GPU in flight",
                        value = telemetry.gpuLeaseInFlightCount?.toString(),
                    )
                    EvidenceLine(
                        label = "Core motion",
                        value = telemetry.captureStageCoreMotionProtected?.let {
                            if (it) "PROTECTED" else "HOLD"
                        },
                    )
                    EvidenceLine(
                        label = "Optional deferred",
                        value = telemetry.captureStageDeferredOptionalCount?.toString(),
                    )
                    EvidenceLine(
                        label = "Predicted frame",
                        value = telemetry.captureStagePredictedFrameTimeMilliseconds?.let {
                            String.format("%.2f ms", it)
                        },
                    )
                    EvidenceLine(
                        label = "Dense coverage",
                        value = telemetry.denseLandmarkCoverage?.percent(),
                    )
                    EvidenceLine(
                        label = "Visible coverage",
                        value = telemetry.denseLandmarkVisibleCoverage?.percent(),
                    )
                    EvidenceLine(
                        label = "Dense uncertainty P95",
                        value = telemetry.denseLandmarkUncertaintyP95?.let {
                            String.format("%.3f", it)
                        },
                    )
                    EvidenceLine(
                        label = "Dense cross-view",
                        value = if (
                            telemetry.denseLandmarkCrossViewEvidenceAvailable == true
                        ) {
                            telemetry.denseLandmarkCrossViewAgreement?.percent()
                        } else {
                            "N/A"
                        },
                    )
                    EvidenceLine(
                        label = "Dense temporal",
                        value = telemetry.denseLandmarkTemporalStability?.percent(),
                    )
                    EvidenceLine(
                        label = "Dense contact",
                        value = telemetry.denseLandmarkContactCoverage?.percent(),
                    )
                    EvidenceLine(
                        label = "Prediction bridge",
                        value = telemetry.denseLandmarkPredictionBridgeRatio?.percent(),
                    )
                    EvidenceLine(
                        label = "Dense rejected",
                        value = telemetry.denseLandmarkRejectedObservations?.toString(),
                    )
                    EvidenceLine(
                        label = "Dense surface view selection",
                        value = telemetry.denseSurfaceAdaptiveViewSelectionScore?.percent(),
                    )
                    EvidenceLine(
                        label = "Dense view diversity",
                        value = telemetry.denseSurfaceAdaptiveViewDiversityScore?.percent(),
                    )
                    EvidenceLine(
                        label = "Weak views excluded",
                        value = telemetry.denseSurfaceExcludedWeakViewCount?.toString(),
                    )
                    EvidenceLine(
                        label = "Dense contract",
                        value = telemetry.denseLandmarkContractScore?.let {
                            "${(it * 100.0).toInt()}% " +
                                (telemetry.denseLandmarkContractState ?: "UNKNOWN")
                        },
                    )
                    EvidenceLine(
                        label = "Dense authoritative",
                        value = telemetry.denseLandmarkContractAuthoritative?.let {
                            if (it) "YES" else "NO"
                        },
                    )
                    EvidenceLine(
                        label = "Dense failure",
                        value = telemetry.denseLandmarkContractFailure,
                    )
                    EvidenceLine(
                        label = "Dense current gate",
                        value = when (telemetry.denseLandmarkHardFailure) {
                            true -> "BLOCKED"
                            false -> "CLEAR"
                            null -> "N/A"
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun EvidenceLine(
    label: String,
    value: String?,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 40.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.labelMedium)
        Text(value ?: "N/A", style = MaterialTheme.typography.labelMedium)
    }
}

private fun evidenceSubtitle(
    telemetry: StudioTelemetryState,
): String {
    val calibration = telemetry.calibrationReady
    val overflow = telemetry.captureOverflowCount ?: 0L
    return when {
        calibration == false ->
            "Calibration evidence is required before reference-grade capture."
        overflow > 0L ->
            "Capture overflow has occurred. Inspect queue pressure before a take."
        telemetry.temporalLagMilliseconds?.let { it > 25.0 } == true ->
            "Temporal lag is above the production contract."
        telemetry.gpuLeaseWatchdogAborts?.let { it > 0L } == true ->
            "GPU lifetime watchdog activity detected."
        telemetry.surfaceTemporalStability?.let { it < 0.50 } == true ->
            "Dense surface temporal stability is below the caution band."
        telemetry.faceEyeMetricState == "HOLD" ->
            "Face/eye metric output is on hold pending stronger geometric evidence."
        telemetry.cameraSessionSupported == false ->
            "Camera stream topology was rejected by the platform session validator."
        telemetry.denseIdentityGateQuality?.let { it < 0.50 } == true ->
            "Cross-view landmark identity consistency is below the caution band."
        telemetry.captureStageCoreMotionProtected == false ->
            "Core motion workload protection is not currently guaranteed."
        telemetry.captureStageFidelityReduced == true ->
            "Optional fidelity is being reduced to preserve capture continuity."
        telemetry.denseLandmarkUncertaintyP95?.let { it > 0.35 } == true ->
            "Dense landmark uncertainty is elevated; inspect visibility and view agreement."
        telemetry.denseLandmarkContractState == "BLOCKED" ->
            "Dense landmark precision contract is blocked. Inspect the failure attribution."
        telemetry.denseLandmarkContractAuthoritative == false &&
            telemetry.denseLandmarkContractScore != null &&
            telemetry.denseLandmarkContractScore < 0.68 ->
            "Dense landmark evidence is below the production authorization target."
        telemetry.denseLandmarkCrossViewEvidenceAvailable == true &&
            telemetry.denseLandmarkCrossViewAgreement?.let { it < 0.55 } == true ->
            "Dense landmark cross-view agreement is below the operational target."
        else ->
            "Measured capture evidence and runtime health."
    }
}

private fun readinessLabel(
    telemetry: StudioTelemetryState,
): String =
    when {
        telemetry.calibrationReady == false -> "HOLD"
        telemetry.cameraSessionSupported == false -> "SESSION REJECTED"
        telemetry.faceEyeMetricState == "HOLD" -> "FACE/EYE HOLD"
        telemetry.faceEyeMetricState == "GEOMETRIC" -> "GEOMETRIC"
        telemetry.faceEyeMetricState == "VISUAL_ONLY" -> "VISUAL ONLY"
        telemetry.nativeEngineHealthy == false -> "ERROR"
        telemetry.thermalConstrained == true -> "THERMAL"
        telemetry.gpuLeaseWatchdogAborts?.let { it > 0L } == true -> "GPU WATCHDOG"
        telemetry.surfaceTemporalStability?.let { it < 0.50 } == true -> "SURFACE"
        telemetry.denseIdentityGateQuality?.let { it < 0.35 } == true -> "IDENTITY HOLD"
        telemetry.captureStageCoreMotionProtected == false -> "CAPTURE HOLD"
        telemetry.captureStageFidelityReduced == true -> "Fidelity Reduced"
        telemetry.denseLandmarkHardFailure == true ||
            telemetry.denseLandmarkContractState == "BLOCKED" ->
            "DENSE BLOCKED"
        telemetry.denseLandmarkUncertaintyP95?.let { it > 0.80 } == true ->
            "DENSE QUALITY"
        telemetry.captureOverflowCount?.let { it > 0L } == true -> "CHECK"
        else -> "READY"
    }

private fun Double.percent(): String =
    "${(coerceIn(0.0, 1.0) * 100.0).toInt()}%"

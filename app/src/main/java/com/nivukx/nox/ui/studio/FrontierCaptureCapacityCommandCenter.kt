package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

/**
 * High-end operator surface for the 128-camera / 50-actor frontier runtime.
 *
 * It intentionally uses LiquidGlassSurface only around the functional control/navigation surface.
 * The reconstruction viewport remains content, avoiding glass-over-content clutter.
 */
@Composable
fun FrontierCaptureCapacityCommandCenter(
    telemetry: StudioTelemetryState,
    modifier: Modifier = Modifier,
) {
    val connected =
        telemetry.camerasConnected?.coerceIn(0, 128) ?: 0
    val expected =
        telemetry.camerasExpected?.coerceIn(1, 128)
    val activeActors =
        telemetry.multiPersonAdmittedCount.coerceIn(0, 64)
    val heldActors =
        telemetry.multiPersonHeldCount.coerceAtLeast(0)

    // 128 is a capacity ceiling, not a claim that the operator requested 128 cameras.
    val cameraRatio =
        connected.toDouble() /
            (expected ?: 128).toDouble()

    val actorRatio =
        activeActors.toDouble() /
            50.0

    // Keep empirical runtime error, modeled feasibility and explicit qualification
    // separate. A modeled score must never be promoted into a measured-accuracy claim.
    val modeledFeasibility =
        telemetry.precisionFeasibilityScore
            ?.takeIf { it.isFinite() }
            ?.coerceIn(0.0, 1.0)
    val measuredMotionP95Millimeters =
        telemetry.motionP95ErrorMillimeters
            ?.takeIf { it.isFinite() && it >= 0.0 }
    val precisionQualified =
        telemetry.captureAuthorityPrecisionQualified == true
    val referenceStatus =
        telemetry.referenceBenchmarkStatus
            ?.takeIf { it.isNotBlank() }
            ?: "NO REFERENCE"

    val uncertaintyMm =
        telemetry.denseSurfaceUncertaintyP95Meters
            ?.times(1000.0)
            ?: telemetry.reconstructionUncertaintyP95Meters
                ?.times(1000.0)

    val state =
        when {
            telemetry.denseLandmarkHardFailure == true ->
                "EVIDENCE-LIMITED"
            telemetry.calibrationRecalibrationRequired == true ->
                "CALIBRATION-LIMITED"
            telemetry.synchronizationResidualP95Milliseconds
                ?.let { it > 12.0 } == true ->
                "TIMING-LIMITED"
            telemetry.thermalConstrained == true ->
                "THERMAL-LIMITED"
            heldActors > 0 ->
                "CAPACITY-DEGRADED"
            telemetry.captureAuthorityHardBlocked == true ->
                "AUTHORITY-BLOCKED"
            precisionQualified ->
                "REFERENCE-QUALIFIED"
            referenceStatus != "NO REFERENCE" ->
                "REFERENCE-REVIEW"
            else ->
                "LIVE / UNQUALIFIED"
        }

    LiquidGlassSurface(
        modifier = modifier.fillMaxWidth(),
        contentPadding = 14.dp,
        materialRole =
            StudioMaterialAdaptationEngine.MaterialRole
                .SECONDARY_TELEMETRY,
        underlyingLuminance = 0.35f,
        contentContrast = 0.96f,
        capturePriority = 1.0f,
        operatorFocus = 1.0f,
        plateProtection = 0.96f,
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column {
                    Text(
                        "FRONTIER CAPTURE FABRIC",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        "128-camera federation • 50-actor target • bounded 64",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Text(
                    state,
                    style = MaterialTheme.typography.labelLarge,
                )
            }

            CapacityMetric(
                label = "CAMERA FABRIC",
                value =
                    if (expected == null) {
                        connected.toString() + " / 128 capacity"
                    } else {
                        connected.toString() + " / " + expected + " expected"
                    },
                progress = cameraRatio,
                detail =
                    "Authority " +
                        (telemetry.selectedCameraCount ?: 0) +
                        " • witness " +
                        ((connected - (telemetry.selectedCameraCount ?: 0)).coerceAtLeast(0)) +
                        " • device concurrency is hardware-limited",
            )

            CapacityMetric(
                label = "ACTOR FABRIC",
                value = activeActors.toString() + " / 50",
                progress = actorRatio.coerceIn(0.0, 1.0),
                detail =
                    "held " +
                        heldActors +
                        " • fairness boosts " +
                        telemetry.multiPersonFairnessBoostedCount,
            )

            CapacityMetric(
                label = "PRECISION AUTHORITY",
                value =
                    when {
                        precisionQualified -> "QUALIFIED"
                        telemetry.captureAuthorityPrecisionQualified == false -> "UNQUALIFIED"
                        else -> "UNKNOWN"
                    },
                progress = if (precisionQualified) 1.0 else 0.0,
                detail = "Reference status: " + referenceStatus,
            )

            CapacityMetric(
                label = "MOTION P95 ERROR",
                value =
                    measuredMotionP95Millimeters?.let {
                        "%.2f mm".format(it)
                    } ?: "UNKNOWN",
                progress =
                    measuredMotionP95Millimeters?.let {
                        (1.0 - it / 10.0).coerceIn(0.0, 1.0)
                    } ?: 0.0,
                detail =
                    uncertaintyMm?.let {
                        "P95 uncertainty " + "%.2f".format(it) + " mm"
                    } ?: "No independent uncertainty estimate reported",
            )

            CapacityMetric(
                label = "MODELED FEASIBILITY",
                value =
                    modeledFeasibility?.let {
                        (it * 100.0).roundToInt().toString() + "%"
                    } ?: "UNKNOWN",
                progress = modeledFeasibility ?: 0.0,
                detail = "Modeled feasibility is not measured physical accuracy",
            )

            val memory =
                telemetry.residentResourceBytes
                    ?.let { bytes ->
                        val megaBytes =
                            bytes.toDouble() / (1024.0 * 1024.0)
                        "%.1f MB resident".format(megaBytes)
                    }
                    ?: "memory telemetry unavailable"

            Text(
                memory,
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun CapacityMetric(
    label: String,
    value: String,
    progress: Double,
    detail: String,
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                label,
                style = MaterialTheme.typography.labelMedium,
            )
            Text(
                value,
                style = MaterialTheme.typography.labelMedium,
            )
        }
        LinearProgressIndicator(
            progress = { progress.coerceIn(0.0, 1.0).toFloat() },
            modifier = Modifier.fillMaxWidth(),
        )
        Text(
            detail,
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

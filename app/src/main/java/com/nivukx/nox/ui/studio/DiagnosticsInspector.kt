package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp

data class DiagnosticsInspectorState(
    val nativeAbi: Int?,
    val nativeHealthy: Boolean?,
    val workloadScale: Double?,
    val queueUtilization: Double?,
    val shedding: Boolean?,
    val thermalConstrained: Boolean?,
    val droppedObservations: Long?,
    val captureOverflowEvents: Long? = null,
    val queueDepth: Int? = null,
    val maximumQueueDepth: Int? = null,
    val activeCameraCount: Int? = null,
    val workerCount: Int? = null,
    val processingEpoch: Long? = null,
    val activeBufferBlocks: Int? = null,
    val freeBufferBlocks: Int? = null,
    val workerProcessedObservations: Long? = null,
    val workerAcceptedRatio: Double? = null,
    val workerQuality: Double? = null,
    val workerTemporalGap: Double? = null,
    val workerQueueAgeMicroseconds: Long? = null,
    val workerProcessingMicroseconds: Long? = null,
    val workerVelocityPixelsPerSecond: Double? = null,
    val workerAccelerationPixelsPerSecond2: Double? = null,
    val workerJerkPixelsPerSecond3: Double? = null,
    val workerP95InnovationPixels: Double? = null,
    val workerMotionDirectionConsistency: Double? = null,
    val workerMicroMotionRetention: Double? = null,
    val workerDynamicStability: Double? = null,
    val workerExposureMotionBlurRisk: Double? = null,
    val workerSubpixelMotionRetention: Double? = null,
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
)

@Composable
fun DiagnosticsInspector(
    state: DiagnosticsInspectorState,
    modifier: Modifier = Modifier,
) {
    LiquidGlassSurface(
        clear = false,
        modifier = modifier.semantics {
            contentDescription = "Native runtime diagnostics"
        },
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column {
                    Text(
                        "NATIVE RUNTIME",
                        style = MaterialTheme.typography.labelMedium,
                    )
                    Text(
                        "Production evidence",
                        style = MaterialTheme.typography.titleMedium,
                    )
                }
                DiagnosticBadge(
                    label =
                        state.nativeHealthy?.let {
                            if (it) "HEALTHY" else "HOLD"
                        } ?: "UNKNOWN",
                )
            }

            DiagnosticGroup("ENGINE") {
                DiagnosticRow("ABI", state.nativeAbi?.toString())
                DiagnosticRow(
                    "Epoch",
                    state.processingEpoch?.toString(),
                )
                DiagnosticRow(
                    "Workers",
                    state.workerCount?.toString(),
                )
                DiagnosticRow(
                    "Active cameras",
                    state.activeCameraCount?.toString(),
                )
            }

            DiagnosticGroup("QUEUE & FRESHNESS") {
                DiagnosticRow(
                    "Depth",
                    state.queueDepth?.toString(),
                )
                DiagnosticRow(
                    "High-water",
                    state.maximumQueueDepth?.toString(),
                )
                DiagnosticRow(
                    "Utilization",
                    state.queueUtilization.percentOrNa(),
                )
                DiagnosticRow(
                    "Queue age",
                    state.workerQueueAgeMicroseconds.microsecondsOrNa(),
                )
                DiagnosticRow(
                    "Worker time",
                    state.workerProcessingMicroseconds.microsecondsOrNa(),
                )
                DiagnosticRow(
                    "Shedding",
                    state.shedding?.let { if (it) "ACTIVE" else "OFF" },
                )
                DiagnosticRow(
                    "Overflow events",
                    state.captureOverflowEvents?.toString(),
                )
            }

            DiagnosticGroup("GPU LEASES") {
                DiagnosticRow(
                    "Active",
                    state.gpuLeaseActiveCount?.toString(),
                )
                DiagnosticRow(
                    "In flight",
                    state.gpuLeaseInFlightCount?.toString(),
                )
                DiagnosticRow(
                    "Watchdog aborts",
                    state.gpuLeaseWatchdogAborts?.toString(),
                )
                DiagnosticRow(
                    "Stale releases",
                    state.gpuLeaseStaleGenerationReleases?.toString(),
                )
            }

            DiagnosticGroup("MEMORY LEASES") {
                DiagnosticRow(
                    "Active",
                    state.memoryLeaseActiveCount?.toString(),
                )
                DiagnosticRow(
                    "Stale",
                    state.memoryLeaseStaleCount?.toString(),
                )
                DiagnosticRow(
                    "Invalid releases",
                    state.memoryLeaseInvalidReleases?.toString(),
                )
                DiagnosticRow(
                    "Double releases",
                    state.memoryLeaseDoubleReleases?.toString(),
                )
                DiagnosticRow(
                    "Forced reclaims",
                    state.memoryLeaseForcedReclaims?.toString(),
                )
                DiagnosticRow(
                    "Active bytes",
                    state.memoryLeaseBytesActive?.toString(),
                )
                DiagnosticRow(
                    "Oldest age",
                    state.memoryLeaseOldestAgeMilliseconds
                        ?.let { "$it ms" },
                )
            }

            DiagnosticGroup("MOTION EVIDENCE") {
                DiagnosticRow(
                    "Velocity",
                    state.workerVelocityPixelsPerSecond?.let {
                        "%.2f px/s".format(it)
                    },
                )
                DiagnosticRow(
                    "Acceleration",
                    state.workerAccelerationPixelsPerSecond2?.let {
                        "%.2f px/s²".format(it)
                    },
                )
                DiagnosticRow(
                    "Jerk",
                    state.workerJerkPixelsPerSecond3?.let {
                        "%.2f px/s³".format(it)
                    },
                )
                DiagnosticRow(
                    "P95 innovation",
                    state.workerP95InnovationPixels?.let {
                        "%.3f px".format(it)
                    },
                )
                DiagnosticRow(
                    "Direction consistency",
                    state.workerMotionDirectionConsistency.percentOrNa(),
                )
                DiagnosticRow(
                    "Micro-motion",
                    state.workerMicroMotionRetention.percentOrNa(),
                )
                DiagnosticRow(
                    "Dynamic stability",
                    state.workerDynamicStability.percentOrNa(),
                )
                DiagnosticRow(
                    "Exposure blur risk",
                    state.workerExposureMotionBlurRisk.percentOrNa(),
                )
                DiagnosticRow(
                    "Subpixel retention",
                    state.workerSubpixelMotionRetention.percentOrNa(),
                )
            }

            DiagnosticGroup("QUALITY & MEMORY") {
                DiagnosticRow(
                    "Workload",
                    state.workloadScale.percentOrNa(),
                )
                DiagnosticRow(
                    "Worker acceptance",
                    state.workerAcceptedRatio.percentOrNa(),
                )
                DiagnosticRow(
                    "Worker quality",
                    state.workerQuality.percentOrNa(),
                )
                DiagnosticRow(
                    "Temporal gap",
                    state.workerTemporalGap.percentOrNa(),
                )
                DiagnosticRow(
                    "Processed",
                    state.workerProcessedObservations?.toString(),
                )
                DiagnosticRow(
                    "Dropped",
                    state.droppedObservations?.toString(),
                )
                DiagnosticRow(
                    "Buffers",
                    state.activeBufferBlocks?.let { active ->
                        val free = state.freeBufferBlocks
                        if (free != null) {
                            active.toString() + " / " + free
                        } else {
                            active.toString()
                        }
                    },
                )
                DiagnosticRow(
                    "Thermal",
                    state.thermalConstrained?.let {
                        if (it) "CONSTRAINED" else "NORMAL"
                    },
                )
            }
        }
    }
}

@Composable
private fun DiagnosticGroup(
    title: String,
    content: @Composable () -> Unit,
) {
    Column(
        Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        Text(
            title,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Surface(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            color = MaterialTheme.colorScheme.surfaceContainerHighest,
            tonalElevation = 1.dp,
        ) {
            Column(
                Modifier.padding(horizontal = 12.dp, vertical = 9.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                content()
            }
        }
    }
}

@Composable
private fun DiagnosticBadge(label: String) {
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surfaceContainerHighest,
        tonalElevation = 1.dp,
    ) {
        Text(
            label,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            style = MaterialTheme.typography.labelSmall,
        )
    }
}

@Composable
private fun DiagnosticRow(
    label: String,
    value: String?,
) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(
            label,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            value ?: "N/A",
            style = MaterialTheme.typography.labelMedium,
        )
    }
}

private fun Double?.percentOrNa(): String =
    this?.let {
        if (!it.isFinite()) "N/A"
        else (it.coerceIn(0.0, 1.0) * 100.0).toInt().toString() + "%"
    } ?: "N/A"

private fun Long?.microsecondsOrNa(): String =
    this?.let { it.toString() + " µs" } ?: "N/A"

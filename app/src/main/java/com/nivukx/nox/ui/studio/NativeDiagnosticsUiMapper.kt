package com.nivukx.nox.ui.studio

import com.nivukx.nox.core.nativeengine.NativeEngineDiagnostics

/**
 * Keeps native runtime evidence transformation centralized so Compose surfaces do not reinterpret
 * counters independently. Missing values remain missing rather than becoming synthetic health.
 */
object NativeDiagnosticsUiMapper {
    const val CURRENT_ABI = 12

    fun toInspectorState(
        diagnostics: NativeEngineDiagnostics,
        nativeHealthy: Boolean? = null,
    ): DiagnosticsInspectorState =
        DiagnosticsInspectorState(
            nativeAbi = CURRENT_ABI,
            nativeHealthy = nativeHealthy,
            workloadScale = diagnostics.workloadScale,
            queueUtilization = diagnostics.queueUtilization,
            shedding = diagnostics.shedding,
            thermalConstrained = diagnostics.thermalConstrained,
            droppedObservations = diagnostics.droppedObservations,
            captureOverflowEvents = diagnostics.captureOverflowEvents,
            queueDepth = diagnostics.queueDepth,
            maximumQueueDepth = diagnostics.maximumQueueDepth,
            activeCameraCount = diagnostics.activeCameraCount,
            workerCount = diagnostics.workerCount,
            processingEpoch = diagnostics.processingEpoch,
            activeBufferBlocks = diagnostics.activeBufferBlocks,
            freeBufferBlocks = diagnostics.freeBufferBlocks,
            workerProcessedObservations = diagnostics.workerProcessedObservations,
            workerAcceptedRatio =
                diagnostics.workerProcessedObservations
                    .takeIf { it > 0L }
                    ?.let { diagnostics.workerAcceptedRatio },
            workerQuality =
                diagnostics.workerProcessedObservations
                    .takeIf { it > 0L }
                    ?.let { diagnostics.workerQuality },
            workerTemporalGap =
                diagnostics.workerProcessedObservations
                    .takeIf { it > 0L }
                    ?.let { diagnostics.workerTemporalGap },
            workerQueueAgeMicroseconds =
                diagnostics.workerProcessedObservations
                    .takeIf { it > 0L }
                    ?.let { diagnostics.workerQueueAgeMicroseconds },
            workerProcessingMicroseconds =
                diagnostics.workerProcessedObservations
                    .takeIf { it > 0L }
                    ?.let { diagnostics.workerProcessingMicroseconds },
            workerVelocityPixelsPerSecond =
                diagnostics.workerVelocityPixelsPerSecond,
            workerAccelerationPixelsPerSecond2 =
                diagnostics.workerAccelerationPixelsPerSecond2,
            workerJerkPixelsPerSecond3 =
                diagnostics.workerJerkPixelsPerSecond3,
            workerP95InnovationPixels =
                diagnostics.workerP95InnovationPixels,
            workerMotionDirectionConsistency =
                diagnostics.workerMotionDirectionConsistency,
            workerMicroMotionRetention =
                diagnostics.workerMicroMotionRetention,
            workerDynamicStability =
                diagnostics.workerDynamicStability,
            workerExposureMotionBlurRisk =
                diagnostics.workerExposureMotionBlurRisk,
            workerSubpixelMotionRetention =
                diagnostics.workerSubpixelMotionRetention,
        )

    fun mergeTelemetry(
        telemetry: StudioTelemetryState,
        diagnostics: NativeEngineDiagnostics,
    ): StudioTelemetryState =
        telemetry.copy(
            droppedObservations = diagnostics.droppedObservations,
            camerasConnected =
                telemetry.camerasConnected ?: diagnostics.activeCameraCount,
            queueDepth = diagnostics.queueDepth,
            thermalConstrained = diagnostics.thermalConstrained,
            nativeQueueAgeMicroseconds =
                diagnostics.workerQueueAgeMicroseconds,
            nativeProcessingMicroseconds =
                diagnostics.workerProcessingMicroseconds,
            nativeMotionVelocityPixelsPerSecond =
                diagnostics.workerVelocityPixelsPerSecond,
            nativeMotionAccelerationPixelsPerSecond2 =
                diagnostics.workerAccelerationPixelsPerSecond2,
            nativeMotionJerkPixelsPerSecond3 =
                diagnostics.workerJerkPixelsPerSecond3,
            nativeMotionInnovationP95Pixels =
                diagnostics.workerP95InnovationPixels,
            nativeMotionDirectionConsistency =
                diagnostics.workerMotionDirectionConsistency,
            nativeMotionMicroMotionRetention =
                diagnostics.workerMicroMotionRetention,
            nativeMotionDynamicStability =
                diagnostics.workerDynamicStability,
            nativeExposureMotionBlurRisk =
                diagnostics.workerExposureMotionBlurRisk,
            nativeSubpixelMotionRetention =
                diagnostics.workerSubpixelMotionRetention,
            captureOverflowCount =
                diagnostics.captureOverflowEvents,
            gpuLeaseActiveCount =
                diagnostics.gpuLeaseActiveCount,
            gpuLeaseInFlightCount =
                diagnostics.gpuLeaseInFlightCount,
            gpuLeaseWatchdogAborts =
                diagnostics.gpuLeaseWatchdogAborts,
            gpuLeaseStaleGenerationReleases =
                diagnostics.gpuLeaseStaleGenerationReleases,
            memoryLeaseActiveCount =
                diagnostics.memoryLeaseActiveCount,
            memoryLeaseStaleCount =
                diagnostics.memoryLeaseStaleCount,
            memoryLeaseInvalidReleases =
                diagnostics.memoryLeaseInvalidReleases,
            memoryLeaseDoubleReleases =
                diagnostics.memoryLeaseDoubleReleases,
            memoryLeaseForcedReclaims =
                diagnostics.memoryLeaseForcedReclaims,
            memoryLeaseBytesActive =
                diagnostics.memoryLeaseBytesActive,
            memoryLeaseOldestAgeMilliseconds =
                diagnostics.memoryLeaseOldestAgeMilliseconds,
            nativeDropRatio =
                (
                    diagnostics.droppedObservations.toDouble() /
                        (
                            diagnostics.processedObservations +
                                diagnostics.droppedObservations
                            ).coerceAtLeast(1L).toDouble()
                    ).coerceIn(0.0, 1.0),
        )
}

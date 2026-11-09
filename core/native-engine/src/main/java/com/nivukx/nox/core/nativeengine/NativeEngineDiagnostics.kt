package com.nivukx.nox.core.nativeengine

data class NativeEngineDiagnostics(
    val processedObservations: Long,
    val droppedObservations: Long,
    val captureOverflowEvents: Long = 0L,
    val queueDepth: Int = 0,
    val maximumQueueDepth: Int = 0,
    val activeCameraCount: Int = 0,
    val workerCount: Int = 0,
    val processingEpoch: Long,
    val activeBufferBlocks: Int,
    val freeBufferBlocks: Int,
    val workloadScale: Double = 1.0,
    val queueUtilization: Double = 0.0,
    val shedding: Boolean = false,
    val thermalConstrained: Boolean = false,
    val workerProcessedObservations: Long = 0L,
    val workerAcceptedRatio: Double = 0.0,
    val workerQuality: Double = 0.0,
    val workerTemporalGap: Double = 0.0,
    val workerQueueAgeMicroseconds: Long = 0L,
    val workerProcessingMicroseconds: Long = 0L,
    val gpuLeaseActiveCount: Int = 0,
    val gpuLeaseInFlightCount: Int = 0,
    val gpuLeaseWatchdogAborts: Long = 0L,
    val gpuLeaseStaleGenerationReleases: Long = 0L,
    val gpuLeaseInvalidReleases: Long = 0L,
    val gpuLeaseAdmissionRejected: Long = 0L,
    val gpuLeaseOldestAgeMilliseconds: Long = 0L,
    val memoryLeaseActiveCount: Int = 0,
    val memoryLeaseStaleCount: Int = 0,
    val memoryLeaseInvalidReleases: Long = 0L,
    val memoryLeaseDoubleReleases: Long = 0L,
    val memoryLeaseForcedReclaims: Long = 0L,
    val memoryLeaseBytesActive: Long = 0L,
    val memoryLeaseOldestAgeMilliseconds: Long = 0L,
    val workerVelocityPixelsPerSecond: Double = 0.0,
    val workerAccelerationPixelsPerSecond2: Double = 0.0,
    val workerJerkPixelsPerSecond3: Double = 0.0,
    val workerP95InnovationPixels: Double = 0.0,
    val workerMotionDirectionConsistency: Double = 1.0,
    val workerMicroMotionRetention: Double = 1.0,
    val workerDynamicStability: Double = 1.0,
    val workerExposureMotionBlurRisk: Double = 0.0,
    val workerSubpixelMotionRetention: Double = 1.0,
) {
    companion object {
        private const val LEGACY_V6_FIELD_COUNT = 20
        private const val LEGACY_V9_FIELD_COUNT = 41
        private const val LEGACY_V10_FIELD_COUNT = 42
        private const val FIELD_COUNT = 43

        fun fromNative(fields: LongArray): NativeEngineDiagnostics {
            require(
                fields.size == FIELD_COUNT ||
                    fields.size == LEGACY_V10_FIELD_COUNT ||
                    fields.size == LEGACY_V9_FIELD_COUNT ||
                    fields.size == LEGACY_V6_FIELD_COUNT,
            ) {
                "Expected $FIELD_COUNT native diagnostic fields or legacy v10/v9/v6 payload with $LEGACY_V10_FIELD_COUNT/$LEGACY_V9_FIELD_COUNT/$LEGACY_V6_FIELD_COUNT fields"
            }
            return NativeEngineDiagnostics(
                processedObservations = fields[0],
                droppedObservations = fields[1],
                captureOverflowEvents = fields[2],
                queueDepth = fields[3].toInt(),
                maximumQueueDepth = fields[4].toInt(),
                activeCameraCount = fields[5].toInt(),
                workerCount = fields[6].toInt(),
                processingEpoch = fields[7],
                activeBufferBlocks = fields[8].toInt(),
                freeBufferBlocks = fields[9].toInt(),
                workloadScale = fields[10] / 1_000_000.0,
                queueUtilization = fields[11] / 1_000_000.0,
                shedding = fields[12] != 0L,
                thermalConstrained = fields[13] != 0L,
                workerProcessedObservations = fields[14],
                workerAcceptedRatio = fields[15] / 1_000_000.0,
                workerQuality = fields[16] / 1_000_000.0,
                workerTemporalGap = fields[17] / 1_000_000.0,
                workerQueueAgeMicroseconds = fields[18],
                workerProcessingMicroseconds = fields[19],
                gpuLeaseActiveCount =
                    if (fields.size == LEGACY_V6_FIELD_COUNT) 0
                    else fields[20].toInt(),
                gpuLeaseInFlightCount = if (fields.size == LEGACY_V6_FIELD_COUNT) 0 else fields[21].toInt(),
                gpuLeaseWatchdogAborts = if (fields.size == LEGACY_V6_FIELD_COUNT) 0L else fields[22],
                gpuLeaseStaleGenerationReleases = if (fields.size == LEGACY_V6_FIELD_COUNT) 0L else fields[23],
                gpuLeaseInvalidReleases = if (fields.size == LEGACY_V6_FIELD_COUNT) 0L else fields[24],
                gpuLeaseAdmissionRejected = if (fields.size == LEGACY_V6_FIELD_COUNT) 0L else fields[25],
                gpuLeaseOldestAgeMilliseconds = if (fields.size == LEGACY_V6_FIELD_COUNT) 0L else fields[26],
                memoryLeaseActiveCount = if (fields.size == LEGACY_V6_FIELD_COUNT) 0 else fields[27].toInt(),
                memoryLeaseStaleCount = if (fields.size == LEGACY_V6_FIELD_COUNT) 0 else fields[28].toInt(),
                memoryLeaseInvalidReleases = if (fields.size == LEGACY_V6_FIELD_COUNT) 0L else fields[29],
                memoryLeaseDoubleReleases = if (fields.size == LEGACY_V6_FIELD_COUNT) 0L else fields[30],
                memoryLeaseForcedReclaims = if (fields.size == LEGACY_V6_FIELD_COUNT) 0L else fields[31],
                memoryLeaseBytesActive = if (fields.size == LEGACY_V6_FIELD_COUNT) 0L else fields[32],
                memoryLeaseOldestAgeMilliseconds = if (fields.size == LEGACY_V6_FIELD_COUNT) 0L else fields[33],
                workerVelocityPixelsPerSecond =
                    if (fields.size == LEGACY_V6_FIELD_COUNT) 0.0
                    else fields[34] / 1_000.0,
                workerAccelerationPixelsPerSecond2 =
                    if (fields.size == LEGACY_V6_FIELD_COUNT) 0.0
                    else fields[35] / 10_000.0,
                workerJerkPixelsPerSecond3 =
                    if (fields.size == LEGACY_V6_FIELD_COUNT) 0.0
                    else fields[36] / 100_000.0,
                workerP95InnovationPixels =
                    if (fields.size == LEGACY_V6_FIELD_COUNT) 0.0
                    else fields[37] / 1_000.0,
                workerMotionDirectionConsistency =
                    if (fields.size == LEGACY_V6_FIELD_COUNT) 1.0
                    else fields[38] / 1_000_000.0,
                workerMicroMotionRetention =
                    if (fields.size == LEGACY_V6_FIELD_COUNT) 1.0
                    else fields[39] / 1_000_000.0,
                workerDynamicStability =
                    if (fields.size == LEGACY_V6_FIELD_COUNT) 1.0
                    else fields[40] / 1_000_000.0,
                workerExposureMotionBlurRisk =
                    if (fields.size < LEGACY_V10_FIELD_COUNT) {
                        0.0
                    } else {
                        fields[41] / 1_000_000.0
                    },
                workerSubpixelMotionRetention =
                    if (fields.size < FIELD_COUNT) {
                        1.0
                    } else {
                        fields[42] / 1_000_000.0
                    },
            )
        }
    }
}

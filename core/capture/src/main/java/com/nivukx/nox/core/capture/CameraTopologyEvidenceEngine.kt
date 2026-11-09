package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.CameraFeatureMatrix

/**
 * Camera topology evidence authority.
 *
 * Distinguishes independent multi-camera capture from logical multi-camera hardware and, where
 * Android exposes it, distinguishes calibrated hardware sensor synchronization from approximate
 * software synchronization.
 *
 * This is topology evidence only. It never converts a logical camera declaration into a metric
 * accuracy claim.
 */
class CameraTopologyEvidenceEngine {

    enum class Mode {
        INDEPENDENT_MULTI_CAMERA,
        LOGICAL_CALIBRATED,
        LOGICAL_APPROXIMATE,
        LOGICAL_UNKNOWN,
        UNKNOWN,
    }

    data class Result(
        val cameraIds: List<String>,
        val mode: Mode,
        val score: Double,
        val logicalCameraCount: Int,
        val calibratedLogicalCameraCount: Int,
        val approximateLogicalCameraCount: Int,
        val hardwareReadoutCameraCount: Int,
        val physicalChildCameraCount: Int,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val hardwareSynchronizedLogicalEvidenceAvailable: Boolean
            get() = calibratedLogicalCameraCount > 0
    }

    fun evaluate(
        matrices: List<CameraFeatureMatrix>,
        selectedCameraIds: Collection<String> = matrices.map { it.cameraId },
    ): Result {
        val selected = selectedCameraIds
            .distinct()
            .mapNotNull { id ->
                matrices.firstOrNull { it.cameraId == id }
            }

        if (selected.size < 2) {
            return Result(
                cameraIds = selected.map { it.cameraId },
                mode = Mode.UNKNOWN,
                score = 0.0,
                logicalCameraCount = selected.count {
                    it.supportsLogicalMultiCamera
                },
                calibratedLogicalCameraCount = selected.count {
                    it.logicalSensorSyncType ==
                        android.hardware.camera2.CameraMetadata
                            .LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_CALIBRATED
                },
                approximateLogicalCameraCount = selected.count {
                    it.logicalSensorSyncType ==
                        android.hardware.camera2.CameraMetadata
                            .LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_APPROXIMATE
                },
                hardwareReadoutCameraCount = selected.count {
                    it.supportsHardwareReadoutTimestamp
                },
                physicalChildCameraCount = selected.sumOf {
                    it.physicalCameraIds.size
                },
                warnings = listOf(
                    "INSUFFICIENT_CAMERA_TOPOLOGY_EVIDENCE",
                ),
                recommendations = listOf(
                    "Select at least two concrete camera endpoints before evaluating topology.",
                ),
            )
        }

        val logicalCount =
            selected.count { it.supportsLogicalMultiCamera }
        val calibratedLogicalCount =
            selected.count {
                it.logicalSensorSyncType ==
                    android.hardware.camera2.CameraMetadata
                        .LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_CALIBRATED
            }
        val approximateLogicalCount =
            selected.count {
                it.logicalSensorSyncType ==
                    android.hardware.camera2.CameraMetadata
                        .LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_APPROXIMATE
            }
        val readoutCount =
            selected.count { it.supportsHardwareReadoutTimestamp }
        val physicalChildren =
            selected.sumOf { it.physicalCameraIds.size }

        val mode = when {
            calibratedLogicalCount > 0 ->
                Mode.LOGICAL_CALIBRATED
            approximateLogicalCount > 0 ->
                Mode.LOGICAL_APPROXIMATE
            logicalCount > 0 ->
                Mode.LOGICAL_UNKNOWN
            selected.all { it.cameraId.isNotBlank() } ->
                Mode.INDEPENDENT_MULTI_CAMERA
            else ->
                Mode.UNKNOWN
        }

        val logicalScore =
            when (mode) {
                Mode.LOGICAL_CALIBRATED -> 1.0
                Mode.LOGICAL_APPROXIMATE -> 0.74
                Mode.LOGICAL_UNKNOWN -> 0.58
                Mode.INDEPENDENT_MULTI_CAMERA -> 0.82
                Mode.UNKNOWN -> 0.35
            }

        val readoutScore =
            if (readoutCount == 0) {
                0.62
            } else {
                readoutCount.toDouble() /
                    selected.size.toDouble()
            }

        val physicalTopologyScore =
            when {
                physicalChildren >= 4 -> 1.0
                physicalChildren >= 2 -> 0.86
                physicalChildren == 1 -> 0.70
                else -> 0.58
            }

        val score = (
            logicalScore * 0.50 +
                readoutScore * 0.25 +
                physicalTopologyScore * 0.25
            ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (mode == Mode.LOGICAL_APPROXIMATE) {
                add(
                    "LOGICAL_CAMERA_SENSOR_SYNC_IS_APPROXIMATE",
                )
            }
            if (mode == Mode.LOGICAL_UNKNOWN) {
                add(
                    "LOGICAL_CAMERA_SENSOR_SYNC_TYPE_UNKNOWN",
                )
            }
            if (readoutCount < selected.size) {
                add(
                    "HARDWARE_READOUT_TIMESTAMP_COVERAGE_INCOMPLETE",
                )
            }
            if (physicalChildren == 0) {
                add(
                    "PHYSICAL_CHILD_CAMERA_TOPOLOGY_NOT_EXPOSED",
                )
            }
        }.distinct()

        val recommendations = buildList {
            when (mode) {
                Mode.LOGICAL_CALIBRATED ->
                    add(
                        "Use logical-camera hardware timestamp evidence as a timing aid, while retaining calibrated rig synchronization for multi-device fusion.",
                    )
                Mode.LOGICAL_APPROXIMATE ->
                    add(
                        "Prefer independent calibrated camera timestamps when available; treat logical synchronization as supporting evidence only.",
                    )
                Mode.LOGICAL_UNKNOWN ->
                    add(
                        "Collect explicit camera timing evidence before promoting logical-camera timing to a production gate.",
                    )
                Mode.INDEPENDENT_MULTI_CAMERA ->
                    add(
                        "Maintain independent calibration and clock alignment per physical camera endpoint.",
                    )
                Mode.UNKNOWN ->
                    add(
                        "Resolve concrete camera topology before enabling reference-grade multi-view policy.",
                    )
            }
            if (readoutCount < selected.size) {
                add(
                    "Enable hardware readout timestamping only on devices that advertise hardware support and keep the coverage state explicit.",
                )
            }
        }.distinct()

        return Result(
            cameraIds = selected.map { it.cameraId }.sorted(),
            mode = mode,
            score = score,
            logicalCameraCount = logicalCount,
            calibratedLogicalCameraCount = calibratedLogicalCount,
            approximateLogicalCameraCount = approximateLogicalCount,
            hardwareReadoutCameraCount = readoutCount,
            physicalChildCameraCount = physicalChildren,
            warnings = warnings,
            recommendations = recommendations,
        )
    }
}

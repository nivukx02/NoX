package com.nivukx.nox.core.nativeengine.calibration

data class NativeCalibrationResult(
    val cameraId: String,
    val reprojectionRmsPx: Double,
    val confidence: Double,
    val valid: Boolean,
)

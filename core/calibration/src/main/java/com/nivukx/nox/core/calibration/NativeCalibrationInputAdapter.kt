package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.nativeengine.calibration.NativeCalibrationConfig

data class NativeCalibrationSeed(
    val config: NativeCalibrationConfig,
    val snapshot: CalibrationSnapshot,
)

/**
 * Additive adapter from the established Kotlin calibration contract to the native numerical layer.
 * The legacy snapshot is preserved exactly; native refinement consumes a copy/seed in later stages.
 */
object NativeCalibrationInputAdapter {
    fun fromSnapshot(
        snapshot: CalibrationSnapshot,
        config: NativeCalibrationConfig = NativeCalibrationConfig(),
    ): NativeCalibrationSeed = NativeCalibrationSeed(
        config = config,
        snapshot = snapshot,
    )
}

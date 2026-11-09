package com.nivukx.nox.core.nativeengine.calibration

data class NativeCalibrationConfig(
    val huberDeltaPx: Double = 2.0,
    val maxIterations: Int = 16,
    val minimumObservationsPerCamera: Int = 12,
    val refineIntrinsics: Boolean = true,
    val refineDistortion: Boolean = true,
    val focalPriorSigmaRatio: Double = 0.10,
    val principalPointPriorSigmaRatio: Double = 0.05,
    val distortionPriorSigma: Double = 0.05,
    val refinementDamping: Double = 1e-3,
    val refinementIterations: Int = 12,
) {
    init {
        require(huberDeltaPx > 0.0)
        require(maxIterations in 1..64)
        require(minimumObservationsPerCamera >= 6)
        require(focalPriorSigmaRatio in 0.005..0.50)
        require(principalPointPriorSigmaRatio in 0.005..0.25)
        require(distortionPriorSigma in 0.001..0.50)
        require(refinementDamping in 1e-8..1e6)
        require(refinementIterations in 1..64)
    }
}
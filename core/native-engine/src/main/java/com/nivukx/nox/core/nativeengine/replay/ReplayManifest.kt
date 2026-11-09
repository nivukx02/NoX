package com.nivukx.nox.core.nativeengine.replay

data class ReplayManifest(
    val cameraCount: Int,
    val frameCount: Long,
    val clockDriftPpm: Double,
    val calibrationResidualPx: Double,
    val occlusionRatio: Double,
    val droppedFrames: Long,
    val thermalStatus: Int,
    val seed: Int,
)
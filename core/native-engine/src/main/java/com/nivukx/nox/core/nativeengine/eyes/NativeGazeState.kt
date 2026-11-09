package com.nivukx.nox.core.nativeengine.eyes

data class NativeGazeState(
    val binocularConfidence: Double,
    val temporalConfidence: Double,
    val consistent: Boolean,
)

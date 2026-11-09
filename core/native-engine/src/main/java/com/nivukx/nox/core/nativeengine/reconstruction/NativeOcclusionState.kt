package com.nivukx.nox.core.nativeengine.reconstruction

enum class NativeOcclusionState { VISIBLE, PARTIALLY_OCCLUDED, PREDICTED, UNSUPPORTED }

data class NativeOcclusionResult(
    val state: NativeOcclusionState,
    val confidence: Double,
    val uncertainty: Double,
)

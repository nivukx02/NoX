package com.nivukx.nox.core.nativeengine.tracking

data class NativeBodyState(
    val timestampNanos: Long,
    val subjectId: Long,
    val confidence: Double,
    val uncertainty: Double,
    val occlusionRatio: Double,
    val predictedJointCount: Int,
    val healthy: Boolean,
)

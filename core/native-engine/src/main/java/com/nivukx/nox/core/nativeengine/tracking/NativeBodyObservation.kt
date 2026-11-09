package com.nivukx.nox.core.nativeengine.tracking

data class NativeBodyObservation(
    val id: Int,
    val x: Double,
    val y: Double,
    val confidence: Double,
    val timestampNanos: Long,
)

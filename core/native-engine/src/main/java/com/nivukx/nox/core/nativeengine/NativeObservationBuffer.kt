package com.nivukx.nox.core.nativeengine

class NativeObservationBuffer(
    private val engine: NativeMotionEngine,
) {
    @Synchronized
    fun submit(packet: NativeCapturePacket): NativeIngestionResult = engine.ingest(packet)
}

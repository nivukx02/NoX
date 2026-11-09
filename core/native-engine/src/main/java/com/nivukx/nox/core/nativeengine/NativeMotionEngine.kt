package com.nivukx.nox.core.nativeengine

import android.hardware.HardwareBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantReadWriteLock

class NativeMotionEngine private constructor(
    @Volatile
    private var handle: Long,
) : AutoCloseable {

    private val lifecycleLock =
        ReentrantReadWriteLock(true)

    companion object {
        private const val EXPECTED_ABI_VERSION = 12

        init {
            System.loadLibrary("nox_native_engine")
        }

        fun create(
            config: NativeEngineConfig = NativeEngineConfig(),
        ): NativeMotionEngine {
            val actualAbi = nativeAbiVersion()
            check(actualAbi == EXPECTED_ABI_VERSION) {
                "Unsupported NoX native engine ABI: $actualAbi"
            }

            val created = nativeCreate(
                config.workerCount,
                config.observationCapacity,
                config.deterministicSeed,
                config.frameBudgetNs,
                config.validationLevel,
            )
            check(created != 0L) {
                "Native motion engine creation failed"
            }
            return NativeMotionEngine(created)
        }

        private external fun nativeAbiVersion(): Int

        private external fun nativeCreate(
            workerCount: Int,
            observationCapacity: Int,
            deterministicSeed: Int,
            frameBudgetNs: Long,
            validationLevel: Int,
        ): Long

        private external fun nativeStart(
            handle: Long,
        ): Boolean

        private external fun nativeIngest(
            handle: Long,
            cameraId: Int,
            frameId: Long,
            sensorTimestampNanos: Long,
            monotonicTimestampNanos: Long,
            exposureDurationNanos: Long,
            rollingShutterSkewNanos: Long,
            nominalFrameIntervalNanos: Long,
            keypointIds: IntArray,
            keypointXy: FloatArray,
            keypointConfidence: FloatArray,
        ): LongArray

        private external fun nativeAdmitHardwareBuffer(
            handle: Long,
            hardwareBuffer: HardwareBuffer,
        ): LongArray?

        private external fun nativeReleaseHardwareBuffer(
            handle: Long,
            leaseId: Long,
        ): Boolean

        private external fun nativeHardwareBufferDiagnostics(
            handle: Long,
        ): LongArray

        private external fun nativeSetEnvironment(
            handle: Long,
            thermalStatus: Int,
            memoryPressure: Double,
            gpuHeadroom: Double,
            npuHeadroom: Double,
            batterySaver: Boolean,
        )

        private external fun nativeEstimateSynchronization(
            handle: Long,
            referenceTimestamps: LongArray,
            remoteTimestamps: LongArray,
            referenceEvents: DoubleArray,
            remoteEvents: DoubleArray,
            eventSampleIntervalNanos: Double,
        ): DoubleArray

        private external fun nativeStop(
            handle: Long,
        )

        private external fun nativeDestroy(
            handle: Long,
        )

        private external fun nativeDiagnostics(
            handle: Long,
        ): LongArray
    }

    fun start(): Boolean =
        lifecycleLock.writeLock().run {
            lock()
            try {
                check(handle != 0L) {
                    "Native motion engine is closed"
                }
                nativeStart(handle)
            } finally {
                unlock()
            }
        }

    fun ingest(
        packet: NativeCapturePacket,
    ): NativeIngestionResult =
        lifecycleLock.readLock().run {
            lock()
            try {
                val nativeHandle = handle
                check(nativeHandle != 0L) {
                    "Native motion engine is closed"
                }

                NativeIngestionResult.fromNative(
                    nativeIngest(
                        nativeHandle,
                        packet.cameraId,
                        packet.frameId,
                        packet.sensorTimestampNanos,
                        packet.monotonicTimestampNanos,
                        packet.exposureDurationNanos,
                        packet.rollingShutterSkewNanos,
                        packet.nominalFrameIntervalNanos,
                        if (packet.keypointIds.isEmpty()) {
                            IntArray(packet.keypointCount) { it }
                        } else {
                            packet.keypointIds
                        },
                        packet.keypointXy,
                        packet.keypointConfidence,
                    ),
                )
            } finally {
                unlock()
            }
        }

    fun ingestLandmarks(
        cameraId: Int,
        observations: List<com.nivukx.nox.core.model.LandmarkObservation>,
    ): NativeIngestionResult? {
        if (observations.isEmpty()) return null
        val finite = observations.filter {
            it.space == com.nivukx.nox.core.model.LandmarkSpace.PIXELS &&
                it.x.isFinite() &&
                it.y.isFinite() &&
                it.confidence.isFinite()
        }
        if (finite.isEmpty()) return null

        val width = finite.first().frameWidth.coerceAtLeast(1)
        val height = finite.first().frameHeight.coerceAtLeast(1)
        val xy = FloatArray(finite.size * 2)
        val confidence = FloatArray(finite.size)

        finite.forEachIndexed { index, landmark ->
            xy[index * 2] =
                (landmark.x / width.toDouble()).toFloat()
            xy[index * 2 + 1] =
                (landmark.y / height.toDouble()).toFloat()
            confidence[index] =
                landmark.confidence.coerceIn(0.0, 1.0).toFloat()
        }

        val reference = finite.first()
        return ingest(
            NativeCapturePacket(
                cameraId = cameraId,
                frameId = reference.timestampNanos,
                sensorTimestampNanos = reference.timestampNanos,
                monotonicTimestampNanos = reference.timestampNanos,
                exposureDurationNanos = reference.exposureTimeNanos,
                rollingShutterSkewNanos = reference.rollingShutterSkewNanos,
                keypointXy = xy,
                keypointConfidence = confidence,
                keypointIds =
                    IntArray(finite.size) { index ->
                        finite[index].landmarkId
                    },
            ),
        )
    }

    data class HardwareBufferLeaseInfo(
        val leaseId: Long,
        val width: Int,
        val height: Int,
        val layers: Int,
        val format: Int,
        val usage: Long,
        val residentBytes: Long,
    )

    class HardwareBufferLease internal constructor(
        private val engine: NativeMotionEngine,
        val info: HardwareBufferLeaseInfo,
    ) : AutoCloseable {
        private val closed = AtomicBoolean(false)

        override fun close() {
            if (closed.compareAndSet(false, true)) {
                engine.releaseHardwareBuffer(info.leaseId)
            }
        }
    }

    /**
     * Acquires a native reference to a Java HardwareBuffer without copying its storage.
     *
     * The returned lease owns one native AHardwareBuffer reference and must be closed after all
     * GPU/accelerator work that uses the buffer has completed.
     */
    fun acquireHardwareBuffer(
        hardwareBuffer: HardwareBuffer,
    ): HardwareBufferLease? =
        lifecycleLock.readLock().run {
            lock()
            try {
                val nativeHandle = handle
                check(nativeHandle != 0L) {
                    "Native motion engine is closed"
                }

                val fields =
                    nativeAdmitHardwareBuffer(
                        nativeHandle,
                        hardwareBuffer,
                    ) ?: return@run null

                if (fields.size < 7) {
                    return@run null
                }

                HardwareBufferLease(
                    engine = this@NativeMotionEngine,
                    info = HardwareBufferLeaseInfo(
                        leaseId = fields[0],
                        width = fields[1].toInt().coerceAtLeast(0),
                        height = fields[2].toInt().coerceAtLeast(0),
                        layers = fields[3].toInt().coerceAtLeast(0),
                        format = fields[4].toInt(),
                        usage = fields[5],
                        residentBytes = fields[6].coerceAtLeast(0L),
                    ),
                )
            } finally {
                unlock()
            }
        }

    fun releaseHardwareBuffer(
        leaseId: Long,
    ): Boolean =
        lifecycleLock.readLock().run {
            lock()
            try {
                val nativeHandle = handle
                if (nativeHandle == 0L || leaseId <= 0L) {
                    return@run false
                }
                nativeReleaseHardwareBuffer(
                    nativeHandle,
                    leaseId,
                )
            } finally {
                unlock()
            }
        }

    fun hardwareBufferDiagnostics(): HardwareBufferDiagnostics {
        val fields =
            lifecycleLock.readLock().run {
                lock()
                try {
                    val nativeHandle = handle
                    check(nativeHandle != 0L) {
                        "Native motion engine is closed"
                    }
                    nativeHardwareBufferDiagnostics(
                        nativeHandle,
                    )
                } finally {
                    unlock()
                }
            }

        return HardwareBufferDiagnostics.fromNative(fields)
    }

    data class HardwareBufferDiagnostics(
        val activeLeaseCount: Long,
        val residentBytes: Long,
        val acquireFailures: Long,
        val invalidReleaseCount: Long,
        val maximumActiveLeaseCount: Long = 0L,
        val maximumResidentBytes: Long = 0L,
    ) {
        companion object {
            fun fromNative(
                fields: LongArray,
            ): HardwareBufferDiagnostics =
                HardwareBufferDiagnostics(
                    activeLeaseCount =
                        fields.getOrElse(0) { 0L },
                    residentBytes =
                        fields.getOrElse(1) { 0L },
                    acquireFailures =
                        fields.getOrElse(2) { 0L },
                    invalidReleaseCount =
                        fields.getOrElse(3) { 0L },
                    maximumActiveLeaseCount =
                        fields.getOrElse(4) { 0L },
                    maximumResidentBytes =
                        fields.getOrElse(5) { 0L },
                )
        }
    }

    fun setEnvironment(
        thermalStatus: Int,
        memoryPressure: Double,
        gpuHeadroom: Double = 1.0,
        npuHeadroom: Double = 1.0,
        batterySaver: Boolean = false,
    ) {
        lifecycleLock.readLock().run {
            lock()
            try {
                val nativeHandle = handle
                check(nativeHandle != 0L) {
                    "Native motion engine is closed"
                }

                nativeSetEnvironment(
                    nativeHandle,
                    thermalStatus,
                    memoryPressure,
                    gpuHeadroom.coerceIn(0.0, 1.0),
                    npuHeadroom.coerceIn(0.0, 1.0),
                    batterySaver,
                )
            } finally {
                unlock()
            }
        }
    }

    fun estimateSynchronization(
        referenceTimestamps: LongArray,
        remoteTimestamps: LongArray,
        referenceEvents: DoubleArray = DoubleArray(0),
        remoteEvents: DoubleArray = DoubleArray(0),
        eventSampleIntervalNanos: Double = 1.0,
    ): com.nivukx.nox.core.nativeengine.synchronization.NativeSyncResult =
        lifecycleLock.readLock().run {
            lock()
            try {
                val nativeHandle = handle
                check(nativeHandle != 0L) {
                    "Native motion engine is closed"
                }

                com.nivukx.nox.core.nativeengine.synchronization.NativeSyncResult
                    .fromNative(
                        nativeEstimateSynchronization(
                            nativeHandle,
                            referenceTimestamps,
                            remoteTimestamps,
                            referenceEvents,
                            remoteEvents,
                            eventSampleIntervalNanos,
                        ),
                    )
            } finally {
                unlock()
            }
        }

    fun stop() {
        lifecycleLock.writeLock().run {
            lock()
            try {
                if (handle != 0L) {
                    nativeStop(handle)
                }
            } finally {
                unlock()
            }
        }
    }

    fun diagnostics(): NativeEngineDiagnostics =
        lifecycleLock.readLock().run {
            lock()
            try {
                val nativeHandle = handle
                check(nativeHandle != 0L) {
                    "Native motion engine is closed"
                }

                NativeEngineDiagnostics.fromNative(
                    nativeDiagnostics(nativeHandle),
                )
            } finally {
                unlock()
            }
        }

    override fun close() {
        lifecycleLock.writeLock().run {
            lock()
            try {
                if (handle != 0L) {
                    nativeStop(handle)
                    nativeDestroy(handle)
                    handle = 0L
                }
            } finally {
                unlock()
            }
        }
    }
}

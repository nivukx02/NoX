package com.nivukx.nox.core.ai

import android.content.Context
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import kotlin.math.max
import kotlin.math.min

/**
 * Adaptive inference scheduler for sustained multi-camera capture.
 *
 * Capture remains real-time regardless of ML load. The scheduler decides whether a frame should
 * enter the perception graph using sensor time, measured inference cost, memory pressure and Android
 * thermal status. It never drops camera frames from the recorder, only inference work.
 */
class InferenceScheduler(
    context: Context,
    captureFps: Int = 30,
    preferredInferenceFps: Int = 30,
) {
    private var captureFps = captureFps.coerceIn(15, 240)
    private var preferredInferenceFps =
        preferredInferenceFps.coerceIn(5, this.captureFps)

    data class Decision(
        val process: Boolean,
        val effectiveInferenceFps: Int,
        val thermalStatus: Int,
        val heapPressure: Double,
        val nativeMemoryPressure: Double = 0.0,
        val memoryLeakRisk: Double = 0.0,
        val reason: String,
    )

    data class Telemetry(
        val frameCount: Long,
        val throttledFrames: Long,
        val throttledFraction: Double,
        val inferenceEwmaMillis: Double,
        val managedMemoryPressure: Double,
        val nativeMemoryPressure: Double,
        val memoryGrowthBytesPerSecond: Double,
        val memoryLeakRisk: Double,
    )

    private val power =
        context.applicationContext.getSystemService(PowerManager::class.java)

    private var lastInferenceTimestampNanos = Long.MIN_VALUE
    private var ewmaInferenceMillis = 0.0
    private var frameCounter = 0L
    private var throttledFrames = 0L
    private var motionClass =
        MotionCaptureEngine.MotionClass.STATIC
    private var motionReadiness = 0.0
    private val memoryPressureModel =
        RuntimeMemoryPressureModel()

    @Synchronized
    fun configure(
        captureFps: Int,
        preferredInferenceFps: Int,
    ) {
        this.captureFps = captureFps.coerceIn(15, 240)
        this.preferredInferenceFps =
            preferredInferenceFps.coerceIn(
                5,
                this.captureFps,
            )
        lastInferenceTimestampNanos = Long.MIN_VALUE
    }

    fun begin(timestampNanos: Long): Decision {
        frameCounter += 1

        val thermal = if (android.os.Build.VERSION.SDK_INT >= 29) {
            power?.currentThermalStatus ?: PowerManager.THERMAL_STATUS_NONE
        } else {
            PowerManager.THERMAL_STATUS_NONE
        }

        val heap = Debug.MemoryInfo().also {
            Debug.getMemoryInfo(it)
        }

        val managedUsedBytes =
            heap.dalvikPss.toLong() * 1024L
        val managedCapacityBytes =
            Runtime.getRuntime().maxMemory()
                .coerceAtLeast(1L)

        val nativeAllocatedBytes =
            Debug.getNativeHeapAllocatedSize()
        val nativeCapacityBytes =
            Debug.getNativeHeapSize()
                .coerceAtLeast(nativeAllocatedBytes + 1L)

        val memorySnapshot =
            memoryPressureModel.observe(
                RuntimeMemoryPressureModel.Input(
                    timestampNanos = timestampNanos,
                    managedUsedBytes = managedUsedBytes,
                    managedCapacityBytes = managedCapacityBytes,
                    nativeAllocatedBytes = nativeAllocatedBytes,
                    nativeCapacityBytes = nativeCapacityBytes,
                ),
            )

        val heapPressure =
            memorySnapshot.combinedPressure

        val thermalScale = when (thermal) {
            PowerManager.THERMAL_STATUS_SEVERE,
            PowerManager.THERMAL_STATUS_CRITICAL -> 0.25

            PowerManager.THERMAL_STATUS_MODERATE -> 0.55
            PowerManager.THERMAL_STATUS_LIGHT -> 0.80
            else -> 1.0
        }

        val memoryScale = when {
            memorySnapshot.band ==
                RuntimeMemoryPressureModel.Band.CRITICAL -> 0.28
            memorySnapshot.band ==
                RuntimeMemoryPressureModel.Band.HIGH -> 0.52
            memorySnapshot.band ==
                RuntimeMemoryPressureModel.Band.ELEVATED -> 0.76
            else -> 1.0
        }

        val leakProtectionScale =
            when {
                memorySnapshot.leakRisk >= 0.80 -> 0.45
                memorySnapshot.leakRisk >= 0.60 -> 0.62
                memorySnapshot.leakRisk >= 0.40 -> 0.80
                else -> 1.0
            }

        val costScale = when {
            ewmaInferenceMillis > 55.0 -> 0.50
            ewmaInferenceMillis > 40.0 -> 0.68
            ewmaInferenceMillis > 28.0 -> 0.82
            else -> 1.0
        }

        val motionScale = when (motionClass) {
            MotionCaptureEngine.MotionClass.EXTREME -> 1.25
            MotionCaptureEngine.MotionClass.FAST -> 1.12
            MotionCaptureEngine.MotionClass.NORMAL -> 1.00
            MotionCaptureEngine.MotionClass.MICRO_MOTION -> 0.86
            MotionCaptureEngine.MotionClass.STATIC -> 0.72
        }

        val readinessScale =
            if (motionReadiness < 0.35) 0.86
            else if (motionReadiness > 0.78) 1.08
            else 1.0

        val effectiveFps = (preferredInferenceFps *
            motionScale *
            readinessScale *
            min(
                thermalScale,
                min(
                    min(memoryScale, leakProtectionScale),
                    costScale,
                ),
            ))
            .toInt()
            .coerceIn(5, captureFps)

        val intervalNanos = 1_000_000_000L / effectiveFps
        val shouldProcess = lastInferenceTimestampNanos == Long.MIN_VALUE ||
            timestampNanos - lastInferenceTimestampNanos >= intervalNanos

        if (!shouldProcess) throttledFrames += 1

        val reason = when {
            thermal >= PowerManager.THERMAL_STATUS_SEVERE -> "thermal-severe"
            thermal >= PowerManager.THERMAL_STATUS_MODERATE -> "thermal-moderate"
            memorySnapshot.band ==
                RuntimeMemoryPressureModel.Band.CRITICAL ->
                "memory-pressure-critical"
            memorySnapshot.nativePressure >
                memorySnapshot.managedPressure + 0.15 ->
                "native-memory-pressure"
            memorySnapshot.leakRisk >= 0.60 ->
                "persistent-memory-growth"
            heapPressure > 0.90 ->
                "memory-pressure"
            ewmaInferenceMillis > 55.0 ->
                "inference-latency"
            motionClass == MotionCaptureEngine.MotionClass.EXTREME -> "high-motion"
            motionClass == MotionCaptureEngine.MotionClass.FAST -> "fast-motion"
            !shouldProcess -> "rate-limit"
            else -> "full-rate"
        }

        if (shouldProcess) {
            lastInferenceTimestampNanos = timestampNanos
        }

        return Decision(
            process = shouldProcess,
            effectiveInferenceFps = effectiveFps,
            thermalStatus = thermal,
            heapPressure = heapPressure,
            nativeMemoryPressure = memorySnapshot.nativePressure,
            memoryLeakRisk = memorySnapshot.leakRisk,
            reason = reason,
        )
    }

    fun updateMotionContext(
        quality: MotionCaptureEngine.Quality,
    ) {
        motionClass = quality.motionClass
        motionReadiness = quality.captureReadiness
    }

    fun recordInferenceDuration(startElapsedNanos: Long, endElapsedNanos: Long) {
        val durationMillis = ((endElapsedNanos - startElapsedNanos).coerceAtLeast(0L)) / 1e6
        ewmaInferenceMillis = if (ewmaInferenceMillis == 0.0) {
            durationMillis
        } else {
            ewmaInferenceMillis * 0.85 + durationMillis * 0.15
        }
    }

    @Synchronized
    fun telemetry(): Telemetry =
        Telemetry(
            frameCount = frameCounter,
            throttledFrames = throttledFrames,
            throttledFraction =
                if (frameCounter <= 0L) {
                    0.0
                } else {
                    throttledFrames.toDouble() /
                        frameCounter.toDouble()
                },
            inferenceEwmaMillis = ewmaInferenceMillis,
            managedMemoryPressure =
                memoryPressureModel.latest()
                    ?.managedPressure
                    ?: 0.0,
            nativeMemoryPressure =
                memoryPressureModel.latest()
                    ?.nativePressure
                    ?: 0.0,
            memoryGrowthBytesPerSecond =
                memoryPressureModel.latest()
                    ?.growthEwmaBytesPerSecond
                    ?: 0.0,
            memoryLeakRisk =
                memoryPressureModel.latest()
                    ?.leakRisk
                    ?: 0.0,
        )

    fun diagnostics(): String =
        "frames=" + frameCounter +
            " throttled=" + throttledFrames +
            " inferenceEwmaMs=" + "%.2f".format(ewmaInferenceMillis)

    fun reset() {
        lastInferenceTimestampNanos = Long.MIN_VALUE
        ewmaInferenceMillis = 0.0
        frameCounter = 0L
        throttledFrames = 0L
        motionClass = MotionCaptureEngine.MotionClass.STATIC
        motionReadiness = 0.0
        memoryPressureModel.reset()
    }
}

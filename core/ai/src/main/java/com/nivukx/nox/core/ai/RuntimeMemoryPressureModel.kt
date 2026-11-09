package com.nivukx.nox.core.ai

import kotlin.math.max
import kotlin.math.min

/**
 * Native + managed memory pressure model for long-running perception workloads.
 *
 * Android camera pipelines frequently allocate outside the Dalvik heap. Looking only at
 * dalvikPss therefore misses an important class of sustained-memory failure. This model
 * combines managed/native ratios, resident growth rate, and pressure persistence without
 * pretending that memory growth alone proves a leak.
 */
class RuntimeMemoryPressureModel(
    private val config: Config = Config(),
) {

    data class Config(
        val managedElevatedThreshold: Double = 0.68,
        val managedHighThreshold: Double = 0.78,
        val managedCriticalThreshold: Double = 0.90,
        val nativeElevatedThreshold: Double = 0.68,
        val nativeHighThreshold: Double = 0.78,
        val nativeCriticalThreshold: Double = 0.90,
        val growthElevatedBytesPerSecond: Double = 4_000_000.0,
        val growthHighBytesPerSecond: Double = 12_000_000.0,
        val growthCriticalBytesPerSecond: Double = 32_000_000.0,
        val pressureAlpha: Double = 0.18,
        val growthAlpha: Double = 0.20,
        val leakPersistenceFrames: Int = 30,
        val maximumSampleIntervalNanos: Long = 5_000_000_000L,
    ) {
        init {
            listOf(
                managedElevatedThreshold,
                managedHighThreshold,
                managedCriticalThreshold,
                nativeElevatedThreshold,
                nativeHighThreshold,
                nativeCriticalThreshold,
                growthElevatedBytesPerSecond,
                growthHighBytesPerSecond,
                growthCriticalBytesPerSecond,
                pressureAlpha,
                growthAlpha,
            ).forEach { require(it.isFinite()) }

            require(managedElevatedThreshold in 0.0..1.0)
            require(managedHighThreshold in managedElevatedThreshold..1.0)
            require(managedCriticalThreshold in managedHighThreshold..1.0)
            require(nativeElevatedThreshold in 0.0..1.0)
            require(nativeHighThreshold in nativeElevatedThreshold..1.0)
            require(nativeCriticalThreshold in nativeHighThreshold..1.0)
            require(growthElevatedBytesPerSecond > 0.0)
            require(growthHighBytesPerSecond >= growthElevatedBytesPerSecond)
            require(growthCriticalBytesPerSecond >= growthHighBytesPerSecond)
            require(pressureAlpha in 0.01..1.0)
            require(growthAlpha in 0.01..1.0)
            require(leakPersistenceFrames >= 1)
            require(maximumSampleIntervalNanos > 0L)
        }
    }

    enum class Band {
        NOMINAL,
        ELEVATED,
        HIGH,
        CRITICAL,
    }

    data class Input(
        val timestampNanos: Long,
        val managedUsedBytes: Long,
        val managedCapacityBytes: Long,
        val nativeAllocatedBytes: Long,
        val nativeCapacityBytes: Long,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(managedUsedBytes >= 0L)
            require(managedCapacityBytes > 0L)
            require(nativeAllocatedBytes >= 0L)
            require(nativeCapacityBytes > 0L)
        }
    }

    data class Snapshot(
        val timestampNanos: Long,
        val managedPressure: Double,
        val nativePressure: Double,
        val combinedPressure: Double,
        val managedGrowthBytesPerSecond: Double,
        val nativeGrowthBytesPerSecond: Double,
        val combinedGrowthBytesPerSecond: Double,
        val pressureEwma: Double,
        val growthEwmaBytesPerSecond: Double,
        val leakPersistenceFrames: Int,
        val leakRisk: Double,
        val band: Band,
        val warningCodes: List<String>,
    )

    private var previousTimestampNanos = Long.MIN_VALUE
    private var previousManagedBytes = 0L
    private var previousNativeBytes = 0L
    private var pressureEwma = 0.0
    private var growthEwmaBytesPerSecond = 0.0
    private var leakPersistence = 0
    private var latest: Snapshot? = null

    @Synchronized
    fun observe(input: Input): Snapshot {
        val managedPressure =
            ratio(
                input.managedUsedBytes,
                input.managedCapacityBytes,
            )
        val nativePressure =
            ratio(
                input.nativeAllocatedBytes,
                input.nativeCapacityBytes,
            )
        val combinedPressure =
            max(managedPressure, nativePressure)

        var managedGrowth = 0.0
        var nativeGrowth = 0.0

        if (previousTimestampNanos != Long.MIN_VALUE) {
            val dtNanos =
                (input.timestampNanos - previousTimestampNanos)
                    .coerceAtLeast(1L)
                    .coerceAtMost(config.maximumSampleIntervalNanos)

            val dtSeconds =
                dtNanos.toDouble() / 1_000_000_000.0

            managedGrowth =
                (
                    input.managedUsedBytes.toDouble() -
                        previousManagedBytes.toDouble()
                    ) / dtSeconds

            nativeGrowth =
                (
                    input.nativeAllocatedBytes.toDouble() -
                        previousNativeBytes.toDouble()
                    ) / dtSeconds
        }

        val combinedGrowth =
            max(
                0.0,
                managedGrowth + nativeGrowth,
            )

        pressureEwma =
            if (latest == null) {
                combinedPressure
            } else {
                pressureEwma * (1.0 - config.pressureAlpha) +
                    combinedPressure * config.pressureAlpha
            }

        growthEwmaBytesPerSecond =
            if (latest == null) {
                combinedGrowth
            } else {
                growthEwmaBytesPerSecond *
                    (1.0 - config.growthAlpha) +
                    combinedGrowth *
                    config.growthAlpha
            }

        val persistentGrowth =
            growthEwmaBytesPerSecond >=
                config.growthElevatedBytesPerSecond

        leakPersistence =
            when {
                persistentGrowth ->
                    (leakPersistence + 1)
                        .coerceAtMost(config.leakPersistenceFrames * 4)
                else ->
                    (leakPersistence - 1).coerceAtLeast(0)
            }

        val leakRisk =
            (
                normalized(
                    growthEwmaBytesPerSecond,
                    config.growthCriticalBytesPerSecond,
                ) * 0.55 +
                    normalized(
                        pressureEwma,
                        config.managedCriticalThreshold,
                    ) * 0.30 +
                    normalized(
                        leakPersistence.toDouble(),
                        config.leakPersistenceFrames.toDouble(),
                    ) * 0.15
                ).coerceIn(0.0, 1.0)

        val band =
            when {
                managedPressure >= config.managedCriticalThreshold ||
                    nativePressure >= config.nativeCriticalThreshold ||
                    growthEwmaBytesPerSecond >=
                        config.growthCriticalBytesPerSecond ->
                    Band.CRITICAL
                managedPressure >= config.managedHighThreshold ||
                    nativePressure >= config.nativeHighThreshold ||
                    growthEwmaBytesPerSecond >=
                        config.growthHighBytesPerSecond ->
                    Band.HIGH
                managedPressure >= config.managedElevatedThreshold ||
                    nativePressure >= config.nativeElevatedThreshold ||
                    growthEwmaBytesPerSecond >=
                        config.growthElevatedBytesPerSecond ->
                    Band.ELEVATED
                else ->
                    Band.NOMINAL
            }

        val warnings =
            buildList {
                if (managedPressure >= config.managedHighThreshold) {
                    add("MANAGED_MEMORY_PRESSURE_HIGH")
                }
                if (nativePressure >= config.nativeHighThreshold) {
                    add("NATIVE_MEMORY_PRESSURE_HIGH")
                }
                if (nativePressure > managedPressure + 0.15) {
                    add("NATIVE_MEMORY_DOMINATES_MANAGED_HEAP")
                }
                if (
                    growthEwmaBytesPerSecond >=
                        config.growthHighBytesPerSecond
                ) {
                    add("RESIDENT_MEMORY_GROWTH_HIGH")
                }
                if (
                    leakPersistence >=
                        config.leakPersistenceFrames
                ) {
                    add("PERSISTENT_MEMORY_GROWTH_REQUIRES_RESOURCE_AUDIT")
                }
                if (band == Band.CRITICAL) {
                    add("MEMORY_PRESSURE_CRITICAL")
                }
            }.distinct()

        val snapshot =
            Snapshot(
                timestampNanos = input.timestampNanos,
                managedPressure = managedPressure,
                nativePressure = nativePressure,
                combinedPressure = combinedPressure,
                managedGrowthBytesPerSecond = managedGrowth,
                nativeGrowthBytesPerSecond = nativeGrowth,
                combinedGrowthBytesPerSecond = combinedGrowth,
                pressureEwma = pressureEwma,
                growthEwmaBytesPerSecond = growthEwmaBytesPerSecond,
                leakPersistenceFrames = leakPersistence,
                leakRisk = leakRisk,
                band = band,
                warningCodes = warnings,
            )

        previousTimestampNanos = input.timestampNanos
        previousManagedBytes = input.managedUsedBytes
        previousNativeBytes = input.nativeAllocatedBytes
        latest = snapshot
        return snapshot
    }

    @Synchronized
    fun latest(): Snapshot? = latest

    @Synchronized
    fun reset() {
        previousTimestampNanos = Long.MIN_VALUE
        previousManagedBytes = 0L
        previousNativeBytes = 0L
        pressureEwma = 0.0
        growthEwmaBytesPerSecond = 0.0
        leakPersistence = 0
        latest = null
    }

    private fun ratio(
        used: Long,
        capacity: Long,
    ): Double =
        (
            used.toDouble() /
                max(1L, capacity).toDouble()
            ).coerceIn(0.0, 1.25)

    private fun normalized(
        value: Double,
        scale: Double,
    ): Double =
        if (!value.isFinite() || scale <= 0.0) {
            1.0
        } else {
            (value / scale).coerceIn(0.0, 1.0)
        }
}

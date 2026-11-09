package com.nivukx.nox.core.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RuntimeMemoryPressureModelTest {

    @Test
    fun nativePressureIsVisibleEvenWhenManagedHeapLooksHealthy() {
        val model = RuntimeMemoryPressureModel()
        val snapshot = model.observe(
            RuntimeMemoryPressureModel.Input(
                timestampNanos = 1_000_000_000L,
                managedUsedBytes = 300_000_000L,
                managedCapacityBytes = 1_000_000_000L,
                nativeAllocatedBytes = 900_000_000L,
                nativeCapacityBytes = 1_000_000_000L,
            ),
        )

        assertTrue(snapshot.nativePressure > snapshot.managedPressure)
        assertEquals(
            RuntimeMemoryPressureModel.Band.CRITICAL,
            snapshot.band,
        )
        assertTrue(
            snapshot.warningCodes.contains(
                "NATIVE_MEMORY_PRESSURE_HIGH",
            ),
        )
    }

    @Test
    fun persistentGrowthRaisesLeakRiskWithoutCallingItASureLeak() {
        val model =
            RuntimeMemoryPressureModel(
                RuntimeMemoryPressureModel.Config(
                    growthElevatedBytesPerSecond = 1_000_000.0,
                    growthHighBytesPerSecond = 2_000_000.0,
                    growthCriticalBytesPerSecond = 4_000_000.0,
                    leakPersistenceFrames = 4,
                ),
            )

        repeat(10) { index ->
            model.observe(
                RuntimeMemoryPressureModel.Input(
                    timestampNanos =
                        (index + 1L) * 1_000_000_000L,
                    managedUsedBytes =
                        100_000_000L +
                            index * 3_000_000L,
                    managedCapacityBytes = 1_000_000_000L,
                    nativeAllocatedBytes =
                        100_000_000L +
                            index * 10_000_000L,
                    nativeCapacityBytes = 1_000_000_000L,
                ),
            )
        }

        val latest = model.latest()!!
        assertTrue(latest.leakRisk > 0.25)
        assertTrue(
            latest.warningCodes.contains(
                "PERSISTENT_MEMORY_GROWTH_REQUIRES_RESOURCE_AUDIT",
            ),
        )
    }

    @Test
    fun resetRemovesHistoricalGrowth() {
        val model = RuntimeMemoryPressureModel()
        model.observe(
            RuntimeMemoryPressureModel.Input(
                1_000_000_000L,
                800_000_000L,
                1_000_000_000L,
                100_000_000L,
                1_000_000_000L,
            ),
        )
        model.reset()
        val latest = model.observe(
            RuntimeMemoryPressureModel.Input(
                5_000_000_000L,
                100_000_000L,
                1_000_000_000L,
                100_000_000L,
                1_000_000_000L,
            ),
        )
        assertEquals(0.0, latest.managedGrowthBytesPerSecond, 0.0)
        assertEquals(0.0, latest.nativeGrowthBytesPerSecond, 0.0)
        assertTrue(
            latest.band ==
                RuntimeMemoryPressureModel.Band.NOMINAL,
        )
    }
}

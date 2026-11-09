package com.nivukx.nox.core.nativeengine

import org.junit.Assert.assertEquals
import org.junit.Test

class NativeEngineConfigTest {
    @Test
    fun defaultConfigurationMatchesFoundationContract() {
        val config = NativeEngineConfig()
        assertEquals(2, config.workerCount)
        assertEquals(4096, config.observationCapacity)
        assertEquals(16_666_667L, config.frameBudgetNs)
        assertEquals(1, config.validationLevel)
    }

    @Test
    fun federated_rig_profile_scales_for_50_actors_and_128_cameras() {
        val config =
            NativeEngineConfig.federatedRig(
                actorCount = 50,
                cameraCount = 128,
                targetFps = 60,
            )

        assertEquals(
            65_536,
            config.observationCapacity,
        )
        assertEquals(
            16_666_666L,
            config.frameBudgetNs,
        )
        assertEquals(2, config.workerCount.coerceAtLeast(2))
        assertEquals(2, config.validationLevel)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsZeroWorkers() {
        NativeEngineConfig(workerCount = 0)
    }
}

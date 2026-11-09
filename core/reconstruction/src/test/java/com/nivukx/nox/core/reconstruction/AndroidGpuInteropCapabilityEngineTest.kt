package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AndroidGpuInteropCapabilityEngineTest {
    private val engine = AndroidGpuInteropCapabilityEngine()

    @Test
    fun fullyValidatedAhardwareBufferPathIsZeroCopyReady() {
        val decision = engine.decide(
            capabilities = AndroidGpuInteropCapabilityEngine.DeviceCapabilities(
                androidApi = 35,
                vulkanAvailable = true,
                externalMemoryAhardwareBuffer = true,
                gpuSampledImageUsage = true,
                cameraProvidesAhardwareBuffer = true,
                cameraUsageContract =
                    AndroidGpuInteropCapabilityEngine.BufferContract.CAMERA_SAMPLE_ONLY,
                bufferLifetimeRetainedUntilGpuComplete = true,
                maxImages = 4,
            ),
            workload = AndroidGpuInteropCapabilityEngine.Workload(
                width = 1920,
                height = 1080,
                batchSize = 1,
            ),
        )

        assertEquals(
            AndroidGpuInteropCapabilityEngine.Backend.VULKAN_AHARDWARE_BUFFER_ZERO_COPY,
            decision.backend,
        )
        assertTrue(decision.ready)
        assertTrue(decision.zeroCopy)
        assertTrue(decision.requiresBufferAcquire)
        assertTrue(decision.requiresGpuCompletionBeforeRelease)
        assertEquals(0, decision.expectedHostCopies)
    }

    @Test
    fun missingExternalMemoryNeverPretendsToBeZeroCopy() {
        val decision = engine.decide(
            capabilities = AndroidGpuInteropCapabilityEngine.DeviceCapabilities(
                androidApi = 35,
                vulkanAvailable = true,
                externalMemoryAhardwareBuffer = false,
                gpuSampledImageUsage = true,
                cameraProvidesAhardwareBuffer = true,
                cameraUsageContract =
                    AndroidGpuInteropCapabilityEngine.BufferContract.CAMERA_SAMPLE_ONLY,
                bufferLifetimeRetainedUntilGpuComplete = true,
                maxImages = 4,
            ),
            workload = AndroidGpuInteropCapabilityEngine.Workload(
                width = 1280,
                height = 720,
                batchSize = 1,
            ),
        )

        assertFalse(decision.zeroCopy)
        assertTrue(
            decision.backend ==
                AndroidGpuInteropCapabilityEngine.Backend.OPENGL_EXTERNAL_IMAGE,
        )
    }

    @Test
    fun unsafeLifetimeFallsBackInsteadOfAllowingUseAfterFree() {
        val decision = engine.decide(
            capabilities = AndroidGpuInteropCapabilityEngine.DeviceCapabilities(
                androidApi = 35,
                vulkanAvailable = true,
                externalMemoryAhardwareBuffer = true,
                gpuSampledImageUsage = true,
                cameraProvidesAhardwareBuffer = true,
                cameraUsageContract =
                    AndroidGpuInteropCapabilityEngine.BufferContract.CAMERA_SAMPLE_ONLY,
                bufferLifetimeRetainedUntilGpuComplete = false,
                maxImages = 4,
            ),
            workload = AndroidGpuInteropCapabilityEngine.Workload(
                width = 1920,
                height = 1080,
                batchSize = 1,
            ),
        )

        assertFalse(decision.ready)
        assertFalse(decision.zeroCopy)
        assertTrue(
            decision.warnings.any {
                it.contains("released before GPU completion")
            },
        )
    }

    @Test
    fun cpuReadbackRemovesEndToEndZeroCopyClaim() {
        val decision = engine.decide(
            capabilities = AndroidGpuInteropCapabilityEngine.DeviceCapabilities(
                androidApi = 35,
                vulkanAvailable = true,
                externalMemoryAhardwareBuffer = true,
                gpuSampledImageUsage = true,
                cameraProvidesAhardwareBuffer = true,
                cameraUsageContract =
                    AndroidGpuInteropCapabilityEngine.BufferContract.CAMERA_SAMPLE_ONLY,
                bufferLifetimeRetainedUntilGpuComplete = true,
                maxImages = 4,
            ),
            workload = AndroidGpuInteropCapabilityEngine.Workload(
                width = 1920,
                height = 1080,
                batchSize = 1,
                cpuReadbackRequired = true,
            ),
        )

        assertFalse(decision.zeroCopy)
        assertEquals(1, decision.expectedHostCopies)
    }

    @Test
    fun insufficientBufferPoolDoesNotCreateUnboundedPinnedImages() {
        val decision = engine.decide(
            capabilities = AndroidGpuInteropCapabilityEngine.DeviceCapabilities(
                androidApi = 35,
                vulkanAvailable = true,
                externalMemoryAhardwareBuffer = true,
                gpuSampledImageUsage = true,
                cameraProvidesAhardwareBuffer = true,
                cameraUsageContract =
                    AndroidGpuInteropCapabilityEngine.BufferContract.CAMERA_SAMPLE_ONLY,
                bufferLifetimeRetainedUntilGpuComplete = true,
                maxImages = 2,
            ),
            workload = AndroidGpuInteropCapabilityEngine.Workload(
                width = 3840,
                height = 2160,
                batchSize = 1,
            ),
        )

        assertFalse(decision.ready)
        assertTrue(decision.pinnedImageBudget <= 2)
        assertTrue(
            decision.backend ==
                AndroidGpuInteropCapabilityEngine.Backend.VULKAN_STAGING_COPY,
        )
    }
}

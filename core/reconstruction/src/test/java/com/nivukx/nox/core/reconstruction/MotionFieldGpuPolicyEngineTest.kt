package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MotionFieldGpuMode
import com.nivukx.nox.core.model.MotionFieldGpuBackend
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class MotionFieldGpuPolicyEngineTest {
    @Test
    fun maximum_mode_selects_vulkan_only_when_implementation_and_interop_are_verified() {
        val decision = MotionFieldGpuPolicyEngine().decide(
            MotionFieldGpuPolicyEngine.Input(
                vulkanAvailable = true,
                vulkanExternalMemoryInterop = true,
                vulkanRendererImplemented = true,
                gles31Available = true,
                gles32Available = true,
                userMode = MotionFieldGpuMode.MAXIMUM,
                qualityScale = 1.0,
                memoryScale = 1.0,
                actorCount = 4,
                detail = 1.0,
            ),
        )

        assertEquals(MotionFieldGpuBackend.VULKAN, decision.backend)
        assertTrue(decision.geometryJointBudget >= 128)
    }

    @Test
    fun maximum_mode_falls_back_to_gles_compute_without_false_vulkan_claim() {
        val decision = MotionFieldGpuPolicyEngine().decide(
            MotionFieldGpuPolicyEngine.Input(
                vulkanAvailable = true,
                vulkanExternalMemoryInterop = false,
                vulkanRendererImplemented = false,
                gles31Available = true,
                gles32Available = true,
                userMode = MotionFieldGpuMode.MAXIMUM,
                qualityScale = 1.0,
                memoryScale = 1.0,
                actorCount = 4,
                detail = 1.0,
            ),
        )

        assertEquals(MotionFieldGpuBackend.OPENGL_ES_COMPUTE, decision.backend)
    }

    @Test
    fun thermal_protection_reduces_detail_without_disabling_camera_render() {
        val decision = MotionFieldGpuPolicyEngine().decide(
            MotionFieldGpuPolicyEngine.Input(
                gles31Available = true,
                userMode = MotionFieldGpuMode.MAXIMUM,
                qualityScale = 0.30,
                memoryScale = 1.0,
                actorCount = 32,
                detail = 1.0,
            ),
        )

        assertEquals(MotionFieldGpuBackend.OPENGL_ES_COMPUTE, decision.backend)
        assertTrue(decision.geometryJointBudget < 32 * 33)
        assertTrue(decision.updateBudgetScale < 1.0)
        assertTrue(decision.warnings.any { it.contains("THERMAL") })
    }

    @Test
    fun vulkanWithoutZeroCopyIsExplicitlyFlagged() {
        val decision = MotionFieldGpuPolicyEngine().decide(
            MotionFieldGpuPolicyEngine.Input(
                vulkanAvailable = true,
                vulkanExternalMemoryInterop = true,
                vulkanRendererImplemented = true,
                zeroCopyCameraInputReady = false,
                expectedCameraHostCopies = 1,
                actorCount = 1,
                detail = 1.0,
            ),
        )
        assertTrue(
            decision.warnings.contains(
                "VULKAN_CAMERA_INPUT_REQUIRES_COPY_PATH",
            ),
        )
        assertTrue(
            decision.warnings.any {
                it.startsWith("CAMERA_HOST_COPIES=")
            },
        )
    }

}

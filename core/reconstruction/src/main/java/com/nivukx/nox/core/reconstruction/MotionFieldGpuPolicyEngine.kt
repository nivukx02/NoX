package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MotionFieldGpuBackend
import com.nivukx.nox.core.model.MotionFieldGpuMode

class MotionFieldGpuPolicyEngine(
    private val maximumJointBudget: Int = 32 * 33,
) {
    data class Input(
        val vulkanAvailable: Boolean = false,
        val vulkanExternalMemoryInterop: Boolean = false,
        val vulkanRendererImplemented: Boolean = false,
        val zeroCopyCameraInputReady: Boolean = false,
        val expectedCameraHostCopies: Int = 1,
        val gles31Available: Boolean = false,
        val gles32Available: Boolean = false,
        val userMode: MotionFieldGpuMode = MotionFieldGpuMode.AUTO,
        val qualityScale: Double = 1.0,
        val memoryScale: Double = 1.0,
        val actorCount: Int = 0,
        val detail: Double = 1.0,
    )

    data class Decision(
        val backend: MotionFieldGpuBackend,
        val geometryJointBudget: Int,
        val updateBudgetScale: Double,
        val renderScale: Double,
        val warnings: List<String>,
    )

    fun decide(input: Input): Decision {
        require(input.qualityScale in 0.0..1.0)
        require(input.memoryScale in 0.0..1.0)
        require(input.actorCount >= 0)
        require(input.expectedCameraHostCopies >= 0)
        require(input.detail in 0.0..1.0)

        val normalizedQuality = input.qualityScale.coerceIn(0.0, 1.0)
        val normalizedMemory = input.memoryScale.coerceIn(0.0, 1.0)
        val protection = minOf(normalizedQuality, normalizedMemory)

        val backend = when {
            input.vulkanAvailable &&
                input.vulkanExternalMemoryInterop &&
                input.vulkanRendererImplemented -> MotionFieldGpuBackend.VULKAN

            input.gles31Available -> MotionFieldGpuBackend.OPENGL_ES_COMPUTE
            input.gles32Available -> MotionFieldGpuBackend.OPENGL_ES_RENDER
            else -> MotionFieldGpuBackend.UNAVAILABLE
        }

        val modeScale = when (input.userMode) {
            MotionFieldGpuMode.AUTO -> 0.90
            MotionFieldGpuMode.MAXIMUM -> 1.0
            MotionFieldGpuMode.HIGH -> 0.86
            MotionFieldGpuMode.BALANCED -> 0.68
        }

        val protectionScale = when {
            protection >= 0.80 -> 1.0
            protection >= 0.65 -> 0.84
            protection >= 0.50 -> 0.68
            protection >= 0.35 -> 0.48
            else -> 0.30
        }

        val detailScale = (
            input.detail.coerceIn(0.0, 1.0) *
                modeScale *
                protectionScale
        ).coerceIn(0.16, 1.0)

        val actorPressure = when {
            input.actorCount <= 4 -> 1.0
            input.actorCount <= 8 -> 0.92
            input.actorCount <= 16 -> 0.78
            else -> 0.62
        }

        val geometryBudget = (
            maximumJointBudget *
                detailScale *
                actorPressure
        ).toInt().coerceIn(16, maximumJointBudget)

        val updateBudgetScale = (
            protectionScale *
                modeScale *
                actorPressure
        ).coerceIn(0.20, 1.0)

        val renderScale = when {
            protection < 0.35 -> 0.70
            protection < 0.50 -> 0.80
            protection < 0.65 -> 0.90
            else -> 1.0
        }

        val warnings = buildList {
            if (backend == MotionFieldGpuBackend.UNAVAILABLE) {
                add("NO_VALIDATED_GPU_BACKEND")
            }
            if (protection < 0.50) {
                add("RUNTIME_QUALITY_OR_MEMORY_PROTECTION_ACTIVE")
            }
            if (
                input.userMode == MotionFieldGpuMode.MAXIMUM &&
                protection < 0.50
            ) {
                add("THERMAL_PROTECTION_ACTIVE")
            }
            if (input.actorCount > 16) {
                add("HIGH_ACTOR_DENSITY")
            }
            if (input.userMode == MotionFieldGpuMode.MAXIMUM &&
                backend != MotionFieldGpuBackend.VULKAN) {
                add("MAXIMUM_GPU_USING_VALIDATED_GLES_PATH")
            }
            if (backend == MotionFieldGpuBackend.VULKAN &&
                !input.zeroCopyCameraInputReady) {
                add("VULKAN_CAMERA_INPUT_REQUIRES_COPY_PATH")
            }
            if (input.expectedCameraHostCopies > 0) {
                add(
                    "CAMERA_HOST_COPIES=" +
                        input.expectedCameraHostCopies,
                )
            }
        }

        return Decision(
            backend = backend,
            geometryJointBudget = geometryBudget,
            updateBudgetScale = updateBudgetScale,
            renderScale = renderScale,
            warnings = warnings,
        )
    }
}

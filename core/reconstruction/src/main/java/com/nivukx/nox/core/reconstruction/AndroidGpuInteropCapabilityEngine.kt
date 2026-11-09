package com.nivukx.nox.core.reconstruction

/**
 * Capability and lifetime policy for camera-buffer -> GPU inference interop.
 *
 * This class does not claim that a device supports zero-copy merely because Vulkan exists.
 * Android AHardwareBuffer import requires several independent capabilities:
 * API level, GPU sampled-image usage, Vulkan external-memory support, camera allocation flags,
 * and an ownership/lifetime policy that keeps the imported buffer alive until GPU work completes.
 *
 * The planner is deliberately platform-neutral at the call site so the native camera/Vulkan layer
 * can supply verified facts and receive a deterministic decision.
 */
class AndroidGpuInteropCapabilityEngine(
    private val config: Config = Config(),
) {
    enum class Backend {
        VULKAN_AHARDWARE_BUFFER_ZERO_COPY,
        VULKAN_STAGING_COPY,
        OPENGL_EXTERNAL_IMAGE,
        CPU_FALLBACK,
    }

    enum class BufferContract {
        CAMERA_SAMPLE_ONLY,
        CAMERA_SAMPLE_AND_CPU_READ,
        SHARED_PROCESS_MEMORY,
        UNSUPPORTED,
    }

    enum class DecisionReason {
        READY_FOR_ZERO_COPY,
        API_LEVEL_TOO_OLD,
        EXTERNAL_MEMORY_UNAVAILABLE,
        GPU_SAMPLED_USAGE_UNAVAILABLE,
        CAMERA_USAGE_MISMATCH,
        BUFFER_LIFETIME_UNSAFE,
        FRAME_POOL_TOO_SMALL,
        VULKAN_READY_BUT_COPY_REQUIRED,
        NO_GPU_PATH,
    }

    data class Config(
        val minimumAndroidApi: Int = 26,
        val recommendedMaxImages: Int = 4,
        val maximumPinnedImages: Int = 6,
        val requireExplicitOwnership: Boolean = true,
        val preferZeroCopy: Boolean = true,
        val maximumLeaseAgeMilliseconds: Double = 250.0,

    ) {
        init {
            require(minimumAndroidApi >= 26)
            require(recommendedMaxImages in 2..16)
            require(maximumPinnedImages >= recommendedMaxImages)
            require(maximumLeaseAgeMilliseconds > 0.0)
        }
    }

    data class DeviceCapabilities(
        val androidApi: Int,
        val vulkanAvailable: Boolean,
        val externalMemoryAhardwareBuffer: Boolean,
        val gpuSampledImageUsage: Boolean,
        val cameraProvidesAhardwareBuffer: Boolean,
        val cameraUsageContract: BufferContract,
        val bufferLifetimeRetainedUntilGpuComplete: Boolean,
        val maxImages: Int,
        val protectedContent: Boolean = false,
    ) {
        init {
            require(androidApi >= 21)
            require(maxImages >= 0)
        }
    }

    data class Workload(
        val width: Int,
        val height: Int,
        val batchSize: Int,
        val cpuReadbackRequired: Boolean = false,
        val longLivedCache: Boolean = true,
    ) {
        init {
            require(width > 0)
            require(height > 0)
            require(batchSize > 0)
        }
    }

    data class Decision(
        val backend: Backend,
        val reason: DecisionReason,
        val ready: Boolean,
        val zeroCopy: Boolean,
        val requiresBufferAcquire: Boolean,
        val requiresGpuCompletionBeforeRelease: Boolean,
        val recommendedMaxImages: Int,
        val pinnedImageBudget: Int,
        val expectedHostCopies: Int,
        val warnings: List<String>,
        val requiresGenerationTrackedLease: Boolean = false,
        val maximumInFlightLeases: Int = 0,
        val leaseWatchdogMilliseconds: Double = 0.0,
    )

    fun decide(
        capabilities: DeviceCapabilities,
        workload: Workload,
    ): Decision {
        val warnings = mutableListOf<String>()

        if (capabilities.protectedContent) {
            warnings += "Protected-content buffers require a separate protected GPU path."
        }

        if (capabilities.androidApi < config.minimumAndroidApi) {
            return fallback(
                DecisionReason.API_LEVEL_TOO_OLD,
                warnings + "Android API level is below the AHardwareBuffer Vulkan import floor.",
            )
        }

        if (!capabilities.vulkanAvailable ||
            !capabilities.externalMemoryAhardwareBuffer
        ) {
            if (capabilities.gpuSampledImageUsage) {
                return Decision(
                    backend = Backend.OPENGL_EXTERNAL_IMAGE,
                    reason = DecisionReason.EXTERNAL_MEMORY_UNAVAILABLE,
                    ready = true,
                    zeroCopy = false,
                    requiresBufferAcquire = true,
                    requiresGpuCompletionBeforeRelease = true,
                    recommendedMaxImages =
                        config.recommendedMaxImages,
                    pinnedImageBudget =
                        minOf(
                            config.maximumPinnedImages,
                            capabilities.maxImages.coerceAtLeast(1),
                        ),
                    expectedHostCopies = 1,
                    warnings =
                        warnings +
                            "Vulkan AHardwareBuffer import is unavailable; using a non-zero-copy GPU path.",
                )
            }
            return fallback(
                DecisionReason.EXTERNAL_MEMORY_UNAVAILABLE,
                warnings,
            )
        }

        if (!capabilities.cameraProvidesAhardwareBuffer) {
            return Decision(
                backend = Backend.VULKAN_STAGING_COPY,
                reason = DecisionReason.CAMERA_USAGE_MISMATCH,
                ready = true,
                zeroCopy = false,
                requiresBufferAcquire = false,
                requiresGpuCompletionBeforeRelease = true,
                recommendedMaxImages = config.recommendedMaxImages,
                pinnedImageBudget = 0,
                expectedHostCopies = 1,
                warnings =
                    warnings +
                        "Camera input is not exposed as AHardwareBuffer; staging import is required.",
            )
        }

        if (!capabilities.gpuSampledImageUsage) {
            return Decision(
                backend = Backend.VULKAN_STAGING_COPY,
                reason = DecisionReason.GPU_SAMPLED_USAGE_UNAVAILABLE,
                ready = true,
                zeroCopy = false,
                requiresBufferAcquire = false,
                requiresGpuCompletionBeforeRelease = true,
                recommendedMaxImages = config.recommendedMaxImages,
                pinnedImageBudget = 0,
                expectedHostCopies = 1,
                warnings =
                    warnings +
                        "Camera buffer lacks GPU sampled-image usage; do not force import.",
            )
        }

        if (
            capabilities.cameraUsageContract ==
                BufferContract.UNSUPPORTED
        ) {
            return fallback(
                DecisionReason.CAMERA_USAGE_MISMATCH,
                warnings,
            )
        }

        if (
            config.requireExplicitOwnership &&
            !capabilities.bufferLifetimeRetainedUntilGpuComplete
        ) {
            return Decision(
                backend = Backend.VULKAN_STAGING_COPY,
                reason = DecisionReason.BUFFER_LIFETIME_UNSAFE,
                ready = false,
                zeroCopy = false,
                requiresBufferAcquire = true,
                requiresGpuCompletionBeforeRelease = true,
                recommendedMaxImages = config.recommendedMaxImages,
                pinnedImageBudget = 0,
                expectedHostCopies = 1,
                warnings =
                    warnings +
                        "The camera image would be released before GPU completion; import is unsafe.",
            )
        }

        if (
            capabilities.maxImages <
                config.recommendedMaxImages
        ) {
            return Decision(
                backend = Backend.VULKAN_STAGING_COPY,
                reason = DecisionReason.FRAME_POOL_TOO_SMALL,
                ready = false,
                zeroCopy = false,
                requiresBufferAcquire = true,
                requiresGpuCompletionBeforeRelease = true,
                recommendedMaxImages = capabilities.maxImages,
                pinnedImageBudget = capabilities.maxImages,
                expectedHostCopies = 1,
                warnings =
                    warnings +
                        "Camera buffer pool is too small for bounded in-flight processing.",
            )
        }

        val pinnedBudget =
            minOf(
                config.maximumPinnedImages,
                capabilities.maxImages,
            ).coerceAtLeast(2)

        val hostCopies =
            if (workload.cpuReadbackRequired) 1 else 0

        if (hostCopies > 0) {
            warnings +=
                "Workload requests CPU readback; end-to-end processing is not zero-copy."
        }

        val maximumInFlightLeases =
            (pinnedBudget - 1).coerceIn(1, pinnedBudget)

        return Decision(
            backend =
                if (config.preferZeroCopy) {
                    Backend.VULKAN_AHARDWARE_BUFFER_ZERO_COPY
                } else {
                    Backend.VULKAN_STAGING_COPY
                },
            reason =
                if (config.preferZeroCopy) {
                    DecisionReason.READY_FOR_ZERO_COPY
                } else {
                    DecisionReason.VULKAN_READY_BUT_COPY_REQUIRED
                },
            ready = true,
            zeroCopy =
                config.preferZeroCopy &&
                    !workload.cpuReadbackRequired,
            requiresBufferAcquire = true,
            requiresGpuCompletionBeforeRelease = true,
            recommendedMaxImages =
                config.recommendedMaxImages,
            pinnedImageBudget = pinnedBudget,
            expectedHostCopies = hostCopies,
            warnings = warnings,
            requiresGenerationTrackedLease =
                config.preferZeroCopy &&
                    !workload.cpuReadbackRequired,
            maximumInFlightLeases =
                if (config.preferZeroCopy) {
                    maximumInFlightLeases
                } else {
                    0
                },
            leaseWatchdogMilliseconds =
                if (config.preferZeroCopy) {
                    config.maximumLeaseAgeMilliseconds
                } else {
                    0.0
                },
        )
    }

    private fun fallback(
        reason: DecisionReason,
        warnings: List<String>,
    ): Decision =
        Decision(
            backend = Backend.CPU_FALLBACK,
            reason = reason,
            ready = false,
            zeroCopy = false,
            requiresBufferAcquire = false,
            requiresGpuCompletionBeforeRelease = false,
            recommendedMaxImages = config.recommendedMaxImages,
            pinnedImageBudget = 0,
            expectedHostCopies = 1,
            warnings = warnings + "Using CPU fallback; no unsafe import is attempted.",
        )
}

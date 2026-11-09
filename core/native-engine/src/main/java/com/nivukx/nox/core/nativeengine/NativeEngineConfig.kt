package com.nivukx.nox.core.nativeengine

data class NativeEngineConfig(
    val workerCount: Int = 2,
    val observationCapacity: Int = 4096,
    val deterministicSeed: Int = 0,
    val frameBudgetNs: Long = 16_666_667L,
    val validationLevel: Int = 1,
) {
    init {
        require(workerCount in 1..64) { "workerCount must be in 1..64" }
        require(observationCapacity in 64..1_000_000) {
            "observationCapacity must be in 64..1,000,000"
        }
        require(frameBudgetNs >= 1_000_000L) { "frameBudgetNs must be at least 1 ms" }
        require(validationLevel in 0..2) { "validationLevel must be in 0..2" }
    }

    companion object {
        fun federatedRig(
            actorCount: Int,
            cameraCount: Int,
            targetFps: Int = 60,
        ): NativeEngineConfig {
            require(actorCount in 1..50)
            require(cameraCount in 2..128)
            require(targetFps in 15..240)

            val processors =
                Runtime.getRuntime()
                    .availableProcessors()
                    .coerceAtLeast(2)

            val workers =
                (processors - 1)
                    .coerceIn(4, 16)

            val observationCapacity =
                maxOf(
                    16_384,
                    cameraCount * 256,
                    actorCount * 512,
                ).coerceAtMost(1_000_000)

            return NativeEngineConfig(
                workerCount = workers,
                observationCapacity = observationCapacity,
                frameBudgetNs =
                    1_000_000_000L /
                        targetFps.toLong(),
                deterministicSeed = 0,
                validationLevel = 2,
            )
        }
    }
}

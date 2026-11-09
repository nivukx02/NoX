package com.nivukx.nox.graphics

enum class MotionFieldRendererLifecycle {
    NEW,
    READY,
    FAILED,
    RELEASED,
}

data class MotionFieldRendererHealth(
    val lifecycle: MotionFieldRendererLifecycle = MotionFieldRendererLifecycle.NEW,
    val lastError: String? = null,
    val frameCount: Long = 0L,
    val droppedFrames: Long = 0L,
) {
    init {
        require(frameCount >= 0L)
        require(droppedFrames >= 0L)
    }
}

package com.nivukx.nox.core.model

enum class FocusTargetMode {
    EYE,
    FACE,
    SUBJECT,
    CENTER,
}

data class FocusTarget(
    val timestampNanos: Long,
    val normalizedX: Double,
    val normalizedY: Double,
    val confidence: Double,
    val mode: FocusTargetMode,
    val trackingId: Int? = null,
    val subjectType: SubjectType = SubjectType.UNKNOWN,
) {
    init {
        require(normalizedX in 0.0..1.0)
        require(normalizedY in 0.0..1.0)
        require(confidence in 0.0..1.0)
    }

    fun distanceTo(other: FocusTarget): Double {
        val dx = normalizedX - other.normalizedX
        val dy = normalizedY - other.normalizedY
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    fun lerp(
        other: FocusTarget,
        alpha: Double,
    ): FocusTarget =
        copy(
            normalizedX =
                normalizedX * (1.0 - alpha) +
                    other.normalizedX * alpha,
            normalizedY =
                normalizedY * (1.0 - alpha) +
                    other.normalizedY * alpha,
            confidence =
                confidence * (1.0 - alpha) +
                    other.confidence * alpha,
            timestampNanos =
                maxOf(timestampNanos, other.timestampNanos),
        )
}

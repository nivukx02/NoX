package com.nivukx.nox.core.ai

/**
 * Resolves display orientation for decoded video frames.
 *
 * The explicit user override is authoritative. Container rotation is the fallback and zero is
 * used only when neither source provides orientation metadata.
 */
internal fun resolveRecordedVideoRotation(
    overrideDegrees: Int?,
    containerDegrees: Int?,
): Int = overrideDegrees ?: containerDegrees ?: 0

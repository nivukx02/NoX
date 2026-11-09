package com.nivukx.nox.core.ai

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertTrue

class EyeTrackingInputBoundarySourceTest {
    @Test
    fun finiteEvidenceValidationPrecedesTrackMutationAndPixelSampling() {
        val candidates = listOf(
            Path.of("core/ai/src/main/java/com/nivukx/nox/core/ai/EyeTrackingEngine.kt"),
            Path.of("src/main/java/com/nivukx/nox/core/ai/EyeTrackingEngine.kt"),
            Path.of("../core/ai/src/main/java/com/nivukx/nox/core/ai/EyeTrackingEngine.kt"),
        )
        val source = candidates.firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("EyeTrackingEngine.kt was not found")
        val processStart = source.indexOf("fun process(")
        val validation = source.indexOf("if (!isUsableFaceEvidence(face)) return emptyFrame(face)", processStart)
        val timing = source.indexOf("temporalQualityScale(", processStart)
        val lumaRead = source.indexOf("LumaPlane(image)", processStart)
        val trackPrune = source.indexOf("pruneExpiredTracks(face.timestampNanos)", processStart)

        assertTrue(processStart >= 0)
        assertTrue(validation > processStart)
        assertTrue(timing > validation)
        assertTrue(lumaRead > validation)
        assertTrue(trackPrune > lumaRead)
        assertTrue(source.contains("minimumFaceDimensionPx.isFinite()"))
    }

    @Test
    fun lumaPlaneUsesLongOffsetsAndRejectsTruncatedBuffers() {
        val candidates = listOf(
            Path.of("core/ai/src/main/java/com/nivukx/nox/core/ai/EyeTrackingEngine.kt"),
            Path.of("src/main/java/com/nivukx/nox/core/ai/EyeTrackingEngine.kt"),
        )
        val source = candidates.firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("EyeTrackingEngine.kt was not found")
        assertTrue(source.contains("x.toLong() * pixelStride.toLong()"))
        assertTrue(source.contains("Eye tracking luma plane is truncated"))
    }
}

package com.nivukx.nox.core.nativeengine

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertTrue

class DepthConfidencePropagationSourceTest {
    private fun read(first: String, second: String): String =
        listOf(Path.of(first), Path.of(second))
            .firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("Source file not found: $first")

    @Test
    fun confidenceMapFlowsFromDepthFrameIntoNativeApis() {
        val bridge = read(
            "core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/NativeSparseTsdfReconstructionBridge.kt",
            "../reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/NativeSparseTsdfReconstructionBridge.kt",
        )
        val engine = read(
            "src/main/java/com/nivukx/nox/core/nativeengine/NativeSparseTsdfEngine.kt",
            "core/native-engine/src/main/java/com/nivukx/nox/core/nativeengine/NativeSparseTsdfEngine.kt",
        )
        val adaptiveBridge =
            bridge.substringAfter("fun integrateDepthFrameAdaptive(")
                .substringBefore("\n    fun integrate(")
        val regularBridge =
            bridge.substringAfter("fun integrateDepthFrame(\n        depthFrame: DepthFrame")
                .substringBefore("\n    fun extract(")
        assertTrue(adaptiveBridge.split("depthConfidence = depthFrame.confidence").size - 1 == 1)
        assertTrue(regularBridge.split("depthConfidence = depthFrame.confidence").size - 1 == 1)
        assertTrue(engine.contains("require(depthConfidence == null || depthConfidence.size == depth.size)"))
        assertTrue(engine.contains("nativeIntegrateDepthFrame(\n            handle,\n            depth,\n            depthConfidence,"))
        assertTrue(engine.contains("nativeIntegrateDepthFrameAdaptive(\n            handle,\n            depth,\n            depthConfidence,"))
    }

    @Test
    fun jniValidatesAndReleasesBothDepthArrays() {
        val jni = read(
            "src/main/cpp/ffi/native_sparse_tsdf_jni.cpp",
            "core/native-engine/src/main/cpp/ffi/native_sparse_tsdf_jni.cpp",
        )
        assertTrue(jni.split("jfloatArray depthConfidence").size - 1 == 2)
        assertTrue(jni.contains("env->GetArrayLength(depthConfidence) != count"))
        assertTrue(jni.contains("env->ReleaseFloatArrayElements(depthConfidence, confidenceValues, JNI_ABORT)"))
        assertTrue(jni.contains("confidenceValues);"))
    }

    @Test
    fun dimensionProductsAreValidatedBeforeNativeSizeArithmetic() {
        val engine = read(
            "src/main/java/com/nivukx/nox/core/nativeengine/NativeSparseTsdfEngine.kt",
            "core/native-engine/src/main/java/com/nivukx/nox/core/nativeengine/NativeSparseTsdfEngine.kt",
        )
        val jni = read(
            "src/main/cpp/ffi/native_sparse_tsdf_jni.cpp",
            "core/native-engine/src/main/cpp/ffi/native_sparse_tsdf_jni.cpp",
        )

        assertTrue(engine.split("val sampleCount = width.toLong() * height.toLong()").size - 1 == 2)
        assertTrue(engine.split("require(sampleCount <= Int.MAX_VALUE.toLong())").size - 1 == 2)
        assertTrue(jni.split("sampleWidth > std::numeric_limits<std::size_t>::max() / sampleHeight").size - 1 == 2)
    }

    @Test
    fun nativeTsdfUsesPerPixelWeightsInBothIntegrationPaths() {
        val native = read(
            "src/main/cpp/scene/sparse_tsdf_native_session.cpp",
            "core/native-engine/src/main/cpp/scene/sparse_tsdf_native_session.cpp",
        )
        assertTrue(native.split("sampleConfidence == nullptr").size - 1 == 2)
        assertTrue(native.split("observationWeight * std::clamp(pixelConfidence, 0.0, 1.0)").size - 1 == 2)
        assertTrue(native.contains("if (!std::isfinite(pixelConfidence) || pixelConfidence <= 0.0)"))
    }
}

package com.nivukx.nox.core.nativeengine

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertTrue

class NativeSparseTsdfLifecycleSourceTest {
    private fun source(): String =
        listOf(
            Path.of("src/main/java/com/nivukx/nox/core/nativeengine/NativeSparseTsdfEngine.kt"),
            Path.of("core/native-engine/src/main/java/com/nivukx/nox/core/nativeengine/NativeSparseTsdfEngine.kt"),
        ).firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("NativeSparseTsdfEngine.kt was not found")

    @Test
    fun nativeHandleEntryPointsSerializeWithClose() {
        val engine = source().substringAfter("class NativeSparseTsdfEngine private constructor")
        val methods = listOf(
            "reset", "releaseMemory", "integrateSurfacePoint", "integrateDepthFrame",
            "integrateDepthFrameAdaptive", "voxelCount", "residentMemoryBytes",
            "voxelUtilization", "rejectedVoxelAllocations", "rejectedTriangleCount",
            "integrateGaussianBatch", "gaussianSurfaceTelemetry", "extractMesh",
        )
        methods.forEach { method ->
            assertTrue(
                Regex("@Synchronized\\s+fun\\s+${method}\\(").containsMatchIn(engine),
                "$method must serialize access to the shared native session handle",
            )
        }
        assertTrue(
            Regex("@Synchronized\\s+override fun close\\(").containsMatchIn(engine),
            "close() must use the same monitor as native entry points",
        )
    }

    @Test
    fun cameraGeometryGuardIsAppliedToBothDepthPaths() {
        val engine = source().substringAfter("class NativeSparseTsdfEngine private constructor")
        val call = "requireValidCameraGeometry(fx, fy, cx, cy, qx, qy, qz, qw, tx, ty, tz)"

        assertTrue(engine.split(call).size - 1 == 2)
        assertTrue(engine.contains("require(depth.isFinite())"))
        assertTrue(engine.contains("directionNormSquared.isFinite()"))
        assertTrue(engine.contains("normSquared.isFinite()"))
    }

    @Test
    fun nativeMeshPayloadCountsAreBoundedBeforeIntegerAllocation() {
        val engine = source().substringAfter("class NativeSparseTsdfEngine private constructor")

        assertTrue(engine.contains("payload[0] != 2.0f"))
        assertTrue(engine.contains("rawVertexCount != rawVertexCount.toInt().toFloat()"))
        assertTrue(engine.contains("rawTriangleCount != rawTriangleCount.toInt().toFloat()"))
        assertTrue(engine.contains("expectedSizeLong != payload.size.toLong()"))
        assertTrue(engine.contains("!rawConfidence.isFinite()"))
        assertTrue(engine.contains("val indices = IntArray(indexFloatCount)"))
    }


    @Test
    fun nativeMeshPayloadCountsAreBoundedBeforeAllocation() {
        val engine = source().substringAfter("class NativeSparseTsdfEngine private constructor")

        assertTrue(engine.contains("payload[0] != 2.0f"))
        assertTrue(engine.contains("rawVertexCount != rawVertexCount.toInt().toFloat()"))
        assertTrue(engine.contains("rawTriangleCount != rawTriangleCount.toInt().toFloat()"))
        assertTrue(engine.contains("expectedSizeLong != payload.size.toLong()"))
        assertTrue(engine.contains("!rawConfidence.isFinite()"))
        assertTrue(engine.contains("IntArray(indexFloatCount)"))
        assertTrue(engine.contains("if (packed.any { !it.isFinite() }) return null"))
    }


    @Test
    fun gaussianEvidencePackingRemainsBoundedForLargeFrames() {
        val engine = source().substringAfter("class NativeSparseTsdfEngine private constructor")

        assertTrue(engine.contains("MAX_GAUSSIAN_BATCH_SIZE = 65_536"))
        assertTrue(engine.contains("minOf(MAX_GAUSSIAN_BATCH_SIZE, observations.size - offset)"))
        assertTrue(engine.contains("DoubleArray(batchCount * stride)"))
        assertTrue(engine.contains("LongArray(batchCount)"))
        assertTrue(!engine.contains("DoubleArray(observations.size * stride)"))
    }


    @Test
    fun nonFiniteNativeTelemetryUsesUnavailableFallback() {
        val engine = source().substringAfter("class NativeSparseTsdfEngine private constructor")

        assertTrue(engine.contains("values.any { !it.isFinite() }"))
        assertTrue(engine.contains("deformation.any { !it.isFinite() }"))
        assertTrue(engine.contains("nativeTelemetryAvailable = false"))
    }


    @Test
    fun closeInvalidatesHandleBeforeNativeDestruction() {
        val engine = source().substringAfter("class NativeSparseTsdfEngine private constructor")
        val close = engine.substringAfter("@Synchronized\n    override fun close()").substringBefore("\n    }")

        assertTrue(close.contains("val closingHandle = handle"))
        assertTrue(close.contains("if (closingHandle == 0L) return"))
        assertTrue(close.indexOf("handle = 0L") < close.indexOf("nativeDestroy(closingHandle)"))
    }

}

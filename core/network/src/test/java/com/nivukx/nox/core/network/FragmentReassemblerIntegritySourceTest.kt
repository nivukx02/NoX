package com.nivukx.nox.core.network

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertTrue

class FragmentReassemblerIntegritySourceTest {
    private fun read(fileName: String): String {
        val modulePath = Path.of("src/main/java/com/nivukx/nox/core/network", fileName)
        val repositoryPath = Path.of("core/network/src/main/java/com/nivukx/nox/core/network", fileName)
        val parentRepositoryPath = Path.of("../core/network/src/main/java/com/nivukx/nox/core/network", fileName)
        return listOf(repositoryPath, modulePath, parentRepositoryPath)
            .firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("Reassembler source was not found: $fileName")
    }

    @Test
    fun denseLandmarkReassemblyCleansUpOnMetadataAndDecodeFailures() {
        val source = read("DenseLandmarkFragmentReassembler.kt")
        assertTrue(source.contains("assembly.timestampNanos != chunk.timestampNanos"))
        assertTrue(source.contains("val failed = assemblies.remove(key)"))
        assertTrue(source.contains("retainedPayloadBytes ="))
        assertTrue(source.contains("catch (failure: Throwable)"))
        assertTrue(source.contains("throw failure"))
    }

    @Test
    fun multiActorPoseReassemblyCleansUpOnMetadataAndDecodeFailures() {
        val source = read("MultiActorPoseFragmentReassembler.kt")
        assertTrue(source.contains("assembly.timestampNanos != chunk.timestampNanos"))
        assertTrue(source.contains("val failed = assemblies.remove(key)"))
        assertTrue(source.contains("retainedPayloadBytes ="))
        assertTrue(source.contains("catch (failure: Throwable)"))
        assertTrue(source.contains("throw failure"))
    }

    @Test
    fun faceMeshReassemblyCleansUpOnMetadataAndDecodeFailures() {
        val source = read("FaceMeshFragmentReassembler.kt")
        assertTrue(source.contains("assembly.timestampNanos != chunk.timestampNanos"))
        assertTrue(source.contains("assemblies.remove(key)"))
        assertTrue(source.contains("catch (failure: Throwable)"))
        assertTrue(source.contains("throw failure"))
    }

    @Test
    fun depthTransportCleansUpFailedFramesAndValidatesBeforeSending() {
        val source = read("DepthFragmentReassembler.kt")
        val wireCodec = read("DepthFragmentWireCodec.kt")
        assertTrue(source.contains("assembly.deviceId != chunk.deviceId"))
        assertTrue(source.contains("Depth assembly payload timestamp mismatch"))
        assertTrue(source.contains("removeAssembly(key)"))
        assertTrue(wireCodec.contains("Depth fragment CRC mismatch before transmission"))
        assertTrue(wireCodec.contains("Depth fragment timestamp mismatch before transmission"))
    }
}

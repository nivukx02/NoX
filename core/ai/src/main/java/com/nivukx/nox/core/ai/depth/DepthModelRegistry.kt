package com.nivukx.nox.core.ai.depth

import android.content.Context

/**
 * Versioned model-artifact registry for on-device AI.
 *
 * Every provisioned model has a name, version, expected checksum and runtime requirements. The
 * registry never silently fetches model weights from the network.
 */
class DepthModelRegistry(
    private val context: Context,
) {

    data class Artifact(
        val id: String,
        val version: String,
        val provider: String,
        val assetName: String,
        val sha256: String?,
        val license: String,
        val modelCardUrl: String?,
        val recommendedInputWidth: Int,
        val recommendedInputHeight: Int,
    )

    data class Availability(
        val artifact: Artifact,
        val provisioned: Boolean,
        val fileBytes: Long,
        val warning: String? = null,
    )

    private val artifacts = linkedMapOf(
        "depth-anything-v2-small" to Artifact(
            id = "depth-anything-v2-small",
            version = "external-onnx",
            provider = "ONNX",
            assetName = "depth_anything_v2_small.onnx",
            sha256 = null,
            license = "Model license must be verified for the exact export",
            modelCardUrl =
                "https://github.com/DepthAnything/Depth-Anything-V2",
            recommendedInputWidth = 518,
            recommendedInputHeight = 518,
        ),
    )

    fun list(): List<Artifact> = artifacts.values.toList()

    fun availability(): List<Availability> =
        artifacts.values.map(::availability)

    fun find(id: String): Artifact? = artifacts[id]

    fun createProvider(
        id: String,
        backend: OnnxMonocularDepthProvider.Backend =
            OnnxMonocularDepthProvider.Backend.AUTO,
    ): MonocularDepthProvider {
        val artifact = artifacts[id]
            ?: return UnavailableMonocularDepthProvider(
                "Unknown depth model: " + id,
            )

        val provisioned = runCatching {
            context.assets.open(artifact.assetName).use { stream ->
                stream.available() > 0
            }
        }.getOrDefault(false)

        if (!provisioned) {
            return UnavailableMonocularDepthProvider(
                "Depth model asset is not provisioned: " +
                    artifact.assetName,
            )
        }

        return OnnxMonocularDepthProvider(
            context = context,
            spec = OnnxMonocularDepthProvider.Spec(
                assetName = artifact.assetName,
                inputWidth = artifact.recommendedInputWidth,
                inputHeight = artifact.recommendedInputHeight,
                backend = backend,
                modelVersion = artifact.version,
                expectedSha256 = artifact.sha256,
            ),
        )
    }

    private fun availability(
        artifact: Artifact,
    ): Availability {
        val file = runCatching {
            context.assets.open(artifact.assetName).use { stream ->
                stream.available().toLong()
            }
        }.getOrDefault(0L)

        return Availability(
            artifact = artifact,
            provisioned = file > 0L,
            fileBytes = file,
            warning =
                if (file > 0L) null
                else "Model weights are not bundled",
        )
    }
}

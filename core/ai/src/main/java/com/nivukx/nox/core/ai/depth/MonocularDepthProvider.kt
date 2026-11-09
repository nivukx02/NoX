package com.nivukx.nox.core.ai.depth

import android.graphics.Bitmap
import com.nivukx.nox.core.model.MonocularDepthFrame
import com.nivukx.nox.core.model.MonocularDepthSample
import com.nivukx.nox.core.model.ScaleSource

/**
 * Provider-neutral learned monocular depth contract.
 *
 * A provider may be backed by ONNX Runtime, TFLite, vendor NPU APIs, or another accelerator.
 * The reconstruction layer consumes only depth samples plus an explicit metric/relative flag.
 */
interface MonocularDepthProvider : AutoCloseable {

    data class Config(
        val inputWidth: Int = 518,
        val inputHeight: Int = 518,
        val source: ScaleSource = ScaleSource.UNKNOWN,
        val metric: Boolean = false,
        val minimumValidConfidence: Double = 0.25,
    )

    data class Result(
        val frame: MonocularDepthFrame,
        val inferenceMillis: Double,
        val backend: String,
        val modelName: String,
        val warnings: List<String> = emptyList(),
    )

    val isReady: Boolean

    fun infer(
        bitmap: Bitmap,
        timestampNanos: Long,
    ): Result

    override fun close()
}

/**
 * Explicit no-op provider used when no learned-depth model has been provisioned.
 */
class UnavailableMonocularDepthProvider(
    private val reason: String,
    private val config: MonocularDepthProvider.Config = MonocularDepthProvider.Config(),
) : MonocularDepthProvider {

    override val isReady: Boolean = false

    override fun infer(
        bitmap: Bitmap,
        timestampNanos: Long,
    ): MonocularDepthProvider.Result =
        MonocularDepthProvider.Result(
            frame = MonocularDepthFrame(
                timestampNanos = timestampNanos,
                width = bitmap.width,
                height = bitmap.height,
                samples = emptyList<MonocularDepthSample>(),
                meanDepthMeters = 0.0,
                medianDepthMeters = 0.0,
                scaleSource = config.source,
                confidence = 0.0,
                temporalStability = 0.0,
                metric = config.metric,
            ),
            inferenceMillis = 0.0,
            backend = "UNAVAILABLE",
            modelName = "none",
            warnings = listOf(reason),
        )

    override fun close() = Unit
}

package com.nivukx.nox.core.ai.depth

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import com.nivukx.nox.core.model.DepthRepresentation
import com.nivukx.nox.core.model.MonocularDepthFrame
import com.nivukx.nox.core.model.MonocularDepthSample
import com.nivukx.nox.core.model.ScaleSource
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.abs

/**
 * Generic ONNX Runtime monocular-depth adapter.
 *
 * Model weights remain external to source control. The provider validates an optional SHA-256,
 * exposes CPU/NNAPI selection, supports common tensor layouts, and converts relative depth into
 * an explicitly non-metric depth prior for the 3D solver.
 */
class OnnxMonocularDepthProvider(
    private val context: Context,
    private val spec: Spec,
) : MonocularDepthProvider {

    enum class Backend { AUTO, CPU, NNAPI }

    enum class InputLayout { NCHW, NHWC }

    enum class OutputInterpretation {
        METRIC_METERS,
        RELATIVE_DEPTH,
        INVERSE_DEPTH,
    }

    data class Spec(
        val assetName: String,
        val inputName: String? = null,
        val outputName: String? = null,
        val inputWidth: Int = 518,
        val inputHeight: Int = 518,
        val inputLayout: InputLayout = InputLayout.NCHW,
        val outputInterpretation: OutputInterpretation =
            OutputInterpretation.RELATIVE_DEPTH,
        val backend: Backend = Backend.CPU,
        val source: ScaleSource = ScaleSource.UNKNOWN,
        val mean: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f),
        val std: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f),
        val confidenceFloor: Double = 0.25,
        val modelVersion: String = "unversioned",
        val expectedSha256: String? = null,
        val runtimeSampleGridX: Int = 32,
        val runtimeSampleGridY: Int = 20,
    ) {
        init {
            require(inputWidth >= 32)
            require(inputHeight >= 32)
            require(mean.size == 3)
            require(std.size == 3)
            require(std.all { it > 0f })
            require(runtimeSampleGridX >= 2)
            require(runtimeSampleGridY >= 2)
        }
    }

    override val isReady: Boolean
        get() = session != null

    private val environment = OrtEnvironment.getEnvironment()
    private var options: OrtSession.SessionOptions? = null
    private var session: OrtSession? = null
    private var inputName: String? = null
    private var outputName: String? = null
    private var activeBackend: Backend = Backend.CPU

    init {
        initialize()
    }

    @Synchronized
    override fun infer(
        bitmap: Bitmap,
        timestampNanos: Long,
    ): MonocularDepthProvider.Result {
        val active = session ?: error(
            "Depth model is not available: " + spec.assetName,
        )
        require(!bitmap.isRecycled) {
            "Cannot infer depth from a recycled bitmap"
        }

        val started = System.nanoTime()
        val data = preprocess(bitmap)

        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(data),
            inputShape(),
        ).use { tensor ->
            val name = inputName ?: error("No ONNX input name")
            val resultName = outputName ?: error("No ONNX output name")

            active.run(mapOf(name to tensor)).use { results ->
                val output = results
                    .get(resultName)
                    .orElseThrow {
                        IllegalStateException(
                            "ONNX output tensor not found: $resultName",
                        )
                    }
                val value = output.getValue()
                val map = decode(value)
                val frame = toDepthFrame(
                    map,
                    timestampNanos,
                )

                return MonocularDepthProvider.Result(
                    frame = frame,
                    inferenceMillis =
                        (System.nanoTime() - started) / 1e6,
                    backend = activeBackend.name,
                    modelName =
                        spec.assetName + "@" + spec.modelVersion,
                )
            }
        }
    }

    @Synchronized
    override fun close() {
        runCatching { session?.close() }
        runCatching { options?.close() }
        session = null
        options = null
    }

    private fun initialize() {
        val file = runCatching { materializeModel() }
            .getOrElse { return }

        val candidates =
            when (spec.backend) {
                Backend.AUTO -> listOf(
                    Backend.NNAPI,
                    Backend.CPU,
                )
                Backend.NNAPI -> listOf(Backend.NNAPI)
                Backend.CPU -> listOf(Backend.CPU)
            }

        for (backend in candidates) {
            try {
                val sessionOptions =
                    OrtSession.SessionOptions().apply {
                        setOptimizationLevel(
                            OrtSession.SessionOptions.OptLevel.ALL_OPT,
                        )
                        if (backend == Backend.NNAPI) {
                            addNnapi()
                        }
                    }

                val created =
                    environment.createSession(
                        file.absolutePath,
                        sessionOptions,
                    )

                options = sessionOptions
                session = created
                activeBackend = backend

                inputName =
                    spec.inputName ?: created.inputNames.firstOrNull()
                outputName =
                    spec.outputName ?: created.outputNames.firstOrNull()

                check(!inputName.isNullOrBlank()) {
                    "Depth model has no input tensor"
                }
                check(!outputName.isNullOrBlank()) {
                    "Depth model has no output tensor"
                }
                return
            } catch (_: Throwable) {
                runCatching { session?.close() }
                runCatching { options?.close() }
                session = null
                options = null
            }
        }
    }

    private fun materializeModel(): File {
        val name = spec.assetName
            .substringAfterLast('/')
            .replace(
                Regex("[^A-Za-z0-9._-]"),
                "_",
            )

        val output = File(
            context.filesDir,
            "models/" + name,
        )

        output.parentFile?.mkdirs()

        if (
            output.isFile &&
            output.length() > 0L &&
            shaMatches(output)
        ) {
            return output
        }

        val partial = File(
            output.parentFile,
            output.name + ".partial",
        )

        context.assets.open(spec.assetName).use { input ->
            FileOutputStream(partial, false).use { out ->
                input.copyTo(out, 1024 * 1024)
                out.fd.sync()
            }
        }

        check(partial.length() > 0L) {
            "Depth model asset is empty: " + spec.assetName
        }
        check(shaMatches(partial)) {
            "Depth model SHA-256 mismatch: " + spec.assetName
        }

        if (!partial.renameTo(output)) {
            partial.copyTo(output, overwrite = true)
            partial.delete()
        }

        return output
    }

    private fun shaMatches(file: File): Boolean {
        val expected = spec.expectedSha256
            ?.trim()
            ?.lowercase()
            ?.takeIf { it.length == 64 }
            ?: return true

        val digest = java.security.MessageDigest
            .getInstance("SHA-256")

        file.inputStream().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val read = input.read(buffer)
                if (read <= 0) break
                digest.update(buffer, 0, read)
            }
        }

        val actual = digest.digest().joinToString("") {
            "%02x".format(it)
        }

        return actual == expected
    }

    private fun preprocess(bitmap: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(
            bitmap,
            spec.inputWidth,
            spec.inputHeight,
            true,
        )

        return try {
            val pixels = IntArray(
                spec.inputWidth * spec.inputHeight,
            )
            scaled.getPixels(
                pixels,
                0,
                spec.inputWidth,
                0,
                0,
                spec.inputWidth,
                spec.inputHeight,
            )

            when (spec.inputLayout) {
                InputLayout.NCHW -> preprocessNchw(pixels)
                InputLayout.NHWC -> preprocessNhwc(pixels)
            }
        } finally {
            if (scaled !== bitmap) scaled.recycle()
        }
    }

    private fun preprocessNchw(
        pixels: IntArray,
    ): FloatArray {
        val plane = spec.inputWidth * spec.inputHeight
        val result = FloatArray(plane * 3)
        var offset = 0

        for (channel in 0..2) {
            pixels.forEach { argb ->
                val raw = when (channel) {
                    0 -> (argb shr 16) and 0xFF
                    1 -> (argb shr 8) and 0xFF
                    else -> argb and 0xFF
                }

                result[offset++] =
                    (
                        raw / 255.0f - spec.mean[channel]
                    ) / spec.std[channel]
            }
        }

        return result
    }

    private fun preprocessNhwc(
        pixels: IntArray,
    ): FloatArray {
        val result = FloatArray(pixels.size * 3)
        var offset = 0

        pixels.forEach { argb ->
            val channels = intArrayOf(
                (argb shr 16) and 0xFF,
                (argb shr 8) and 0xFF,
                argb and 0xFF,
            )

            for (channel in 0..2) {
                result[offset++] =
                    (
                        channels[channel] / 255.0f -
                            spec.mean[channel]
                    ) / spec.std[channel]
            }
        }

        return result
    }

    private fun inputShape(): LongArray =
        when (spec.inputLayout) {
            InputLayout.NCHW ->
                longArrayOf(
                    1,
                    3,
                    spec.inputHeight.toLong(),
                    spec.inputWidth.toLong(),
                )

            InputLayout.NHWC ->
                longArrayOf(
                    1,
                    spec.inputHeight.toLong(),
                    spec.inputWidth.toLong(),
                    3,
                )
        }

    private data class DepthMap(
        val width: Int,
        val height: Int,
        val values: FloatArray,
    )

    private fun decode(value: Any?): DepthMap =
        when (value) {
            is FloatArray ->
                decodeFlat(value)

            is DoubleArray ->
                decodeFlat(
                    FloatArray(value.size) {
                        value[it].toFloat()
                    },
                )

            is Array<*> ->
                decodeNested(value)

            else ->
                error(
                    "Unsupported ONNX depth output type: " +
                        (value?.javaClass?.name ?: "null"),
                )
        }

    private fun decodeFlat(
        values: FloatArray,
    ): DepthMap {
        val required =
            spec.inputWidth * spec.inputHeight

        require(values.size >= required) {
            "Depth tensor contains too few values"
        }

        return DepthMap(
            spec.inputWidth,
            spec.inputHeight,
            values.copyOf(required),
        )
    }

    private fun decodeNested(
        value: Array<*>,
    ): DepthMap {
        val flat = ArrayList<Float>()
        flatten(value, flat)

        return decodeFlat(
            FloatArray(flat.size) { flat[it] },
        )
    }

    private fun flatten(
        value: Any?,
        output: MutableList<Float>,
    ) {
        when (value) {
            is FloatArray ->
                value.forEach(output::add)

            is DoubleArray ->
                value.forEach {
                    output += it.toFloat()
                }

            is IntArray ->
                value.forEach {
                    output += it.toFloat()
                }

            is Array<*> ->
                value.forEach {
                    flatten(it, output)
                }

            is Number ->
                output += value.toFloat()
        }
    }

    private fun toDepthFrame(
        map: DepthMap,
        timestampNanos: Long,
    ): MonocularDepthFrame {
        val normalized = normalize(map.values)

        val representation =
            if (
                spec.outputInterpretation ==
                    OutputInterpretation.METRIC_METERS
            ) {
                DepthRepresentation.METRIC_METERS
            } else {
                DepthRepresentation.RELATIVE_ORDERED_DEPTH
            }

        val samples = ArrayList<MonocularDepthSample>(
            spec.runtimeSampleGridX *
                spec.runtimeSampleGridY,
        )

        for (gy in 0 until spec.runtimeSampleGridY) {
            val y = (
                (gy + 0.5) /
                    spec.runtimeSampleGridY *
                    map.height
                ).toInt().coerceIn(0, map.height - 1)

            for (gx in 0 until spec.runtimeSampleGridX) {
                val x = (
                    (gx + 0.5) /
                        spec.runtimeSampleGridX *
                        map.width
                    ).toInt().coerceIn(0, map.width - 1)

                val index = y * map.width + x
                val base = normalized[index]
                val ordered =
                    if (
                        spec.outputInterpretation ==
                            OutputInterpretation.INVERSE_DEPTH
                    ) {
                        1f - base
                    } else {
                        base
                    }

                samples += MonocularDepthSample(
                    x = x.toDouble() / map.width,
                    y = y.toDouble() / map.height,
                    depthMeters = ordered.toDouble(),
                    confidence = localConfidence(
                        normalized,
                        map.width,
                        map.height,
                        x,
                        y,
                    ),
                    source = spec.source,
                )
            }
        }

        val values = normalized.map { it.toDouble() }
        return MonocularDepthFrame(
            timestampNanos = timestampNanos,
            width = map.width,
            height = map.height,
            samples = samples,
            meanDepthMeters = values.averageOrZero(),
            medianDepthMeters = values.medianOrZero(),
            scaleSource = spec.source,
            confidence = samples.map { it.confidence }.averageOrZero(),
            temporalStability = 0.50,
            metric =
                representation ==
                    DepthRepresentation.METRIC_METERS,
            representation = representation,
        )
    }

    private fun normalize(
        raw: FloatArray,
    ): FloatArray {
        val finite = raw.filter { it.isFinite() }
        if (finite.isEmpty()) return FloatArray(raw.size)

        val minValue = finite.minOrNull() ?: 0f
        val maxValue = finite.maxOrNull() ?: 1f
        val range = (maxValue - minValue)
            .coerceAtLeast(1e-6f)

        return FloatArray(raw.size) { index ->
            (
                (raw[index] - minValue) /
                    range
                ).coerceIn(0f, 1f)
        }
    }

    private fun localConfidence(
        values: FloatArray,
        width: Int,
        height: Int,
        x: Int,
        y: Int,
    ): Double {
        val center = values[y * width + x]
        var gradient = 0.0
        var count = 0

        listOf(
            x - 1 to y,
            x + 1 to y,
            x to y - 1,
            x to y + 1,
        ).forEach { (nx, ny) ->
            if (
                nx in 0 until width &&
                ny in 0 until height
            ) {
                gradient += abs(
                    center -
                        values[ny * width + nx],
                )
                count += 1
            }
        }

        if (count == 0) return spec.confidenceFloor

        val stability = 1.0 -
            (gradient / count).coerceIn(0.0, 1.0)

        return (
            spec.confidenceFloor * 0.30 +
                stability * 0.70
            ).coerceIn(0.0, 1.0)
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private fun List<Double>.medianOrZero(): Double {
        if (isEmpty()) return 0.0
        val sorted = sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }
}

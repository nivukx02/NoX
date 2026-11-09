package com.nivukx.nox.core.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import androidx.annotation.WorkerThread
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Production live person-segmentation authority for dense multi-view reconstruction.
 *
 * The task is intentionally separated from pose detection. Pose landmarks provide sparse semantic
 * geometry; segmentation supplies a dense support field around the actor. The two signals are useful
 * for different failure modes:
 * - silhouette support rejects background points that accidentally satisfy a landmark-only gate;
 * - body/hair/clothing regions provide spatial evidence around partially occluded joints;
 * - temporal mask continuity makes a stable shape prior available when a detector briefly drops;
 * - the mask never creates a 3D point by itself, so it cannot silently manufacture metric geometry.
 *
 * MediaPipe's LIVE_STREAM mode is used because it provides asynchronous processing with flow limiting.
 * The callback is reduced to one bounded 64x64 evidence field before it escapes the MediaPipe callback,
 * preventing unbounded retained mask images and making the memory lifetime explicit.
 */
class PersonSegmentationEngine(
    private val context: Context,
    private val config: Config = Config(),
) : AutoCloseable {

    enum class DelegateMode {
        CPU,
        GPU,
    }

    enum class State {
        READY,
        DEGRADED,
        CLOSED,
    }

    data class Config(
        val modelAssetPath: String =
            "selfie_multiclass_256x256.tflite",
        val delegate: DelegateMode = DelegateMode.GPU,
        val allowGpuFallbackToCpu: Boolean = true,
        val outputGridWidth: Int = 64,
        val outputGridHeight: Int = 64,
        val foregroundThreshold: Double = 0.45,
        val strongForegroundThreshold: Double = 0.72,
        val minimumQuality: Double = 0.22,
        val maximumCallbackAgeMillis: Long = 250L,
        val personLabelHints: Set<String> =
            setOf(
                "hair",
                "body",
                "face",
                "clothes",
                "person",
                "selfie",
                "skin",
                "others",
            ),
        val backgroundLabelHints: Set<String> =
            setOf(
                "background",
                "bg",
            ),
    ) {
        init {
            require(modelAssetPath.isNotBlank())
            require(outputGridWidth in 16..128)
            require(outputGridHeight in 16..128)
            require(foregroundThreshold in 0.0..1.0)
            require(strongForegroundThreshold >= foregroundThreshold)
            require(minimumQuality in 0.0..1.0)
            require(maximumCallbackAgeMillis > 0L)
            require(personLabelHints.isNotEmpty())
            require(backgroundLabelHints.isNotEmpty())
        }
    }

    data class Result(
        val timestampMs: Long,
        val inferenceLatencyMs: Long,
        val sourceWidth: Int,
        val sourceHeight: Int,
        val foregroundCoverage: Double,
        val strongForegroundCoverage: Double,
        val meanForegroundConfidence: Double,
        val boundaryStability: Double,
        val spatialCompactness: Double,
        val horizontalCentroid: Double,
        val verticalCentroid: Double,
        val modelQuality: Double,
        val personMaskIndices: List<Int>,
        val labels: List<String>,
        val activeDelegate: DelegateMode?,
        val gpuFallbackCount: Long,
        val temporalMaskConsistency: Double,
        val gridWidth: Int,
        val gridHeight: Int,
        val foregroundMask: FloatArray,
        val state: State,
        val warnings: List<String>,
    ) {
        val pixelCount: Int
            get() = gridWidth * gridHeight

        fun maskValue(
            x: Int,
            y: Int,
        ): Float {
            if (
                x !in 0 until gridWidth ||
                    y !in 0 until gridHeight
            ) {
                return 0f
            }
            return foregroundMask[
                y * gridWidth + x
            ]
        }

        fun isForeground(
            x: Int,
            y: Int,
            threshold: Double = 0.45,
        ): Boolean =
            maskValue(x, y) >= threshold

        fun sampleNormalized(
            normalizedX: Double,
            normalizedY: Double,
        ): Double {
            val x =
                (
                    normalizedX.coerceIn(0.0, 1.0) *
                        (gridWidth - 1)
                )
                    .coerceAtLeast(0.0)
            val y =
                (
                    normalizedY.coerceIn(0.0, 1.0) *
                        (gridHeight - 1)
                )
                    .coerceAtLeast(0.0)

            val x0 = x.toInt().coerceIn(0, gridWidth - 1)
            val y0 = y.toInt().coerceIn(0, gridHeight - 1)
            val x1 = min(x0 + 1, gridWidth - 1)
            val y1 = min(y0 + 1, gridHeight - 1)

            val tx = x - x0
            val ty = y - y0

            val v00 = maskValue(x0, y0).toDouble()
            val v10 = maskValue(x1, y0).toDouble()
            val v01 = maskValue(x0, y1).toDouble()
            val v11 = maskValue(x1, y1).toDouble()

            val top = v00 + (v10 - v00) * tx
            val bottom = v01 + (v11 - v01) * tx
            return (top + (bottom - top) * ty)
                .coerceIn(0.0, 1.0)
        }

        fun silhouetteSupport(
            normalizedX: Double,
            normalizedY: Double,
        ): Double {
            val center =
                sampleNormalized(
                    normalizedX,
                    normalizedY,
                )
            val stepX =
                1.0 /
                    gridWidth.coerceAtLeast(1).toDouble()
            val stepY =
                1.0 /
                    gridHeight.coerceAtLeast(1).toDouble()
            val neighbors =
                listOf(
                    sampleNormalized(
                        normalizedX - stepX,
                        normalizedY,
                    ),
                    sampleNormalized(
                        normalizedX + stepX,
                        normalizedY,
                    ),
                    sampleNormalized(
                        normalizedX,
                        normalizedY - stepY,
                    ),
                    sampleNormalized(
                        normalizedX,
                        normalizedY + stepY,
                    ),
                )

            val localMean =
                neighbors.average()
            val localAgreement =
                (
                    1.0 -
                        neighbors
                            .map { abs(it - center) }
                            .average()
                    )
                    .coerceIn(0.0, 1.0)

            return (
                center * 0.70 +
                    localMean * 0.20 +
                    localAgreement * 0.10
                ).coerceIn(0.0, 1.0)
        }

        fun boundingBox(
            threshold: Double = 0.45,
        ): BoundingBox? {
            var minX = gridWidth
            var minY = gridHeight
            var maxX = -1
            var maxY = -1

            for (y in 0 until gridHeight) {
                for (x in 0 until gridWidth) {
                    if (maskValue(x, y) < threshold) continue
                    minX = min(minX, x)
                    minY = min(minY, y)
                    maxX = max(maxX, x)
                    maxY = max(maxY, y)
                }
            }

            if (maxX < minX || maxY < minY) {
                return null
            }

            return BoundingBox(
                left = minX.toDouble() /
                    gridWidth.coerceAtLeast(1),
                top = minY.toDouble() /
                    gridHeight.coerceAtLeast(1),
                right = (maxX + 1).toDouble() /
                    gridWidth.coerceAtLeast(1),
                bottom = (maxY + 1).toDouble() /
                    gridHeight.coerceAtLeast(1),
            )
        }
    }

    data class BoundingBox(
        val left: Double,
        val top: Double,
        val right: Double,
        val bottom: Double,
    ) {
        val width: Double
            get() = (right - left).coerceAtLeast(0.0)

        val height: Double
            get() = (bottom - top).coerceAtLeast(0.0)

        val area: Double
            get() = width * height
    }

    data class Diagnostics(
        val state: State,
        val submittedFrames: Long,
        val emittedResults: Long,
        val droppedNonMonotonicFrames: Long,
        val callbackFailures: Long,
        val latestTimestampMs: Long,
        val latestForegroundCoverage: Double,
        val latestMeanConfidence: Double,
        val latestTemporalMaskConsistency: Double,
        val latestLatencyMs: Long,
        val latestWarnings: List<String>,
        val modelLabels: List<String>,
        val activeDelegate: DelegateMode?,
        val gpuFallbackCount: Long,
        val initializationFailures: Long,
    )

    private var segmenter: ImageSegmenter? = null
    private var activeDelegate: DelegateMode? = null
    private var modelLabels: List<String> = emptyList()
    private var personMaskIndices: List<Int> = emptyList()

    private var latestResult: Result? = null
    private var lastSubmittedTimestampMs = Long.MIN_VALUE
    private val pendingBitmapsByTimestamp =
        LinkedHashMap<Long, Bitmap>()
    private val maximumPendingBitmaps = 8

    private var submittedFrames = 0L
    private var emittedResults = 0L
    private var droppedNonMonotonicFrames = 0L
    private var callbackFailures = 0L
    private var gpuFallbackCount = 0L
    private var initializationFailures = 0L
    private var latestArrivalUptimeMillis = Long.MIN_VALUE
    private val submittedUptimeByTimestamp =
        LinkedHashMap<Long, Long>()

    private val closed =
        AtomicBoolean(false)

    init {
        initialize()
    }

    @Synchronized
    private fun initialize() {
        if (closed.get()) return

        val preferred = config.delegate
        val first =
            tryCreateSegmenter(preferred)
        if (first != null) {
            segmenter = first
            activeDelegate = preferred
            updateModelLabels(first)
            return
        }

        initializationFailures += 1L

        if (
            preferred == DelegateMode.GPU &&
                config.allowGpuFallbackToCpu
        ) {
            val cpu =
                tryCreateSegmenter(
                    DelegateMode.CPU,
                )
            if (cpu != null) {
                segmenter = cpu
                activeDelegate = DelegateMode.CPU
                gpuFallbackCount += 1L
                updateModelLabels(cpu)
                return
            }
            initializationFailures += 1L
        }
    }

    private fun tryCreateSegmenter(
        delegateMode: DelegateMode,
    ): ImageSegmenter? =
        try {
            val baseOptions =
                BaseOptions
                    .builder()
                    .setModelAssetPath(
                        config.modelAssetPath,
                    )
                    .setDelegate(
                        when (delegateMode) {
                            DelegateMode.CPU -> Delegate.CPU
                            DelegateMode.GPU -> Delegate.GPU
                        },
                    )
                    .build()

            val options =
                ImageSegmenter
                    .ImageSegmenterOptions
                    .builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setOutputCategoryMask(false)
                    .setOutputConfidenceMasks(true)
                    .setResultListener(
                        this::onSegmentationResult,
                    )
                    .setErrorListener(
                        this::onSegmentationError,
                    )
                    .build()

            ImageSegmenter.createFromOptions(
                context.applicationContext,
                options,
            )
        } catch (_: RuntimeException) {
            null
        }

    @Synchronized
    private fun updateModelLabels(
        active: ImageSegmenter,
    ) {
        modelLabels =
            active.getLabels()?.toList().orEmpty()

        personMaskIndices =
            resolvePersonMaskIndices(
                modelLabels,
            )
    }

    @WorkerThread
    @Synchronized
    fun submit(
        image: Image,
        timestampMs: Long,
    ): Boolean {
        check(!closed.get()) {
            "PersonSegmentationEngine is closed"
        }

        if (
            image.format !=
                ImageFormat.YUV_420_888
        ) {
            callbackFailures += 1L
            return false
        }

        val bitmap =
            runCatching {
                imageToArgbBitmap(image)
            }.getOrNull()
                ?: return false

        val mpImage =
            runCatching {
                BitmapImageBuilder(
                    bitmap,
                ).build()
            }.getOrElse {
                bitmap.recycle()
                callbackFailures += 1L
                return false
            }

        synchronized(this) {
            pendingBitmapsByTimestamp[timestampMs] =
                bitmap

            while (
                pendingBitmapsByTimestamp.size >
                    maximumPendingBitmaps
            ) {
                pendingBitmapsByTimestamp
                    .entries
                    .first()
                    .key
                    .let(
                        pendingBitmapsByTimestamp::remove,
                    )
            }
        }

        val submitted =
            try {
                submit(
                    mpImage,
                    timestampMs,
                )
            } catch (error: RuntimeException) {
                runCatching { mpImage.close() }
                false
            }

        if (!submitted) {
            runCatching { mpImage.close() }
            synchronized(this) {
                pendingBitmapsByTimestamp
                    .remove(timestampMs)
                    ?.let { bitmapToRecycle ->
                        runCatching {
                            bitmapToRecycle.recycle()
                        }
                    }
            }
        }

        return submitted
    }

    @WorkerThread
    @Synchronized
    fun submit(
        image: MPImage,
        timestampMs: Long,
    ): Boolean {
        check(!closed.get()) {
            "PersonSegmentationEngine is closed"
        }

        if (
            timestampMs <=
                lastSubmittedTimestampMs
        ) {
            droppedNonMonotonicFrames += 1L
            runCatching { image.close() }
            return false
        }

        val activeSegmenter =
            segmenter
                ?: run {
                    runCatching { image.close() }
                    return false
                }

        lastSubmittedTimestampMs =
            timestampMs
        submittedFrames += 1L
        submittedUptimeByTimestamp[timestampMs] =
            android.os.SystemClock.uptimeMillis()
        while (submittedUptimeByTimestamp.size > 8) {
            submittedUptimeByTimestamp.remove(
                submittedUptimeByTimestamp
                    .entries
                    .first()
                    .key,
            )
        }

        return try {
            activeSegmenter.segmentAsync(
                image,
                timestampMs,
            )
            true
        } catch (error: RuntimeException) {
            callbackFailures += 1L
            runCatching {
                image.close()
            }
            false
        }
    }

    private fun imageToArgbBitmap(
        image: Image,
    ): Bitmap {
        val sourceWidth =
            image.width.coerceAtLeast(1)
        val sourceHeight =
            image.height.coerceAtLeast(1)

        val targetWidth = 256
        val targetHeight = 256

        val planes = image.planes
        check(planes.size >= 3) {
            "YUV_420_888 requires three planes"
        }

        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val yBuffer =
            yPlane.buffer.duplicate()
        val uBuffer =
            uPlane.buffer.duplicate()
        val vBuffer =
            vPlane.buffer.duplicate()

        val pixels =
            IntArray(
                targetWidth *
                    targetHeight,
            )

        for (ty in 0 until targetHeight) {
            val sourceY =
                (
                    ty *
                        sourceHeight /
                        targetHeight
                ).coerceIn(
                    0,
                    sourceHeight - 1,
                )

            val chromaY =
                sourceY / 2

            for (tx in 0 until targetWidth) {
                val sourceX =
                    (
                        tx *
                            sourceWidth /
                            targetWidth
                    ).coerceIn(
                        0,
                        sourceWidth - 1,
                    )

                val chromaX =
                    sourceX / 2

                val yIndex =
                    sourceY *
                        yPlane.rowStride +
                        sourceX *
                            yPlane.pixelStride
                val uIndex =
                    chromaY *
                        uPlane.rowStride +
                        chromaX *
                            uPlane.pixelStride
                val vIndex =
                    chromaY *
                        vPlane.rowStride +
                        chromaX *
                            vPlane.pixelStride

                val y =
                    (
                        readUnsignedByte(
                            yBuffer,
                            yIndex,
                        ) - 16
                    ).coerceAtLeast(0)
                val u =
                    readUnsignedByte(
                        uBuffer,
                        uIndex,
                    ) - 128
                val v =
                    readUnsignedByte(
                        vBuffer,
                        vIndex,
                    ) - 128

                val red =
                    (
                        y +
                            1.402 * v
                    )
                        .toInt()
                        .coerceIn(
                            0,
                            255,
                        )

                val green =
                    (
                        y -
                            0.344136 * u -
                            0.714136 * v
                    )
                        .toInt()
                        .coerceIn(
                            0,
                            255,
                        )

                val blue =
                    (
                        y +
                            1.772 * u
                    )
                        .toInt()
                        .coerceIn(
                            0,
                            255,
                        )

                pixels[
                    ty * targetWidth +
                        tx
                ] =
                    (
                        0xFF shl 24 or
                            (red shl 16) or
                            (green shl 8) or
                            blue
                    )
            }
        }

        return Bitmap.createBitmap(
            pixels,
            targetWidth,
            targetHeight,
            Bitmap.Config.ARGB_8888,
        )
    }

    private fun readUnsignedByte(
        buffer: java.nio.ByteBuffer,
        index: Int,
    ): Int =
        if (
            index < 0 ||
                index >= buffer.limit()
        ) {
            0
        } else {
            buffer.get(index).toInt() and 0xFF
        }

    @Synchronized
    fun latest(): Result? =
        latestResult

    @Synchronized
    fun diagnostics(): Diagnostics =
        Diagnostics(
            state = state(),
            submittedFrames = submittedFrames,
            emittedResults = emittedResults,
            droppedNonMonotonicFrames =
                droppedNonMonotonicFrames,
            callbackFailures = callbackFailures,
            latestTimestampMs =
                latestResult?.timestampMs ?: 0L,
            latestForegroundCoverage =
                latestResult?.foregroundCoverage ?: 0.0,
            latestMeanConfidence =
                latestResult?.meanForegroundConfidence
                    ?: 0.0,
            latestTemporalMaskConsistency =
                latestResult?.temporalMaskConsistency
                    ?: 0.0,
            latestLatencyMs =
                latestResult?.inferenceLatencyMs ?: 0L,
            latestWarnings =
                latestResult?.warnings.orEmpty(),
            modelLabels = modelLabels,
            activeDelegate = activeDelegate,
            gpuFallbackCount = gpuFallbackCount,
            initializationFailures =
                initializationFailures,
        )

    @Synchronized
    fun state(): State {
        if (closed.get()) return State.CLOSED
        val latest = latestResult
            ?: return State.DEGRADED

        val age =
            if (
                latestArrivalUptimeMillis ==
                    Long.MIN_VALUE
            ) {
                Long.MAX_VALUE
            } else {
                (
                    android.os.SystemClock.uptimeMillis() -
                        latestArrivalUptimeMillis
                    ).coerceAtLeast(0L)
            }

        return if (
            age <= config.maximumCallbackAgeMillis &&
                latest.modelQuality >=
                    config.minimumQuality
        ) {
            State.READY
        } else {
            State.DEGRADED
        }
    }

    @Synchronized
    fun reset() {
        if (closed.get()) return
        latestResult = null
        lastSubmittedTimestampMs = Long.MIN_VALUE
        submittedFrames = 0L
        emittedResults = 0L
        droppedNonMonotonicFrames = 0L
        callbackFailures = 0L
        gpuFallbackCount = 0L
        initializationFailures = 0L
        latestArrivalUptimeMillis = Long.MIN_VALUE
        submittedUptimeByTimestamp.clear()
        pendingBitmapsByTimestamp.values.forEach { bitmap ->
            runCatching { bitmap.recycle() }
        }
        pendingBitmapsByTimestamp.clear()
    }

    private fun onSegmentationResult(
        result: ImageSegmenterResult,
        inputImage: MPImage,
    ) {
        try {
            val confidenceMasks =
                result
                    .confidenceMasks()
                    .orElse(emptyList())

            try {
                if (confidenceMasks.isEmpty()) {
                    callbackFailures += 1L
                    return
                }

                val timestampMs =
                    result.timestampMs()

                val retainedBitmap =
                    synchronized(this) {
                        pendingBitmapsByTimestamp
                            .remove(timestampMs)
                    }

            val sourceWidth =
                confidenceMasks
                    .firstOrNull()
                    ?.width
                    ?.coerceAtLeast(1)
                    ?: inputImage.width.coerceAtLeast(1)

            val sourceHeight =
                confidenceMasks
                    .firstOrNull()
                    ?.height
                    ?.coerceAtLeast(1)
                    ?: inputImage.height.coerceAtLeast(1)

            val grid =
                buildForegroundGrid(
                    confidenceMasks =
                        confidenceMasks,
                    width = sourceWidth,
                    height = sourceHeight,
                )

            val previous =
                synchronized(this) {
                    latestResult
                }

            val stabilization =
                stabilizeGrid(
                    current = grid,
                    previous = previous,
                )

            val quality =
                modelQuality(
                    result,
                    personMaskIndices,
                )

            val arrivalUptime =
                android.os.SystemClock.uptimeMillis()
            val submittedUptime =
                synchronized(this) {
                    submittedUptimeByTimestamp
                        .remove(timestampMs)
                }

            val latest =
                summarizeGrid(
                    timestampMs = timestampMs,
                    sourceWidth = sourceWidth,
                    sourceHeight = sourceHeight,
                    inferenceLatencyMs =
                        if (submittedUptime == null) {
                            0L
                        } else {
                            (
                                arrivalUptime -
                                    submittedUptime
                                ).coerceAtLeast(0L)
                        },
                    grid = stabilization.grid,
                    temporalMaskConsistency =
                        stabilization.consistency,
                    modelQuality = quality,
                )

                synchronized(this) {
                    latestResult = latest
                    latestArrivalUptimeMillis =
                        arrivalUptime
                    emittedResults += 1L
                }
            } finally {
                confidenceMasks.forEach { mask ->
                    runCatching { mask.close() }
                }
            }
        } catch (error: RuntimeException) {
            synchronized(this) {
                callbackFailures += 1L
            }
        } finally {
            runCatching {
                inputImage.close()
            }
        }
    }

    private fun onSegmentationError(
        error: RuntimeException,
    ) {
        synchronized(this) {
            callbackFailures += 1L
            val existing = latestResult
            if (existing != null) {
                latestResult =
                    existing.copy(
                        state = State.DEGRADED,
                        warnings =
                            (
                                existing.warnings +
                                    (
                                        error.message
                                            ?: "SEGMENTATION_RUNTIME_ERROR"
                                    )
                                )
                                    .distinct()
                                    .take(12),
                    )
            }
        }
    }

    private fun buildForegroundGrid(
        confidenceMasks: List<MPImage>,
        width: Int,
        height: Int,
    ): FloatArray {
        val output =
            FloatArray(
                config.outputGridWidth *
                    config.outputGridHeight,
            )

        val buffers =
            confidenceMasks.map { mask ->
                ByteBufferExtractor
                    .extract(mask)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer()
            }

        val selected =
            if (personMaskIndices.isEmpty()) {
                buffers.indices.toList()
            } else {
                personMaskIndices
            }

        val validSelections =
            selected.filter {
                it in buffers.indices
            }

        if (validSelections.isEmpty()) {
            return output
        }

        for (gy in 0 until config.outputGridHeight) {
            val sourceY =
                (
                    (
                        gy + 0.5
                    ) *
                        height.toDouble() /
                        config.outputGridHeight
                ).toInt().coerceIn(
                    0,
                    height - 1,
                )

            for (gx in 0 until config.outputGridWidth) {
                val sourceX =
                    (
                        (
                            gx + 0.5
                        ) *
                            width.toDouble() /
                            config.outputGridWidth
                    ).toInt().coerceIn(
                        0,
                        width - 1,
                    )

                val sourceIndex =
                    sourceY * width +
                        sourceX

                var foreground =
                    0.0

                validSelections.forEach { index ->
                    val buffer =
                        buffers[index]
                    if (
                        sourceIndex <
                            buffer.limit()
                    ) {
                        foreground =
                            max(
                                foreground,
                                buffer.get(
                                    sourceIndex,
                                ).toDouble().coerceIn(
                                    0.0,
                                    1.0,
                                ),
                            )
                    }
                }

                output[
                    gy * config.outputGridWidth +
                        gx
                ] = foreground.toFloat()
            }
        }

        return output
    }

    private fun summarizeGrid(
        timestampMs: Long,
        sourceWidth: Int,
        sourceHeight: Int,
        inferenceLatencyMs: Long,
        grid: FloatArray,
        temporalMaskConsistency: Double,
        modelQuality: Double,
    ): Result {
        var foregroundCount = 0
        var strongForegroundCount = 0
        var confidenceSum = 0.0
        var weightSum = 0.0
        var weightedX = 0.0
        var weightedY = 0.0

        var boundaryVariation = 0.0
        var boundaryPairs = 0

        val width =
            config.outputGridWidth
        val height =
            config.outputGridHeight

        for (y in 0 until height) {
            for (x in 0 until width) {
                val value =
                    grid[
                        y * width + x
                    ].toDouble().coerceIn(
                        0.0,
                        1.0,
                    )

                if (
                    value >=
                        config.foregroundThreshold
                ) {
                    foregroundCount += 1
                }

                if (
                    value >=
                        config.strongForegroundThreshold
                ) {
                    strongForegroundCount += 1
                }

                if (
                    value >=
                        config.foregroundThreshold
                ) {
                    confidenceSum += value
                    weightSum += 1.0
                    val normalizedX =
                        (
                            x + 0.5
                        ) / width.toDouble()
                    val normalizedY =
                        (
                            y + 0.5
                        ) / height.toDouble()
                    weightedX +=
                        normalizedX * value
                    weightedY +=
                        normalizedY * value
                }

                if (x + 1 < width) {
                    boundaryVariation +=
                        abs(
                            value -
                                grid[
                                    y * width +
                                        x + 1
                                ].toDouble(),
                        )
                    boundaryPairs += 1
                }

                if (y + 1 < height) {
                    boundaryVariation +=
                        abs(
                            value -
                                grid[
                                    (y + 1) * width +
                                        x
                                ].toDouble(),
                        )
                    boundaryPairs += 1
                }
            }
        }

        val total =
            (width * height)
                .coerceAtLeast(1)

        val coverage =
            foregroundCount.toDouble() /
                total

        val strongCoverage =
            strongForegroundCount.toDouble() /
                total

        val meanConfidence =
            if (weightSum <= 0.0) {
                0.0
            } else {
                confidenceSum /
                    weightSum
            }

        val centroidX =
            if (weightSum <= 0.0) {
                0.5
            } else {
                (
                    weightedX /
                        weightSum
                ).coerceIn(0.0, 1.0)
            }

        val centroidY =
            if (weightSum <= 0.0) {
                0.5
            } else {
                (
                    weightedY /
                        weightSum
                ).coerceIn(0.0, 1.0)
            }

        val meanBoundaryVariation =
            if (boundaryPairs <= 0) {
                0.0
            } else {
                boundaryVariation /
                    boundaryPairs.toDouble()
            }

        val boundaryStability =
            (
                1.0 -
                    meanBoundaryVariation *
                    1.35
            ).coerceIn(0.0, 1.0)

        val compactness =
            estimateSpatialCompactness(
                grid,
                coverage,
            )

        val warnings =
            buildList {
                if (coverage < 0.01) {
                    add("NO_FOREGROUND_SUPPORT")
                }
                if (coverage > 0.92) {
                    add("FOREGROUND_NEAR_FULL_FRAME")
                }
                if (
                    meanConfidence <
                        config.minimumQuality
                ) {
                    add("SEGMENTATION_CONFIDENCE_DEGRADED")
                }
                if (
                    boundaryStability < 0.45
                ) {
                    add("SILHOUETTE_BOUNDARY_UNSTABLE")
                }
                if (
                    compactness < 0.18 &&
                    coverage > 0.02
                ) {
                    add("SCATTERED_FOREGROUND")
                }
            }.distinct()

        return Result(
            timestampMs = timestampMs,
            inferenceLatencyMs =
                inferenceLatencyMs,
            sourceWidth = sourceWidth,
            sourceHeight = sourceHeight,
            foregroundCoverage = coverage,
            strongForegroundCoverage =
                strongCoverage,
            meanForegroundConfidence =
                meanConfidence.coerceIn(
                    0.0,
                    1.0,
                ),
            boundaryStability =
                boundaryStability,
            spatialCompactness =
                compactness,
            horizontalCentroid =
                centroidX,
            verticalCentroid =
                centroidY,
            modelQuality =
                modelQuality.coerceIn(
                    0.0,
                    1.0,
                ),
            personMaskIndices =
                personMaskIndices,
            labels = modelLabels,
            activeDelegate = activeDelegate,
            gpuFallbackCount = gpuFallbackCount,
            temporalMaskConsistency =
                temporalMaskConsistency.coerceIn(0.0, 1.0),
            gridWidth = width,
            gridHeight = height,
            foregroundMask =
                grid.copyOf(),
            state =
                if (
                    modelQuality >= config.minimumQuality &&
                        meanConfidence >=
                        config.minimumQuality * 0.85
                ) {
                    State.READY
                } else {
                    State.DEGRADED
                },
            warnings = warnings,
        )
    }

    private fun stabilizeGrid(
        current: FloatArray,
        previous: Result?,
    ): StabilizationResult {
        if (
            previous == null ||
            previous.foregroundMask.size != current.size
        ) {
            return StabilizationResult(
                grid = current.copyOf(),
                consistency = 0.0,
            )
        }

        val width = config.outputGridWidth
        val height = config.outputGridHeight
        val currentCentroidX =
            weightedCentroidAxis(
                current,
                width,
                height,
                horizontal = true,
            )
        val currentCentroidY =
            weightedCentroidAxis(
                current,
                width,
                height,
                horizontal = false,
            )

        val previousCentroidX =
            previous.horizontalCentroid
                .coerceIn(0.0, 1.0)
        val previousCentroidY =
            previous.verticalCentroid
                .coerceIn(0.0, 1.0)

        val shiftX =
            (
                currentCentroidX -
                    previousCentroidX
                )
        val shiftY =
            (
                currentCentroidY -
                    previousCentroidY
                )

        val compensationScale =
            1.0 / 3.0

        val compensated =
            FloatArray(current.size)

        var absoluteDifference = 0.0
        var agreementWeight = 0.0

        for (y in 0 until height) {
            val normalizedY =
                (
                    y + 0.5
                ) / height.toDouble()

            for (x in 0 until width) {
                val normalizedX =
                    (
                        x + 0.5
                    ) / width.toDouble()

                val currentValue =
                    current[
                        y * width + x
                    ].toDouble().coerceIn(0.0, 1.0)

                val previousValue =
                    sampleGrid(
                        previous.foregroundMask,
                        width,
                        height,
                        normalizedX -
                            shiftX * compensationScale,
                        normalizedY -
                            shiftY * compensationScale,
                    )

                absoluteDifference +=
                    abs(
                        currentValue -
                            previousValue,
                    )
                agreementWeight +=
                    max(
                        currentValue,
                        previousValue,
                    ) + 1e-4

                val motionMagnitude =
                    (
                        abs(shiftX) +
                            abs(shiftY)
                    ).coerceIn(0.0, 0.25)

                val temporalWeight =
                    (
                        (
                            previous.temporalMaskConsistency *
                                0.42
                        ) +
                            (
                                previous.meanForegroundConfidence *
                                    0.38
                            ) +
                            (
                                previous.boundaryStability *
                                    0.20
                            )
                    )
                        .coerceIn(0.0, 1.0)

                val alpha =
                    (
                        0.18 +
                            motionMagnitude * 0.72 +
                            (
                                1.0 -
                                    temporalWeight
                            ) *
                                0.22
                    )
                        .coerceIn(0.18, 0.94)

                compensated[
                    y * width + x
                ] =
                    (
                        previousValue *
                            (1.0 - alpha) +
                            currentValue *
                                alpha
                    )
                        .coerceIn(0.0, 1.0)
                        .toFloat()
            }
        }

        val meanDifference =
            (
                absoluteDifference /
                    agreementWeight.coerceAtLeast(1e-6)
            ).coerceIn(0.0, 1.0)

        val consistency =
            (
                1.0 -
                    meanDifference * 1.45
            ).coerceIn(0.0, 1.0)

        return StabilizationResult(
            grid = compensated,
            consistency = consistency,
        )
    }

    private fun weightedCentroidAxis(
        grid: FloatArray,
        width: Int,
        height: Int,
        horizontal: Boolean,
    ): Double {
        var weighted = 0.0
        var weight = 0.0

        for (y in 0 until height) {
            for (x in 0 until width) {
                val value =
                    grid[
                        y * width + x
                    ].toDouble().coerceIn(0.0, 1.0)

                val axis =
                    if (horizontal) {
                        (x + 0.5) /
                            width.toDouble()
                    } else {
                        (y + 0.5) /
                            height.toDouble()
                    }

                weighted +=
                    axis * value
                weight += value
            }
        }

        return if (weight <= 1e-9) {
            0.5
        } else {
            (
                weighted /
                    weight
            ).coerceIn(0.0, 1.0)
        }
    }

    private fun sampleGrid(
        grid: FloatArray,
        width: Int,
        height: Int,
        normalizedX: Double,
        normalizedY: Double,
    ): Double {
        val x =
            (
                normalizedX.coerceIn(0.0, 1.0) *
                    (width - 1)
            ).coerceAtLeast(0.0)
        val y =
            (
                normalizedY.coerceIn(0.0, 1.0) *
                    (height - 1)
            ).coerceAtLeast(0.0)

        val x0 = x.toInt().coerceIn(0, width - 1)
        val y0 = y.toInt().coerceIn(0, height - 1)
        val x1 = min(x0 + 1, width - 1)
        val y1 = min(y0 + 1, height - 1)
        val tx = x - x0
        val ty = y - y0

        val v00 =
            grid[
                y0 * width + x0
            ].toDouble().coerceIn(0.0, 1.0)
        val v10 =
            grid[
                y0 * width + x1
            ].toDouble().coerceIn(0.0, 1.0)
        val v01 =
            grid[
                y1 * width + x0
            ].toDouble().coerceIn(0.0, 1.0)
        val v11 =
            grid[
                y1 * width + x1
            ].toDouble().coerceIn(0.0, 1.0)

        val top =
            v00 +
                (
                    v10 - v00
                ) * tx
        val bottom =
            v01 +
                (
                    v11 - v01
                ) * tx

        return (
            top +
                (
                    bottom - top
                ) * ty
            ).coerceIn(0.0, 1.0)
    }

    private data class StabilizationResult(
        val grid: FloatArray,
        val consistency: Double,
    )

    private fun estimateSpatialCompactness(
        grid: FloatArray,
        coverage: Double,
    ): Double {
        if (coverage <= 0.0) return 0.0

        var neighborAgreement = 0.0
        var pairCount = 0

        for (y in 0 until config.outputGridHeight) {
            for (x in 0 until config.outputGridWidth) {
                val value =
                    grid[
                        y *
                            config.outputGridWidth +
                            x
                    ].toDouble()
                if (
                    value <
                        config.foregroundThreshold
                ) {
                    continue
                }

                if (
                    x + 1 <
                    config.outputGridWidth
                ) {
                    val neighbor =
                        grid[
                            y *
                                config.outputGridWidth +
                                x + 1
                        ].toDouble()
                    if (
                        neighbor >=
                            config.foregroundThreshold
                    ) {
                        neighborAgreement += 1.0
                    }
                    pairCount += 1
                }

                if (
                    y + 1 <
                    config.outputGridHeight
                ) {
                    val neighbor =
                        grid[
                            (y + 1) *
                                config.outputGridWidth +
                                x
                        ].toDouble()
                    if (
                        neighbor >=
                            config.foregroundThreshold
                    ) {
                        neighborAgreement += 1.0
                    }
                    pairCount += 1
                }
            }
        }

        return if (pairCount <= 0) {
            0.0
        } else {
            (
                neighborAgreement /
                    pairCount.toDouble()
            ).coerceIn(0.0, 1.0)
        }
    }

    private fun modelQuality(
        result: ImageSegmenterResult,
        indices: List<Int>,
    ): Double {
        val quality =
            result.qualityScores()
        if (quality.isEmpty()) return 1.0

        val usable =
            if (indices.isEmpty()) {
                quality
            } else {
                indices.mapNotNull {
                    quality.getOrNull(it)
                }
            }

        return if (usable.isEmpty()) {
            quality.average()
        } else {
            usable.average()
        }.coerceIn(0.0, 1.0)
    }

    private fun resolvePersonMaskIndices(
        labels: List<String>,
    ): List<Int> {
        if (labels.isEmpty()) {
            return emptyList()
        }

        val normalized =
            labels.map {
                it
                    .trim()
                    .lowercase()
            }

        val matched =
            normalized
                .mapIndexedNotNull { index, label ->
                    if (
                        config.personLabelHints.any {
                            label.contains(it)
                        } &&
                        config.backgroundLabelHints.none {
                            label.contains(it)
                        }
                    ) {
                        index
                    } else {
                        null
                    }
                }

        if (matched.isNotEmpty()) {
            return matched
        }

        val background =
            normalized
                .mapIndexedNotNull { index, label ->
                    if (
                        config.backgroundLabelHints.any {
                            label.contains(it)
                        }
                    ) {
                        index
                    } else {
                        null
                    }
                }
                .toSet()

        return normalized.indices
            .filter { it !in background }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) {
            return
        }

        synchronized(this) {
            segmenter?.close()
            segmenter = null
            activeDelegate = null
            latestResult = null
            modelLabels = emptyList()
            latestArrivalUptimeMillis =
                Long.MIN_VALUE
            submittedUptimeByTimestamp.clear()
            personMaskIndices = emptyList()
        }
    }
}

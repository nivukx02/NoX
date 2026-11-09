package com.nivukx.nox.core.ai

import android.media.Image
import android.graphics.ImageFormat
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaExtractor
import android.media.MediaFormat
import android.os.Handler
import android.os.HandlerThread
import com.google.android.gms.tasks.Tasks
import com.nivukx.nox.core.model.MotionCaptureCustomization
import java.io.File
import java.util.Collections
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Offline visual-evidence decoder that re-enters the existing NoX multimodal perception graph.
 *
 * The decoder preserves a bounded queue of Android Image objects, keeps timestamps in nanoseconds,
 * and delegates body/multi-person/face/eye/hand/dense-landmark processing to PerceptionPipeline.
 * It deliberately does not duplicate the reconstruction mathematics already implemented in NoX.
 */
class RecordedVideoPostProcessor(
    private val perception: PerceptionPipeline,
    private val reconstructionCallback:
        ((PerceptionPipeline.Result) -> Int)? = null,
    private val queueCapacity: Int = 4,
) : AutoCloseable {

    data class Configuration(
        val cameraId: String = "offline-0",
        val rotationDegreesOverride: Int? = null,
        val maxFrames: Long = Long.MAX_VALUE,
        val queueOfferTimeoutMillis: Long = 1_000L,
        val taskTimeoutSeconds: Long = 30L,
        val abortOnConsecutiveFrameFailures: Int = 12,
        val customization: MotionCaptureCustomization =
            MotionCaptureCustomization.production(),
        val rollingShutterDirectionSign: Int = 1,
    ) {
        init {
            require(maxFrames > 0L)
            require(queueOfferTimeoutMillis in 50L..10_000L)
            require(taskTimeoutSeconds in 1L..300L)
            require(abortOnConsecutiveFrameFailures in 1..100)
            require(
                rollingShutterDirectionSign == 1 ||
                    rollingShutterDirectionSign == -1,
            )
        }
    }

    data class FrameSummary(
        val timestampNanos: Long,
        val sequence: Long,
        val subjectCount: Int,
        val actorCount: Int,
        val poseConfidence: Double,
        val inferenceMillis: Long,
        val warnings: List<String>,
    )

    data class Result(
        val videoFile: File,
        val width: Int,
        val height: Int,
        val sourceFps: Double,
        val frameDurationNanos: Long,
        val rotationDegrees: Int,
        val codecMimeType: String,
        val decodedFrames: Long,
        val processedFrames: Long,
        val droppedFrames: Long,
        val failedFrames: Long,
        val maxObservedActors: Int,
        val maxObservedSubjects: Int,
        val reconstructedFrames: Long,
        val acceptedReconstructedActors: Long,
        val totalProcessingMillis: Long,
        val cancelled: Boolean,
        val completed: Boolean,
        val warnings: List<String>,
        val frameSummaries: List<FrameSummary>,
    ) {
        val processingCoverage: Double
            get() =
                if (decodedFrames <= 0L) 0.0
                else processedFrames.toDouble() / decodedFrames.toDouble()
    }

    private data class QueuedFrame(
        val image: Image,
        val sequence: Long,
        val timestampNanos: Long,
    )

    private data class RunContext(
        val configuration: Configuration,
        val videoFile: File,
        val resultLatch: CountDownLatch,
    ) {
        @Volatile
        var sourceRotationDegrees: Int =
            configuration.rotationDegreesOverride ?: 0
        @Volatile
        var decodeStarted: Boolean = false
        @Volatile
        var decodeFinished: Boolean = false
        @Volatile
        var callerFinished: Boolean = false
    }

    private val runAdmissionLock = Any()
    private val closed = AtomicBoolean(false)
    private val cancelled = AtomicBoolean(false)
    private val inFlight = AtomicLong(0L)
    private val decodedFrames = AtomicLong(0L)
    private val processedFrames = AtomicLong(0L)
    private val droppedFrames = AtomicLong(0L)
    private val failedFrames = AtomicLong(0L)
    private val maxObservedActors = AtomicLong(0L)
    private val maxObservedSubjects = AtomicLong(0L)
    private val reconstructedFrames = AtomicLong(0L)
    private val acceptedReconstructedActors = AtomicLong(0L)
    private val totalProcessingMillis = AtomicLong(0L)
    private val consecutiveFailures = AtomicLong(0L)
    private val runFailed = AtomicBoolean(false)

    private val processingQueue =
        ArrayBlockingQueue<QueuedFrame>(queueCapacity)
    private val decodeThread =
        HandlerThread("NoX-OfflineVideoDecode").apply { start() }
    private val decodeHandler = Handler(decodeThread.looper)
    private val processingThread =
        Thread({ processingLoop() }, "NoX-OfflineVideoPerception").apply {
            isDaemon = true
            priority = Thread.NORM_PRIORITY
        }

    private val summariesLock = Any()
    private val frameSummaries = ArrayList<FrameSummary>(2_048)
    @Volatile private var currentRun: RunContext? = null
    @Volatile private var currentCallback:
        ((FrameSummary, PerceptionPipeline.Result) -> Unit)? = null
    private val runWarnings =
        Collections.synchronizedList(mutableListOf<String>())

    init {
        require(queueCapacity in 2..32)
        processingThread.start()
    }

    fun process(
        videoFile: File,
        configuration: Configuration = Configuration(),
        onFrame: ((FrameSummary, PerceptionPipeline.Result) -> Unit)? = null,
    ): Result {
        val run =
            synchronized(runAdmissionLock) {
                check(!closed.get()) { "RecordedVideoPostProcessor is closed" }
                check(currentRun == null) {
                    "RecordedVideoPostProcessor does not permit concurrent reconstruction runs"
                }
                check(inFlight.get() == 0L && processingQueue.isEmpty()) {
                    "RecordedVideoPostProcessor cannot start a new run while prior frames are still draining"
                }
                require(videoFile.isFile) {
                    "Recorded video does not exist: " + videoFile.absolutePath
                }

                perception.configureCustomization(configuration.customization)
                resetRunState()
                currentCallback = onFrame

                RunContext(
                    configuration = configuration,
                    videoFile = videoFile,
                    resultLatch = CountDownLatch(1),
                ).also { currentRun = it }
            }

        if (!decodeHandler.post { decode(run) }) {
            runFailed.set(true)
            addWarning("OFFLINE_DECODE_SCHEDULING_FAILED")
            finishRun(run)
        }

        val waited = run.resultLatch.await(
            configuration.taskTimeoutSeconds * 10L,
            TimeUnit.SECONDS,
        )
        if (!waited) {
            cancelled.set(true)
            // Keep the decode task queued if it has not started yet. It must run cleanup
            // and release currentRun; removing it here can permanently poison this instance.
            addWarning("OFFLINE_DECODE_RUN_TIMEOUT")

            run.resultLatch.await(
                5L,
                TimeUnit.SECONDS,
            )
            if (currentRun === run) {
                runFailed.set(true)
                addWarning("OFFLINE_DECODE_DID_NOT_DRAIN_AFTER_TIMEOUT")
            }
        }

        return try {
            buildResult(videoFile, configuration, run)
        } finally {
            synchronized(runAdmissionLock) {
                run.callerFinished = true
                if (run.decodeFinished && currentRun === run) {
                    currentRun = null
                }
            }
        }
    }

    fun cancel() {
        cancelled.set(true)
        addWarning("OFFLINE_DECODE_CANCELLED")
    }

    private fun decode(run: RunContext) {
        var extractor: MediaExtractor? = null
        var decoder: MediaCodec? = null
        var reader: ImageReader? = null
        var readerThread: HandlerThread? = null
        var readerHandler: Handler? = null

        try {
            val mayStart =
                synchronized(runAdmissionLock) {
                    if (closed.get() || cancelled.get() || currentRun !== run) {
                        false
                    } else {
                        run.decodeStarted = true
                        true
                    }
                }
            if (!mayStart) {
                addWarning("OFFLINE_DECODE_CANCELLED_BEFORE_START")
                return
            }

            val activeExtractor = MediaExtractor()
            extractor = activeExtractor
            activeExtractor.setDataSource(run.videoFile.absolutePath)

            val trackIndex = selectVideoTrack(activeExtractor)
            require(trackIndex >= 0) {
                "No video track found in " + run.videoFile.name
            }
            activeExtractor.selectTrack(trackIndex)

            val sourceFormat = activeExtractor.getTrackFormat(trackIndex)
            val mime = sourceFormat.getString(MediaFormat.KEY_MIME)
                ?: error("Video track has no MIME type")
            val width = sourceFormat.getInteger(MediaFormat.KEY_WIDTH)
            val height = sourceFormat.getInteger(MediaFormat.KEY_HEIGHT)
            require(width > 0 && height > 0) {
                "Invalid video dimensions: " + width + "x" + height
            }

            val frameRate = if (sourceFormat.containsKey(MediaFormat.KEY_FRAME_RATE)) {
                sourceFormat.getInteger(MediaFormat.KEY_FRAME_RATE).toDouble()
            } else 30.0

            val rotation = resolveRecordedVideoRotation(
                overrideDegrees = run.configuration.rotationDegreesOverride,
                containerDegrees =
                    if (sourceFormat.containsKey(MediaFormat.KEY_ROTATION)) {
                        sourceFormat.getInteger(MediaFormat.KEY_ROTATION)
                    } else {
                        null
                    },
            )
            run.sourceRotationDegrees = rotation

            val decoderInfo = selectDecoder(mime, width, height)
            val activeDecoder = MediaCodec.createByCodecName(decoderInfo.name)
            decoder = activeDecoder

            readerThread = HandlerThread("NoX-OfflineVideoImageReader").apply { start() }
            readerHandler = Handler(readerThread.looper)

            val imageReader = ImageReader.newInstance(
                width,
                height,
                ImageFormat.YUV_420_888,
                queueCapacity + 2,
            )
            reader = imageReader
            imageReader.setOnImageAvailableListener(
                { source -> acquireDecodedImage(source, run) },
                readerHandler,
            )

            activeDecoder.configure(
                sourceFormat,
                imageReader.surface,
                null,
                0,
            )
            activeDecoder.start()

            val bufferInfo = MediaCodec.BufferInfo()
            var inputEos = false
            var outputEos = false

            while (!outputEos && !cancelled.get() && !closed.get()) {
                if (!inputEos) {
                    val inputIndex = activeDecoder.dequeueInputBuffer(20_000L)
                    if (inputIndex >= 0) {
                        val inputBuffer = activeDecoder.getInputBuffer(inputIndex)
                        if (inputBuffer != null) {
                            val sampleSize = activeExtractor.readSampleData(inputBuffer, 0)
                            if (sampleSize < 0) {
                                activeDecoder.queueInputBuffer(
                                    inputIndex, 0, 0, 0L,
                                    MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                                )
                                inputEos = true
                            } else {
                                val sampleTimeUs = activeExtractor.sampleTime.coerceAtLeast(0L)
                                activeDecoder.queueInputBuffer(
                                    inputIndex,
                                    0,
                                    sampleSize,
                                    sampleTimeUs,
                                    activeExtractor.sampleFlags,
                                )
                                activeExtractor.advance()
                            }
                        }
                    }
                }

                val outputIndex = activeDecoder.dequeueOutputBuffer(bufferInfo, 20_000L)
                when {
                    outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        addWarning("OFFLINE_DECODER_FORMAT_READY")
                    }
                    outputIndex >= 0 -> {
                        val shouldRender =
                            bufferInfo.size > 0 ||
                                (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                        activeDecoder.releaseOutputBuffer(outputIndex, shouldRender)
                        outputEos =
                            (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                    }
                }

                if (decodedFrames.get() >= run.configuration.maxFrames) {
                    cancelled.set(true)
                    addWarning("MAX_OFFLINE_FRAME_LIMIT_REACHED")
                }
            }

            if (cancelled.get() && !outputEos) {
                addWarning("OFFLINE_DECODER_STOPPED_BEFORE_EOS")
            }

            awaitProcessingQueue(run.configuration.taskTimeoutSeconds)
        } catch (error: Throwable) {
            runFailed.set(true)
            addWarning(
                "OFFLINE_DECODE_FAILURE:" +
                    (error.message ?: error.javaClass.simpleName)
            )
        } finally {
            runCatching { reader?.setOnImageAvailableListener(null, null) }
            runCatching { reader?.close() }
            runCatching { decoder?.stop() }
            runCatching { decoder?.release() }
            runCatching { extractor?.release() }
            runCatching { readerThread?.quitSafely(); readerThread?.join(2_000L) }
            finishRun(run)
        }
    }

    private fun acquireDecodedImage(
        source: ImageReader,
        run: RunContext,
    ) {
        val image = runCatching { source.acquireNextImage() }.getOrNull() ?: return
        val sequence = decodedFrames.getAndIncrement()
        val timestampNanos = image.timestamp.coerceAtLeast(0L)

        if (sequence >= run.configuration.maxFrames || cancelled.get() || closed.get()) {
            image.close()
            return
        }

        val accepted = runCatching {
            processingQueue.offer(
                QueuedFrame(image, sequence, timestampNanos),
                run.configuration.queueOfferTimeoutMillis,
                TimeUnit.MILLISECONDS,
            )
        }.getOrDefault(false)

        if (!accepted) {
            image.close()
            droppedFrames.incrementAndGet()
            addWarning("OFFLINE_PROCESSING_QUEUE_PRESSURE")
        }
    }

    private fun processingLoop() {
        while (!closed.get()) {
            val queued = try {
                processingQueue.poll(100L, TimeUnit.MILLISECONDS)
            } catch (_: InterruptedException) {
                if (closed.get()) return
                null
            } ?: continue

            val run = currentRun
            if (run == null) {
                queued.image.close()
                continue
            }

            inFlight.incrementAndGet()
            val startedNanos = System.nanoTime()
            try {
                val task = perception.process(
                    image = queued.image,
                    rotationDegrees = resolveRecordedVideoRotation(
                        overrideDegrees = run.configuration.rotationDegreesOverride,
                        containerDegrees = run.sourceRotationDegrees,
                    ),
                    timestampNanos = queued.timestampNanos,
                    cameraId = run.configuration.cameraId,
                    rollingShutterDirectionSign =
                        run.configuration.rollingShutterDirectionSign,
                )

                val result = Tasks.await(
                    task,
                    run.configuration.taskTimeoutSeconds,
                    TimeUnit.SECONDS,
                )

                val actorCount = result.actorTracks.size
                val subjectCount = result.subjectCount
                maxObservedActors.updateAndGet { maxOf(it, actorCount.toLong()) }
                maxObservedSubjects.updateAndGet { maxOf(it, subjectCount.toLong()) }

                reconstructionCallback?.let { callback ->
                    val acceptedActors =
                        runCatching {
                            callback(result)
                        }.getOrDefault(0)

                    if (acceptedActors > 0) {
                        reconstructedFrames.incrementAndGet()
                        acceptedReconstructedActors.addAndGet(
                            acceptedActors.toLong(),
                        )
                    }
                }

                val elapsedMillis =
                    (System.nanoTime() - startedNanos).coerceAtLeast(0L) / 1_000_000L
                totalProcessingMillis.addAndGet(elapsedMillis)
                processedFrames.incrementAndGet()
                consecutiveFailures.set(0L)

                val summary = FrameSummary(
                    timestampNanos = queued.timestampNanos,
                    sequence = queued.sequence,
                    subjectCount = subjectCount,
                    actorCount = actorCount,
                    poseConfidence = result.pose.confidence.toDouble(),
                    inferenceMillis = elapsedMillis,
                    warnings = result.multiActorPose?.warnings.orEmpty(),
                )

                synchronized(summariesLock) {
                    if (frameSummaries.size < 50_000) frameSummaries += summary
                }
                currentCallback?.invoke(summary, result)
            } catch (error: Throwable) {
                val failures = consecutiveFailures.incrementAndGet()
                failedFrames.incrementAndGet()
                runFailed.set(true)
                addWarning(
                    "OFFLINE_FRAME_FAILURE:" +
                        (error.message ?: error.javaClass.simpleName),
                )
                if (failures >= run.configuration.abortOnConsecutiveFrameFailures) {
                    cancelled.set(true)
                    addWarning("OFFLINE_CONSECUTIVE_FAILURE_ABORT")
                }
            } finally {
                queued.image.close()
                inFlight.decrementAndGet()
            }
        }
    }

    private fun awaitProcessingQueue(timeoutSeconds: Long) {
        val deadlineNanos = System.nanoTime() +
            TimeUnit.SECONDS.toNanos(timeoutSeconds.coerceIn(1L, 300L))
        while (
            !closed.get() &&
            System.nanoTime() < deadlineNanos &&
            (processingQueue.isNotEmpty() || inFlight.get() > 0L)
        ) {
            Thread.sleep(10L)
        }
        if (processingQueue.isNotEmpty() || inFlight.get() > 0L) {
            runFailed.set(true)
            addWarning("OFFLINE_PROCESSING_DRAIN_TIMEOUT")
            while (true) {
                val stale = processingQueue.poll() ?: break
                stale.image.close()
            }
        }
    }

    private fun finishRun(run: RunContext) {
        synchronized(runAdmissionLock) {
            run.decodeFinished = true
            if (run.callerFinished && currentRun === run) {
                currentRun = null
            }
            run.resultLatch.countDown()
        }
    }

    private fun buildResult(
        videoFile: File,
        configuration: Configuration,
        run: RunContext,
    ): Result {
        val dimensions = runCatching {
            val extractor = MediaExtractor()
            try {
                extractor.setDataSource(videoFile.absolutePath)
                val track = selectVideoTrack(extractor)
                if (track < 0) {
                    null
                } else {
                    extractor.getTrackFormat(track)
                }
            } finally {
                extractor.release()
            }
        }.getOrNull()

        val width = dimensions?.getInteger(MediaFormat.KEY_WIDTH) ?: 0
        val height = dimensions?.getInteger(MediaFormat.KEY_HEIGHT) ?: 0
        val fps = dimensions?.let { format ->
            if (format.containsKey(MediaFormat.KEY_FRAME_RATE))
                format.getInteger(MediaFormat.KEY_FRAME_RATE).toDouble()
            else 0.0
        } ?: 0.0
        val mime = dimensions?.getString(MediaFormat.KEY_MIME) ?: "unknown"

        return Result(
            videoFile = videoFile,
            width = width,
            height = height,
            sourceFps = fps,
            frameDurationNanos = if (fps > 0.0) (1_000_000_000.0 / fps).toLong() else 0L,
            rotationDegrees = resolveRecordedVideoRotation(
                overrideDegrees = configuration.rotationDegreesOverride,
                containerDegrees =
                    dimensions?.let { format ->
                        if (format.containsKey(MediaFormat.KEY_ROTATION)) {
                            format.getInteger(MediaFormat.KEY_ROTATION)
                        } else {
                            null
                        }
                    },
            ),
            codecMimeType = mime,
            decodedFrames = decodedFrames.get(),
            processedFrames = processedFrames.get(),
            droppedFrames = droppedFrames.get(),
            failedFrames = failedFrames.get(),
            maxObservedActors = maxObservedActors.get().toInt(),
            maxObservedSubjects = maxObservedSubjects.get().toInt(),
            reconstructedFrames = reconstructedFrames.get(),
            acceptedReconstructedActors = acceptedReconstructedActors.get(),
            totalProcessingMillis = totalProcessingMillis.get(),
            cancelled = cancelled.get(),
            completed =
                run.decodeFinished &&
                    !cancelled.get() &&
                    !runFailed.get(),
            warnings = synchronized(runWarnings) { runWarnings.distinct().takeLast(64) },
            frameSummaries = synchronized(summariesLock) { frameSummaries.toList() },
        )
    }

    private fun resetRunState() {
        cancelled.set(false)
        while (true) {
            val stale = processingQueue.poll() ?: break
            stale.image.close()
        }
        currentRun = null
        runWarnings.clear()
        runFailed.set(false)
        decodedFrames.set(0L)
        processedFrames.set(0L)
        droppedFrames.set(0L)
        failedFrames.set(0L)
        maxObservedActors.set(0L)
        maxObservedSubjects.set(0L)
        reconstructedFrames.set(0L)
        acceptedReconstructedActors.set(0L)
        totalProcessingMillis.set(0L)
        consecutiveFailures.set(0L)
        inFlight.set(0L)
        currentCallback = null
        synchronized(summariesLock) { frameSummaries.clear() }
    }

    private fun selectVideoTrack(extractor: MediaExtractor): Int {
        for (index in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(index)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("video/")) return index
        }
        return -1
    }

    private fun selectDecoder(
        mime: String,
        width: Int,
        height: Int,
    ): MediaCodecInfo {
        val infos = MediaCodecList(MediaCodecList.ALL_CODECS).codecInfos
        val candidate = infos.firstOrNull { info ->
            !info.isEncoder &&
                info.supportedTypes.any { it.equals(mime, ignoreCase = true) } &&
                runCatching {
                    val capabilities = info.getCapabilitiesForType(mime)
                    val video = capabilities.videoCapabilities ?: return@runCatching false
                    video.isSizeSupported(width, height) && capabilities.colorFormats.isNotEmpty()
                }.getOrDefault(false)
        }
        return candidate ?: throw IllegalStateException(
            "No decoder for " + mime + " at " + width + "x" + height
        )
    }

    private fun addWarning(message: String) {
        if (message.isBlank()) return
        runWarnings.add(message)
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        cancelled.set(true)

        val queuedRun =
            synchronized(runAdmissionLock) {
                val run = currentRun
                if (run != null && !run.decodeStarted) {
                    decodeHandler.removeCallbacksAndMessages(null)
                    run
                } else {
                    null
                }
            }
        queuedRun?.let { run ->
            runFailed.set(true)
            addWarning("OFFLINE_DECODE_CANCELLED_BEFORE_START")
            finishRun(run)
        }

        decodeHandler.removeCallbacksAndMessages(null)
        processingThread.interrupt()
        runCatching { decodeThread.quitSafely(); decodeThread.join(2_000L) }
        runCatching { processingThread.join(2_000L) }
        while (true) {
            val stale = processingQueue.poll() ?: break
            runCatching { stale.image.close() }
        }
    }
}

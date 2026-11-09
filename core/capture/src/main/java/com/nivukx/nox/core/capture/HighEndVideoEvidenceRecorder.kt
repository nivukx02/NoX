package com.nivukx.nox.core.capture

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.Build
import android.view.Surface
import com.nivukx.nox.core.model.RecordedVideoCodec
import com.nivukx.nox.core.model.RecordedVideoEvidenceProfile
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Surface-input Android evidence recorder for high-end NoX capture.
 *
 * The camera is connected directly to the encoder input Surface, avoiding a CPU-side YUV copy for
 * the recorded path. The recorder is deliberately independent from the motion-analysis ImageReader:
 * analysis remains low-latency while encoded visual evidence is drained on a separate bounded worker.
 *
 * Important production behavior:
 *  - codec choice is negotiated against actual MediaCodec encoders;
 *  - HEVC/AVC/AV1 preferences are fallible and exposed as actual state;
 *  - output buffers are drained continuously so the camera producer cannot be starved by encoder
 *    backpressure;
 *  - muxer startup waits for the encoder output format;
 *  - stop is EOS-driven and bounded by a timeout;
 *  - all ownership is explicit and idempotent;
 *  - the recorder never invents a timestamp or accuracy guarantee.
 *
 * This component must be prepared before Camera2CaptureController.open() so its input Surface is
 * included when the CameraCaptureSession is created.
 */
class HighEndVideoEvidenceRecorder(
    private val requested: RecordedVideoEvidenceProfile,
    private val outputFile: File,
) : AutoCloseable {

    enum class State {
        IDLE,
        PREPARED,
        RUNNING,
        STOPPING,
        STOPPED,
        FAILED,
        CLOSED,
    }

    data class NegotiatedFormat(
        val mimeType: String,
        val width: Int,
        val height: Int,
        val fps: Int,
        val bitrateBitsPerSecond: Int,
        val requestTenBit: Boolean,
        val codecName: String,
    )

    data class Result(
        val outputFile: File,
        val state: State,
        val bytesWritten: Long,
        val encodedSamples: Long,
        val durationUs: Long,
        val negotiatedFormat: NegotiatedFormat?,
        val error: String?,
    )

    private val stateRef = AtomicReference(State.IDLE)
    private val bytesWritten = AtomicLong(0L)
    private val encodedSamples = AtomicLong(0L)
    private val durationUs = AtomicLong(0L)

    @Volatile
    private var encoder: MediaCodec? = null

    @Volatile
    private var muxer: MediaMuxer? = null

    @Volatile
    private var inputSurface: Surface? = null

    @Volatile
    private var negotiated: NegotiatedFormat? = null

    @Volatile
    private var trackIndex = -1

    @Volatile
    private var muxerStarted = false

    @Volatile
    private var failureMessage: String? = null

    private var drainExecutor: ExecutorService? = null
    private var stopLatch: CountDownLatch? = null

    val state: State
        get() = stateRef.get()

    val surface: Surface
        get() =
            inputSurface
                ?: error("Recorder is not prepared")

    val negotiatedFormat: NegotiatedFormat?
        get() = negotiated

    fun prepare() {
        check(stateRef.get() == State.IDLE) {
            "Recorder can only be prepared from IDLE"
        }

        outputFile.parentFile?.mkdirs()
        require(
            outputFile.parentFile?.isDirectory != false,
        ) {
            "Unable to create evidence output directory: " +
                (outputFile.parentFile?.absolutePath ?: "<none>")
        }

        val candidate = selectEncoder(requested.codec)
        val mime = candidate.first
        val codecInfo = candidate.second

        val actualBitrate =
            requested.bitrateBitsPerSecond.coerceIn(
                1_000_000,
                250_000_000,
            )

        val format =
            MediaFormat.createVideoFormat(
                mime,
                requested.width,
                requested.height,
            ).apply {
                setInteger(
                    MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface,
                )
                setInteger(
                    MediaFormat.KEY_BIT_RATE,
                    actualBitrate,
                )
                setInteger(
                    MediaFormat.KEY_FRAME_RATE,
                    requested.fps,
                )
                setInteger(
                    MediaFormat.KEY_I_FRAME_INTERVAL,
                    requested.iFrameIntervalSeconds,
                )

                runCatching {
                    setInteger(
                        MediaFormat.KEY_BITRATE_MODE,
                        MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR,
                    )
                }

                if (
                    requested.requestTenBit &&
                    mime == MediaFormat.MIMETYPE_VIDEO_HEVC
                ) {
                    runCatching {
                        setInteger(
                            MediaFormat.KEY_PROFILE,
                            MediaCodecInfo.CodecProfileLevel.HEVCProfileMain10,
                        )
                    }
                }

                if (requested.requestHdrTransfer && Build.VERSION.SDK_INT >= 29) {
                    runCatching {
                        setInteger(
                            MediaFormat.KEY_COLOR_STANDARD,
                            MediaFormat.COLOR_STANDARD_BT2020,
                        )
                        setInteger(
                            MediaFormat.KEY_COLOR_TRANSFER,
                            MediaFormat.COLOR_TRANSFER_HLG,
                        )
                        setInteger(
                            MediaFormat.KEY_COLOR_RANGE,
                            MediaFormat.COLOR_RANGE_LIMITED,
                        )
                    }
                }
            }

        try {
            val selectedCodec =
                MediaCodec.createByCodecName(codecInfo.name)

            selectedCodec.configure(
                format,
                null,
                null,
                MediaCodec.CONFIGURE_FLAG_ENCODE,
            )

            val surface =
                selectedCodec.createInputSurface()

            encoder = selectedCodec
            inputSurface = surface
            negotiated =
                NegotiatedFormat(
                    mimeType = mime,
                    width = requested.width,
                    height = requested.height,
                    fps = requested.fps,
                    bitrateBitsPerSecond = actualBitrate,
                    requestTenBit = requested.requestTenBit,
                    codecName = codecInfo.name,
                )

            muxer =
                MediaMuxer(
                    outputFile.absolutePath,
                    MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4,
                )

            stateRef.set(State.PREPARED)
        } catch (error: Throwable) {
            cleanupCodec()
            failureMessage =
                error.message ?: error.javaClass.simpleName
            stateRef.set(State.FAILED)
            throw IllegalStateException(
                "Unable to prepare video evidence encoder: " +
                    failureMessage,
                error,
            )
        }
    }

    fun start() {
        check(stateRef.get() == State.PREPARED) {
            "Recorder must be PREPARED before start"
        }

        val activeEncoder =
            encoder ?: error("Missing prepared encoder")
        val executor =
            Executors.newSingleThreadExecutor { runnable ->
                Thread(runnable, "NoX-VideoEvidenceDrain").apply {
                    isDaemon = true
                    priority = Thread.NORM_PRIORITY
                }
            }

        drainExecutor = executor
        stopLatch = CountDownLatch(1)
        stateRef.set(State.RUNNING)

        try {
            activeEncoder.start()
        } catch (error: Throwable) {
            failureMessage =
                error.message ?: error.javaClass.simpleName
            stateRef.set(State.FAILED)
            runCatching {
                executor.shutdownNow()
            }
            cleanupCodec()
            throw IllegalStateException(
                "Unable to start video evidence encoder: " +
                    failureMessage,
                error,
            )
        }

        executor.execute {
            drainLoop(activeEncoder)
        }
    }

    /**
     * Signals the camera input surface to finish and waits for the encoder EOS.
     *
     * The camera should normally be stopped immediately before this method so no additional
     * frames are being submitted to the input surface.
     */
    fun stop(timeoutSeconds: Long = 8L): Result {
        val state = stateRef.get()
        if (state == State.STOPPED || state == State.CLOSED) {
            return result()
        }
        if (state == State.FAILED) {
            cleanupCodec()
            return result()
        }

        check(state == State.RUNNING) {
            "Recorder must be RUNNING before stop"
        }

        stateRef.set(State.STOPPING)

        runCatching {
            encoder?.signalEndOfInputStream()
        }.onFailure {
            failureMessage =
                it.message ?: it.javaClass.simpleName
        }

        val latch = stopLatch
        if (latch != null) {
            val completed =
                latch.await(
                    timeoutSeconds.coerceIn(1L, 30L),
                    TimeUnit.SECONDS,
                )
            if (!completed && failureMessage == null) {
                failureMessage =
                    "Timed out waiting for encoder EOS"
                stateRef.set(State.FAILED)
            }
        }

        runCatching {
            drainExecutor?.shutdown()
            drainExecutor?.awaitTermination(
                2L,
                TimeUnit.SECONDS,
            )
        }

        cleanupCodec()
        if (stateRef.get() != State.FAILED) {
            stateRef.set(State.STOPPED)
        }
        return result()
    }

    override fun close() {
        val current = stateRef.get()
        if (current == State.CLOSED) return

        if (current == State.RUNNING) {
            runCatching {
                stop()
            }.onFailure {
                failureMessage =
                    it.message ?: it.javaClass.simpleName
            }
        } else {
            cleanupCodec()
        }

        stateRef.set(State.CLOSED)
    }

    private fun drainLoop(codec: MediaCodec) {
        val info = MediaCodec.BufferInfo()
        var eos = false

        try {
            while (!eos) {
                val index =
                    codec.dequeueOutputBuffer(
                        info,
                        25_000L,
                    )

                when {
                    index == MediaCodec.INFO_TRY_AGAIN_LATER -> Unit

                    index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        if (muxerStarted) {
                            throw IllegalStateException(
                                "Encoder output format changed more than once",
                            )
                        }

                        val activeMuxer =
                            muxer
                                ?: throw IllegalStateException(
                                    "Muxer unavailable on format change",
                                )

                        val outputFormat =
                            codec.outputFormat
                        trackIndex =
                            activeMuxer.addTrack(outputFormat)
                        activeMuxer.start()
                        muxerStarted = true
                    }

                    index == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit

                    index >= 0 -> {
                        val buffer =
                            codec.getOutputBuffer(index)
                                ?: throw IllegalStateException(
                                    "Encoder returned null output buffer",
                                )

                        val codecConfig =
                            (info.flags and
                                MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0

                        if (
                            info.size > 0 &&
                            muxerStarted &&
                            trackIndex >= 0 &&
                            !codecConfig
                        ) {
                            buffer.position(info.offset)
                            buffer.limit(
                                info.offset + info.size,
                            )

                            muxer?.writeSampleData(
                                trackIndex,
                                buffer,
                                info,
                            )

                            bytesWritten.addAndGet(
                                info.size.toLong(),
                            )
                            encodedSamples.incrementAndGet()

                            durationUs.updateAndGet {
                                max(it, info.presentationTimeUs)
                            }
                        }

                        eos =
                            (info.flags and
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM) !=
                                0

                        codec.releaseOutputBuffer(
                            index,
                            false,
                        )
                    }
                }

                if (outputFile.exists() &&
                    outputFile.length() >
                    requested.maxFileSizeBytes
                ) {
                    failureMessage =
                        "Video evidence file exceeded configured maximum size"
                    stateRef.set(State.FAILED)
                    break
                }
            }
        } catch (error: Throwable) {
            failureMessage =
                error.message ?: error.javaClass.simpleName
            stateRef.set(State.FAILED)
        } finally {
            runCatching {
                if (muxerStarted) {
                    muxer?.stop()
                }
            }.onFailure {
                if (failureMessage == null) {
                    failureMessage =
                        it.message ?: it.javaClass.simpleName
                }
            }
            muxerStarted = false
            stopLatch?.countDown()
        }
    }

    private fun selectEncoder(
        preferred: RecordedVideoCodec,
    ): Pair<String, MediaCodecInfo> {
        val preferences =
            when (preferred) {
                RecordedVideoCodec.HEVC ->
                    listOf(
                        MediaFormat.MIMETYPE_VIDEO_HEVC,
                        MediaFormat.MIMETYPE_VIDEO_AVC,
                    )

                RecordedVideoCodec.AV1 ->
                    listOf(
                        MediaFormat.MIMETYPE_VIDEO_AV1,
                        MediaFormat.MIMETYPE_VIDEO_HEVC,
                        MediaFormat.MIMETYPE_VIDEO_AVC,
                    )

                RecordedVideoCodec.AVC ->
                    listOf(
                        MediaFormat.MIMETYPE_VIDEO_AVC,
                        MediaFormat.MIMETYPE_VIDEO_HEVC,
                    )
            }

        val infos =
            MediaCodecList(
                MediaCodecList.ALL_CODECS,
            ).codecInfos

        for (mime in preferences) {
            val candidates =
                infos
                    .filter { codecInfo ->
                        codecInfo.isEncoder &&
                            codecInfo.supportedTypes.any {
                                it.equals(
                                    mime,
                                    ignoreCase = true,
                                )
                            } &&
                            supportsRequestedSizeAndRate(
                                codecInfo,
                                mime,
                            )
                    }
                    .sortedByDescending { codecInfo ->
                        hardwareAccelerationScore(
                            codecInfo,
                            mime,
                        )
                    }

            val info = candidates.firstOrNull()
            if (info != null) {
                return mime to info
            }
        }

        throw IllegalStateException(
            "No suitable hardware/software encoder found for " +
                preferences.joinToString(),
        )
    }

    private fun hardwareAccelerationScore(
        info: MediaCodecInfo,
        mime: String,
    ): Int {
        if (Build.VERSION.SDK_INT < 29) {
            return if (
                !info.name.startsWith(
                    "OMX.google.",
                    ignoreCase = true,
                )
            ) {
                1
            } else {
                0
            }
        }

        val hardware =
            runCatching {
                info.isHardwareAccelerated
            }.getOrDefault(false)

        val softwareOnly =
            runCatching {
                info.isSoftwareOnly
            }.getOrDefault(false)

        val tenBitBonus =
            if (
                requested.requestTenBit &&
                mime == MediaFormat.MIMETYPE_VIDEO_HEVC &&
                supportsHevcMain10(info)
            ) {
                2
            } else {
                0
            }

        return when {
            hardware && !softwareOnly -> 10 + tenBitBonus
            hardware -> 6 + tenBitBonus
            !softwareOnly -> 3
            else -> 0
        }
    }

    private fun supportsHevcMain10(
        info: MediaCodecInfo,
    ): Boolean =
        runCatching {
            info.getCapabilitiesForType(
                MediaFormat.MIMETYPE_VIDEO_HEVC,
            ).profileLevels.any {
                it.profile ==
                    MediaCodecInfo.CodecProfileLevel.HEVCProfileMain10
            }
        }.getOrDefault(false)

    private fun supportsRequestedSizeAndRate(
        info: MediaCodecInfo,
        mime: String,
    ): Boolean {
        val caps =
            runCatching {
                info.getCapabilitiesForType(mime)
            }.getOrNull()
                ?: return false

        val video =
            caps.videoCapabilities
                ?: return false

        if (
            !video.isSizeSupported(
                requested.width,
                requested.height,
            )
        ) {
            return false
        }

        if (
            !caps.colorFormats.contains(
                MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface,
            )
        ) {
            return false
        }

        if (
            requested.requestTenBit &&
            mime == MediaFormat.MIMETYPE_VIDEO_HEVC &&
            !supportsHevcMain10(info)
        ) {
            return false
        }

        return runCatching {
            video.getSupportedFrameRatesFor(
                requested.width,
                requested.height,
            ).contains(
                requested.fps.toDouble(),
            )
        }.getOrDefault(true)
    }

    private fun cleanupCodec() {
        runCatching {
            inputSurface?.release()
        }
        inputSurface = null

        runCatching {
            encoder?.stop()
        }
        runCatching {
            encoder?.release()
        }
        encoder = null

        runCatching {
            if (muxerStarted) {
                muxer?.stop()
            }
        }
        muxerStarted = false

        runCatching {
            muxer?.release()
        }
        muxer = null
        trackIndex = -1
    }

    private fun result(): Result =
        Result(
            outputFile = outputFile,
            state = stateRef.get(),
            bytesWritten = bytesWritten.get(),
            encodedSamples = encodedSamples.get(),
            durationUs = durationUs.get(),
            negotiatedFormat = negotiated,
            error = failureMessage,
        )

    private fun AtomicReference<State>.setIfCurrent(
        expected: State,
        next: State,
    ) {
        compareAndSet(expected, next)
    }
}

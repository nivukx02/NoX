package com.nivukx.nox.core.capture

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.hardware.camera2.params.MeteringRectangle
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraConstrainedHighSpeedCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.Image
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Range
import android.util.Size
import android.view.Surface
import java.util.concurrent.Executor
import android.hardware.Sensor
import android.hardware.SensorManager
import com.nivukx.nox.core.model.CameraCaptureProfile
import com.nivukx.nox.core.model.CameraExposureMode
import com.nivukx.nox.core.model.CameraFeatureMatrix
import com.nivukx.nox.core.model.CameraFocusMode
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.FrameStamp
import com.nivukx.nox.core.model.DepthFrame
import com.nivukx.nox.core.model.DepthSource
import com.nivukx.nox.core.model.FrameTelemetry
import com.nivukx.nox.core.model.CaptureTimingEvidence
import com.nivukx.nox.core.sync.ExposureMidpointTimingAuthorityEngine
import com.nivukx.nox.core.model.FocusTarget
import com.nivukx.nox.core.nativeengine.NativeCapturePacket
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.receiveAsFlow
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

private const val MAX_DEPTH16_PIXELS = 4_194_304L
private const val DEPTH16_SAMPLE_BYTES = 2
private const val DEPTH16_ROW_ALIGNMENT_BYTES = 32

/**
 * Motion Capture NoX Camera2 acquisition controller.
 *
 * Responsibilities:
 *  - provide a low-latency YUV analysis stream for pose, face and eye inference
 *  - preserve sensor timestamps, exposure, focus and frame timing telemetry
 *  - optionally acquire RAW_SENSOR and DEPTH16 sidecars for offline reconstruction
 *  - negotiate optional sensor outputs against the device's real Camera2 capabilities
 *  - keep camera/HAL work ordered through a bounded shared callback handler pool
 *
 * This controller intentionally has no video encoder or cinematic color pipeline.
 */
class Camera2CaptureController(
    context: Context,
    private val nativeObservationSink: ((NativeCapturePacket) -> Unit)? = null,
) : AutoCloseable {
    private val appContext = context.applicationContext
    private val manager = appContext.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val sensors = appContext.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val capabilityEngine = CameraCapabilityEngine(appContext)
    private val timestampIntegrity = CaptureTimestampIntegrityEngine()
    private val exposureMidpointTimingAuthority = ExposureMidpointTimingAuthorityEngine()
    private val sessionNegotiation = CameraSessionNegotiationEngine(manager)

    private val cameraHandlerLease =
        CameraCaptureHandlerPool.shared.acquire()
    private val thread = cameraHandlerLease.worker
    private val handler = Handler(thread.looper)
    private val cameraSessionExecutor = Executor { command -> handler.post(command) }

    private val frameSequence = AtomicLong(0L)
    private val negotiatedCaptureFps = AtomicInteger(30)
    private val frames =
        Channel<Pair<FrameStamp, Image>>(
            capacity = 4,
            onUndeliveredElement = { (_, image) ->
                runCatching { image.close() }
            },
        )
    private val depthFrames = Channel<DepthFrame>(
        capacity = 2,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    private val closed = AtomicBoolean(false)
    private val openGeneration = AtomicLong(0L)
    private val lastTelemetry = AtomicReference<FrameTelemetry?>(null)
    private val timingQuality = AtomicReference<CaptureTimestampIntegrityEngine.Quality?>(null)
    private val timingAssessment = AtomicReference<CaptureTimestampIntegrityEngine.Assessment?>(null)
    private val frontierTimingEvidence = AtomicReference<CaptureTimingEvidence?>(null)
    private val analysisFramesSubmitted = AtomicLong(0L)
    private val analysisFramesDropped = AtomicLong(0L)
    private val sessionPlanState = AtomicReference<CameraSessionNegotiationEngine.SessionPlan?>(null)
    private val highSpeedSessionPlanState =
        AtomicReference<CameraSessionNegotiationEngine.HighSpeedPlan?>(null)
    private val highSpeedSessionActive =
        AtomicBoolean(false)
    private val readoutTimestampByFrameNumber = HashMap<Long, Long>()
    private val telemetryByTimestamp = LinkedHashMap<Long, FrameTelemetry>(128, 0.75f, true)


    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null

    private var reader: ImageReader? = null
    private var rawReader: ImageReader? = null
    private var depthReader: ImageReader? = null
    @Volatile private var activeDepthReader = false

    private var rawWriter: RawSensorSidecarWriter? = null
    private var depthWriter: DepthSidecarWriter? = null

    private var previewSurface: Surface? = null
    /** Encoder input surface supplied by HighEndVideoEvidenceRecorder before camera open. */
    private var recordingSurface: Surface? = null
    private var activeCharacteristics: CameraCharacteristics? = null
    private var activeCameraId: String? = null
    private var currentFocusTarget: FocusTarget? = null
    private var lastFocusTriggerNanos = Long.MIN_VALUE
    private var lastFocusCenterX = Double.NaN
    private var lastFocusCenterY = Double.NaN

    private var rawOutput: File? = null
    private var depthOutput: File? = null
    private var activeProfile: CameraCaptureProfile = CameraCaptureProfile.adaptive(
        1920,
        1080,
        30,
    )

    val frameFlow = frames.receiveAsFlow()
    val depthFlow = depthFrames.receiveAsFlow()
    val isDepthReaderActive: Boolean get() = activeDepthReader
    val activeRawOutput: File? get() = rawOutput
    val activeDepthOutput: File? get() = depthOutput
    val latestTelemetry: FrameTelemetry? get() = lastTelemetry.get()
    val latestTimingQuality: CaptureTimestampIntegrityEngine.Quality?
        get() = timingQuality.get()
    val latestTimingAssessment: CaptureTimestampIntegrityEngine.Assessment?
        get() = timingAssessment.get()
    val latestFrontierTimingEvidence: CaptureTimingEvidence?
        get() = frontierTimingEvidence.get()
    val latestSessionPlan: CameraSessionNegotiationEngine.SessionPlan?
        get() = sessionPlanState.get()
    val latestHighSpeedSessionPlan:
        CameraSessionNegotiationEngine.HighSpeedPlan?
        get() = highSpeedSessionPlanState.get()
    val isHighSpeedSessionActive: Boolean
        get() = highSpeedSessionActive.get()
    val hardwareReadoutTimestampSupported: Boolean
        get() = activeCameraId?.let { sessionNegotiation.supportsHardwareReadoutTimestamp(it) } == true
    val captureProfile: CameraCaptureProfile get() = activeProfile
    val actualNegotiatedCaptureFps: Int
        get() = negotiatedCaptureFps.get()
    fun resetAnalysisBackpressureTelemetry() {
        analysisFramesSubmitted.set(0L)
        analysisFramesDropped.set(0L)
    }
    val analysisFramesSubmitted: Long get() = analysisFramesSubmitted.get()
    val analysisFramesDropped: Long get() = analysisFramesDropped.get()
    val analysisDropRatio: Double
        get() {
            val submitted = analysisFramesSubmitted.get()
            val dropped = analysisFramesDropped.get()
            val total = (submitted + dropped).coerceAtLeast(1L)
            return dropped.toDouble() / total.toDouble()
        }

    fun setPreviewSurface(surface: Surface?) {
        handler.post {
            previewSurface = surface
        }
    }

    /**
     * Attaches the optional evidence encoder surface for the next CameraCaptureSession.
     *
     * Camera2 sessions cannot safely add a new output Surface after configuration on all API 30+
     * devices, so production callers must set this before open(). The surface is owned by the
     * encoder and is never released by this controller.
     */
    fun setRecordingSurface(surface: Surface?) {
        handler.post {
            recordingSurface = surface
        }
    }

    val activeRecordingSurface: Surface?
        get() = recordingSurface

    /**
     * Applies an AI-selected focus target on the camera handler thread.
     *
     * The method updates the continuous AF metering region and rate-limits explicit AF triggers.
     * Capture remains continuous when the target is unchanged, preventing autofocus hunting and
     * avoiding a Camera2 control transaction on every inference frame.
     */
    fun setFocusTarget(target: FocusTarget?) {
        handler.post {
            currentFocusTarget = target
            val activeSession = session ?: return@post
            val camera = device ?: return@post
            val characteristics = activeCharacteristics ?: return@post

            if (target == null) {
                rebuildRepeatingRequest(
                    camera = camera,
                    characteristics = characteristics,
                    session = activeSession,
                )
                return@post
            }

            val dx = if (lastFocusCenterX.isNaN()) {
                1.0
            } else {
                kotlin.math.abs(
                    target.normalizedX - lastFocusCenterX,
                )
            }
            val dy = if (lastFocusCenterY.isNaN()) {
                1.0
            } else {
                kotlin.math.abs(
                    target.normalizedY - lastFocusCenterY,
                )
            }

            val now = System.nanoTime()
            val targetChangedEnough =
                dx + dy >= 0.018
            val triggerAllowed =
                now -
                    lastFocusTriggerNanos >=
                    120_000_000L

            rebuildRepeatingRequest(
                camera = camera,
                characteristics = characteristics,
                session = activeSession,
            )

            if (
                targetChangedEnough &&
                triggerAllowed &&
                target.confidence >= 0.55
            ) {
                lastFocusCenterX = target.normalizedX
                lastFocusCenterY = target.normalizedY
                lastFocusTriggerNanos = now

                val trigger =
                    buildCaptureRequest(
                        camera = camera,
                        characteristics = characteristics,
                        targets = focusTriggerTargets(),
                        profile = activeProfile,
                        focusTrigger =
                            CaptureRequest.CONTROL_AF_TRIGGER_START,
                    )

                runCatching {
                    activeSession.capture(
                        trigger,
                        object :
                            CameraCaptureSession.CaptureCallback() {
                            override fun onCaptureCompleted(
                                session: CameraCaptureSession,
                                request: CaptureRequest,
                                result: android.hardware.camera2.TotalCaptureResult,
                            ) {
                                rebuildRepeatingRequest(
                                    camera = camera,
                                    characteristics = characteristics,
                                    session = session,
                                )
                            }
                        },
                        handler,
                    )
                }
            }
        }
    }

    fun capabilities(): com.nivukx.nox.core.model.DeviceCapabilities {
        val cameraIds = cameraIds()
        val matrices = cameraIds.mapNotNull { id ->
            runCatching { capabilityEngine.inspect(id) }.getOrNull()
        }

        val sizes = cameraIds.flatMap { id ->
            manager.getCameraCharacteristics(id)
                .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?.getOutputSizes(ImageFormat.YUV_420_888)
                ?.toList()
                .orEmpty()
        }

        val maxSize = sizes.maxByOrNull {
            it.width.toLong() * it.height.toLong()
        } ?: Size(1920, 1080)

        val maxFps =
            matrices.maxOfOrNull { it.maxFps } ?: 30
        val raw =
            matrices.any { it.supportsRaw }
        val concurrent =
            matrices.any { it.supportsConcurrentCamera }
        val cameraIdSet =
            cameraIds.toSet()

        val maximumVerifiedConcurrentCameras =
            sessionNegotiation
                .declaredConcurrentCameraSets()
                .maxOfOrNull { declared ->
                    declared.intersect(cameraIdSet).size
                }
                ?.coerceAtLeast(1)
                ?: 1

        val positiveMinimumExposureTimes =
            matrices.mapNotNull {
                it.minimumExposureTimeNanos
                    .takeIf { value -> value > 0L }
            }
        val positiveMaximumExposureTimes =
            matrices.mapNotNull {
                it.maximumExposureTimeNanos
                    .takeIf { value -> value > 0L }
            }

        return com.nivukx.nox.core.model.DeviceCapabilities(
            model = Build.MODEL,
            manufacturer = Build.MANUFACTURER,
            androidApi = Build.VERSION.SDK_INT,
            cameraIds = cameraIds,
            supportsRaw = raw,
            supportsConcurrentCamera = concurrent,
            supportsGyroscope = sensors.getDefaultSensor(
                Sensor.TYPE_GYROSCOPE,
            ) != null,
            supportsAccelerometer = sensors.getDefaultSensor(
                Sensor.TYPE_ACCELEROMETER,
            ) != null,
            maxAnalysisWidth = maxSize.width,
            maxAnalysisHeight = maxSize.height,
            maxFps = maxFps,
            supportsHardwareReadoutTimestamp =
                matrices.any {
                    it.supportsHardwareReadoutTimestamp
                },
            maximumVerifiedConcurrentCameras =
                maximumVerifiedConcurrentCameras,
            minimumExposureTimeNanos =
                positiveMinimumExposureTimes.minOrNull() ?: 0L,
            maximumExposureTimeNanos =
                positiveMaximumExposureTimes.maxOrNull() ?: 0L,
        )
    }

    fun cameraFeatureMatrix(cameraId: String): CameraFeatureMatrix =
        capabilityEngine.inspect(cameraId)

    fun recommendedProfile(cameraId: String): CameraCaptureProfile =
        capabilityEngine.inspect(cameraId).recommendedProfile

    fun sensorOrientation(cameraId: String): Int =
        manager.getCameraCharacteristics(cameraId)
            .get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0

    fun lensFacing(cameraId: String): Int =
        manager.getCameraCharacteristics(cameraId)
            .get(CameraCharacteristics.LENS_FACING)
            ?: CameraCharacteristics.LENS_FACING_BACK

    fun intrinsics(cameraId: String): FloatArray? =
        manager.getCameraCharacteristics(cameraId).let { c ->
            c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)?.copyOf()
        }

    fun intrinsicsModel(
        cameraId: String,
        imageWidth: Int,
        imageHeight: Int,
    ): CameraIntrinsics? {
        val characteristics = manager.getCameraCharacteristics(cameraId)
        val calibration = characteristics.get(
            CameraCharacteristics.LENS_INTRINSIC_CALIBRATION,
        ) ?: return null

        if (calibration.size < 4) return null

        val distortion = characteristics.get(
            CameraCharacteristics.LENS_DISTORTION,
        ) ?: FloatArray(0)

        return CameraIntrinsics(
            fx = calibration[0].toDouble(),
            fy = calibration[1].toDouble(),
            cx = calibration[2].toDouble(),
            cy = calibration[3].toDouble(),
            k1 = distortion.getOrNull(0)?.toDouble() ?: 0.0,
            k2 = distortion.getOrNull(1)?.toDouble() ?: 0.0,
            p1 = distortion.getOrNull(2)?.toDouble() ?: 0.0,
            p2 = distortion.getOrNull(3)?.toDouble() ?: 0.0,
            k3 = distortion.getOrNull(4)?.toDouble() ?: 0.0,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
        )
    }

    @SuppressLint("MissingPermission")
    fun open(
        cameraId: String,
        width: Int = 1920,
        height: Int = 1080,
        fps: Int = 30,
        sidecarDirectory: File? = null,
        allowPersistentSensorSidecars: Boolean = false,
        onReady: (Throwable?) -> Unit = {},
    ) {
        val profile = CameraCaptureProfile.adaptive(
            requestedWidth = width,
            requestedHeight = height,
            requestedFps = fps,
        )
        open(
            cameraId = cameraId,
            profile = profile,
            sidecarDirectory = sidecarDirectory,
            allowPersistentSensorSidecars = allowPersistentSensorSidecars,
            onReady = onReady,
        )
    }

    @SuppressLint("MissingPermission")
    fun open(
        cameraId: String,
        profile: CameraCaptureProfile,
        sidecarDirectory: File? = null,
        allowPersistentSensorSidecars: Boolean = false,
        enableLiveDepthStream: Boolean = false,
        onReady: (Throwable?) -> Unit = {},
    ) {
        check(!closed.get()) { "Camera controller is closed" }
        val requestGeneration = openGeneration.incrementAndGet()

        handler.post {
            if (!isCurrentOpen(requestGeneration)) return@post
            try {
                closeCameraOnlyInternal()
                if (!isCurrentOpen(requestGeneration)) return@post
                timestampIntegrity.configureFrameRate(profile.fps)
                timingQuality.set(null)
                timingAssessment.set(null)
        frontierTimingEvidence.set(null)
                val characteristics = manager.getCameraCharacteristics(cameraId)
                activeCharacteristics = characteristics
                activeCameraId = cameraId
                currentFocusTarget = null
                lastFocusTriggerNanos = Long.MIN_VALUE
                lastFocusCenterX = Double.NaN
                lastFocusCenterY = Double.NaN
                activeProfile = profile
                negotiatedCaptureFps.set(profile.fps.coerceIn(15, 240))

                val streamMap = characteristics.get(
                    CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP,
                ) ?: throw IllegalStateException(
                    "No stream configuration map for $cameraId",
                )

                val analysisSizes = streamMap
                    .getOutputSizes(ImageFormat.YUV_420_888)
                    ?.toList()
                    .orEmpty()

                val analysisSize = selectSize(
                    analysisSizes,
                    profile.analysisWidth,
                    profile.analysisHeight,
                )

                reader = createAnalysisReader(
                    analysisSize = analysisSize,
                    requestGeneration = requestGeneration,
                )

                val capability = runCatching {
                    capabilityEngine.inspect(cameraId)
                }.getOrNull()

                prepareOptionalReaders(
                    cameraId = cameraId,
                    requestGeneration = requestGeneration,
                    capability = capability,
                    profile = profile,
                    sidecarBase =
                        if (allowPersistentSensorSidecars) {
                            sidecarDirectory
                        } else {
                            null
                        },
                    streamMap = streamMap,
                    enableLiveDepthStream = enableLiveDepthStream,
                )

                val negotiatedOutputs = buildOutputSpecs(
                    targets = captureTargets(),
                )
                val negotiatedPlan = sessionNegotiation.verifySingleCameraSession(
                    cameraId = cameraId,
                    outputs = negotiatedOutputs,
                )
                sessionPlanState.set(negotiatedPlan)

                if (
                    !negotiatedPlan.supported &&
                    negotiatedPlan.verification ==
                        CameraSessionNegotiationEngine.Verification.REJECTED
                ) {
                    closeOptionalReaders()
                    val mandatoryPlan = sessionNegotiation.verifySingleCameraSession(
                        cameraId = cameraId,
                        outputs = buildOutputSpecs(
                            targets = listOf(reader!!.surface) +
                                listOfNotNull(
                                    previewSurface,
                                    recordingSurface,
                                ),
                        ),
                    )
                    sessionPlanState.set(mandatoryPlan)
                    if (!mandatoryPlan.supported) {
                        throw IllegalStateException(
                            sessionNegotiation.summarize(mandatoryPlan),
                        )
                    }
                }

                manager.openCamera(
                    cameraId,
                    object : CameraDevice.StateCallback() {
                        override fun onOpened(camera: CameraDevice) {
                            if (!isCurrentOpen(requestGeneration)) {
                                camera.close()
                                return
                            }
                            device = camera

                            val mandatoryTargets = buildList {
                                add(reader!!.surface)
                                previewSurface?.let(::add)
                                recordingSurface?.let(::add)
                            }

                            val optionalTargets = buildList {
                                rawReader?.surface?.let(::add)
                                depthReader?.surface?.let(::add)
                            }

                            val request = buildCaptureRequest(
                                camera = camera,
                                characteristics = characteristics,
                                targets = mandatoryTargets + optionalTargets,
                                profile = profile,
                            )

                            val highSpeedPlan =
                                highSpeedPlanForLiveTopology(
                                    cameraId = cameraId,
                                    analysisSize = analysisSize,
                                    mandatoryTargets = mandatoryTargets,
                                    optionalTargets = optionalTargets,
                                    profile = profile,
                                )
                            highSpeedSessionPlanState.set(
                                highSpeedPlan,
                            )

                            val highSpeedRequest =
                                highSpeedPlan?.selectedRange?.let { range ->
                                    buildCaptureRequest(
                                        camera = camera,
                                        characteristics = characteristics,
                                        targets = mandatoryTargets,
                                        profile = profile,
                                        fpsRangeOverride = range,
                                    )
                                }

                            configureSession(
                                requestGeneration = requestGeneration,
                                camera = camera,
                                request =
                                    highSpeedRequest ?: request,
                                mandatoryTargets = mandatoryTargets,
                                optionalTargets = optionalTargets,
                                characteristics = characteristics,
                                profile = profile,
                                allowOptionalRetry = optionalTargets.isNotEmpty(),
                                highSpeedPlan = highSpeedPlan,
                                onReady = { error ->
                                    if (isCurrentOpen(requestGeneration)) onReady(error)
                                },
                            )
                        }

                        override fun onDisconnected(camera: CameraDevice) {
                            camera.close()
                            if (!isCurrentOpen(requestGeneration)) return
                            // Device loss terminates the whole capture session, not just CameraDevice.
                            // Close analysis/optional ImageReaders and sidecar writers as well.
                            closeCameraResources()
                            onReady(
                                IllegalStateException(
                                    "Camera disconnected",
                                ),
                            )
                        }

                        override fun onError(camera: CameraDevice, error: Int) {
                            camera.close()
                            if (!isCurrentOpen(requestGeneration)) return
                            closeCameraResources()
                            onReady(
                                IllegalStateException(
                                    "Camera error=$error",
                                ),
                            )
                        }
                    },
                    handler,
                )
            } catch (t: Throwable) {
                if (isCurrentOpen(requestGeneration)) {
                    closeCameraOnlyInternal()
                    if (isCurrentOpen(requestGeneration)) onReady(t)
                }
            }
        }
    }

    private fun isCurrentOpen(requestGeneration: Long): Boolean =
        !closed.get() &&
            requestGeneration > 0L &&
            openGeneration.get() == requestGeneration

    private fun createAnalysisReader(
        analysisSize: Size,
        requestGeneration: Long,
    ): ImageReader =
        ImageReader.newInstance(
            analysisSize.width,
            analysisSize.height,
            ImageFormat.YUV_420_888,
            5,
        ).apply {
            setOnImageAvailableListener(
                { source ->
                    val image = source.acquireLatestImage()
                        ?: return@setOnImageAvailableListener
                    if (!isCurrentOpen(requestGeneration)) {
                        image.close()
                        return@setOnImageAvailableListener
                    }

                    try {
                        val telemetry = telemetryFor(image.timestamp)
                        val stamp = FrameStamp(
                            sequence = frameSequence.incrementAndGet(),
                            sensorTimestampNanos = image.timestamp,
                            monotonicTimestampNanos =
                                telemetry?.monotonicTimestampNanos
                                    ?: System.nanoTime(),
                            exposureTimeNanos =
                                telemetry?.exposureTimeNanos ?: 0L,
                            iso = telemetry?.sensorSensitivityIso ?: 0,
                        )

                        if (!isCurrentOpen(requestGeneration)) {
                            image.close()
                            return@setOnImageAvailableListener
                        }
                        nativeObservationSink?.let { sink ->
                            runCatching {
                                sink(
                                    NativeCapturePacket.fromCapture(
                                        cameraId = activeCameraId,
                                        stamp = stamp,
                                        telemetry = telemetry,
                                        nominalFrameIntervalNanos =
                                            telemetry?.frameDurationNanos
                                                ?.takeIf {
                                                    it in
                                                        1_000_000L..1_000_000_000L
                                                }
                                                ?: (
                                                    1_000_000_000L /
                                                        negotiatedCaptureFps
                                                            .get()
                                                            .coerceAtLeast(1)
                                                            .toLong()
                                                    ),
                                    ),
                                )
                            }
                        }

                        // The sink may be slow or re-entrant. Recheck after it returns so
                        // a superseded generation cannot enqueue its frame into the next session.
                        if (!isCurrentOpen(requestGeneration)) {
                            image.close()
                            return@setOnImageAvailableListener
                        }
                        if (frames.trySend(stamp to image).isSuccess) {
                            analysisFramesSubmitted.incrementAndGet()
                        } else {
                            analysisFramesDropped.incrementAndGet()
                            image.close()
                        }
                    } catch (t: Throwable) {
                        image.close()
                    }
                },
                handler,
            )
        }

    private fun prepareOptionalReaders(
        cameraId: String,
        requestGeneration: Long,
        capability: CameraFeatureMatrix?,
        profile: CameraCaptureProfile,
        sidecarBase: File?,
        streamMap: android.hardware.camera2.params.StreamConfigurationMap,
        enableLiveDepthStream: Boolean,
    ) {
        closeOptionalReaders()

        if (capability == null) return

        // RAW_SENSOR is only created when a persistent sidecar destination exists.
        if (profile.enableRawSidecar && capability.supportsRaw && sidecarBase != null) {
            val size = Size(
                capability.rawWidth,
                capability.rawHeight,
            )
            if (size.width > 0 && size.height > 0) {
                val target = sidecarBase.resolveSibling(
                    sidecarBase.nameWithoutExtension + ".raw.noxraw",
                )

                rawOutput = target
                rawWriter = RawSensorSidecarWriter(target)

                rawReader = ImageReader.newInstance(
                    size.width,
                    size.height,
                    ImageFormat.RAW_SENSOR,
                    2,
                ).apply {
                    setOnImageAvailableListener(
                        { source ->
                            val image = source.acquireLatestImage()
                                ?: return@setOnImageAvailableListener
                            if (!isCurrentOpen(requestGeneration)) {
                                image.close()
                                return@setOnImageAvailableListener
                            }
                            runCatching {
                                if (!isCurrentOpen(requestGeneration)) return@runCatching
                                val plane = image.planes.firstOrNull()
                                    ?: return@runCatching
                                if (!isCurrentOpen(requestGeneration)) return@runCatching
                                rawWriter?.append(
                                    width = image.width,
                                    height = image.height,
                                    timestampNanos = image.timestamp,
                                    telemetry = telemetryFor(image.timestamp),
                                    rowStride = plane.rowStride,
                                    pixelStride = plane.pixelStride,
                                    buffer = plane.buffer,
                                )
                            }
                            image.close()
                        },
                        handler,
                    )
                }
            }
        }

        // The live depth stream is a reconstruction input. Persistent sidecar configuration only
        // controls whether those same samples are written to disk, not whether they are captured.
        if (
            (profile.enableDepthSidecar || enableLiveDepthStream) &&
            capability.supportsDepth16 &&
            Build.VERSION.SDK_INT >= 23
        ) {
            val width = capability.depthWidth
            val height = capability.depthHeight
            if (width > 0 && height > 0) {
                // Live DEPTH16 streaming does not depend on file recording. Persist only when a
                // sidecar destination was explicitly provided; the ImageReader remains available
                // to depthFlow even when persistent sensor sidecars are disabled.
                val target =
                    sidecarBase?.resolveSibling(
                        sidecarBase.nameWithoutExtension + ".depth.noxdepth",
                    )
                if (target != null) {
                    depthOutput = target
                    depthWriter = DepthSidecarWriter(target)
                }

                depthReader = ImageReader.newInstance(
                    width,
                    height,
                    ImageFormat.DEPTH16,
                    2,
                ).apply {
                    setOnImageAvailableListener(
                        { source ->
                            val image = source.acquireLatestImage()
                                ?: return@setOnImageAvailableListener
                            if (!isCurrentOpen(requestGeneration)) {
                                image.close()
                                return@setOnImageAvailableListener
                            }
                            runCatching {
                                if (!isCurrentOpen(requestGeneration)) return@runCatching
                                depthWriter?.append(image)
                                if (!isCurrentOpen(requestGeneration)) return@runCatching
                                decodeDepthFrame(
                                    image,
                                )?.let {
                                    depthFrames.trySend(it)
                                }
                            }
                            image.close()
                        },
                        handler,
                    )
                }
                activeDepthReader = depthReader != null
            }
        }
    }

    private fun buildCaptureRequest(
        camera: CameraDevice,
        characteristics: CameraCharacteristics,
        targets: List<Surface>,
        profile: CameraCaptureProfile,
        focusTrigger: Int =
            CaptureRequest.CONTROL_AF_TRIGGER_IDLE,
        fpsRangeOverride: Range<Int>? = null,
    ): CaptureRequest {
        val request = camera.createCaptureRequest(
            CameraDevice.TEMPLATE_RECORD,
        )

        targets.distinct().forEach(request::addTarget)

        val requestedFps = profile.fps.coerceIn(15, 240)
        val ranges = characteristics.get(
            CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES,
        ) ?: emptyArray()

        val selectedRange =
            fpsRangeOverride
                ?: ranges
                    .filter {
                        requestedFps in
                            it.lower..it.upper
                    }
                    .minByOrNull {
                        abs(
                            it.lower -
                                requestedFps,
                        ) +
                            abs(
                                it.upper -
                                    requestedFps,
                            )
                    }
                ?: ranges.minByOrNull {
                    abs(
                        it.lower -
                            requestedFps,
                    ) +
                        abs(
                            it.upper -
                                requestedFps,
                        )
                }

        selectedRange?.let {
            val negotiatedFps =
                requestedFps.coerceIn(
                    it.lower,
                    it.upper,
                )
            negotiatedCaptureFps.set(
                negotiatedFps.coerceIn(15, 240),
            )
            timestampIntegrity.configureFrameRate(
                negotiatedCaptureFps.get(),
            )
            request.set(
                CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                it,
            )
        } ?: run {
            negotiatedCaptureFps.set(
                requestedFps.coerceIn(15, 240),
            )
            timestampIntegrity.configureFrameRate(
                negotiatedCaptureFps.get(),
            )
        }

        val availableAntiBandingModes =
            characteristics.get(
                CameraCharacteristics
                    .CONTROL_AE_AVAILABLE_ANTIBANDING_MODES,
            ) ?: intArrayOf()

        val antiBandingMode =
            when (profile.antiBandingMode) {
                com.nivukx.nox.core.model.CameraAntibandingMode.AUTO ->
                    CaptureRequest.CONTROL_AE_ANTIBANDING_MODE_AUTO
                com.nivukx.nox.core.model.CameraAntibandingMode.HZ_50 ->
                    CaptureRequest.CONTROL_AE_ANTIBANDING_MODE_50HZ
                com.nivukx.nox.core.model.CameraAntibandingMode.HZ_60 ->
                    CaptureRequest.CONTROL_AE_ANTIBANDING_MODE_60HZ
                com.nivukx.nox.core.model.CameraAntibandingMode.OFF ->
                    CaptureRequest.CONTROL_AE_ANTIBANDING_MODE_OFF
            }

        if (
            antiBandingMode in
                availableAntiBandingModes
        ) {
            request.set(
                CaptureRequest.CONTROL_AE_ANTIBANDING_MODE,
                antiBandingMode,
            )
        } else if (
            CaptureRequest
                .CONTROL_AE_ANTIBANDING_MODE_AUTO in
                availableAntiBandingModes
        ) {
            request.set(
                CaptureRequest.CONTROL_AE_ANTIBANDING_MODE,
                CaptureRequest
                    .CONTROL_AE_ANTIBANDING_MODE_AUTO,
            )
        }

        val afModes = characteristics.get(
            CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES,
        ) ?: intArrayOf()

        when (profile.focusMode) {
            CameraFocusMode.MANUAL -> {
                if (
                    afModes.contains(
                        CaptureRequest.CONTROL_AF_MODE_OFF,
                    )
                ) {
                    request.set(
                        CaptureRequest.CONTROL_AF_MODE,
                        CaptureRequest.CONTROL_AF_MODE_OFF,
                    )

                    profile.manualFocusDistanceDiopters?.let { diopters ->
                        val minimumFocusDistance =
                            characteristics.get(
                                CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE,
                            ) ?: 0.0f
                        if (minimumFocusDistance > 0.0f) {
                            request.set(
                                CaptureRequest.LENS_FOCUS_DISTANCE,
                                diopters
                                    .toFloat()
                                    .coerceIn(
                                        0.0f,
                                        minimumFocusDistance,
                                    ),
                            )
                        }
                    }
                }
            }

            CameraFocusMode.CONTINUOUS -> {
                if (
                    afModes.contains(
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO,
                    )
                ) {
                    request.set(
                        CaptureRequest.CONTROL_AF_MODE,
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO,
                    )
                }
            }

            CameraFocusMode.TRACKED_SUBJECT -> {
                if (
                    afModes.contains(
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO,
                    )
                ) {
                    request.set(
                        CaptureRequest.CONTROL_AF_MODE,
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO,
                    )
                }

                currentFocusTarget?.let { target ->
                    applyFocusRegion(
                        request = request,
                        characteristics = characteristics,
                        target = target,
                    )
                }
            }
        }

        request.set(
            CaptureRequest.CONTROL_AF_TRIGGER,
            focusTrigger,
        )

        val oisModes = characteristics.get(
            CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION,
        ) ?: intArrayOf()

        if (
            profile.oisEnabled &&
            oisModes.contains(
                CameraCharacteristics.LENS_OPTICAL_STABILIZATION_MODE_ON,
            )
        ) {
            request.set(
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                CameraCharacteristics.LENS_OPTICAL_STABILIZATION_MODE_ON,
            )
        } else if (oisModes.contains(
                CameraCharacteristics.LENS_OPTICAL_STABILIZATION_MODE_OFF,
            )
        ) {
            request.set(
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                CameraCharacteristics.LENS_OPTICAL_STABILIZATION_MODE_OFF,
            )
        }

        val manualSensor =
            characteristics.get(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES,
            )?.contains(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR,
            ) == true

        val manualExposureNanos = profile.manualExposureNanos
        val manualIso = profile.manualIso

        if (
            profile.exposureMode == CameraExposureMode.MANUAL &&
            manualSensor &&
            manualExposureNanos != null &&
            manualIso != null
        ) {
            request.set(
                CaptureRequest.CONTROL_AE_MODE,
                CaptureRequest.CONTROL_AE_MODE_OFF,
            )
            request.set(
                CaptureRequest.SENSOR_EXPOSURE_TIME,
                manualExposureNanos.coerceIn(
                    1_000L,
                    500_000_000L,
                ),
            )
            request.set(
                CaptureRequest.SENSOR_SENSITIVITY,
                manualIso.coerceIn(25, 12_800),
            )
        } else {
            request.set(
                CaptureRequest.CONTROL_AE_MODE,
                CaptureRequest.CONTROL_AE_MODE_ON,
            )

            val compensationRange =
                characteristics.get(
                    CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE,
                )

            if (
                compensationRange != null &&
                compensationRange.lower <= compensationRange.upper
            ) {
                val step =
                    characteristics.get(
                        CameraCharacteristics.CONTROL_AE_COMPENSATION_STEP,
                    )

                if (step != null && !step.isZero()) {
                    val index =
                        (
                            profile.exposureCompensationEv /
                                step.toDouble()
                            )
                            .toInt()
                            .coerceIn(
                                compensationRange.lower,
                                compensationRange.upper,
                            )
                    request.set(
                        CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION,
                        index,
                    )
                }
            }
        }

        request.set(
            CaptureRequest.CONTROL_AE_LOCK,
            profile.autoExposureLock,
        )
        request.set(
            CaptureRequest.CONTROL_AWB_LOCK,
            profile.autoWhiteBalanceLock,
        )

        request.set(
            CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
            if (profile.electronicStabilizationEnabled) {
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON
            } else {
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF
            },
        )

                if (
            profile.exposureMode ==
                CameraExposureMode.SENSITIVITY_PRIORITY
        ) {
            applySensitivityPriorityExposure(
                request = request,
                characteristics = characteristics,
                profile = profile,
            )
        }

applyPreciseColorTemperature(
            request = request,
            characteristics = characteristics,
            profile = profile,
        )

        return request.build()
    }

    private fun applySensitivityPriorityExposure(
        request: CaptureRequest.Builder,
        characteristics: CameraCharacteristics,
        profile: CameraCaptureProfile,
    ) {
        if (Build.VERSION.SDK_INT < 36) return

        val modes =
            characteristics.get(
                CameraCharacteristics.CONTROL_AE_AVAILABLE_PRIORITY_MODES,
            )?.toSet().orEmpty()

        val sensitivityPriority =
            android.hardware.camera2.CameraMetadata
                .CONTROL_AE_PRIORITY_MODE_SENSOR_SENSITIVITY_PRIORITY

        if (sensitivityPriority !in modes) return

        val iso =
            profile.manualIso
                ?.coerceIn(25, 12_800)
                ?: return

        request.set(
            CaptureRequest.CONTROL_AE_MODE,
            CaptureRequest.CONTROL_AE_MODE_ON,
        )
        request.set(
            CaptureRequest.CONTROL_AE_PRIORITY_MODE,
            sensitivityPriority,
        )
        request.set(
            CaptureRequest.SENSOR_SENSITIVITY,
            iso,
        )
        request.set(
            CaptureRequest.CONTROL_AE_LOCK,
            false,
        )
    }

    private fun applyPreciseColorTemperature(
        request: CaptureRequest.Builder,
        characteristics: CameraCharacteristics,
        profile: CameraCaptureProfile,
    ) {
        if (
            Build.VERSION.SDK_INT < 36 ||
            !profile.preciseColorTemperatureEnabled
        ) {
            return
        }

        val range =
            characteristics.get(
                CameraCharacteristics
                    .COLOR_CORRECTION_COLOR_TEMPERATURE_RANGE,
            ) ?: return

        if (range.lower > range.upper) return

        val temperature =
            profile.colorTemperatureKelvin.coerceIn(
                range.lower,
                range.upper,
            )
        val tint =
            profile.colorTint.coerceIn(-100, 100)

        request.set(
            CaptureRequest.CONTROL_AWB_MODE,
            CaptureRequest.CONTROL_AWB_MODE_OFF,
        )
        request.set(
            CaptureRequest.COLOR_CORRECTION_MODE,
            android.hardware.camera2.CameraMetadata
                .COLOR_CORRECTION_MODE_CCT,
        )
        request.set(
            CaptureRequest.COLOR_CORRECTION_COLOR_TEMPERATURE,
            temperature,
        )
        request.set(
            CaptureRequest.COLOR_CORRECTION_COLOR_TINT,
            tint,
        )
    }

    private fun highSpeedPlanForLiveTopology(
        cameraId: String,
        analysisSize: Size,
        mandatoryTargets: List<Surface>,
        optionalTargets: List<Surface>,
        profile: CameraCaptureProfile,
    ): CameraSessionNegotiationEngine.HighSpeedPlan? {
        if (profile.fps < 120) return null
        if (optionalTargets.isNotEmpty()) return null
        if (mandatoryTargets.size != 1) return null
        if (profile.dynamicRangeProfile != null) return null
        if (
            sessionNegotiation.supportsHardwareReadoutTimestamp(
                cameraId,
            )
        ) {
            return null
        }

        return sessionNegotiation
            .inspectHighSpeedCapability(
                cameraId = cameraId,
                outputSize = analysisSize,
                requestedFps = profile.fps,
                outputCount = mandatoryTargets.size,
            )
            .takeIf { it.admissible }
    }

    private fun captureTargets(): List<Surface> =
        buildList {
            reader?.surface?.let(::add)
            previewSurface?.let(::add)
            recordingSurface?.let(::add)
            rawReader?.surface?.let(::add)
            depthReader?.surface?.let(::add)
        }.distinct()

    private fun buildOutputSpecs(
        targets: List<Surface>,
    ): List<CameraSessionNegotiationEngine.OutputSpec> =
        targets.distinct().mapIndexed { index, surface ->
            val name = when {
                surface === reader?.surface -> "analysis"
                surface === previewSurface -> "preview"
                surface === recordingSurface -> "recording"
                surface === rawReader?.surface -> "raw"
                surface === depthReader?.surface -> "depth"
                else -> "surface-" + index
            }
            val hardwareReadoutSupported =
                activeCameraId?.let {
                    sessionNegotiation.supportsHardwareReadoutTimestamp(it)
                } == true

            CameraSessionNegotiationEngine.OutputSpec(
                surface = surface,
                name = name,
                /*
                 * When hardware readout timestamps are supported, every output in this Camera2
                 * session opts into the same readout timing policy. Android 17 rejects mixed
                 * readout enablement across session outputs.
                 */
                enableReadoutTimestamp = hardwareReadoutSupported,
                hardwareReadoutSupported = hardwareReadoutSupported,
                dynamicRangeProfile =
                    activeProfile.dynamicRangeProfile,
            )
        }

    private fun focusTriggerTargets(): List<Surface> =
        buildList {
            reader?.surface?.let(::add)
            previewSurface?.let(::add)
        }.distinct()

    private fun rebuildRepeatingRequest(
        camera: CameraDevice,
        characteristics: CameraCharacteristics,
        session: CameraCaptureSession,
    ) {
        val targets = captureTargets()
        if (targets.isEmpty()) return

        val request =
            buildCaptureRequest(
                camera = camera,
                characteristics = characteristics,
                targets = targets,
                profile = activeProfile,
            )

        runCatching {
            session.setRepeatingRequest(
                request,
                captureCallback(characteristics),
                handler,
            )
        }
    }

    private fun applyFocusRegion(
        request: CaptureRequest.Builder,
        characteristics: CameraCharacteristics,
        target: FocusTarget,
    ) {
        val maxRegions =
            characteristics.get(
                CameraCharacteristics.CONTROL_MAX_REGIONS_AF,
            ) ?: 0

        if (maxRegions <= 0) return

        val activeArray =
            characteristics.get(
                CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE,
            ) ?: return

        val width =
            activeArray.width().coerceAtLeast(1)
        val height =
            activeArray.height().coerceAtLeast(1)

        val cx =
            (
                target.normalizedX.coerceIn(0.0, 1.0) *
                    width
                ).toInt()
        val cy =
            (
                target.normalizedY.coerceIn(0.0, 1.0) *
                    height
                ).toInt()

        val size =
            (
                min(width, height) *
                    (
                        0.055 +
                            0.030 *
                            (1.0 -
                                target.confidence)
                        )
                ).toInt().coerceAtLeast(24)

        val half =
            size / 2

        val left =
            (cx - half)
                .coerceIn(0, max(0, width - 2))
        val top =
            (cy - half)
                .coerceIn(0, max(0, height - 2))
        val right =
            (cx + half)
                .coerceIn(left + 1, width)
        val bottom =
            (cy + half)
                .coerceIn(top + 1, height)

        request.set(
            CaptureRequest.CONTROL_AF_REGIONS,
            arrayOf(
                MeteringRectangle(
                    Rect(
                        activeArray.left + left,
                        activeArray.top + top,
                        activeArray.left + right,
                        activeArray.top + bottom,
                    ),
                    1000,
                ),
            ),
        )
    }

    private fun configureSession(
        requestGeneration: Long,
        camera: CameraDevice,
        request: CaptureRequest,
        mandatoryTargets: List<Surface>,
        optionalTargets: List<Surface>,
        characteristics: CameraCharacteristics,
        profile: CameraCaptureProfile,
        allowOptionalRetry: Boolean,
        highSpeedPlan:
            CameraSessionNegotiationEngine.HighSpeedPlan? = null,
        onReady: (Throwable?) -> Unit,
    ) {
        val allTargets = (mandatoryTargets + optionalTargets).distinct()

        val stateCallback = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(active: CameraCaptureSession) {
                if (!isCurrentOpen(requestGeneration) || device !== camera) {
                    active.close()
                    return
                }
                session = active

                if (
                    highSpeedPlan?.admissible == true &&
                        active is CameraConstrainedHighSpeedCaptureSession
                ) {
                    var highSpeedStarted = false
                    runCatching {
                        val burst =
                            active.createHighSpeedRequestList(request)
                        active.setRepeatingBurst(
                            burst,
                            captureCallback(characteristics),
                            handler,
                        )
                        highSpeedStarted = true
                    }.onFailure { failure ->
                        active.close()
                        session = null
                        configureSession(
                            requestGeneration = requestGeneration,
                            camera = camera,
                            request = buildCaptureRequest(
                                camera = camera,
                                characteristics = characteristics,
                                targets = allTargets,
                                profile = profile,
                            ),
                            mandatoryTargets = mandatoryTargets,
                            optionalTargets = optionalTargets,
                            characteristics = characteristics,
                            profile = profile,
                            allowOptionalRetry = allowOptionalRetry,
                            highSpeedPlan = null,
                            onReady = onReady,
                        )
                    }
                    if (highSpeedStarted) {
                        if (isCurrentOpen(requestGeneration) && device === camera) {
                            highSpeedSessionActive.set(true)
                            onReady(null)
                        } else {
                            active.close()
                            if (session === active) session = null
                            highSpeedSessionActive.set(false)
                        }
                    }
                    return
                }

                highSpeedSessionActive.set(false)
                active.setRepeatingRequest(
                    request,
                    captureCallback(characteristics),
                    handler,
                )

                if (!isCurrentOpen(requestGeneration) || device !== camera) {
                    active.close()
                    if (session === active) session = null
                    return
                }
                onReady(null)
            }

            override fun onConfigureFailed(
                active: CameraCaptureSession,
            ) {
                active.close()
                if (!isCurrentOpen(requestGeneration) || device !== camera) return

                if (highSpeedPlan?.admissible == true) {
                    highSpeedSessionActive.set(false)
                    highSpeedSessionPlanState.set(
                        highSpeedPlan.copy(
                            supported = false,
                            verification =
                                CameraSessionNegotiationEngine.Verification.REJECTED,
                            reason =
                                "constrained high-speed HAL configuration was rejected; regular session fallback selected",
                        ),
                    )
                    configureSession(
                        requestGeneration = requestGeneration,
                        camera = camera,
                        request = buildCaptureRequest(
                            camera = camera,
                            characteristics = characteristics,
                            targets = allTargets,
                            profile = profile,
                        ),
                        mandatoryTargets = mandatoryTargets,
                        optionalTargets = optionalTargets,
                        characteristics = characteristics,
                        profile = profile,
                        allowOptionalRetry = allowOptionalRetry,
                        highSpeedPlan = null,
                        onReady = onReady,
                    )
                    return
                }

                if (allowOptionalRetry) {
                    closeOptionalReaders()

                    val retryRequest = buildCaptureRequest(
                        camera = camera,
                        characteristics = characteristics,
                        targets = mandatoryTargets,
                        profile = profile,
                    )

                    configureSession(
                        requestGeneration = requestGeneration,
                        camera = camera,
                        request = retryRequest,
                        mandatoryTargets = mandatoryTargets,
                        optionalTargets = emptyList(),
                        characteristics = characteristics,
                        profile = profile,
                        allowOptionalRetry = false,
                        onReady = onReady,
                    )
                } else {
                    onReady(
                        IllegalStateException(
                            "Camera session configuration failed",
                        ),
                    )
                }
            }
        }

        if (
            highSpeedPlan?.admissible == true &&
                allTargets.size == 1 &&
                Build.VERSION.SDK_INT >= 23
        ) {
            runCatching {
                camera.createConstrainedHighSpeedCaptureSession(
                    allTargets,
                    stateCallback,
                    handler,
                )
            }.onFailure { failure ->
                highSpeedSessionPlanState.set(
                    highSpeedPlan.copy(
                        supported = false,
                        verification =
                            CameraSessionNegotiationEngine.Verification.REJECTED,
                        reason =
                            "constrained high-speed session creation failed safely: " +
                                failure.javaClass.simpleName,
                    ),
                )
                camera.createCaptureSession(
                    sessionNegotiation.buildRegularSessionConfiguration(
                        outputs = buildOutputSpecs(allTargets),
                        callback = stateCallback,
                        executor = cameraSessionExecutor,
                    ),
                )
            }
            return
        }

        val sessionConfiguration =
            sessionNegotiation.buildRegularSessionConfiguration(
                outputs = buildOutputSpecs(allTargets),
                callback = stateCallback,
                executor = cameraSessionExecutor,
            )
        camera.createCaptureSession(sessionConfiguration)
    }

    private fun captureCallback(
        characteristics: CameraCharacteristics,
    ): CameraCaptureSession.CaptureCallback =
        object : CameraCaptureSession.CaptureCallback() {
            override fun onReadoutStarted(
                session: CameraCaptureSession,
                request: CaptureRequest,
                timestamp: Long,
                frameNumber: Long,
            ) {
                if (Build.VERSION.SDK_INT >= 34) {
                    readoutTimestampByFrameNumber[frameNumber] = timestamp
                    while (readoutTimestampByFrameNumber.size > 128) {
                        readoutTimestampByFrameNumber.remove(
                            readoutTimestampByFrameNumber.keys.minOrNull(),
                        )
                    }
                }
            }

            override fun onCaptureCompleted(
                session: CameraCaptureSession,
                request: CaptureRequest,
                result: android.hardware.camera2.TotalCaptureResult,
            ) {
                val frameNumber = result.frameNumber
                val readoutTimestamp =
                    if (Build.VERSION.SDK_INT >= 34) {
                        readoutTimestampByFrameNumber.remove(frameNumber)
                    } else {
                        null
                    }

                val sensorTimestamp =
                    result.get(
                        android.hardware.camera2.CaptureResult.SENSOR_TIMESTAMP,
                    ) ?: System.nanoTime()

                val exposure =
                    result.get(
                        android.hardware.camera2.CaptureResult.SENSOR_EXPOSURE_TIME,
                    ) ?: 0L

                val iso =
                    result.get(
                        android.hardware.camera2.CaptureResult.SENSOR_SENSITIVITY,
                    ) ?: 0

                val duration =
                    result.get(
                        android.hardware.camera2.CaptureResult.SENSOR_FRAME_DURATION,
                    ) ?: 0L

                val skew =
                    if (Build.VERSION.SDK_INT >= 28) {
                        result.get(
                            android.hardware.camera2.CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW,
                        ) ?: 0L
                    } else {
                        0L
                    }

                val monotonicTimestamp = System.nanoTime()
                val telemetry = FrameTelemetry(
                    sensorTimestampNanos = sensorTimestamp,
                    monotonicTimestampNanos = monotonicTimestamp,
                    exposureTimeNanos = exposure,
                    sensorSensitivityIso = iso,
                    frameDurationNanos = duration,
                    rollingShutterSkewNanos = skew,
                    aeState = result.get(
                        android.hardware.camera2.CaptureResult.CONTROL_AE_STATE,
                    ),
                    afState = result.get(
                        android.hardware.camera2.CaptureResult.CONTROL_AF_STATE,
                    ),
                    awbState = result.get(
                        android.hardware.camera2.CaptureResult.CONTROL_AWB_STATE,
                    ),
                    lensFocusDistanceDiopters = result.get(
                        android.hardware.camera2.CaptureResult.LENS_FOCUS_DISTANCE,
                    ),
                    focusDistanceMeters = result.get(
                        android.hardware.camera2.CaptureResult.LENS_FOCUS_DISTANCE,
                    )?.let {
                        if (it <= 1e-6f) {
                            null
                        } else {
                            1f / it
                        }
                    },
                    readoutTimestampNanos = readoutTimestamp,
                )

                val timing = timestampIntegrity.observe(
                    CaptureTimestampIntegrityEngine.Sample(
                        sensorTimestampNanos = sensorTimestamp,
                        monotonicTimestampNanos = monotonicTimestamp,
                        frameDurationNanos = duration,
                        exposureTimeNanos = exposure,
                        rollingShutterSkewNanos = skew,
                    ),
                )

                timingAssessment.set(timing)
                timingQuality.set(timestampIntegrity.quality())
                exposureMidpointTimingAuthority
                    .resolve(telemetry, activeProfile.analysisHeight)
                    .evidence
                    ?.let(frontierTimingEvidence::set)

                while (telemetryByTimestamp.size > 96) {
                    telemetryByTimestamp.remove(
                        telemetryByTimestamp.keys.minOrNull(),
                    )
                }

                lastTelemetry.set(telemetry)
            }
        }

    private fun telemetryFor(sensorTimestamp: Long): FrameTelemetry? {
        telemetryByTimestamp[sensorTimestamp]?.let { return it }

        val nearest = telemetryByTimestamp.entries.minByOrNull {
            abs(it.key - sensorTimestamp)
        }

        return nearest?.value?.takeIf {
            abs(it.sensorTimestampNanos - sensorTimestamp) <= 25_000_000L
        }
    }

    fun stopCapture() {
        // Stopping is terminal for the current open request. Any Camera2 callback already queued
        // for that request must be treated as stale even if it arrives after resources are closed.
        openGeneration.incrementAndGet()

        if (Thread.currentThread() === thread) {
            runCatching { session?.stopRepeating() }
            runCatching { session?.abortCaptures() }
            closeCameraResources()
            return
        }

        val latch = CountDownLatch(1)
        val posted = handler.post {
            try {
                runCatching { session?.stopRepeating() }
                runCatching { session?.abortCaptures() }
                closeCameraResources()
            } finally {
                latch.countDown()
            }
        }

        if (!posted) {
            // The worker is already terminating; there is no queued callback left to serialize.
            closeCameraResources()
            return
        }
        latch.await(5, TimeUnit.SECONDS)
    }

    fun closeCameraOnly() {
        openGeneration.incrementAndGet()
        closeCameraOnlyInternal()
    }

    private fun closeCameraOnlyInternal() {
        val latch = CountDownLatch(1)

        if (Thread.currentThread() === thread) {
            closeCameraResources()
            latch.countDown()
        } else {
            val posted =
                handler.post {
                    try {
                        closeCameraResources()
                    } finally {
                        latch.countDown()
                    }
                }

            if (!posted) {
                // A stopped HandlerThread cannot drain this cleanup task. Close the HAL resources
                // directly rather than silently retaining ImageReaders and recording writers.
                closeCameraResources()
                return
            }

            latch.await(3, TimeUnit.SECONDS)
        }
    }

    private fun closeCameraResources() {
        runCatching { session?.stopRepeating() }
        runCatching { session?.abortCaptures() }
        runCatching { session?.close() }
        session = null


        runCatching { device?.close() }
        device = null

        runCatching { reader?.close() }
        reader = null

        closeOptionalReaders()

        telemetryByTimestamp.clear()
        readoutTimestampByFrameNumber.clear()
        sessionPlanState.set(null)
        highSpeedSessionPlanState.set(null)
        activeCharacteristics = null
        activeCameraId = null
        currentFocusTarget = null
        lastFocusTriggerNanos = Long.MIN_VALUE
        lastFocusCenterX = Double.NaN
        lastFocusCenterY = Double.NaN
        timestampIntegrity.reset()
        recordingSurface = null
        timingQuality.set(null)
        timingAssessment.set(null)
        lastTelemetry.set(null)
    }

    private fun closeOptionalReaders() {
        runCatching { rawReader?.close() }
        rawReader = null

        runCatching { depthReader?.close() }
        depthReader = null
        activeDepthReader = false

        runCatching { rawWriter?.close() }
        rawWriter = null

        runCatching { depthWriter?.close() }
        depthWriter = null

        rawOutput = null
        depthOutput = null
    }

    /**
     * Decode Android DEPTH16 as two separate signals: metric range and sample confidence.
     *
     * Android packs range in the low 13 bits and confidence in the high 3 bits. Treating the
     * packed word as a distance overestimates range whenever the confidence code is non-zero.
     *
     * Depth buffers are external hardware input. Validate dimensions, stride arithmetic and the
     * last byte needed before allocating per-pixel storage. A truncated image is rejected rather
     * than silently converted into a partially valid reconstruction frame.
     *
     * The maximum pixel count is an allocation guard, not a capture-resolution setting. The
     * camera stream negotiation remains responsible for choosing the supported output size.
     * Invalid zero-range samples remain zero; confidence is retained independently for weighting.
     */
    private fun decodeDepthFrame(
        image: Image,
    ): DepthFrame? {
        val plane = image.planes.firstOrNull()
            ?: return null
        val width = image.width
        val height = image.height
        val timestampNanos = image.timestamp
        if (
            timestampNanos <= 0L ||
            width <= 0 ||
            height <= 0 ||
            (width and 1) != 0 ||
            (height and 1) != 0
        ) {
            return null
        }

        val pixelCountLong = width.toLong() * height.toLong()
        if (
            pixelCountLong > MAX_DEPTH16_PIXELS ||
            pixelCountLong > Int.MAX_VALUE.toLong()
        ) {
            return null
        }
        val pixelCount = pixelCountLong.toInt()

        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        // Android DEPTH16 is a packed 16-bit sample. The platform format contract requires an
        // even-sized image and horizontal stride aligned to 16 pixels. A noncanonical pixel stride
        // is not a safe substitute for this packed layout, even if the backing buffer is long enough.
        if (pixelStride != DEPTH16_SAMPLE_BYTES) return null
        if (
            rowStride <= 0 ||
            rowStride % DEPTH16_ROW_ALIGNMENT_BYTES != 0 ||
            rowStride.toLong() < width.toLong() * DEPTH16_SAMPLE_BYTES
        ) {
            return null
        }

        // Use Long for every offset calculation so corrupt image metadata cannot overflow Int.
        val lastByteExclusive =
            (height - 1L) * rowStride.toLong() +
                (width - 1L) * pixelStride.toLong() +
                2L
        val buffer = plane.buffer.duplicate().apply { position(0) }
        if (
            lastByteExclusive <= 0L ||
            lastByteExclusive > buffer.limit().toLong()
        ) {
            return null
        }

        val depthMeters = FloatArray(pixelCount)
        val confidence = FloatArray(pixelCount)
        for (y in 0 until height) {
            val rowOffset = y.toLong() * rowStride.toLong()
            val outputRow = y * width
            for (x in 0 until width) {
                val byteOffset =
                    rowOffset + x.toLong() * pixelStride.toLong()
                if (byteOffset < 0L || byteOffset + 2L > buffer.limit().toLong()) {
                    return null
                }
                val index = byteOffset.toInt()
                // Use a shared bit decoder so the byte order and confidence mapping stay testable.
                // Both primitives inline and allocate no per-pixel objects in this hot capture loop.
                val packed =
                    Depth16SampleDecoder.unpackLittleEndian(
                        lowByte = buffer.get(index).toInt(),
                        highByte = buffer.get(index + 1).toInt(),
                    )
                val rangeMillimeters =
                    Depth16SampleDecoder.rangeMillimeters(packed)
                val outputIndex = outputRow + x

                depthMeters[outputIndex] =
                    if (rangeMillimeters == 0) 0f else rangeMillimeters / 1000.0f
                confidence[outputIndex] =
                    Depth16SampleDecoder.confidence(packed)
            }
        }

        return DepthFrame(
            timestampNanos = timestampNanos,
            source = DepthSource.DEPTH16,
            width = width,
            height = height,
            depthMeters = depthMeters,
            confidence = confidence,
        )
    }

    private fun cameraIds(): List<String> =
        manager.cameraIdList.toList()

    private fun selectSize(
        supported: List<Size>,
        requestedWidth: Int,
        requestedHeight: Int,
    ): Size {
        if (supported.isEmpty()) {
            return Size(
                requestedWidth.coerceAtLeast(320),
                requestedHeight.coerceAtLeast(240),
            )
        }

        val requestedPixels =
            requestedWidth.toLong() * requestedHeight.toLong()

        return supported
            .filter {
                it.width <= requestedWidth &&
                    it.height <= requestedHeight
            }
            .maxByOrNull {
                it.width.toLong() * it.height.toLong()
            }
            ?: supported.minByOrNull {
                abs(
                    it.width.toLong() * it.height.toLong() -
                        requestedPixels,
                )
            }
            ?: supported.maxBy {
                it.width.toLong() * it.height.toLong()
            }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        openGeneration.incrementAndGet()

        closeCameraOnlyInternal()
        // Cancellation drains buffered undelivered elements immediately. The frame channel's
        // onUndeliveredElement closes each Image, preventing the final four frames from holding
        // ImageReader buffers after the capture owner and its collectors have stopped.
        runCatching { frames.cancel() }
        runCatching { depthFrames.cancel() }
        cameraHandlerLease.close()
    }
}

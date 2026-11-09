package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DepthFrame
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.nativeengine.DepthSampleConvention
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Timestamp-aware adapter from live camera depth frames to the native TSDF reconstruction engine.
 *
 * This class deliberately requires an explicit, frame-time calibration result. DEPTH16 frames are
 * not integrated using fabricated identity poses or guessed intrinsics. The resolver is expected
 * to return calibration valid at exactly the supplied sensor timestamp, including the corresponding
 * camera-to-world pose and intrinsics for the actual depth output dimensions.
 *
 * Calls are serialized because one sparse TSDF volume is shared by the whole camera federation.
 * The upstream Camera2 depth channel is bounded and keeps recent frames during overload, so a slow
 * native integration does not create an unbounded queue of old sensor images.
 */
class NativeDepthFrameReconstructionSink(
    private val calibrationResolver: CalibrationResolver,
    private val config: Config = Config(),
    private val bridge: NativeSparseTsdfReconstructionBridge? = null,
    private val backendOverride: Backend? = null,
) : AutoCloseable {

    /**
     * Backend seam for deterministic tests and platform-specific reconstruction implementations.
     * The default backend delegates to the existing native sparse TSDF bridge.
     */
    interface Backend : AutoCloseable {
        fun integrateDepthFrame(
            depthFrame: DepthFrame,
            intrinsics: CameraIntrinsics,
            pose: Pose,
            stride: Int,
            confidence: Double,
            depthConvention: DepthSampleConvention,
        )

        fun extract(
            timestampNanos: Long,
            isoLevel: Float,
        ): NativeSparseTsdfReconstructionBridge.ReconstructionMeshWithQuality?

        fun reset()

        override fun close()
    }

    private class NativeBridgeBackend(
        private val bridge: NativeSparseTsdfReconstructionBridge,
    ) : Backend {
        override fun integrateDepthFrame(
            depthFrame: DepthFrame,
            intrinsics: CameraIntrinsics,
            pose: Pose,
            stride: Int,
            confidence: Double,
            depthConvention: DepthSampleConvention,
        ) {
            bridge.integrateDepthFrame(
                depthFrame = depthFrame,
                intrinsics = intrinsics,
                pose = pose,
                stride = stride,
                confidence = confidence,
                depthConvention = depthConvention,
            )
        }

        override fun extract(
            timestampNanos: Long,
            isoLevel: Float,
        ): NativeSparseTsdfReconstructionBridge.ReconstructionMeshWithQuality? =
            bridge.extract(timestampNanos = timestampNanos, isoLevel = isoLevel)

        override fun reset() = bridge.reset()

        override fun close() = bridge.close()
    }

    init {
        require(bridge == null || backendOverride == null) {
            "Provide either a native bridge or a backend override, not both"
        }
    }

    private val backendDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        backendOverride ?: NativeBridgeBackend(
            bridge ?: NativeSparseTsdfReconstructionBridge(),
        )
    }

    fun interface CalibrationResolver {
        fun resolve(
            cameraId: String,
            frameTimestampNanos: Long,
        ): CameraCalibration?
    }

    data class Config(
        val maximumTrackedCameras: Int = 128,
        val sampleStride: Int = 2,
    ) {
        init {
            require(maximumTrackedCameras in 1..1_024)
            require(sampleStride in 1..16)
        }
    }

    data class CameraCalibration(
        val resolvedTimestampNanos: Long,
        val intrinsics: CameraIntrinsics,
        val cameraToWorld: Pose,
        val observationWeight: Double = 1.0,
        val depthConvention: DepthSampleConvention = DepthSampleConvention.EUCLIDEAN_RAY,
        val depthImageRectified: Boolean,
    ) {
        internal fun isCompatibleWith(frame: DepthFrame): Boolean {
            if (resolvedTimestampNanos != frame.timestampNanos) return false
            if (!depthImageRectified) return false
            if (intrinsics.imageWidth != frame.width || intrinsics.imageHeight != frame.height) {
                return false
            }
            if (
                !intrinsics.fx.isFinite() || intrinsics.fx <= 0.0 ||
                !intrinsics.fy.isFinite() || intrinsics.fy <= 0.0 ||
                !intrinsics.cx.isFinite() || !intrinsics.cy.isFinite()
            ) {
                return false
            }
            if (
                !intrinsics.k1.isFinite() || !intrinsics.k2.isFinite() ||
                !intrinsics.k3.isFinite() || !intrinsics.p1.isFinite() ||
                !intrinsics.p2.isFinite()
            ) {
                return false
            }
            if (
                !cameraToWorld.position.x.isFinite() ||
                !cameraToWorld.position.y.isFinite() ||
                !cameraToWorld.position.z.isFinite()
            ) {
                return false
            }
            val rotation = cameraToWorld.rotation
            if (
                !rotation.w.isFinite() || !rotation.x.isFinite() ||
                !rotation.y.isFinite() || !rotation.z.isFinite()
            ) {
                return false
            }
            val quaternionNormSquared =
                rotation.w * rotation.w +
                    rotation.x * rotation.x +
                    rotation.y * rotation.y +
                    rotation.z * rotation.z
            if (!quaternionNormSquared.isFinite() || quaternionNormSquared <= 1e-12) {
                return false
            }
            // A zero-weight sample cannot change TSDF evidence. Reject it instead of
            // reporting a no-op as an integrated frame.
            return observationWeight.isFinite() && observationWeight > 0.0 && observationWeight <= 1.0
        }
    }

    enum class SubmissionResult {
        INTEGRATED,
        NO_CALIBRATION,
        INVALID_CALIBRATION,
        STALE_FRAME,
        CAMERA_CAPACITY_EXCEEDED,
        INTEGRATION_FAILED,
        CLOSED,
    }

    data class Diagnostics(
        val integratedFrames: Long,
        val framesWithoutCalibration: Long,
        val invalidCalibrations: Long,
        val staleFrames: Long,
        val integrationFailures: Long,
        val calibrationResolutionFailures: Long,
        val cameraCapacityRejections: Long,
        val trackedCameras: Int,
        val lastIntegratedTimestampNanos: Long?,
        val lastFailure: String?,
    )

    private val lock = Any()
    private val closed = AtomicBoolean(false)
    private val cameraTimestamps =
        CameraDepthTimestampWatermarks(config.maximumTrackedCameras)

    private var integratedFrames = 0L
    private var framesWithoutCalibration = 0L
    private var invalidCalibrations = 0L
    private var staleFrames = 0L
    private var integrationFailures = 0L
    private var calibrationResolutionFailures = 0L
    private var cameraCapacityRejections = 0L
    private var lastIntegratedTimestampNanos: Long? = null
    private var lastFailure: String? = null

    /**
     * Submit one sensor frame. A frame is marked accepted only after native integration completes.
     * Stale or uncalibrated frames are explicitly rejected and never silently fused at identity pose.
     */
    fun submit(
        cameraId: String,
        frame: DepthFrame,
    ): SubmissionResult {
        require(cameraId.isNotBlank())
        require(cameraId.length <= MAX_CAMERA_ID_LENGTH)
        require(frame.timestampNanos > 0L)
        if (closed.get()) return SubmissionResult.CLOSED

        val calibration =
            try {
                calibrationResolver.resolve(cameraId, frame.timestampNanos)
            } catch (failure: Exception) {
                synchronized(lock) {
                    calibrationResolutionFailures = increment(calibrationResolutionFailures)
                    lastFailure = "calibration-resolver-failed:${failure.javaClass.simpleName}"
                }
                return SubmissionResult.INVALID_CALIBRATION
            }

        if (closed.get()) return SubmissionResult.CLOSED
        if (calibration == null) {
            synchronized(lock) {
                framesWithoutCalibration = increment(framesWithoutCalibration)
            }
            return SubmissionResult.NO_CALIBRATION
        }

        return synchronized(lock) {
            if (closed.get()) return@synchronized SubmissionResult.CLOSED

            if (!calibration.isCompatibleWith(frame)) {
                invalidCalibrations = increment(invalidCalibrations)
                lastFailure = "calibration-does-not-match-depth-frame"
                return@synchronized SubmissionResult.INVALID_CALIBRATION
            }

            when (cameraTimestamps.admission(cameraId, frame.timestampNanos)) {
                CameraDepthTimestampWatermarks.Admission.STALE -> {
                    staleFrames = increment(staleFrames)
                    return@synchronized SubmissionResult.STALE_FRAME
                }
                CameraDepthTimestampWatermarks.Admission.CAPACITY_EXCEEDED -> {
                    cameraCapacityRejections = increment(cameraCapacityRejections)
                    lastFailure = "maximum-tracked-depth-cameras-exceeded"
                    return@synchronized SubmissionResult.CAMERA_CAPACITY_EXCEEDED
                }
                CameraDepthTimestampWatermarks.Admission.ACCEPT -> Unit
            }

            try {
                backendDelegate.value.integrateDepthFrame(
                    depthFrame = frame,
                    intrinsics = calibration.intrinsics,
                    pose = calibration.cameraToWorld,
                    stride = config.sampleStride,
                    confidence = calibration.observationWeight,
                    depthConvention = calibration.depthConvention,
                )
            } catch (failure: Exception) {
                integrationFailures = increment(integrationFailures)
                lastFailure = "native-depth-integration-failed:${failure.javaClass.simpleName}"
                return@synchronized SubmissionResult.INTEGRATION_FAILED
            }

            // Commit a timestamp watermark only after native integration succeeds. Never evict an
            // existing camera watermark to admit a new camera: doing so would make an old delayed
            // frame look new and corrupt the shared TSDF volume.
            cameraTimestamps.markIntegrated(cameraId, frame.timestampNanos)
            integratedFrames = increment(integratedFrames)
            lastIntegratedTimestampNanos =
                maxOf(lastIntegratedTimestampNanos ?: 0L, frame.timestampNanos)
            lastFailure = null
            SubmissionResult.INTEGRATED
        }
    }

    /** Mesh extraction shares the same lock as integration to keep the native session serialized. */
    fun extract(
        timestampNanos: Long,
        isoLevel: Float = 0.0f,
    ): ReconstructionMeshWithQuality? =
        synchronized(lock) {
            if (closed.get()) return@synchronized null
            require(timestampNanos > 0L)
            backendDelegate.value.extract(
                timestampNanos = timestampNanos,
                isoLevel = isoLevel,
            )
        }

    /** Reset the accumulated volume and timestamp watermarks for a new capture session. */
    fun reset() {
        synchronized(lock) {
            check(!closed.get()) { "Native depth reconstruction sink is closed" }
            backendDelegate.value.reset()
            cameraTimestamps.clear()
            lastIntegratedTimestampNanos = null
            lastFailure = null
        }
    }

    fun diagnostics(): Diagnostics =
        synchronized(lock) {
            Diagnostics(
                integratedFrames = integratedFrames,
                framesWithoutCalibration = framesWithoutCalibration,
                invalidCalibrations = invalidCalibrations,
                staleFrames = staleFrames,
                integrationFailures = integrationFailures,
                calibrationResolutionFailures = calibrationResolutionFailures,
                cameraCapacityRejections = cameraCapacityRejections,
                trackedCameras = cameraTimestamps.size,
                lastIntegratedTimestampNanos = lastIntegratedTimestampNanos,
                lastFailure = lastFailure,
            )
        }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        synchronized(lock) {
            cameraTimestamps.clear()
            when {
                backendDelegate.isInitialized() -> backendDelegate.value.close()
                backendOverride != null -> backendOverride.close()
                bridge != null -> bridge.close()
                else -> Unit
            }
        }
    }

    private fun increment(value: Long): Long =
        if (value == Long.MAX_VALUE) value else value + 1L

    private companion object {
        const val MAX_CAMERA_ID_LENGTH = 128
    }
}
/**
 * Bounded per-camera monotonic timestamp admission.
 *
 * The sink serializes access under its own lock. A camera watermark is committed only after native
 * integration succeeds, and an existing camera watermark is never evicted merely to admit a new
 * camera. This preserves out-of-order rejection throughout the entire tracking window.
 */
internal class CameraDepthTimestampWatermarks(
    private val maximumTrackedCameras: Int,
) {
    enum class Admission {
        ACCEPT,
        STALE,
        CAPACITY_EXCEEDED,
    }

    private val lastTimestampByCamera = HashMap<String, Long>(maximumTrackedCameras)

    init {
        require(maximumTrackedCameras > 0)
    }

    val size: Int
        get() = lastTimestampByCamera.size

    fun admission(
        cameraId: String,
        timestampNanos: Long,
    ): Admission {
        require(cameraId.isNotBlank())
        require(timestampNanos > 0L)
        val previousTimestamp = lastTimestampByCamera[cameraId]
        if (previousTimestamp != null && timestampNanos <= previousTimestamp) {
            return Admission.STALE
        }
        if (previousTimestamp == null && lastTimestampByCamera.size >= maximumTrackedCameras) {
            return Admission.CAPACITY_EXCEEDED
        }
        return Admission.ACCEPT
    }

    fun markIntegrated(
        cameraId: String,
        timestampNanos: Long,
    ) {
        check(admission(cameraId, timestampNanos) == Admission.ACCEPT) {
            "Camera timestamp was not admitted for integration"
        }
        lastTimestampByCamera[cameraId] = timestampNanos
    }

    fun clear() {
        lastTimestampByCamera.clear()
    }
}

package com.nivukx.nox.core.model

import kotlin.math.abs
import kotlin.math.sqrt

@JvmInline
value class DeviceId(val value: String)

@JvmInline
value class SessionId(val value: String)

enum class DeviceRole {
    MASTER, CAMERA, FACE, TOP, BOTTOM, FRONT, REAR, LEFT, RIGHT, HAND, AUXILIARY
}

enum class CaptureState {
    IDLE, DISCOVERING, CALIBRATING, ARMED, RECORDING, FINALIZING, ERROR
}

enum class SubjectType { HUMAN, ANIMAL, OBJECT, UNKNOWN }

enum class LandmarkSpace { PIXELS, NORMALIZED, CAMERA, WORLD }

data class DeviceCapabilities(
    val model: String,
    val manufacturer: String,
    val androidApi: Int,
    val cameraIds: List<String>,
    val supportsRaw: Boolean,
    val supportsConcurrentCamera: Boolean,
    val supportsGyroscope: Boolean,
    val supportsAccelerometer: Boolean,
    val maxAnalysisWidth: Int,
    val maxAnalysisHeight: Int,
    val maxFps: Int,
    val supportsHardwareReadoutTimestamp: Boolean = false,
    val maximumVerifiedConcurrentCameras: Int = 1,
    val minimumExposureTimeNanos: Long = 0L,
    val maximumExposureTimeNanos: Long = 0L,
) {
    init {
        require(androidApi >= 1)
        require(cameraIds.isNotEmpty())
        require(maxAnalysisWidth > 0)
        require(maxAnalysisHeight > 0)
        require(maxFps > 0)
        require(maximumVerifiedConcurrentCameras >= 1)
        require(
            maximumExposureTimeNanos >=
                minimumExposureTimeNanos,
        )
        require(minimumExposureTimeNanos >= 0L)
    }
}

data class DeviceDescriptor(
    val id: DeviceId,
    val role: DeviceRole,
    val displayName: String,
    val capabilities: DeviceCapabilities,
    val connected: Boolean = false,
    val clockOffsetNanos: Long = 0L,
    val roundTripNanos: Long = 0L,
    val packetLoss: Double = 0.0,
)

data class FrameStamp(
    val sequence: Long,
    val sensorTimestampNanos: Long,
    val monotonicTimestampNanos: Long,
    val exposureTimeNanos: Long = 0L,
    val iso: Int = 0,
)

data class Vec2(val x: Double, val y: Double) {
    operator fun plus(o: Vec2) = Vec2(x + o.x, y + o.y)
    operator fun minus(o: Vec2) = Vec2(x - o.x, y - o.y)
    operator fun times(s: Double) = Vec2(x * s, y * s)
    operator fun div(s: Double) = Vec2(x / s, y / s)
    fun dot(o: Vec2): Double = x * o.x + y * o.y
    fun norm(): Double = kotlin.math.sqrt(dot(this))
    fun distanceTo(o: Vec2): Double = (this - o).norm()
    fun normalized(): Vec2 = if (norm() < 1e-12) Vec2(0.0, 0.0) else this * (1.0 / norm())
    fun lerp(o: Vec2, alpha: Double): Vec2 = this * (1.0 - alpha) + o * alpha
}

data class CameraIntrinsics(
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
    val k1: Double = 0.0,
    val k2: Double = 0.0,
    val p1: Double = 0.0,
    val p2: Double = 0.0,
    val k3: Double = 0.0,
    val imageWidth: Int = 0,
    val imageHeight: Int = 0,
)

data class Vec3(val x: Double, val y: Double, val z: Double) {
    operator fun plus(o: Vec3) = Vec3(x + o.x, y + o.y, z + o.z)
    operator fun minus(o: Vec3) = Vec3(x - o.x, y - o.y, z - o.z)
    operator fun times(s: Double) = Vec3(x * s, y * s, z * s)
    operator fun div(s: Double) = Vec3(x / s, y / s, z / s)
    fun dot(o: Vec3): Double = x * o.x + y * o.y + z * o.z
    fun cross(o: Vec3): Vec3 = Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x)
    fun norm(): Double = sqrt(dot(this))
    fun distanceTo(o: Vec3): Double = (this - o).norm()
    fun normalized(): Vec3 = if (norm() < 1e-12) Vec3(0.0, 0.0, 0.0) else this * (1.0 / norm())
    fun lerp(o: Vec3, alpha: Double): Vec3 = this * (1.0 - alpha) + o * alpha
}

data class Quat(val w: Double, val x: Double, val y: Double, val z: Double) {
    fun normalized(): Quat {
        val n = sqrt(w*w + x*x + y*y + z*z)
        return if (n < 1e-12) IDENTITY else Quat(w/n, x/n, y/n, z/n)
    }

    operator fun times(o: Quat): Quat = Quat(
        w*o.w - x*o.x - y*o.y - z*o.z,
        w*o.x + x*o.w + y*o.z - z*o.y,
        w*o.y - x*o.z + y*o.w + z*o.x,
        w*o.z + x*o.y - y*o.x + z*o.w,
    )

    fun conjugate(): Quat = Quat(w, -x, -y, -z)

    fun rotate(v: Vec3): Vec3 {
        val qv = Quat(0.0, v.x, v.y, v.z)
        val r = this.normalized() * qv * normalized().conjugate()
        return Vec3(r.x, r.y, r.z)
    }

    fun slerp(target: Quat, alpha: Double): Quat {
        var t = target
        var cosTheta = w*t.w + x*t.x + y*t.y + z*t.z
        if (cosTheta < 0.0) {
            t = Quat(-t.w, -t.x, -t.y, -t.z)
            cosTheta = -cosTheta
        }
        if (cosTheta > 0.9995) {
            return Quat(
                w + alpha*(t.w-w), x + alpha*(t.x-x),
                y + alpha*(t.y-y), z + alpha*(t.z-z)
            ).normalized()
        }
        val theta = kotlin.math.acos(cosTheta.coerceIn(-1.0, 1.0))
        val sinTheta = kotlin.math.sin(theta)
        val a = kotlin.math.sin((1.0-alpha)*theta) / sinTheta
        val b = kotlin.math.sin(alpha*theta) / sinTheta
        return Quat(w*a+t.w*b, x*a+t.x*b, y*a+t.y*b, z*a+t.z*b).normalized()
    }

    companion object { val IDENTITY = Quat(1.0, 0.0, 0.0, 0.0) }
}

data class Pose(
    val position: Vec3 = Vec3(0.0, 0.0, 0.0),
    val rotation: Quat = Quat.IDENTITY,
)

data class CameraExtrinsics(
    val pose: Pose,
    val referenceDevice: DeviceId?,
)

data class CalibrationUncertaintyProfile(
    /**
     * 1-sigma translational uncertainty of the calibrated camera center in meters.
     * A zero value means no uncertainty estimate has been supplied, not perfect calibration.
     */
    val translationSigmaMeters: Double = 0.0,
    /**
     * 1-sigma rotational uncertainty in radians.
     */
    val rotationSigmaRadians: Double = 0.0,
    /**
     * Dimensionless 1-sigma fractional intrinsic uncertainty.
     */
    val intrinsicRelativeSigma: Double = 0.0,
    /**
     * Confidence that the uncertainty profile is measurement-derived rather than a default.
     */
    val evidenceConfidence: Double = 0.0,
) {
    init {
        require(
            translationSigmaMeters.isFinite() &&
                translationSigmaMeters >= 0.0,
        )
        require(
            rotationSigmaRadians.isFinite() &&
                rotationSigmaRadians >= 0.0,
        )
        require(
            intrinsicRelativeSigma.isFinite() &&
                intrinsicRelativeSigma >= 0.0,
        )
        require(evidenceConfidence in 0.0..1.0)
    }

    val hasEvidence: Boolean
        get() = evidenceConfidence > 0.0
}

data class CalibrationSnapshot(
    val intrinsics: Map<DeviceId, CameraIntrinsics>,
    val extrinsics: Map<DeviceId, CameraExtrinsics>,
    val reprojectionErrorPx: Double,
    val calibratedAtNanos: Long,
    val uncertaintyProfiles: Map<DeviceId, CalibrationUncertaintyProfile> = emptyMap(),
    /**
     * Per-camera lens focus state at calibration time, in diopters.
     * Missing entries mean the calibration artifact did not contain a trustworthy focus reading.
     */
    val calibrationFocusDistanceDiopters: Map<DeviceId, Double> = emptyMap(),
    /**
     * Provenance for the numerical calibration stage that produced this snapshot.
     * Defaults preserve compatibility with legacy snapshots while allowing reference workflows
     * to distinguish manufacturer calibration, self-calibration, and native intrinsic refinement.
     */
    val calibrationMethod: String = "UNKNOWN",
    val nativeIntrinsicRefinementApplied: Boolean = false,
    val nativeIntrinsicRefinementConfidence: Double = 0.0,
    val nativeIntrinsicRefinementImprovementPx: Double = 0.0,
    val timingAuthorityConfidence: Double = 0.0,
    val timingAuthorityResidualP95Milliseconds: Double = 50.0,
) {
    init {
        require(reprojectionErrorPx.isFinite() && reprojectionErrorPx >= 0.0)
        require(calibratedAtNanos >= 0L)
        require(intrinsics.isNotEmpty())
        require(intrinsics.keys == extrinsics.keys)
        require(intrinsics.size <= 128)
        require(uncertaintyProfiles.keys.all { it in intrinsics.keys })
        require(calibrationFocusDistanceDiopters.keys.all { it in intrinsics.keys })
        require(
            calibrationFocusDistanceDiopters.values.all {
                it.isFinite() && it >= 0.0
            },
        )
        require(calibrationMethod.isNotBlank() && calibrationMethod.length <= 64)
        require(nativeIntrinsicRefinementConfidence in 0.0..1.0)
        require(nativeIntrinsicRefinementImprovementPx.isFinite())
        require(timingAuthorityConfidence in 0.0..1.0)
        require(
            timingAuthorityResidualP95Milliseconds.isFinite() &&
                timingAuthorityResidualP95Milliseconds >= 0.0,
        )
    }
}

data class CaptureManifest(
    val sessionId: SessionId,
    val createdAtNanos: Long,
    val targetFps: Int,
    val width: Int,
    val height: Int,
    val devices: List<DeviceDescriptor>,
    val calibration: CalibrationSnapshot?,
)

data class LandmarkObservation(
    val landmarkId: Int,
    val x: Double,
    val y: Double,
    val confidence: Double,
    val timestampNanos: Long,
    val space: LandmarkSpace = LandmarkSpace.PIXELS,
    val trackingId: Int? = null,
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    val rollingShutterSkewNanos: Long = 0L,
    val frameDurationNanos: Long = 0L,
    val exposureTimeNanos: Long = 0L,
    val captureTimingConfidence: Double = 1.0,
    val sensorRowNormalized: Double? = null,
) {
    init {
        require(confidence in 0.0..1.0)
        require(frameWidth >= 0)
        require(frameHeight >= 0)
        require(rollingShutterSkewNanos >= 0L)
        require(frameDurationNanos >= 0L)
        require(exposureTimeNanos >= 0L)
        require(captureTimingConfidence in 0.0..1.0)
        require(
            sensorRowNormalized == null ||
                (
                    sensorRowNormalized.isFinite() &&
                    sensorRowNormalized in 0.0..1.0
                ),
        )
    }

    /**
     * Conservative quality estimate for the capture timestamp attached to this observation.
     * It is a temporal-evidence score only, not a physical-accuracy guarantee.
     */
    val captureTimingQuality: Double
        get() {
            val frameDuration =
                frameDurationNanos
                    .toDouble()
                    .coerceAtLeast(1.0)
            val rollingFraction =
                (
                    rollingShutterSkewNanos.toDouble() /
                        frameDuration
                    ).coerceIn(0.0, 1.0)
            val exposureFraction =
                (
                    exposureTimeNanos.toDouble() /
                        frameDuration
                    ).coerceIn(0.0, 1.0)

            return (
                captureTimingConfidence * 0.60 +
                    (1.0 - rollingFraction) * 0.24 +
                    (1.0 - exposureFraction) * 0.16
                ).coerceIn(0.0, 1.0)
        }
}

data class MultiViewLandmark(
    val landmarkId: Int,
    val timestampNanos: Long,
    val observations: List<CameraObservation>,
)

data class CameraObservation(
    val deviceId: DeviceId,
    val point: Vec2,
    val confidence: Double,
    val intrinsics: CameraIntrinsics,
    val pose: Pose,
    val rollingShutterFraction: Double = 0.0,
    val landmarkId: Int = -1,
    val temporalAlignmentConfidence: Double = 1.0,
    val temporalUncertaintyNanos: Long = 0L,
    val rowTimeOffsetNanos: Long = 0L,
    /** Frame duration used to estimate exposure-integration blur risk. */
    val frameDurationNanos: Long = 0L,
    /** Exposure integration duration used to estimate optical blur risk. */
    val exposureTimeNanos: Long = 0L,
    /** Capture timing confidence associated with this observation. */
    val captureTimingConfidence: Double = 1.0,
    /**
     * Per-camera calibration uncertainty envelope propagated into metric reconstruction.
     * The default represents unknown calibration uncertainty, not perfect calibration.
     */
    val calibrationUncertainty:
        CalibrationUncertaintyProfile =
        CalibrationUncertaintyProfile(),
) {
    val cameraPose: Pose
        get() = pose

    init {
        require(confidence in 0.0..1.0)
        require(rollingShutterFraction.isFinite())
        require(temporalAlignmentConfidence in 0.0..1.0)
        require(temporalUncertaintyNanos >= 0L)
        require(frameDurationNanos >= 0L)
        require(exposureTimeNanos >= 0L)
        require(captureTimingConfidence in 0.0..1.0)
    }

    /**
     * Conservative optical integration risk inferred from exposure duty cycle, row timing and
     * capture confidence. This is evidence for weighting only, not a physical blur measurement.
     * When a native angular-velocity estimator is available upstream, its measured risk should
     * remain authoritative and may override this conservative estimate.
     */
    val estimatedMotionBlurRisk: Double
        get() = estimateOpticalMotionBlurRisk(
            captureTimingConfidence = captureTimingConfidence,
            rollingShutterSkewNanos = rollingShutterFraction
                .coerceAtLeast(0.0)
                .let {
                    (
                        it *
                            frameDurationNanos.toDouble()
                    ).toLong()
                }.coerceAtLeast(0L),
            frameDurationNanos = frameDurationNanos,
            exposureTimeNanos = exposureTimeNanos,
        )
}

data class ReconstructionJoint(
    val landmarkId: Int,
    val position: Vec3,
    val velocity: Vec3,
    val confidence: Double,
    val reprojectionErrorPx: Double = 0.0,
    val observationCount: Int = 1,
    val positionUncertaintyMeters: Double = 0.0,
    val acceleration: Vec3 = Vec3(0.0, 0.0, 0.0),
) {
    val uncertaintyMeters: Double
        get() = positionUncertaintyMeters
}

data class SkeletonFrame(
    val timestampNanos: Long,
    val subjectType: SubjectType = SubjectType.HUMAN,
    val trackId: Long = 0L,
    val joints: List<ReconstructionJoint>,
    val globalConfidence: Double = 1.0,
)

data class SubjectTrack(
    val trackId: Long,
    val subjectType: SubjectType,
    val confidence: Float,
    val keypoints: List<Vec3>,
    val timestampNanos: Long,
)

data class FaceMeshPoint(
    val index: Int,
    val point: Vec2,
    val confidence: Double,
)

data class EyeObservation(
    val eye: EyeSide,
    val center: Vec2,
    val irisCenter: Vec2?,
    val normalizedOffset: Vec2,
    val openness: Double,
    val blink: Boolean,
    val confidence: Double,
    val gazeDirection: Vec3 = Vec3(0.0, 0.0, 1.0),
    /** Apparent iris radius relative to inter-eye distance, when locally resolved. */
    val irisRadiusNormalized: Double = 0.0,
    /** Local iris-structure confidence from darkness, annulus contrast and symmetry. */
    val irisStructureConfidence: Double = 0.0,
    /** Estimated eyelid occlusion fraction derived from eye openness. */
    val eyelidOcclusion: Double = 0.0,
)

enum class EyeSide { LEFT, RIGHT }

data class EyeGazeCalibrationSample(
    val offset: Vec2,
    val targetYawRadians: Double,
    val targetPitchRadians: Double,
    val weight: Double = 1.0,
)

data class EyeGazeCalibrationProfile(
    val yawCoefficients: DoubleArray,
    val pitchCoefficients: DoubleArray,
    val samples: Int,
    val rmsAngularErrorRadians: Double,
    val calibratedAtNanos: Long,
) {
    init {
        require(yawCoefficients.size == 3)
        require(pitchCoefficients.size == 3)
    }

    fun predict(offset: Vec2): Vec2 =
        Vec2(
            yawCoefficients[0] * offset.x +
                yawCoefficients[1] * offset.y +
                yawCoefficients[2],
            pitchCoefficients[0] * offset.x +
                pitchCoefficients[1] * offset.y +
                pitchCoefficients[2],
        )
}

data class EyeTrackingFrame(
    val timestampNanos: Long,
    val trackingId: Int?,
    val left: EyeObservation?,
    val right: EyeObservation?,
    val combinedGaze: Vec3,
    val fixationConfidence: Double,
    /** Capture timing evidence retained for downstream metric face/eye reconstruction. */
    val captureTimingConfidence: Double = 1.0,
    val rollingShutterSkewNanos: Long = 0L,
    val frameDurationNanos: Long = 0L,
    val exposureTimeNanos: Long = 0L,
) {
    init {
        require(captureTimingConfidence in 0.0..1.0)
        require(rollingShutterSkewNanos >= 0L)
        require(frameDurationNanos >= 0L)
        require(exposureTimeNanos >= 0L)
    }

    /**
     * Conservative optical risk used when native motion-blur telemetry is not yet available.
     * It deliberately treats missing frame timing as unknown rather than claiming a measured blur.
     */
    val estimatedMotionBlurRisk: Double
        get() = estimateOpticalMotionBlurRisk(
            captureTimingConfidence = captureTimingConfidence,
            rollingShutterSkewNanos = rollingShutterSkewNanos,
            frameDurationNanos = frameDurationNanos,
            exposureTimeNanos = exposureTimeNanos,
        )
}

data class FaceFrame(
    val timestampNanos: Long,
    val trackingId: Int?,
    val boundsLeft: Float,
    val boundsTop: Float,
    val boundsRight: Float,
    val boundsBottom: Float,
    val yawDegrees: Float,
    val rollDegrees: Float,
    val points: List<FaceMeshPoint>,
    val leftEyeCenter: Vec2? = null,
    val rightEyeCenter: Vec2? = null,
    val leftEyeOpenProbability: Double = 1.0,
    val rightEyeOpenProbability: Double = 1.0,
    val captureTimingConfidence: Double = 1.0,
    val rollingShutterSkewNanos: Long = 0L,
    val frameDurationNanos: Long = 0L,
    val exposureTimeNanos: Long = 0L,
    val rollingShutterDirectionSign: Int = 1,
) {
    init {
        require(timestampNanos >= 0L)
        require(captureTimingConfidence in 0.0..1.0)
        require(rollingShutterSkewNanos >= 0L)
        require(frameDurationNanos >= 0L)
        require(exposureTimeNanos >= 0L)
    }

    val estimatedMotionBlurRisk: Double
        get() = estimateOpticalMotionBlurRisk(
            captureTimingConfidence = captureTimingConfidence,
            rollingShutterSkewNanos = rollingShutterSkewNanos,
            frameDurationNanos = frameDurationNanos,
            exposureTimeNanos = exposureTimeNanos,
        )
}

enum class FaceContourKind {
    FACE_OVAL,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_EYEBROW_TOP,
    RIGHT_EYEBROW_TOP,
    NOSE_BRIDGE,
    NOSE_BOTTOM,
    UPPER_LIP_TOP,
    UPPER_LIP_BOTTOM,
    LOWER_LIP_TOP,
    LOWER_LIP_BOTTOM,
}

data class FaceContour(
    val kind: FaceContourKind,
    val points: List<Vec2>,
    val confidence: Double,
)

data class FaceContourFrame(
    val timestampNanos: Long,
    val boundsLeft: Float,
    val boundsTop: Float,
    val boundsRight: Float,
    val boundsBottom: Float,
    val yawDegrees: Float,
    val rollDegrees: Float,
    val contours: List<FaceContour>,
    val imageWidth: Int = 0,
    val imageHeight: Int = 0,
    val rotationDegrees: Int = 0,
    val captureTimingConfidence: Double = 1.0,
    val rollingShutterSkewNanos: Long = 0L,
    val frameDurationNanos: Long = 0L,
    val exposureTimeNanos: Long = 0L,
    val rollingShutterDirectionSign: Int = 1,
) {
    init {
        require(timestampNanos >= 0L)
        require(captureTimingConfidence in 0.0..1.0)
        require(rollingShutterSkewNanos >= 0L)
        require(frameDurationNanos >= 0L)
        require(exposureTimeNanos >= 0L)
        require(rollingShutterDirectionSign == 1 || rollingShutterDirectionSign == -1)
        require(imageWidth >= 0)
        require(imageHeight >= 0)
    }

    val estimatedMotionBlurRisk: Double
        get() = estimateOpticalMotionBlurRisk(
            captureTimingConfidence = captureTimingConfidence,
            rollingShutterSkewNanos = rollingShutterSkewNanos,
            frameDurationNanos = frameDurationNanos,
            exposureTimeNanos = exposureTimeNanos,
        )
}

internal fun estimateOpticalMotionBlurRisk(
    captureTimingConfidence: Double,
    rollingShutterSkewNanos: Long,
    frameDurationNanos: Long,
    exposureTimeNanos: Long,
): Double {
    val frame =
        frameDurationNanos.toDouble().coerceAtLeast(1.0)
    val exposureRatio =
        (
            exposureTimeNanos.toDouble() /
                frame
        ).coerceIn(0.0, 1.0)
    val exposureRisk =
        when {
            frameDurationNanos <= 0L -> 0.0
            exposureRatio <= 0.20 -> 0.0
            exposureRatio >= 0.85 -> 1.0
            else ->
                (
                    exposureRatio - 0.20
                ) / 0.65
        }
    val readoutRisk =
        (
            rollingShutterSkewNanos.toDouble().coerceAtLeast(0.0) /
                frameDurationNanos.toDouble().coerceAtLeast(1.0) *
                0.72
            ).coerceIn(0.0, 1.0)
    val timingRisk =
        (
            1.0 -
                captureTimingConfidence
        ).coerceIn(0.0, 1.0)
    return (
        1.0 -
            (
                (1.0 - exposureRisk) *
                    (1.0 - readoutRisk) *
                    (1.0 - timingRisk * 0.50)
                )
        ).coerceIn(0.0, 1.0)
}

data class TrackedObject(
    val trackingId: Int,
    val subjectType: SubjectType,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val confidence: Double,
    val timestampNanos: Long,
)

data class SilhouetteMask(
    val width: Int,
    val height: Int,
    val alpha: ByteArray,
    val timestampNanos: Long,
    val deviceId: DeviceId,
) {
    init { require(alpha.size == width * height) { "Silhouette alpha buffer must match image dimensions" } }
    fun isInside(x: Int, y: Int, threshold: Int = 128): Boolean =
        x in 0 until width && y in 0 until height && (alpha[y * width + x].toInt() and 0xFF) >= threshold
}

data class MeshVertex(
    val position: Vec3,
    val uncertaintyMeters: Double = 0.0,
) {
    init { require(uncertaintyMeters.isFinite() && uncertaintyMeters >= 0.0) }
}

data class MeshTriangle(val a: Int, val b: Int, val c: Int)

data class ReconstructionMesh(
    val vertices: List<MeshVertex>,
    val triangles: List<MeshTriangle>,
    val timestampNanos: Long,
    val confidence: Double,
    val normals: List<Vec3> = emptyList(),
    val vertexUncertaintyMeters: List<Double> = emptyList(),
)

data class ReconstructionDiagnostics(
    val jointsTriangulated: Int,
    val meanReprojectionErrorPx: Double,
    val medianReprojectionErrorPx: Double,
    val clockSkewNanos: Long,
    val droppedObservations: Int,
    val camerasUsed: Int,
)

data class MotionFieldJoint(
    val landmarkId: Int,
    val x: Double,
    val y: Double,
    val confidence: Double,
    val predicted: Boolean,
    val uncertaintyNormalized: Double = 0.0,
    val microMotionScore: Double = 0.0,
) {
    init {
        require(x.isFinite() && x in 0.0..1.0)
        require(y.isFinite() && y in 0.0..1.0)
        require(confidence.isFinite() && confidence in 0.0..1.0)
        require(uncertaintyNormalized.isFinite() && uncertaintyNormalized >= 0.0)
        require(microMotionScore.isFinite() && microMotionScore in 0.0..1.0)
    }
}

data class MotionFieldWorldJoint(
    val landmarkId: Int,
    val xMeters: Double,
    val yMeters: Double,
    val zMeters: Double,
    val confidence: Double,
    val predicted: Boolean = false,
) {
    init {
        require(xMeters.isFinite())
        require(yMeters.isFinite())
        require(zMeters.isFinite())
        require(confidence.isFinite() && confidence in 0.0..1.0)
    }
}

data class MotionFieldActor(
    val actorId: Long,
    val confidence: Double,
    val predicted: Boolean,
    val missedFrames: Int = 0,
    val microMotionScore: Double = 0.0,
    val velocityNormalizedPerSecond: Double = 0.0,
    val velocityXNormalizedPerSecond: Double = 0.0,
    val velocityYNormalizedPerSecond: Double = 0.0,
    val joints: List<MotionFieldJoint>,
) {
    init {
        require(actorId >= 0L)
        require(confidence.isFinite() && confidence in 0.0..1.0)
        require(missedFrames >= 0)
        require(microMotionScore.isFinite() && microMotionScore in 0.0..1.0)
        require(velocityNormalizedPerSecond.isFinite() && velocityNormalizedPerSecond >= 0.0)
        require(velocityXNormalizedPerSecond.isFinite())
        require(velocityYNormalizedPerSecond.isFinite())
        require(joints.size <= 256)
    }
}

enum class MotionFieldGpuBackend {
    VULKAN,
    OPENGL_ES_COMPUTE,
    OPENGL_ES_RENDER,
    CPU,
    UNAVAILABLE,
}

enum class MotionFieldGpuMode {
    AUTO,
    MAXIMUM,
    HIGH,
    BALANCED,
}

data class MotionFieldGpuState(
    val backend: MotionFieldGpuBackend = MotionFieldGpuBackend.UNAVAILABLE,
    val mode: MotionFieldGpuMode = MotionFieldGpuMode.AUTO,
    val renderLatencyMillis: Double = 0.0,
    val gpuUtilizationEstimate: Double = 0.0,
    val droppedFrames: Long = 0L,
    val warning: String? = null,
) {
    init {
        require(renderLatencyMillis.isFinite() && renderLatencyMillis >= 0.0)
        require(gpuUtilizationEstimate.isFinite() && gpuUtilizationEstimate in 0.0..1.0)
        require(droppedFrames >= 0L)
    }
}

private const val MAX_VISIBLE_ACTORS = 64
private const val MAX_JOINTS_PER_ACTOR = 64

data class MotionFieldPreviewFrame(
    val timestampNanos: Long,
    val cameraWidth: Int,
    val cameraHeight: Int,
    val displayTimestampNanos: Long,
    val actors: List<MotionFieldActor>,
    val reconstructionJoints3d: List<MotionFieldWorldJoint> = emptyList(),
    val reconstructionConfidence: Double,
    val reconstructionUncertaintyMeters: Double = 0.0,
    val cameraMotionRisk: Double = 0.0,
    val stale: Boolean = false,
    val predictedToDisplay: Boolean = false,
    val gpu: MotionFieldGpuState = MotionFieldGpuState(),
) {
    init {
        require(timestampNanos >= 0L)
        require(cameraWidth > 0)
        require(cameraHeight > 0)
        require(displayTimestampNanos >= 0L)
        require(actors.size <= MAX_VISIBLE_ACTORS)
        require(
            reconstructionJoints3d.size <=
                MAX_VISIBLE_ACTORS * MAX_JOINTS_PER_ACTOR,
        )
        require(reconstructionConfidence.isFinite() && reconstructionConfidence in 0.0..1.0)
        require(reconstructionUncertaintyMeters.isFinite() && reconstructionUncertaintyMeters >= 0.0)
        require(cameraMotionRisk.isFinite() && cameraMotionRisk in 0.0..1.0)
    }

    val actorCount: Int
        get() = actors.size

    val totalJointCount: Int
        get() = actors.sumOf { it.joints.size }
}


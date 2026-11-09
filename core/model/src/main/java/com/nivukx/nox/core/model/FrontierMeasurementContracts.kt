package com.nivukx.nox.core.model

import kotlin.math.abs

/**
 * Frontier measurement contracts shared by high-end motion, reconstruction and federation layers.
 *
 * The contracts deliberately distinguish observation, geometric constraint, temporal prediction and
 * completion-derived state. This prevents a visually plausible reconstruction from being mistaken
 * for directly measured geometry.
 */

enum class MeasurementAuthority {
    OBSERVED,
    GEOMETRIC,
    TEMPORAL,
    LEARNED,
    COMPLETED,
    REJECTED,
}

enum class MeasurementFailure {
    NONE,
    INVALID_TIMESTAMP,
    INVALID_CAMERA,
    INVALID_LANDMARK,
    INSUFFICIENT_VIEWS,
    POOR_VIEW_DIVERSITY,
    POOR_CALIBRATION,
    POOR_TIMING,
    HIGH_REPROJECTION_ERROR,
    HIGH_UNCERTAINTY,
    IDENTITY_AMBIGUITY,
    MEMORY_PRESSURE,
    THERMAL_PRESSURE,
    NETWORK_PRESSURE,
}

data class CameraFederationEvidence(
    val cameraId: String,
    val sector: Int,
    val positionMeters: Vec3,
    val forward: Vec3,
    val quality: Double,
    val synchronizationConfidence: Double,
    val calibrationConfidence: Double,
    val timingResidualMillis: Double = 0.0,
    val calibrationTranslationSigmaMeters: Double = 0.0,
    val calibrationRotationSigmaRadians: Double = 0.0,
    val networkHeadroom: Double = 1.0,
    val thermalHeadroom: Double = 1.0,
    val available: Boolean = true,
    val witnessOnly: Boolean = false,
) {
    init {
        require(cameraId.isNotBlank())
        require(sector >= 0)
        require(positionMeters.x.isFinite() && positionMeters.y.isFinite() && positionMeters.z.isFinite())
        require(forward.x.isFinite() && forward.y.isFinite() && forward.z.isFinite())
        require(quality in 0.0..1.0)
        require(synchronizationConfidence in 0.0..1.0)
        require(calibrationConfidence in 0.0..1.0)
        require(timingResidualMillis.isFinite() && timingResidualMillis >= 0.0)
        require(calibrationTranslationSigmaMeters.isFinite() && calibrationTranslationSigmaMeters >= 0.0)
        require(calibrationRotationSigmaRadians.isFinite() && calibrationRotationSigmaRadians >= 0.0)
        require(networkHeadroom in 0.0..1.0)
        require(thermalHeadroom in 0.0..1.0)
    }

    val timingAuthority: Double
        get() =
            (
                1.0 -
                    timingResidualMillis / 12.0
                ).coerceIn(0.0, 1.0)

    val calibrationUncertaintyQuality: Double
        get() {
            val translation =
                (
                    1.0 -
                        calibrationTranslationSigmaMeters /
                            0.010
                    ).coerceIn(0.0, 1.0)
            val rotation =
                (
                    1.0 -
                        calibrationRotationSigmaRadians /
                            0.002
                    ).coerceIn(0.0, 1.0)
            return (
                0.65 * translation +
                    0.35 * rotation
                ).coerceIn(0.0, 1.0)
        }

    val authorityScore: Double
        get() = (
            0.28 * quality +
                0.19 * synchronizationConfidence +
                0.18 * calibrationConfidence +
                0.12 * timingAuthority +
                0.10 * calibrationUncertaintyQuality +
                0.07 * networkHeadroom +
                0.06 * thermalHeadroom
            ).coerceIn(0.0, 1.0)

    val usable: Boolean
        get() =
            available &&
                authorityScore >= 0.35 &&
                synchronizationConfidence >= 0.45 &&
                calibrationConfidence >= 0.45
}

data class FrontierLandmarkMeasurement(
    val landmarkId: Int,
    val cameraId: String,
    val point: Vec2,
    val confidence: Double,
    val intrinsics: CameraIntrinsics,
    val cameraPose: Pose,
    val temporalAlignmentConfidence: Double = 1.0,
    val temporalUncertaintyNanos: Long = 0L,
    val rollingShutterFraction: Double = 0.0,
    val rollingShutterSkewNanos: Long = 0L,
    val exposureTimeNanos: Long = 0L,
    val calibrationTranslationSigmaMeters: Double = 0.0,
    val calibrationRotationSigmaRadians: Double = 0.0,
    val observedAtNanos: Long,
) {
    init {
        require(landmarkId >= 0)
        require(cameraId.isNotBlank())
        require(point.x.isFinite() && point.y.isFinite())
        require(confidence in 0.0..1.0)
        require(temporalAlignmentConfidence in 0.0..1.0)
        require(temporalUncertaintyNanos >= 0L)
        require(rollingShutterFraction.isFinite())
        require(rollingShutterSkewNanos >= 0L)
        require(exposureTimeNanos >= 0L)
        require(calibrationTranslationSigmaMeters.isFinite() && calibrationTranslationSigmaMeters >= 0.0)
        require(calibrationRotationSigmaRadians.isFinite() && calibrationRotationSigmaRadians >= 0.0)
        require(observedAtNanos > 0L)
    }

    val timingQuality: Double
        get() {
            val uncertaintyMillis =
                temporalUncertaintyNanos.toDouble() / 1_000_000.0
            val timingPenalty =
                (1.0 - uncertaintyMillis / 12.0).coerceIn(0.0, 1.0)
            val rowPenalty =
                (1.0 - abs(rollingShutterFraction).coerceIn(0.0, 1.0) * 0.40)
                    .coerceIn(0.0, 1.0)
            val exposurePenalty =
                if (exposureTimeNanos <= 0L) {
                    0.80
                } else {
                    (1.0 - exposureTimeNanos.toDouble() / 33_333_333.0)
                        .coerceIn(0.35, 1.0)
                }
            return (
                0.55 * temporalAlignmentConfidence +
                    0.25 * timingPenalty +
                    0.12 * rowPenalty +
                    0.08 * exposurePenalty
                ).coerceIn(0.0, 1.0)
        }

    val calibrationQuality: Double
        get() {
            val translation =
                (1.0 - calibrationTranslationSigmaMeters / 0.010)
                    .coerceIn(0.0, 1.0)
            val rotation =
                (1.0 - calibrationRotationSigmaRadians / 0.002)
                    .coerceIn(0.0, 1.0)
            return (0.60 * translation + 0.40 * rotation).coerceIn(0.0, 1.0)
        }
}

data class FrontierLandmark3dResult(
    val landmarkId: Int,
    val timestampNanos: Long,
    val positionMeters: Vec3,
    val covarianceMeters2: Double,
    val p95UncertaintyMeters: Double,
    val reprojectionErrorPixels: Double,
    val viewCount: Int,
    val angularDiversityDegrees: Double,
    val cameraIds: List<String>,
    val authority: MeasurementAuthority,
    val failure: MeasurementFailure = MeasurementFailure.NONE,
) {
    init {
        require(landmarkId >= 0)
        require(timestampNanos > 0L)
        require(positionMeters.x.isFinite() && positionMeters.y.isFinite() && positionMeters.z.isFinite())
        require(covarianceMeters2.isFinite() && covarianceMeters2 >= 0.0)
        require(p95UncertaintyMeters.isFinite() && p95UncertaintyMeters >= 0.0)
        require(reprojectionErrorPixels.isFinite() && reprojectionErrorPixels >= 0.0)
        require(viewCount >= 0)
        require(angularDiversityDegrees.isFinite() && angularDiversityDegrees >= 0.0)
        require(cameraIds.distinct().size == cameraIds.size)
    }

    val precisionFeasibility: Double
        get() {
            val views = (viewCount.toDouble() / 6.0).coerceIn(0.0, 1.0)
            val parallax = (angularDiversityDegrees / 30.0).coerceIn(0.0, 1.0)
            val uncertainty =
                (1.0 - p95UncertaintyMeters / 0.002).coerceIn(0.0, 1.0)
            val reprojection =
                (1.0 - reprojectionErrorPixels / 1.0).coerceIn(0.0, 1.0)
            return (
                0.28 * views +
                    0.26 * parallax +
                    0.30 * uncertainty +
                    0.16 * reprojection
                ).coerceIn(0.0, 1.0)
        }

    val qualificationSafe: Boolean
        get() =
            authority == MeasurementAuthority.GEOMETRIC &&
                failure == MeasurementFailure.NONE &&
                viewCount >= 3 &&
                angularDiversityDegrees >= 8.0 &&
                p95UncertaintyMeters <= 0.001
}

data class FrontierActorRuntimeContract(
    val actorId: Long,
    val priority: Double,
    val visibleRatio: Double,
    val identityConfidence: Double,
    val occlusionRatio: Double,
    val interactionRisk: Double,
    val requestedSurfaceDetail: Boolean,
    val requestedFaceDetail: Boolean,
    val requestedEyeDetail: Boolean,
    val requestedHandDetail: Boolean,
    val faceGeometryConfidence: Double = 1.0,
    val eyeGeometryConfidence: Double = 1.0,
    val handGeometryConfidence: Double = 1.0,
    val surfaceGeometryConfidence: Double = 1.0,
    val occlusionRecoveryConfidence: Double = 1.0,
    val identityStability: Double = 1.0,
) {
    init {
        require(actorId >= 0L)
        require(priority in 0.0..1.0)
        require(visibleRatio in 0.0..1.0)
        require(identityConfidence in 0.0..1.0)
        require(occlusionRatio in 0.0..1.0)
        require(interactionRisk in 0.0..1.0)
        listOf(
            faceGeometryConfidence,
            eyeGeometryConfidence,
            handGeometryConfidence,
            surfaceGeometryConfidence,
            occlusionRecoveryConfidence,
            identityStability,
        ).forEach { require(it in 0.0..1.0) }
    }

    val modalityUncertainty: Double
        get() =
            (
                0.20 * (1.0 - faceGeometryConfidence) +
                    0.18 * (1.0 - eyeGeometryConfidence) +
                    0.16 * (1.0 - handGeometryConfidence) +
                    0.22 * (1.0 - surfaceGeometryConfidence) +
                    0.14 * (1.0 - occlusionRecoveryConfidence) +
                    0.10 * (1.0 - identityStability)
                ).coerceIn(0.0, 1.0)

    val urgency: Double
        get() =
            (
                0.22 * priority +
                    0.18 * (1.0 - visibleRatio) +
                    0.18 * (1.0 - identityConfidence) +
                    0.14 * occlusionRatio +
                    0.12 * interactionRisk +
                    0.12 * modalityUncertainty +
                    0.10 * (1.0 - occlusionRecoveryConfidence) +
                    0.08 * (1.0 - identityStability)
                ).coerceIn(0.0, 1.0)
}

data class FrontierReconstructionProvenance(
    val observedRatio: Double,
    val geometricRatio: Double,
    val temporalRatio: Double,
    val learnedRatio: Double,
    val completedRatio: Double,
    val meanUncertaintyMeters: Double,
    val p95UncertaintyMeters: Double,
    val topologyContinuity: Double,
    val temporalStability: Double,
    val viewDiversity: Double,
) {
    init {
        listOf(
            observedRatio,
            geometricRatio,
            temporalRatio,
            learnedRatio,
            completedRatio,
            topologyContinuity,
            temporalStability,
            viewDiversity,
        ).forEach { require(it in 0.0..1.0) }
        require(meanUncertaintyMeters.isFinite() && meanUncertaintyMeters >= 0.0)
        require(p95UncertaintyMeters.isFinite() && p95UncertaintyMeters >= 0.0)
    }

    val directlyMeasuredRatio: Double
        get() = (observedRatio + geometricRatio).coerceIn(0.0, 1.0)

    val conservativeConfidence: Double
        get() =
            (
                0.34 * directlyMeasuredRatio +
                    0.22 * topologyContinuity +
                    0.22 * temporalStability +
                    0.22 * viewDiversity
                ).coerceIn(0.0, 1.0)
}

data class FrontierCaptureHealth(
    val timestampNanos: Long,
    val cameraCount: Int,
    val selectedAuthorityCameraCount: Int,
    val witnessCameraCount: Int,
    val activeActorCount: Int,
    val heldActorCount: Int,
    val meanGeometryConfidence: Double,
    val p95UncertaintyMeters: Double,
    val meanReprojectionErrorPixels: Double,
    val synchronizationConfidence: Double,
    val calibrationConfidence: Double,
    val viewDiversity: Double,
    val memoryHeadroom: Double,
    val thermalHeadroom: Double,
    val networkHeadroom: Double,
    val precisionFeasibility: Double,
    val authoritative: Boolean,
    val failure: MeasurementFailure = MeasurementFailure.NONE,
    val warnings: List<String> = emptyList(),
) {
    init {
        require(timestampNanos > 0L)
        require(cameraCount in 0..128)
        require(selectedAuthorityCameraCount in 0..32)
        require(witnessCameraCount in 0..128)
        require(activeActorCount in 0..64)
        require(heldActorCount >= 0)
        listOf(
            meanGeometryConfidence,
            synchronizationConfidence,
            calibrationConfidence,
            viewDiversity,
            memoryHeadroom,
            thermalHeadroom,
            networkHeadroom,
            precisionFeasibility,
        ).forEach { require(it in 0.0..1.0) }
        require(p95UncertaintyMeters.isFinite() && p95UncertaintyMeters >= 0.0)
        require(meanReprojectionErrorPixels.isFinite() && meanReprojectionErrorPixels >= 0.0)
    }
}

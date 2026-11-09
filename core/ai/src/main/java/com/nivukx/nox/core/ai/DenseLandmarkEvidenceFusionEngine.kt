package com.nivukx.nox.core.ai

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import com.nivukx.nox.core.model.Vec3
import kotlin.math.acos
import kotlin.math.sqrt

/**
 * Production dense-landmark evidence fusion for multi-view markerless capture.
 *
 * The engine is model-agnostic: learned landmark detectors, segmentation models, geometry
 * trackers and depth estimators can all contribute evidence. It does not invent physical
 * accuracy claims. Its job is to produce a bounded, uncertainty-aware evidence surface that
 * downstream body fitting and reconstruction engines can trust or reject.
 *
 * Invariants:
 * - non-finite evidence is rejected;
 * - per-camera timestamps are monotonic;
 * - stale evidence cannot become authoritative;
 * - robust weighting prevents one bad view from dominating;
 * - visibility, uncertainty and contact probability are first-class;
 * - short occlusions may be bridged with bounded constant-acceleration prediction;
 * - state and diagnostic memory are explicitly bounded.
 */
class DenseLandmarkEvidenceFusionEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        // Camera-state memory is bounded but aligned with the distributed 128-node capture envelope.
        val maximumCameras: Int = 128,
        val maximumSubjects: Int = 64,
        val maximumLandmarksPerSubject: Int = 1024,
        val maximumStoredLandmarks: Int = 16_384,
        val maximumObservationsPerFrame: Int = 65_536,
        val maximumObservationsPerLandmark: Int = 64,
        val maximumHistoryFrames: Int = 12,
        val maximumCameraStates: Int = 128,
        val diagnosticHistoryLimit: Int = 128,
        val staleObservationMillis: Double = 120.0,
        val staleCameraMillis: Double = 800.0,
        val temporalHalfLifeMillis: Double = 90.0,
        val robustResidualScale: Double = 0.018,
        val hardResidualNormalized: Double = 0.18,
        val hardDepthResidualMeters: Double = 0.15,
        val minimumObservationWeight: Double = 0.025,
        val minimumFusedConfidence: Double = 0.30,
        val minimumViewCountForCrossView: Int = 2,
        val occlusionPredictionHorizonMillis: Double = 90.0,
        val maximumPredictionSpeedNormalizedPerSecond: Double = 4.0,
        val maximumPredictionAccelerationNormalizedPerSecond2: Double = 40.0,
        val maximumPredictionDepthSpeedMetersPerSecond: Double = 12.0,
        val maximumPredictionDepthAccelerationMetersPerSecond2: Double = 60.0,
        val contactEnterProbability: Double = 0.78,
        val contactExitProbability: Double = 0.34,
        val contactHysteresisFrames: Int = 2,
        val motionBoundaryWeight: Double = 0.04,
        val appearanceConsistencyWeight: Double = 0.04,
        val photometricConsistencyWeight: Double = 0.05,
        val depthConfidenceWeight: Double = 0.06,
        val temporalInnovationWeight: Double = 0.10,
        val geometricConfidenceWeight: Double = 0.14,
        val calibratedRayAgreementWeight: Double = 0.18,
        val maximumSymmetricRayDistanceMeters: Double = 0.08,
        val minimumRayAngleDegrees: Double = 0.25,
        val calibrationTranslationSigmaScaleMeters: Double = 0.020,
        val calibrationRotationSigmaScaleRadians: Double = 0.010,
        val calibrationIntrinsicSigmaScale: Double = 0.010,
    ) {
        init {
            require(maximumCameras > 0)
            require(maximumSubjects > 0)
            require(maximumLandmarksPerSubject > 0)
            require(maximumStoredLandmarks >= maximumLandmarksPerSubject)
            require(maximumObservationsPerFrame > 0)
            require(maximumObservationsPerLandmark > 0)
            require(maximumHistoryFrames >= 2)
            require(maximumCameraStates >= maximumCameras)
            require(diagnosticHistoryLimit >= 8)
            require(staleObservationMillis > 0.0)
            require(staleCameraMillis >= staleObservationMillis)
            require(temporalHalfLifeMillis > 0.0)
            require(robustResidualScale > 0.0)
            require(hardResidualNormalized > robustResidualScale)
            require(hardDepthResidualMeters > 0.0)
            require(minimumObservationWeight in 0.0..1.0)
            require(minimumFusedConfidence in 0.0..1.0)
            require(minimumViewCountForCrossView > 0)
            require(occlusionPredictionHorizonMillis >= 0.0)
            require(maximumPredictionSpeedNormalizedPerSecond > 0.0)
            require(maximumPredictionAccelerationNormalizedPerSecond2 > 0.0)
            require(maximumPredictionDepthSpeedMetersPerSecond > 0.0)
            require(maximumPredictionDepthAccelerationMetersPerSecond2 > 0.0)
            require(contactEnterProbability in 0.0..1.0)
            require(contactExitProbability in 0.0..1.0)
            require(contactExitProbability < contactEnterProbability)
            require(contactHysteresisFrames > 0)
            require(motionBoundaryWeight >= 0.0)
            require(appearanceConsistencyWeight >= 0.0)
            require(photometricConsistencyWeight >= 0.0)
            require(depthConfidenceWeight >= 0.0)
            require(temporalInnovationWeight >= 0.0)
            require(geometricConfidenceWeight >= 0.0)
            require(calibratedRayAgreementWeight in 0.0..0.5)
            require(maximumSymmetricRayDistanceMeters > 0.0)
            require(minimumRayAngleDegrees > 0.0)
            require(calibrationTranslationSigmaScaleMeters > 0.0)
            require(calibrationRotationSigmaScaleRadians > 0.0)
            require(calibrationIntrinsicSigmaScale > 0.0)
        }
    }

    data class LandmarkKey(
        val subjectId: Int,
        val landmarkId: Int,
    )

    enum class EvidenceState {
        OBSERVED,
        TEMPORALLY_PREDICTED,
        OCCLUDED,
        LOW_CONFIDENCE,
        REJECTED,
    }

    enum class ContactState {
        UNKNOWN,
        FREE,
        APPROACHING,
        CONTACT,
        SEPARATING,
        OCCLUDED,
    }

    data class CameraObservation(
        val cameraId: Int,
        val timestampNanos: Long,
        val normalizedX: Double,
        val normalizedY: Double,
        val depthMeters: Double? = null,
        val confidence: Double,
        val uncertainty: Double = 0.0,
        val visibility: Double = 1.0,
        val contactProbability: Double = 0.0,
        val cameraQuality: Double = 1.0,
        val reprojectionResidualNormalized: Double = 0.0,
        val depthResidualMeters: Double = 0.0,
        val sourceTag: String = "unknown",
        val rollingShutterCompensated: Boolean = true,
        val segmentationConsistency: Double = 0.0,
        val motionBoundaryConfidence: Double = 1.0,
        val appearanceConsistency: Double = 1.0,
        val photometricConsistency: Double = 1.0,
        val depthConfidence: Double = 1.0,
        val viewSelectionScore: Double = 1.0,
        val probabilitySigmaXNormalized: Double = 0.0,
        val probabilitySigmaYNormalized: Double = 0.0,
        val probabilityCalibrationScore: Double = 1.0,
        val worldRayOriginMeters: Vec3? = null,
        val worldRayDirection: Vec3? = null,
        val calibrationUncertaintyMeters: Double = 0.0,
        val calibrationConfidence: Double = 0.0,
        val motionBlurRisk: Double = 0.0,
        val segmentationEvidenceAvailable: Boolean = false,
        val calibrationEvidenceAvailable: Boolean = false,
    ) {
        fun finite(): Boolean =
            timestampNanos >= 0L &&
                normalizedX.isFinite() &&
                normalizedY.isFinite() &&
                (depthMeters == null || depthMeters.isFinite()) &&
                confidence.isFinite() &&
                uncertainty.isFinite() &&
                visibility.isFinite() &&
                contactProbability.isFinite() &&
                cameraQuality.isFinite() &&
                reprojectionResidualNormalized.isFinite() &&
                depthResidualMeters.isFinite() &&
                segmentationConsistency.isFinite() &&
                motionBoundaryConfidence.isFinite() &&
                appearanceConsistency.isFinite() &&
                photometricConsistency.isFinite() &&
                depthConfidence.isFinite() &&
                viewSelectionScore.isFinite() &&
                probabilitySigmaXNormalized.isFinite() &&
                probabilitySigmaYNormalized.isFinite() &&
                probabilityCalibrationScore.isFinite() &&
                calibrationUncertaintyMeters.isFinite() &&
                calibrationConfidence.isFinite() &&
                motionBlurRisk.isFinite() &&
                (worldRayOriginMeters == null || finite(worldRayOriginMeters)) &&
                (worldRayDirection == null || finite(worldRayDirection)) &&
                (worldRayDirection == null || worldRayDirection.norm() > 1e-9) &&
                (worldRayOriginMeters == null) == (worldRayDirection == null)
    }

    data class FrameInput(
        val timestampNanos: Long,
        val observations: List<Pair<LandmarkKey, CameraObservation>>,
    )

    data class FusedLandmark(
        val key: LandmarkKey,
        val timestampNanos: Long,
        val normalizedX: Double,
        val normalizedY: Double,
        val depthMeters: Double?,
        val confidence: Double,
        val uncertainty: Double,
        val visibility: Double,
        val contactProbability: Double,
        val contactState: ContactState,
        val crossViewAgreement: Double,
        val supportingCameraCount: Int,
        val temporalStability: Double,
        val acceptedViewCount: Int,
        val rejectedViewCount: Int,
        val occludedViewCount: Int,
        val state: EvidenceState,
        val sourceDiversity: Int,
        val weightedReprojectionResidualNormalized: Double,
        val weightedDepthResidualMeters: Double,
        val predictedDistanceNormalized: Double,
        val motionBoundaryConfidence: Double = 0.0,
        val appearanceConsistency: Double = 0.0,
        val depthConfidence: Double = 0.0,
        val viewSelectionScore: Double = 0.0,
        val temporalInnovationScore: Double = 0.0,
        val probabilitySigmaXNormalized: Double = 0.0,
        val probabilitySigmaYNormalized: Double = 0.0,
        val probabilityCalibrationScore: Double = 1.0,
        val calibrationConfidence: Double = 0.0,
        val diagnostics: List<String>,
        val segmentationConsistency: Double = 0.0,
        val segmentationEvidenceAvailable: Boolean = false,
        val calibrationEvidenceAvailable: Boolean = false,
    )

    data class FrameResult(
        val timestampNanos: Long,
        val landmarks: List<FusedLandmark>,
        val coverage: Double,
        val visibleCoverage: Double,
        val meanConfidence: Double,
        val uncertaintyP50: Double,
        val uncertaintyP95: Double,
        val crossViewAgreement: Double,
        val supportingCameraCount: Int,
        val crossViewEvidenceAvailable: Boolean,
        val temporalStability: Double,
        val contactCoverage: Double,
        val predictionBridgeRatio: Double,
        val acceptedObservations: Long,
        val rejectedObservations: Long,
        val staleObservations: Long,
        val nonFiniteObservations: Long,
        val duplicateCameraObservations: Long,
        val warningCount: Int,
        val hardFailure: Boolean,
        val warnings: List<String>,
        val diagnostics: List<String>,
        val geometricConfidence: Double = 0.0,
        val viewDiversity: Double = 0.0,
        val motionBoundaryConfidence: Double = 0.0,
        val appearanceConsistency: Double = 0.0,
        val depthConfidence: Double = 0.0,
        val temporalInnovationP95: Double = 0.0,
    ) {
        val authoritative: Boolean
            get() =
                !hardFailure &&
                    coverage >= 0.50 &&
                    meanConfidence >= 0.55 &&
                    uncertaintyP95 <= 0.25 &&
                    (
                        !crossViewEvidenceAvailable ||
                            crossViewAgreement >= 0.55
                    )
    }

    data class CameraTelemetry(
        val cameraId: Int,
        val lastTimestampNanos: Long,
        val acceptedObservations: Long,
        val rejectedObservations: Long,
        val staleObservations: Long,
        val nonFiniteObservations: Long,
        val hardResidualObservations: Long,
    )

    data class TelemetrySnapshot(
        val activeCameraCount: Int,
        val activeLandmarkCount: Int,
        val latestTimestampNanos: Long?,
        val coverage: Double,
        val visibleCoverage: Double,
        val meanConfidence: Double,
        val uncertaintyP95: Double,
        val crossViewAgreement: Double,
        val crossViewEvidenceAvailable: Boolean,
        val temporalStability: Double,
        val contactCoverage: Double,
        val predictionBridgeRatio: Double,
        val rejectedObservations: Long,
        val staleObservations: Long,
        val nonFiniteObservations: Long,
        val duplicateCameraObservations: Long,
        val hardFailure: Boolean,
        val cameraTelemetry: List<CameraTelemetry>,
        val geometricConfidence: Double = 0.0,
        val viewDiversity: Double = 0.0,
        val motionBoundaryConfidence: Double = 0.0,
        val appearanceConsistency: Double = 0.0,
        val depthConfidence: Double = 0.0,
        val temporalInnovationP95: Double = 0.0,
    )

    private data class WeightedObservation(
        val observation: CameraObservation,
        val weight: Double,
        val temporalInnovationScore: Double,
    )

    private data class LandmarkState(
        val key: LandmarkKey,
        var timestampNanos: Long = Long.MIN_VALUE,
        var observedTimestampNanos: Long = Long.MIN_VALUE,
        var x: Double = 0.0,
        var y: Double = 0.0,
        var z: Double? = null,
        var vx: Double = 0.0,
        var vy: Double = 0.0,
        var vz: Double = 0.0,
        var ax: Double = 0.0,
        var ay: Double = 0.0,
        var az: Double = 0.0,
        var confidence: Double = 0.0,
        var uncertainty: Double = 1.0,
        var visibility: Double = 0.0,
        var contactProbability: Double = 0.0,
        var contactState: ContactState = ContactState.UNKNOWN,
        var contactCandidateFrames: Int = 0,
        var separationFrames: Int = 0,
        var predictedFrames: Long = 0L,
        var observedFrames: Long = 0L,
    )

    private data class CameraState(
        val cameraId: Int,
        var lastTimestampNanos: Long = Long.MIN_VALUE,
        var acceptedObservations: Long = 0L,
        var rejectedObservations: Long = 0L,
        var staleObservations: Long = 0L,
        var nonFiniteObservations: Long = 0L,
        var hardResidualObservations: Long = 0L,
    )

    private data class FrameDigest(
        val timestampNanos: Long,
        val coverage: Double,
        val visibleCoverage: Double,
        val meanConfidence: Double,
        val uncertaintyP95: Double,
        val crossViewAgreement: Double,
        val crossViewEvidenceAvailable: Boolean,
        val temporalStability: Double,
        val hardFailure: Boolean,
    )

    private val landmarkStates = LinkedHashMap<LandmarkKey, LandmarkState>()
    private val cameraStates = LinkedHashMap<Int, CameraState>()
    private val frameHistory = ArrayDeque<FrameDigest>()
    private val diagnosticHistory = ArrayDeque<String>()

    private var closed = false
    private var lastFrameTimestampNanos = Long.MIN_VALUE
    private var latestResult: FrameResult? = null

    private var cumulativeRejected = 0L
    private var cumulativeStale = 0L
    private var cumulativeNonFinite = 0L
    private var cumulativeDuplicates = 0L

    @Synchronized
    fun process(input: FrameInput): FrameResult {
        check(!closed) { "DenseLandmarkEvidenceFusionEngine is closed" }

        val timestamp = input.timestampNanos
        if (
            lastFrameTimestampNanos != Long.MIN_VALUE &&
            timestamp < lastFrameTimestampNanos
        ) {
            appendDiagnostic(
                "frame timestamp reversal rejected: previous=" +
                    lastFrameTimestampNanos + " current=" + timestamp,
            )
            val failed = emptyResult(
                timestamp = timestamp,
                warning = "Frame timestamp reversal was rejected.",
                hardFailure = true,
            )
            latestResult = failed
            return failed
        }

        lastFrameTimestampNanos = timestamp
        prune(timestamp)

        val grouped = HashMap<LandmarkKey, MutableList<CameraObservation>>()
        var frameStale = 0L
        var frameNonFinite = 0L
        var frameRejected = 0L
        var frameDuplicate = 0L

        var acceptedInputObservations = 0

        for ((key, raw) in input.observations) {
            if (acceptedInputObservations >= config.maximumObservationsPerFrame) {
                frameRejected++
                cumulativeRejected++
                continue
            }

            val camera = cameraState(raw.cameraId)

            if (!raw.finite()) {
                frameNonFinite++
                cumulativeNonFinite++
                camera.nonFiniteObservations++
                continue
            }

            if (
                key.subjectId !in 0 until config.maximumSubjects ||
                key.landmarkId !in 0 until config.maximumLandmarksPerSubject
            ) {
                frameRejected++
                cumulativeRejected++
                camera.rejectedObservations++
                continue
            }

            if (
                camera.lastTimestampNanos != Long.MIN_VALUE &&
                raw.timestampNanos < camera.lastTimestampNanos
            ) {
                frameRejected++
                cumulativeRejected++
                camera.rejectedObservations++
                appendDiagnostic(
                    "camera timestamp reversal rejected: camera=" + raw.cameraId,
                )
                continue
            }

            if (raw.timestampNanos > timestamp) {
                frameRejected++
                cumulativeRejected++
                camera.rejectedObservations++
                continue
            }

            val ageMillis =
                (timestamp - raw.timestampNanos).toDouble() / 1_000_000.0

            if (!ageMillis.isFinite() || ageMillis > config.staleObservationMillis) {
                frameStale++
                cumulativeStale++
                camera.staleObservations++
                continue
            }

            camera.lastTimestampNanos = raw.timestampNanos
            acceptedInputObservations++

            if (
                raw.reprojectionResidualNormalized >
                    config.hardResidualNormalized ||
                abs(raw.depthResidualMeters) >
                    config.hardDepthResidualMeters
            ) {
                frameRejected++
                cumulativeRejected++
                camera.rejectedObservations++
                camera.hardResidualObservations++
                continue
            }

            val observationsForLandmark =
                grouped.getOrPut(key) { mutableListOf() }
            if (
                observationsForLandmark.size <
                    config.maximumObservationsPerLandmark
            ) {
                observationsForLandmark += sanitize(raw)
            } else {
                cumulativeRejected++
                camera.rejectedObservations++
            }
        }

        val fused = ArrayList<FusedLandmark>(grouped.size)

        for ((key, observations) in grouped) {
            fused += fuseLandmark(
                timestampNanos = timestamp,
                key = key,
                observations = observations,
            )
        }

        val observedKeys = grouped.keys
        for ((key, state) in landmarkStates) {
            if (key !in observedKeys) {
                predictOccluded(
                    timestampNanos = timestamp,
                    state = state,
                )?.let(fused::add)
            }
        }

        fused.sortWith(
            compareBy<FusedLandmark> { it.key.subjectId }
                .thenBy { it.key.landmarkId },
        )

        val result = summarize(
            timestamp = timestamp,
            landmarks = fused,
            frameAccepted = fused.sumOf { it.acceptedViewCount }.toLong(),
            frameRejected = frameRejected +
                fused.sumOf { it.rejectedViewCount }.toLong(),
            frameStale = frameStale,
            frameNonFinite = frameNonFinite,
            frameDuplicates = frameDuplicate,
        )

        latestResult = result
        frameHistory.addLast(
            FrameDigest(
                timestampNanos = result.timestampNanos,
                coverage = result.coverage,
                visibleCoverage = result.visibleCoverage,
                meanConfidence = result.meanConfidence,
                uncertaintyP95 = result.uncertaintyP95,
                crossViewAgreement = result.crossViewAgreement,
                crossViewEvidenceAvailable =
                    result.crossViewEvidenceAvailable,
                temporalStability = result.temporalStability,
                hardFailure = result.hardFailure,
            ),
        )
        while (frameHistory.size > config.maximumHistoryFrames) {
            frameHistory.removeFirst()
        }

        if (frameStale > 0L || frameRejected > 0L || frameNonFinite > 0L) {
            appendDiagnostic(
                "frame gate: rejected=" + frameRejected +
                    " stale=" + frameStale +
                    " nonFinite=" + frameNonFinite,
            )
        }

        return result
    }

    @Synchronized
    fun latest(): FrameResult? = latestResult

    @Synchronized
    fun diagnostics(): List<String> = diagnosticHistory.toList()

    @Synchronized
    fun telemetry(): TelemetrySnapshot {
        val result = latestResult
        return TelemetrySnapshot(
            activeCameraCount = cameraStates.size,
            activeLandmarkCount = landmarkStates.size,
            latestTimestampNanos = result?.timestampNanos,
            coverage = result?.coverage ?: 0.0,
            visibleCoverage = result?.visibleCoverage ?: 0.0,
            meanConfidence = result?.meanConfidence ?: 0.0,
            uncertaintyP95 = result?.uncertaintyP95 ?: 1.0,
            crossViewAgreement = result?.crossViewAgreement ?: 0.0,
            crossViewEvidenceAvailable =
                result?.crossViewEvidenceAvailable ?: false,
            temporalStability = result?.temporalStability ?: 0.0,
            contactCoverage = result?.contactCoverage ?: 0.0,
            predictionBridgeRatio = result?.predictionBridgeRatio ?: 0.0,
            rejectedObservations = cumulativeRejected,
            staleObservations = cumulativeStale,
            nonFiniteObservations = cumulativeNonFinite,
            duplicateCameraObservations = cumulativeDuplicates,
            hardFailure = result?.hardFailure ?: false,
            cameraTelemetry = cameraStates.values.map {
                CameraTelemetry(
                    cameraId = it.cameraId,
                    lastTimestampNanos = it.lastTimestampNanos,
                    acceptedObservations = it.acceptedObservations,
                    rejectedObservations = it.rejectedObservations,
                    staleObservations = it.staleObservations,
                    nonFiniteObservations = it.nonFiniteObservations,
                    hardResidualObservations = it.hardResidualObservations,
                )
            },
            geometricConfidence = result?.geometricConfidence ?: 0.0,
            viewDiversity = result?.viewDiversity ?: 0.0,
            motionBoundaryConfidence =
                result?.motionBoundaryConfidence ?: 0.0,
            appearanceConsistency =
                result?.appearanceConsistency ?: 0.0,
            depthConfidence = result?.depthConfidence ?: 0.0,
            temporalInnovationP95 =
                result?.temporalInnovationP95 ?: 0.0,
        )
    }

    @Synchronized
    fun reset() {
        check(!closed) { "DenseLandmarkEvidenceFusionEngine is closed" }
        landmarkStates.clear()
        cameraStates.clear()
        frameHistory.clear()
        diagnosticHistory.clear()
        lastFrameTimestampNanos = Long.MIN_VALUE
        latestResult = null
        cumulativeRejected = 0L
        cumulativeStale = 0L
        cumulativeNonFinite = 0L
        cumulativeDuplicates = 0L
    }

    @Synchronized
    override fun close() {
        if (closed) return
        reset()
        closed = true
    }

    private fun fuseLandmark(
        timestampNanos: Long,
        key: LandmarkKey,
        observations: List<CameraObservation>,
    ): FusedLandmark {
        val state = landmarkStates.getOrPut(key) {
            LandmarkState(key = key)
        }

        val uniqueByCamera = HashMap<Int, CameraObservation>()
        var duplicateCount = 0

        for (observation in observations) {
            val existing = uniqueByCamera[observation.cameraId]
            if (existing == null) {
                uniqueByCamera[observation.cameraId] = observation
            } else {
                duplicateCount++
                cumulativeDuplicates++
                cameraState(observation.cameraId).rejectedObservations++
            }
        }

        val candidates = uniqueByCamera.values.mapNotNull { observation ->
            val ageMillis =
                (timestampNanos - observation.timestampNanos)
                    .coerceAtLeast(0L)
                    .toDouble() / 1_000_000.0

            val freshness = temporalFreshness(ageMillis)
            val confidence = observation.confidence.coerceIn(0.0, 1.0)
            val uncertaintyQuality =
                (1.0 - observation.uncertainty).coerceIn(0.0, 1.0)
            val visibility =
                observation.visibility.coerceIn(0.0, 1.0)
            val cameraQuality =
                observation.cameraQuality.coerceIn(0.0, 1.0)
            val residualQuality =
                residualQuality(observation.reprojectionResidualNormalized)
            val segmentation =
                observation.segmentationConsistency.coerceIn(0.0, 1.0)
            // Missing mask evidence is not a perfect score and not an observed failure. Keep
            // its contribution neutral until a real segmentation measurement is available.
            val segmentationContribution =
                if (observation.segmentationEvidenceAvailable) segmentation else 0.5
            val rollingShutterQuality =
                if (observation.rollingShutterCompensated) 1.0 else 0.72

            val motionBoundary =
                observation.motionBoundaryConfidence.coerceIn(0.0, 1.0)
            val appearance =
                observation.appearanceConsistency.coerceIn(0.0, 1.0)
            val photometric =
                observation.photometricConsistency
                    .coerceIn(0.0, 1.0)
            val depthConfidence =
                observation.depthConfidence.coerceIn(0.0, 1.0)
            val viewSelection =
                observation.viewSelectionScore.coerceIn(0.0, 1.0)
            val opticalQuality =
                (
                    1.0 -
                        0.72 *
                            observation.motionBlurRisk
                    ).coerceIn(0.22, 1.0)

            val predicted =
                predictPosition(timestampNanos, state)

            val temporalInnovationScore =
                predicted?.let { (px, py) ->
                    val innovation =
                        sqrt(
                            square(observation.normalizedX - px) +
                                square(observation.normalizedY - py),
                        )
                    exp(
                        -innovation /
                            (config.robustResidualScale * 2.4)
                                .coerceAtLeast(1e-6),
                    ).coerceIn(0.0, 1.0)
                } ?: 1.0

            val baseWeight = (
                confidence * 0.18 +
                    uncertaintyQuality * 0.19 +
                    visibility * 0.15 +
                    cameraQuality * 0.11 +
                    freshness * 0.08 +
                    residualQuality * 0.07 +
                    segmentationContribution * 0.04 +
                    rollingShutterQuality * 0.02 +
                    opticalQuality * 0.06 +
                    motionBoundary * config.motionBoundaryWeight +
                    appearance * config.appearanceConsistencyWeight +
                    photometric * config.photometricConsistencyWeight +
                    depthConfidence * config.depthConfidenceWeight +
                    viewSelection * 0.04
                ).coerceIn(0.0, 1.0)

            val weight = (
                baseWeight *
                    (
                        1.0 -
                            config.temporalInnovationWeight *
                                (1.0 - temporalInnovationScore)
                    )
                ).coerceIn(0.0, 1.0)

            if (!weight.isFinite() || weight < config.minimumObservationWeight) {
                cumulativeRejected++
                cameraState(observation.cameraId).rejectedObservations++
                null
            } else {
                WeightedObservation(
                    observation = observation,
                    weight = weight,
                    temporalInnovationScore = temporalInnovationScore,
                )
            }
        }

        if (candidates.isEmpty()) {
            return predictOccluded(timestampNanos, state)
                ?: rejectedLandmark(timestampNanos, key)
        }

        val xMedian = robustMedian(
            candidates.map { it.observation.normalizedX },
        )
        val yMedian = robustMedian(
            candidates.map { it.observation.normalizedY },
        )
        val depthValues = candidates.mapNotNull {
            it.observation.depthMeters
        }
        val zMedian = depthValues.takeIf { it.isNotEmpty() }
            ?.let(::robustMedian)

        val robustCandidates = candidates.map { candidate ->
            val observation = candidate.observation
            val xyResidual = sqrt(
                square(observation.normalizedX - xMedian) +
                    square(observation.normalizedY - yMedian),
            )
            val depthResidual = if (
                zMedian != null && observation.depthMeters != null
            ) {
                abs(observation.depthMeters - zMedian)
            } else {
                0.0
            }

            candidate.copy(
                weight = candidate.weight *
                    huberWeight(
                        xyResidual,
                        config.robustResidualScale,
                    ) *
                    huberWeight(
                        depthResidual,
                        config.hardDepthResidualMeters * 0.25,
                    ),
            )
        }.filter {
            val observation = it.observation
            val xyResidual = sqrt(
                square(observation.normalizedX - xMedian) +
                    square(observation.normalizedY - yMedian),
            )
            val depthResidual = if (
                zMedian != null && observation.depthMeters != null
            ) {
                abs(observation.depthMeters - zMedian)
            } else {
                0.0
            }

            it.weight >= config.minimumObservationWeight * 0.40 &&
                xyResidual <= config.hardResidualNormalized &&
                depthResidual <= config.hardDepthResidualMeters
        }

        if (robustCandidates.isEmpty()) {
            return predictOccluded(timestampNanos, state)
                ?: rejectedLandmark(timestampNanos, key)
        }

        val totalWeight =
            robustCandidates.sumOf { it.weight }.coerceAtLeast(1e-9)

        val x = robustCandidates.sumOf {
            it.observation.normalizedX * it.weight
        } / totalWeight

        val y = robustCandidates.sumOf {
            it.observation.normalizedY * it.weight
        } / totalWeight

        val depthCandidates = robustCandidates.filter {
            it.observation.depthMeters != null
        }

        val depth = depthCandidates.takeIf { it.isNotEmpty() }?.let {
            val denominator = it.sumOf { candidate -> candidate.weight }
                .coerceAtLeast(1e-9)
            it.sumOf {
                it.observation.depthMeters!! * it.weight
            } / denominator
        }

        val rawConfidence = weightedAverage(robustCandidates) {
            it.observation.confidence
        }
        val calibrationEvidenceAvailable =
            robustCandidates.any { it.observation.calibrationEvidenceAvailable }
        val calibrationConfidence =
            aggregateCalibrationConfidence(robustCandidates)
        // A 2D landmark can still be tracked without metric calibration, but its confidence
        // must be reduced and its calibration state must remain explicitly unknown.
        val calibrationConfidenceForWeight =
            if (calibrationEvidenceAvailable) calibrationConfidence else 0.65
        val opticalConfidence =
            weightedAverage(robustCandidates) {
                (
                    1.0 -
                        0.72 *
                            it.observation.motionBlurRisk
                    ).coerceIn(0.22, 1.0)
            }
        val confidence =
            (
                rawConfidence *
                    calibrationConfidenceForWeight *
                    (
                        0.78 +
                            0.22 *
                                opticalConfidence
                    )
                ).coerceIn(0.0, 1.0)

        val visibility = weightedAverage(robustCandidates) {
            it.observation.visibility
        }

        val contactProbability = weightedAverage(robustCandidates) {
            it.observation.contactProbability
        }

        val motionBoundaryConfidence =
            weightedAverage(robustCandidates) {
                it.observation.motionBoundaryConfidence
            }

        val appearanceConsistency =
            weightedAverage(robustCandidates) {
                it.observation.appearanceConsistency
            }

        val depthConfidence =
            weightedAverage(robustCandidates) {
                it.observation.depthConfidence
            }

        val viewSelectionScore =
            weightedAverage(robustCandidates) {
                it.observation.viewSelectionScore
            }

        val temporalInnovationScore =
            weightedAverage(robustCandidates) {
                it.temporalInnovationScore
            }

        val uncertainty = uncertainty(
            candidates = robustCandidates,
            fusedX = x,
            fusedY = y,
            fusedDepth = depth,
            timestampNanos = timestampNanos,
        )

        val crossViewAgreement = crossViewAgreement(
            candidates = robustCandidates,
            fusedX = x,
            fusedY = y,
            fusedDepth = depth,
        )
        val supportingCameraCount =
            robustCandidates.map { it.observation.cameraId }.distinct().size

        val previousPrediction = predictPosition(
            timestampNanos,
            state,
        )

        val predictionDistance = previousPrediction?.let {
            sqrt(
                square(x - it.first) +
                    square(y - it.second),
            )
        } ?: 0.0

        val temporalStability = temporalStability(
            timestampNanos = timestampNanos,
            x = x,
            y = y,
            depthMeters = depth,
            state = state,
            supportViews = robustCandidates.size,
        )

        val contactState = updateContactState(
            state = state,
            timestampNanos = timestampNanos,
            probability = contactProbability,
        )

        state.updateFromObservation(
            timestampNanos = timestampNanos,
            x = x,
            y = y,
            depth = depth,
            confidence = confidence,
            uncertainty = uncertainty,
            visibility = visibility,
            contactProbability = contactProbability,
        )

        robustCandidates.forEach {
            cameraState(it.observation.cameraId).acceptedObservations++
        }

        val diagnostics = buildList {
            if (robustCandidates.size < candidates.size) {
                add(
                    "robust consensus removed " +
                        (candidates.size - robustCandidates.size) +
                        " geometrically inconsistent view(s)",
                )
            }
            if (visibility < 0.50) add(
                "landmark visibility is degraded",
            )
            if (motionBoundaryConfidence < 0.45) add(
                "motion boundary evidence is degraded",
            )
            if (appearanceConsistency < 0.45) add(
                "appearance consistency is degraded",
            )
            if (depthConfidence < 0.45) add(
                "depth confidence is degraded",
            )
            if (temporalInnovationScore < 0.45) add(
                "landmark innovation diverges from causal temporal prediction",
            )
            if (uncertainty > 0.35) add(
                "landmark uncertainty is high",
            )
            if (!calibrationEvidenceAvailable) {
                add("camera calibration confidence is unknown for this landmark")
            } else if (calibrationConfidence < 0.55) {
                add("camera calibration uncertainty limits dense landmark authority")
            }
            if (
                crossViewAgreement < 0.50 &&
                robustCandidates.size >= 2
            ) {
                add("cross-view landmark agreement is degraded")
            }
            if (predictionDistance > 0.08) add(
                "landmark motion diverges from temporal prediction",
            )
            if (contactProbability >= config.contactEnterProbability) {
                add("model contact evidence is active")
            }
            if (duplicateCount > 0) {
                add(
                    "duplicate camera evidence suppressed: " +
                        duplicateCount,
                )
            }
        }

        val stateBand = when {
            confidence < config.minimumFusedConfidence ->
                EvidenceState.LOW_CONFIDENCE
            visibility < 0.30 ->
                EvidenceState.OCCLUDED
            else ->
                EvidenceState.OBSERVED
        }

        val sourceDiversity =
            robustCandidates.map {
                it.observation.sourceTag
            }.toSet().size

        return FusedLandmark(
            key = key,
            timestampNanos = timestampNanos,
            normalizedX = x.coerceIn(-4.0, 4.0),
            normalizedY = y.coerceIn(-4.0, 4.0),
            depthMeters = depth?.takeIf {
                it.isFinite() && it > 0.0
            },
            confidence = confidence,
            uncertainty = uncertainty,
            visibility = visibility,
            contactProbability = contactProbability,
            contactState = contactState,
            crossViewAgreement = crossViewAgreement,
            supportingCameraCount = supportingCameraCount,
            temporalStability = temporalStability,
            acceptedViewCount = robustCandidates.size,
            rejectedViewCount =
                max(
                    0,
                    observations.size - robustCandidates.size,
                ) + duplicateCount,
            occludedViewCount = candidates.count {
                it.observation.visibility < 0.35
            },
            state = stateBand,
            sourceDiversity = sourceDiversity,
            weightedReprojectionResidualNormalized =
                weightedAverage(robustCandidates) {
                    it.observation.reprojectionResidualNormalized
                },
            weightedDepthResidualMeters =
                weightedAverage(robustCandidates) {
                    abs(it.observation.depthResidualMeters)
                },
            predictedDistanceNormalized = predictionDistance,
            motionBoundaryConfidence = motionBoundaryConfidence,
            appearanceConsistency = appearanceConsistency,
            depthConfidence = depthConfidence,
            viewSelectionScore = viewSelectionScore,
            temporalInnovationScore = temporalInnovationScore,
            probabilitySigmaXNormalized =
                weightedAverage(robustCandidates) {
                    it.observation.probabilitySigmaXNormalized
                }.coerceIn(0.0, 0.25),
            probabilitySigmaYNormalized =
                weightedAverage(robustCandidates) {
                    it.observation.probabilitySigmaYNormalized
                }.coerceIn(0.0, 0.25),
            probabilityCalibrationScore =
                weightedAverage(robustCandidates) {
                    it.observation.probabilityCalibrationScore
                }.coerceIn(0.0, 1.0),
            diagnostics = diagnostics.distinct(),
            segmentationConsistency =
                robustCandidates
                    .filter { it.observation.segmentationEvidenceAvailable }
                    .let { measured ->
                        if (measured.isEmpty()) {
                            0.0
                        } else {
                            weightedAverage(measured) {
                                it.observation.segmentationConsistency
                            }.coerceIn(0.0, 1.0)
                        }
                    },
            segmentationEvidenceAvailable =
                robustCandidates.any { it.observation.segmentationEvidenceAvailable },
            calibrationEvidenceAvailable = calibrationEvidenceAvailable,
        )
    }

    private fun predictOccluded(
        timestampNanos: Long,
        state: LandmarkState,
    ): FusedLandmark? {
        if (state.observedTimestampNanos == Long.MIN_VALUE) {
            return null
        }

        val ageMillis =
            (timestampNanos - state.observedTimestampNanos)
                .coerceAtLeast(0L)
                .toDouble() / 1_000_000.0

        if (ageMillis > config.occlusionPredictionHorizonMillis) {
            state.contactState = ContactState.OCCLUDED
            return null
        }

        val predicted = predictPosition(
            timestampNanos,
            state,
        ) ?: return null

        val predictedDepth =
            predictDepth(
                timestampNanos,
                state,
            )

        val decay = temporalFreshness(ageMillis) * 0.78
        val uncertaintyGrowth =
            (
                state.uncertainty +
                    (
                        ageMillis /
                            config.occlusionPredictionHorizonMillis
                                .coerceAtLeast(1.0)
                        ) * 0.22
                ).coerceIn(0.0, 1.0)

        state.predictedFrames++
        state.contactState = ContactState.OCCLUDED

        return FusedLandmark(
            key = state.key,
            timestampNanos = timestampNanos,
            normalizedX = predicted.first,
            normalizedY = predicted.second,
            depthMeters =
                predictedDepth ?: state.z,
            confidence = (
                state.confidence * decay
            ).coerceIn(0.0, 1.0),
            uncertainty = uncertaintyGrowth,
            visibility = (
                state.visibility * decay
            ).coerceIn(0.0, 1.0),
            contactProbability = (
                state.contactProbability * decay
            ).coerceIn(0.0, 1.0),
            contactState = ContactState.OCCLUDED,
            crossViewAgreement = 0.0,
            supportingCameraCount = 0,
            temporalStability = (
                state.confidence * decay
            ).coerceIn(0.0, 1.0),
            acceptedViewCount = 0,
            rejectedViewCount = 0,
            occludedViewCount = 1,
            state = EvidenceState.TEMPORALLY_PREDICTED,
            sourceDiversity = 0,
            weightedReprojectionResidualNormalized = 0.0,
            weightedDepthResidualMeters = 0.0,
            predictedDistanceNormalized = 0.0,
            motionBoundaryConfidence = 0.0,
            appearanceConsistency = 0.0,
            depthConfidence = 0.0,
            viewSelectionScore = 0.0,
            temporalInnovationScore = 1.0,
            probabilitySigmaXNormalized =
                sqrt(
                    state.uncertainty
                ).coerceIn(1e-4, 0.25),
            probabilitySigmaYNormalized =
                sqrt(
                    state.uncertainty
                ).coerceIn(1e-4, 0.25),
            probabilityCalibrationScore =
                (
                    1.0 -
                        uncertaintyGrowth
                ).coerceIn(0.0, 1.0),
            diagnostics = listOf(
                "short occlusion bridged by bounded temporal prediction",
            ),
        )
    }

    private fun rejectedLandmark(
        timestampNanos: Long,
        key: LandmarkKey,
    ): FusedLandmark =
        FusedLandmark(
            key = key,
            timestampNanos = timestampNanos,
            normalizedX = 0.0,
            normalizedY = 0.0,
            depthMeters = null,
            confidence = 0.0,
            uncertainty = 1.0,
            visibility = 0.0,
            contactProbability = 0.0,
            contactState = ContactState.UNKNOWN,
            crossViewAgreement = 0.0,
            supportingCameraCount = 0,
            temporalStability = 0.0,
            acceptedViewCount = 0,
            rejectedViewCount = 1,
            occludedViewCount = 0,
            state = EvidenceState.REJECTED,
            sourceDiversity = 0,
            weightedReprojectionResidualNormalized = 1.0,
            weightedDepthResidualMeters = config.hardDepthResidualMeters,
            predictedDistanceNormalized = 0.0,
            motionBoundaryConfidence = 0.0,
            appearanceConsistency = 0.0,
            depthConfidence = 0.0,
            viewSelectionScore = 0.0,
            temporalInnovationScore = 0.0,
            probabilitySigmaXNormalized = 0.25,
            probabilitySigmaYNormalized = 0.25,
            probabilityCalibrationScore = 0.0,
            diagnostics = listOf(
                "no trustworthy landmark evidence survived gating",
            ),
        )

    private fun summarize(
        timestamp: Long,
        landmarks: List<FusedLandmark>,
        frameAccepted: Long,
        frameRejected: Long,
        frameStale: Long,
        frameNonFinite: Long,
        frameDuplicates: Long,
    ): FrameResult {
        if (landmarks.isEmpty()) {
            return emptyResult(
                timestamp = timestamp,
                warning = "No dense landmark evidence survived fusion.",
                hardFailure = frameRejected > 0L ||
                    frameStale > 0L ||
                    frameNonFinite > 0L,
            ).copy(
                acceptedObservations = frameAccepted,
                rejectedObservations = frameRejected,
                staleObservations = frameStale,
                nonFiniteObservations = frameNonFinite,
                duplicateCameraObservations = frameDuplicates,
            )
        }

        val observed = landmarks.count {
            it.state == EvidenceState.OBSERVED
        }
        val visible = landmarks.count {
            it.visibility >= 0.35 &&
                it.state != EvidenceState.REJECTED
        }
        val predicted = landmarks.count {
            it.state == EvidenceState.TEMPORALLY_PREDICTED
        }
        val occluded = landmarks.count {
            it.state == EvidenceState.OCCLUDED ||
                it.contactState == ContactState.OCCLUDED ||
                it.occludedViewCount > 0
        }

        val coverage =
            observed.toDouble() /
                landmarks.size.toDouble().coerceAtLeast(1.0)

        val visibleCoverage =
            visible.toDouble() /
                landmarks.size.toDouble().coerceAtLeast(1.0)

        val confidenceValues =
            landmarks.map { it.confidence.coerceIn(0.0, 1.0) }

        val uncertaintyValues =
            landmarks.map { it.uncertainty.coerceIn(0.0, 1.0) }

        val meanConfidence = confidenceValues.average()
        val uncertaintyP50 = percentile(
            uncertaintyValues,
            0.50,
        )
        val uncertaintyP95 = percentile(
            uncertaintyValues,
            0.95,
        )

        val multiView = landmarks.filter {
            it.acceptedViewCount >= config.minimumViewCountForCrossView
        }

        val crossViewEvidenceAvailable = multiView.isNotEmpty()
        val supportingCameraCount =
            multiView.maxOfOrNull { it.supportingCameraCount } ?: 0

        val crossViewAgreement =
            multiView.map { it.crossViewAgreement }
                .takeIf { it.isNotEmpty() }
                ?.average()
                ?: 0.0

        val temporalStability = landmarks.map {
            it.temporalStability
        }.average().coerceIn(0.0, 1.0)

        val contactCoverage =
            landmarks.count {
                it.contactProbability >= 0.50 &&
                    it.contactState != ContactState.UNKNOWN
            }.toDouble() /
                landmarks.size.toDouble().coerceAtLeast(1.0)

        val geometricConfidence =
            (
                crossViewAgreement * 0.42 +
                    (1.0 - uncertaintyP95) * 0.34 +
                    temporalStability * 0.24
                ).coerceIn(0.0, 1.0)

        val viewDiversity =
            min(1.0, supportingCameraCount.toDouble() / 6.0)

        val motionBoundaryConfidence =
            landmarks.map { it.motionBoundaryConfidence }
                .average().coerceIn(0.0, 1.0)
        val appearanceConsistency =
            landmarks.map { it.appearanceConsistency }
                .average().coerceIn(0.0, 1.0)
        val depthConfidence =
            landmarks.map { it.depthConfidence }
                .average().coerceIn(0.0, 1.0)

        val innovationValues =
            landmarks.map {
                (1.0 - it.temporalInnovationScore)
                    .coerceIn(0.0, 1.0)
            }
        val temporalInnovationP95 =
            percentile(innovationValues, 0.95)

        val predictionBridgeRatio =
            if (occluded <= 0) {
                0.0
            } else {
                predicted.toDouble() / occluded.toDouble()
            }.coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (coverage < 0.50) add(
                "dense landmark coverage is below the authoritative threshold",
            )
            if (visibleCoverage < 0.50) add(
                "visible dense landmark coverage is below the caution threshold",
            )
            if (uncertaintyP95 > 0.35) add(
                "dense landmark uncertainty P95 is high",
            )
            if (
                multiView.isNotEmpty() &&
                crossViewAgreement < 0.55
            ) {
                add(
                    "cross-view landmark agreement is degraded",
                )
            }
            if (temporalStability < 0.55) add(
                "dense landmark temporal stability is degraded",
            )
            if (geometricConfidence < 0.55) add(
                "geometric confidence is below the high-end evidence threshold",
            )
            if (temporalInnovationP95 > 0.55) add(
                "temporal innovation P95 is elevated",
            )
            if (frameStale > 0L) add(
                "stale landmark observations were rejected",
            )
            if (frameNonFinite > 0L) add(
                "non-finite landmark observations were rejected",
            )
            if (frameDuplicates > 0L) add(
                "duplicate same-camera landmark evidence was suppressed",
            )
        }

        val hardFailure =
            (coverage < 0.20 && visibleCoverage < 0.20) ||
                uncertaintyP95 >= 0.95 ||
                (
                    frameNonFinite > 0L &&
                        frameAccepted == 0L
                    )

        val diagnostics = listOf(
            "landmarks=" + landmarks.size,
            "acceptedViews=" + frameAccepted,
            "rejectedViews=" + frameRejected,
            "coverage=" + metric(coverage),
            "visibleCoverage=" + metric(visibleCoverage),
            "meanConfidence=" + metric(meanConfidence),
            "uncertaintyP95=" + metric(uncertaintyP95),
            "crossViewAgreement=" + metric(crossViewAgreement),
            "temporalStability=" + metric(temporalStability),
            "contactCoverage=" + metric(contactCoverage),
            "predictionBridgeRatio=" + metric(predictionBridgeRatio),
        )

        return FrameResult(
            timestampNanos = timestamp,
            landmarks = landmarks,
            coverage = coverage,
            visibleCoverage = visibleCoverage,
            meanConfidence = meanConfidence,
            uncertaintyP50 = uncertaintyP50,
            uncertaintyP95 = uncertaintyP95,
            crossViewAgreement = crossViewAgreement,
            supportingCameraCount = supportingCameraCount,
            crossViewEvidenceAvailable = crossViewEvidenceAvailable,
            temporalStability = temporalStability,
            contactCoverage = contactCoverage,
            predictionBridgeRatio = predictionBridgeRatio,
            acceptedObservations = frameAccepted,
            rejectedObservations = frameRejected,
            staleObservations = frameStale,
            nonFiniteObservations = frameNonFinite,
            duplicateCameraObservations = frameDuplicates,
            warningCount = warnings.size,
            hardFailure = hardFailure,
            warnings = warnings,
            diagnostics = diagnostics,
            geometricConfidence = geometricConfidence,
            viewDiversity = viewDiversity,
            motionBoundaryConfidence = motionBoundaryConfidence,
            appearanceConsistency = appearanceConsistency,
            depthConfidence = depthConfidence,
            temporalInnovationP95 = temporalInnovationP95,
        )
    }

    private fun temporalStability(
        timestampNanos: Long,
        x: Double,
        y: Double,
        depthMeters: Double?,
        state: LandmarkState,
        supportViews: Int,
    ): Double {
        if (state.timestampNanos == Long.MIN_VALUE) {
            return (
                0.58 +
                    min(0.30, supportViews.toDouble() * 0.07)
                ).coerceIn(0.0, 1.0)
        }

        val dt =
            (timestampNanos - state.timestampNanos)
                .coerceAtLeast(1L)
                .toDouble() / 1_000_000_000.0

        if (!dt.isFinite() || dt > 0.5) return 0.30

        val predictedX =
            state.x +
                state.vx * dt +
                0.5 * state.ax * dt * dt

        val predictedY =
            state.y +
                state.vy * dt +
                0.5 * state.ay * dt * dt

        val xyInnovation =
            sqrt(
                square(x - predictedX) +
                    square(y - predictedY),
            )

        val depthInnovation =
            if (
                depthMeters != null &&
                    state.z != null
            ) {
                val predictedDepth =
                    state.z!! +
                        state.vz * dt +
                        0.5 *
                            state.az *
                            dt *
                            dt
                abs(
                    depthMeters -
                        predictedDepth,
                ).coerceAtMost(
                    config.hardDepthResidualMeters *
                        2.0,
                )
            } else {
                0.0
            }

        val innovation =
            xyInnovation +
                if (
                    depthMeters != null &&
                        state.z != null
                ) {
                    depthInnovation * 0.45
                } else {
                    0.0
                }

        val innovationScore =
            exp(
                -innovation /
                    config.robustResidualScale.coerceAtLeast(1e-6),
            ).coerceIn(0.0, 1.0)

        val supportScore =
            min(1.0, supportViews.toDouble() / 4.0)

        return (
            innovationScore * 0.72 +
                supportScore * 0.28
            ).coerceIn(0.0, 1.0)
    }

    private fun uncertainty(
        candidates: List<WeightedObservation>,
        fusedX: Double,
        fusedY: Double,
        fusedDepth: Double?,
        timestampNanos: Long,
    ): Double {
        val totalWeight =
            candidates.sumOf { it.weight }.coerceAtLeast(1e-9)

        var normalizedVariance = 0.0
        var residual = 0.0
        var inputUncertainty = 0.0
        var oldestEvidencePenalty = 0.0

        for (candidate in candidates) {
            val observation = candidate.observation
            val dx = observation.normalizedX - fusedX
            val dy = observation.normalizedY - fusedY
            val xyTerm =
                square(dx) + square(dy)

            val depthTerm =
                if (
                    fusedDepth != null &&
                    observation.depthMeters != null
                ) {
                    square(
                        (
                            observation.depthMeters - fusedDepth
                        ) / config.hardDepthResidualMeters,
                    )
                } else {
                    0.0
                }

            normalizedVariance += (
                xyTerm +
                    depthTerm * 0.40
                ) * candidate.weight

            residual += (
                observation.reprojectionResidualNormalized /
                    config.hardResidualNormalized
            ).coerceIn(0.0, 1.0) * candidate.weight

            inputUncertainty +=
                observation.uncertainty.coerceIn(0.0, 1.0) *
                    candidate.weight

            val ageMillis =
                (timestampNanos - observation.timestampNanos)
                    .coerceAtLeast(0L)
                    .toDouble() / 1_000_000.0

            oldestEvidencePenalty = max(
                oldestEvidencePenalty,
                1.0 - temporalFreshness(ageMillis),
            )
        }

        val spatial =
            sqrt(
                normalizedVariance /
                    totalWeight.coerceAtLeast(1e-9),
            ).coerceIn(0.0, 1.0)

        val residualQualityLoss =
            (residual / totalWeight).coerceIn(0.0, 1.0)

        val sourceUncertainty =
            (inputUncertainty / totalWeight).coerceIn(0.0, 1.0)

        return (
            spatial * 0.40 +
                residualQualityLoss * 0.24 +
                sourceUncertainty * 0.28 +
                oldestEvidencePenalty * 0.08
            ).coerceIn(0.0, 1.0)
    }

    private fun crossViewAgreement(
        candidates: List<WeightedObservation>,
        fusedX: Double,
        fusedY: Double,
        fusedDepth: Double?,
    ): Double {
        if (candidates.size < config.minimumViewCountForCrossView) {
            return 0.0
        }

        val residuals = candidates.map {
            val xyResidual = sqrt(
                square(it.observation.normalizedX - fusedX) +
                    square(it.observation.normalizedY - fusedY),
            )
            val depthResidual =
                if (
                    fusedDepth != null &&
                    it.observation.depthMeters != null
                ) {
                    (
                        abs(
                            it.observation.depthMeters - fusedDepth,
                        ) / config.hardDepthResidualMeters
                    ).coerceIn(0.0, 1.0) * 0.15
                } else {
                    0.0
                }
            xyResidual + depthResidual
        }

        val p50 = percentile(residuals, 0.50)
        val p95 = percentile(residuals, 0.95)

        val p50Score = exp(
            -p50 /
                config.robustResidualScale.coerceAtLeast(1e-6),
        ).coerceIn(0.0, 1.0)

        val p95Score = exp(
            -p95 /
                (config.robustResidualScale * 2.8),
        ).coerceIn(0.0, 1.0)

        val supportScore =
            min(1.0, candidates.size.toDouble() / 6.0)

        val imageAgreement = (
            p50Score * 0.48 +
                p95Score * 0.34 +
                supportScore * 0.18
            ).coerceIn(0.0, 1.0)

        val rays = candidates.mapNotNull { candidate ->
            val origin = candidate.observation.worldRayOriginMeters
            val direction = candidate.observation.worldRayDirection
            if (origin == null || direction == null) {
                null
            } else {
                Pair(
                    Pair(origin, direction.normalized()),
                    candidate.weight,
                )
            }
        }

        if (rays.size < 2) {
            return imageAgreement
        }

        var weightedDistance = 0.0
        var totalWeight = 0.0
        var validPairs = 0

        for (i in 0 until rays.lastIndex) {
            for (j in i + 1 until rays.size) {
                val first = rays[i]
                val second = rays[j]
                val originA = first.first.first
                val directionA = first.first.second
                val originB = second.first.first
                val directionB = second.first.second

                val cosine = directionA.dot(directionB)
                    .coerceIn(-1.0, 1.0)
                val angleDegrees = Math.toDegrees(acos(cosine))
                if (
                    !angleDegrees.isFinite() ||
                    angleDegrees < config.minimumRayAngleDegrees
                ) {
                    continue
                }

                val distance = symmetricRayDistance(
                    originA = originA,
                    directionA = directionA,
                    originB = originB,
                    directionB = directionB,
                )
                if (!distance.isFinite()) continue

                val pairWeight =
                    first.second.coerceAtLeast(1e-6) *
                        second.second.coerceAtLeast(1e-6)

                weightedDistance += distance * pairWeight
                totalWeight += pairWeight
                validPairs++
            }
        }

        if (validPairs == 0 || totalWeight <= 1e-9) {
            return 0.0
        }

        val rayDistance = weightedDistance / totalWeight
        val rayAgreement = (
            1.0 -
                rayDistance /
                config.maximumSymmetricRayDistanceMeters
            ).coerceIn(0.0, 1.0)

        val rayCalibrationConfidence =
            calibrationConfidenceForRays(rays)

        val rayWeight =
            config.calibratedRayAgreementWeight *
                rayCalibrationConfidence
        val imageWeight =
            (1.0 - config.calibratedRayAgreementWeight)
                .coerceAtLeast(0.0)

        return (
            imageAgreement * imageWeight +
                rayAgreement * rayWeight +
                imageAgreement * (
                    config.calibratedRayAgreementWeight -
                        rayWeight
                )
            ).coerceIn(0.0, 1.0)
    }

    private fun calibrationConfidenceForRays(
        rays: List<WeightedObservation>,
    ): Double {
        val measuredRays =
            rays.filter { it.observation.calibrationEvidenceAvailable }
        if (measuredRays.isEmpty()) return 0.0
        val confidences =
            measuredRays.map {
                (
                    it.observation.calibrationConfidence *
                        (
                            1.0 /
                                (
                                    1.0 +
                                        it.observation.calibrationUncertaintyMeters /
                                            config.calibrationTranslationSigmaScaleMeters
                                )
                        )
                    ).coerceIn(0.0, 1.0)
            }
        return confidences.averageOrZero().coerceIn(0.0, 1.0)
    }

    private fun aggregateCalibrationConfidence(
        observations: List<WeightedObservation>,
    ): Double {
        val measured =
            observations.filter { it.observation.calibrationEvidenceAvailable }
        if (measured.isEmpty()) return 0.0
        return measured.map {
            it.observation.calibrationConfidence
        }.averageOrZero().coerceIn(0.0, 1.0)
    }

    private fun symmetricRayDistance(
        originA: Vec3,
        directionA: Vec3,
        originB: Vec3,
        directionB: Vec3,
    ): Double {
        val a = directionA.normalized()
        val b = directionB.normalized()
        val w0 = originA - originB
        val ab = a.dot(b)
        val denominator = 1.0 - ab * ab

        if (!denominator.isFinite() || denominator < 1e-8) {
            return w0.cross(a).norm()
        }

        val d = a.dot(w0)
        val e = b.dot(w0)
        val s = (ab * e - d) / denominator
        val t = (e - ab * d) / denominator
        val pointA = originA + a * s
        val pointB = originB + b * t
        return pointA.distanceTo(pointB)
    }

    private fun updateContactState(
        state: LandmarkState,
        timestampNanos: Long,
        probability: Double,
    ): ContactState {
        if (
            state.timestampNanos != Long.MIN_VALUE &&
            timestampNanos < state.timestampNanos
        ) {
            state.contactState = ContactState.UNKNOWN
            state.contactCandidateFrames = 0
            state.separationFrames = 0
            return state.contactState
        }

        val p = probability.coerceIn(0.0, 1.0)

        when (state.contactState) {
            ContactState.UNKNOWN,
            ContactState.FREE -> {
                if (p >= config.contactEnterProbability) {
                    state.contactCandidateFrames++
                    state.contactState =
                        if (
                            state.contactCandidateFrames >=
                            config.contactHysteresisFrames
                        ) {
                            ContactState.CONTACT
                        } else {
                            ContactState.APPROACHING
                        }
                } else {
                    state.contactCandidateFrames = 0
                    state.contactState = ContactState.FREE
                }
            }

            ContactState.APPROACHING -> {
                if (p >= config.contactEnterProbability) {
                    state.contactCandidateFrames++
                    if (
                        state.contactCandidateFrames >=
                        config.contactHysteresisFrames
                    ) {
                        state.contactState = ContactState.CONTACT
                    }
                } else if (p <= config.contactExitProbability) {
                    state.contactCandidateFrames = 0
                    state.contactState = ContactState.FREE
                }
            }

            ContactState.CONTACT -> {
                if (p <= config.contactExitProbability) {
                    state.separationFrames++
                    state.contactState =
                        if (
                            state.separationFrames >=
                            config.contactHysteresisFrames
                        ) {
                            ContactState.SEPARATING
                        } else {
                            ContactState.CONTACT
                        }
                } else {
                    state.separationFrames = 0
                }
            }

            ContactState.SEPARATING -> {
                if (p >= config.contactEnterProbability) {
                    state.contactCandidateFrames++
                    if (
                        state.contactCandidateFrames >=
                        config.contactHysteresisFrames
                    ) {
                        state.contactState = ContactState.CONTACT
                        state.separationFrames = 0
                    }
                } else if (p <= config.contactExitProbability) {
                    state.contactState = ContactState.FREE
                }
            }

            ContactState.OCCLUDED -> {
                if (p >= config.contactEnterProbability) {
                    state.contactCandidateFrames++
                    if (
                        state.contactCandidateFrames >=
                        config.contactHysteresisFrames
                    ) {
                        state.contactState = ContactState.CONTACT
                    }
                } else if (p <= config.contactExitProbability) {
                    state.contactState = ContactState.FREE
                }
            }
        }

        return state.contactState
    }

    private fun predictDepth(
        timestampNanos: Long,
        state: LandmarkState,
    ): Double? {
        val z = state.z ?: return null

        val dt =
            (
                timestampNanos -
                    state.timestampNanos
            ).coerceAtLeast(0L)
                .toDouble() /
                1_000_000_000.0

        if (!dt.isFinite()) return null

        val limitedDt =
            min(
                dt,
                config.occlusionPredictionHorizonMillis /
                    1000.0,
            )

        val speed =
            state.vz.coerceIn(
                -config.maximumPredictionDepthSpeedMetersPerSecond,
                config.maximumPredictionDepthSpeedMetersPerSecond,
            )

        val acceleration =
            state.az.coerceIn(
                -config.maximumPredictionDepthAccelerationMetersPerSecond2,
                config.maximumPredictionDepthAccelerationMetersPerSecond2,
            )

        return (
            z +
                speed * limitedDt +
                0.5 *
                    acceleration *
                    limitedDt *
                    limitedDt
            ).coerceAtLeast(0.01)
    }

    private fun predictPosition(
        timestampNanos: Long,
        state: LandmarkState,
    ): Pair<Double, Double>? {
        if (state.timestampNanos == Long.MIN_VALUE) return null

        val dt =
            (timestampNanos - state.timestampNanos)
                .coerceAtLeast(0L)
                .toDouble() / 1_000_000_000.0

        if (!dt.isFinite()) return null

        val horizonSeconds =
            config.occlusionPredictionHorizonMillis / 1000.0

        val limitedDt = min(dt, horizonSeconds)

        val speed = sqrt(
            square(state.vx) +
                square(state.vy),
        ).coerceAtLeast(1e-9)

        val speedScale = min(
            1.0,
            config.maximumPredictionSpeedNormalizedPerSecond / speed,
        )

        val acceleration = sqrt(
            square(state.ax) +
                square(state.ay),
        ).coerceAtLeast(1e-9)

        val accelerationScale = min(
            1.0,
            config.maximumPredictionAccelerationNormalizedPerSecond2 /
                acceleration,
        )

        val vx = state.vx * speedScale
        val vy = state.vy * speedScale
        val ax = state.ax * accelerationScale
        val ay = state.ay * accelerationScale

        return (
            state.x +
                vx * limitedDt +
                0.5 * ax * limitedDt * limitedDt
            ) to (
            state.y +
                vy * limitedDt +
                0.5 * ay * limitedDt * limitedDt
            )
    }

    private fun LandmarkState.updateFromObservation(
        timestampNanos: Long,
        x: Double,
        y: Double,
        depth: Double?,
        confidence: Double,
        uncertainty: Double,
        visibility: Double,
        contactProbability: Double,
    ) {
        val dt =
            if (this.timestampNanos == Long.MIN_VALUE) {
                0.0
            } else {
                (timestampNanos - this.timestampNanos)
                    .coerceAtLeast(1L)
                    .toDouble() / 1_000_000_000.0
            }

        if (dt.isFinite() && dt > 1e-6) {
                val nextVx =
                    (x - this.x) / dt
                val nextVy =
                    (y - this.y) / dt

                val nextAx =
                    (nextVx - this.vx) / dt
                val nextAy =
                    (nextVy - this.vy) / dt

                this.ax =
                    this.ax * 0.65 +
                        nextAx.coerceIn(-60.0, 60.0) * 0.35

                this.ay =
                    this.ay * 0.65 +
                        nextAy.coerceIn(-60.0, 60.0) * 0.35

                this.vx =
                    this.vx * 0.55 +
                        nextVx.coerceIn(-8.0, 8.0) * 0.45

                this.vy =
                    this.vy * 0.55 +
                        nextVy.coerceIn(-8.0, 8.0) * 0.45
            }
        // Keep subsequent coordinate and depth updates in the same observation transition.
        this.x = x
        this.y = y
        if (depth != null) {
            if (this.z != null && dt.isFinite() && dt > 1e-6) {
                val previousVz = this.vz
                val nextVz =
                    (
                        depth -
                            this.z!!
                    ) / dt
                this.vz =
                    nextVz.coerceIn(
                        -config.maximumPredictionDepthSpeedMetersPerSecond,
                        config.maximumPredictionDepthSpeedMetersPerSecond,
                    )
                this.az =
                    (
                        (this.vz - previousVz) / dt
                    ).coerceIn(
                        -config.maximumPredictionDepthAccelerationMetersPerSecond2,
                        config.maximumPredictionDepthAccelerationMetersPerSecond2,
                    )
            }
            this.z = depth.coerceAtLeast(0.01)
        }
        this.timestampNanos = timestampNanos
        this.observedTimestampNanos = timestampNanos
        this.confidence =
            this.confidence * 0.36 +
                confidence * 0.64
        this.uncertainty =
            this.uncertainty * 0.38 +
                uncertainty * 0.62
        this.visibility =
            this.visibility * 0.32 +
                visibility * 0.68
        this.contactProbability =
            this.contactProbability * 0.65 +
                contactProbability * 0.35
        this.observedFrames++
    }

    private fun prune(timestampNanos: Long) {
        val landmarkCutoff =
            timestampNanos -
                (
                    config.staleObservationMillis *
                        4.0 *
                        1_000_000.0
                    ).toLong().coerceAtLeast(0L)

        val landmarkIterator = landmarkStates.entries.iterator()
        while (landmarkIterator.hasNext()) {
            val state = landmarkIterator.next().value
            if (
                state.observedTimestampNanos != Long.MIN_VALUE &&
                state.observedTimestampNanos < landmarkCutoff
            ) {
                landmarkIterator.remove()
            }
        }

        while (landmarkStates.size > config.maximumStoredLandmarks) {
            val key = landmarkStates.entries.minByOrNull {
                it.value.observedTimestampNanos
                    .takeIf { value -> value != Long.MIN_VALUE }
                    ?: Long.MIN_VALUE
            }?.key ?: break
            landmarkStates.remove(key)
        }

        val cameraCutoff =
            timestampNanos -
                (
                    config.staleCameraMillis *
                        1_000_000.0
                    ).toLong().coerceAtLeast(0L)

        val cameraIterator = cameraStates.entries.iterator()
        while (cameraIterator.hasNext()) {
            val state = cameraIterator.next().value
            if (
                state.lastTimestampNanos != Long.MIN_VALUE &&
                state.lastTimestampNanos < cameraCutoff
            ) {
                cameraIterator.remove()
            }
        }

        while (cameraStates.size > config.maximumCameraStates) {
            val cameraId = cameraStates.entries.minByOrNull {
                it.value.lastTimestampNanos
                    .takeIf { value -> value != Long.MIN_VALUE }
                    ?: Long.MIN_VALUE
            }?.key ?: break
            cameraStates.remove(cameraId)
        }
    }

    private fun sanitize(
        observation: CameraObservation,
    ): CameraObservation =
        observation.copy(
            normalizedX = observation.normalizedX.coerceIn(-4.0, 4.0),
            normalizedY = observation.normalizedY.coerceIn(-4.0, 4.0),
            depthMeters = observation.depthMeters?.takeIf {
                it.isFinite() && it > 0.0 && it < 1000.0
            },
            confidence = observation.confidence.coerceIn(0.0, 1.0),
            uncertainty = observation.uncertainty.coerceIn(0.0, 1.0),
            visibility = observation.visibility.coerceIn(0.0, 1.0),
            contactProbability =
                observation.contactProbability.coerceIn(0.0, 1.0),
            cameraQuality =
                observation.cameraQuality.coerceIn(0.0, 1.0),
            reprojectionResidualNormalized =
                observation.reprojectionResidualNormalized
                    .coerceAtLeast(0.0),
            depthResidualMeters =
                observation.depthResidualMeters
                    .takeIf { it.isFinite() }
                    ?: config.hardDepthResidualMeters,
            segmentationConsistency =
                observation.segmentationConsistency
                    .coerceIn(0.0, 1.0),
        )

    private fun cameraState(cameraId: Int): CameraState =
        cameraStates.getOrPut(cameraId) {
            if (cameraStates.size >= config.maximumCameras) {
                val oldest = cameraStates.entries.minByOrNull {
                    it.value.lastTimestampNanos
                        .takeIf { value -> value != Long.MIN_VALUE }
                        ?: Long.MIN_VALUE
                }?.key
                if (oldest != null) cameraStates.remove(oldest)
            }
            CameraState(cameraId)
        }

    private fun temporalFreshness(ageMillis: Double): Double {
        if (!ageMillis.isFinite()) return 0.0
        if (ageMillis <= 0.0) return 1.0
        return exp(
            -ln(2.0) *
                ageMillis /
                config.temporalHalfLifeMillis,
        ).coerceIn(0.0, 1.0)
    }

    private fun residualQuality(residual: Double): Double =
        exp(
            -residual.coerceAtLeast(0.0) /
                config.robustResidualScale,
        ).coerceIn(0.0, 1.0)

    private fun huberWeight(
        residual: Double,
        scale: Double,
    ): Double {
        val normalized =
            abs(residual) / scale.coerceAtLeast(1e-9)
        return if (normalized <= 1.0) {
            1.0
        } else {
            1.0 / normalized
        }.coerceIn(0.0, 1.0)
    }

    private fun robustMedian(values: List<Double>): Double {
        val sorted = values.filter { it.isFinite() }.sorted()
        if (sorted.isEmpty()) return 0.0
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }

    private fun weightedAverage(
        observations: List<WeightedObservation>,
        selector: (WeightedObservation) -> Double,
    ): Double {
        var numerator = 0.0
        var denominator = 0.0

        for (observation in observations) {
            val value = selector(observation)
            if (!value.isFinite()) continue
            numerator += value * observation.weight
            denominator += observation.weight
        }

        return if (denominator <= 1e-9) {
            0.0
        } else {
            (
                numerator / denominator
            ).coerceIn(-10.0, 10.0)
        }
    }

    private fun percentile(
        values: List<Double>,
        quantile: Double,
    ): Double {
        val sorted = values.filter { it.isFinite() }.sorted()
        if (sorted.isEmpty()) return 1.0
        if (sorted.size == 1) return sorted.first()
        val position =
            quantile.coerceIn(0.0, 1.0) *
                (sorted.size - 1).toDouble()
        val lower = position.toInt()
        val upper = min(sorted.lastIndex, lower + 1)
        val fraction = position - lower
        return (
            sorted[lower] +
                (sorted[upper] - sorted[lower]) * fraction
            ).coerceAtLeast(0.0)
    }

    private fun square(value: Double): Double =
        value * value

    private fun metric(value: Double): String =
        "%.4f".format(
            value.coerceIn(-99.0, 99.0),
        )

    private fun appendDiagnostic(message: String) {
        diagnosticHistory.addLast(message)
        while (diagnosticHistory.size > config.diagnosticHistoryLimit) {
            diagnosticHistory.removeFirst()
        }
    }

    private fun emptyResult(
        timestamp: Long,
        warning: String,
        hardFailure: Boolean,
    ): FrameResult =
        FrameResult(
            timestampNanos = timestamp,
            landmarks = emptyList(),
            coverage = 0.0,
            visibleCoverage = 0.0,
            meanConfidence = 0.0,
            uncertaintyP50 = 1.0,
            uncertaintyP95 = 1.0,
            crossViewAgreement = 0.0,
            supportingCameraCount = 0,
            crossViewEvidenceAvailable = false,
            temporalStability = 0.0,
            contactCoverage = 0.0,
            predictionBridgeRatio = 0.0,
            acceptedObservations = 0L,
            rejectedObservations = 0L,
            staleObservations = 0L,
            nonFiniteObservations = 0L,
            duplicateCameraObservations = 0L,
            warningCount = 1,
            hardFailure = hardFailure,
            warnings = listOf(warning),
            diagnostics = listOf(
                "dense landmark fusion produced no trusted evidence",
            ),
        )
}

package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

/**
 * Reliability-aware face/eye fusion engine.
 *
 * This engine sits above the individual face, dense-face, binocular-gaze and temporal engines.
 * Its job is not to reconstruct another face or another gaze vector. Its job is to determine which
 * already-reconstructed signals are trustworthy enough to influence the face/eye state.
 *
 * The design deliberately separates:
 *  - face geometry quality,
 *  - dense face surface quality,
 *  - binocular eye geometry quality,
 *  - eye temporal stability,
 *  - uncertainty,
 *  - rapid gaze dynamics,
 *  - persistent modality failure.
 *
 * That separation prevents one broken stream from contaminating a healthy stream.
 *
 * No synthetic measurements are created here. Missing evidence contributes zero evidence and is
 * reported explicitly. Prediction-only or quarantined evidence never teaches the reliability prior.
 */
class FaceEyeEvidenceFusionEngine(
    private var config: Config = Config(),
) {

    data class Config(
        val faceGeometryWeight: Double = 0.34,
        val faceTemporalWeight: Double = 0.30,
        val denseMeshWeight: Double = 0.20,
        val faceShapeWeight: Double = 0.10,
        val faceHistoryWeight: Double = 0.16,
        val leftEyeWeight: Double = 0.16,
        val rightEyeWeight: Double = 0.16,
        val binocularWeight: Double = 0.28,
        val eyeTemporalWeight: Double = 0.20,
        val eyeHistoryWeight: Double = 0.20,
        val uncertaintyWeight: Double = 0.12,
        val angularAgreementWeight: Double = 0.10,
        val parallaxWeight: Double = 0.06,
        val saccadeWeight: Double = 0.04,
        val gazeAccelerationWeight: Double = 0.04,
        val blinkWeight: Double = 0.04,
        val faceEyeAgreementWeight: Double = 0.08,
        val faceFreshnessWeight: Double = 0.10,
        val faceReprojectionWeight: Double = 0.08,
        val captureLightingWeight: Double = 0.06,
        val captureScaleWeight: Double = 0.05,
        val gazeStabilityWeight: Double = 0.08,
        val gazeAgreementWeight: Double = 0.10,
        val faceShapeRejectThreshold: Double = 0.38,
        val faceShapeWarningThreshold: Double = 0.62,
        val denseMeshMissingScore: Double = 0.72,
        val gazeAccelerationComfortDegreesPerSecond2: Double = 3200.0,
        val gazeAccelerationHardDegreesPerSecond2: Double = 9000.0,
        val staleFaceThreshold: Double = 0.45,
        val hardStaleFaceThreshold: Double = 0.18,
        val faceReprojectionHardErrorPx: Double = 8.0,
        val uncertaintyScaleMeters: Double = 0.06,
        val angularSpreadWarningDegrees: Double = 6.0,
        val angularSpreadRejectDegrees: Double = 12.0,
        val minimumParallaxDegrees: Double = 0.60,
        val strongParallaxDegrees: Double = 2.0,
        val saccadeComfortDegreesPerSecond: Double = 420.0,
        val saccadeHardDegreesPerSecond: Double = 900.0,
        val binocularRejectThreshold: Double = 0.42,
        val minimumEyeEvidence: Double = 0.18,
        val minimumFaceEvidence: Double = 0.18,
        val rejectFusedConfidence: Double = 0.48,
        val quarantineAfterBadFrames: Int = 3,
        val recoveryFrames: Int = 4,
        val recoveryEyeConfidence: Double = 0.68,
        val recoveryBinocularConfidence: Double = 0.64,
        val historyAlpha: Double = 0.20,
        val eyeQuarantineScale: Double = 0.16,
        val faceQuarantineScale: Double = 0.24,
        val ipdCalibrationWarmupSamples: Int = 8,
        val minimumIpdCalibrationEyeConfidence: Double = 0.68,
        val minimumIpdCalibrationBinocularConfidence: Double = 0.70,
        val minimumIpdCalibrationParallaxDegrees: Double = 0.50,
        val maximumIpdCalibrationAngularSpreadDegrees: Double = 7.0,
        val maximumIpdDeviationMeters: Double = 0.012,
        val ipdConsistencyWeight: Double = 0.08,
    ) {
        init {
            require(faceGeometryWeight >= 0.0)
            require(faceTemporalWeight >= 0.0)
            require(denseMeshWeight >= 0.0)
            require(faceShapeWeight >= 0.0)
            require(faceHistoryWeight >= 0.0)
            require(leftEyeWeight >= 0.0)
            require(rightEyeWeight >= 0.0)
            require(binocularWeight >= 0.0)
            require(eyeTemporalWeight >= 0.0)
            require(eyeHistoryWeight >= 0.0)
            require(uncertaintyWeight >= 0.0)
            require(angularAgreementWeight >= 0.0)
            require(parallaxWeight >= 0.0)
            require(saccadeWeight >= 0.0)
            require(gazeAccelerationWeight >= 0.0)
            require(blinkWeight >= 0.0)
            require(faceEyeAgreementWeight >= 0.0)
            require(faceFreshnessWeight >= 0.0)
            require(faceReprojectionWeight >= 0.0)
            require(captureLightingWeight >= 0.0)
            require(captureScaleWeight >= 0.0)
            require(gazeStabilityWeight >= 0.0)
            require(gazeAgreementWeight >= 0.0)
            require(faceShapeRejectThreshold in 0.0..faceShapeWarningThreshold)
            require(faceShapeWarningThreshold in 0.0..1.0)
            require(denseMeshMissingScore in 0.0..1.0)
            require(gazeAccelerationComfortDegreesPerSecond2 > 0.0)
            require(gazeAccelerationHardDegreesPerSecond2 >
                gazeAccelerationComfortDegreesPerSecond2)
            require(staleFaceThreshold in 0.0..1.0)
            require(hardStaleFaceThreshold in 0.0..staleFaceThreshold)
            require(faceReprojectionHardErrorPx > 0.0)
            require(uncertaintyScaleMeters > 0.0)
            require(angularSpreadWarningDegrees >= 0.0)
            require(angularSpreadRejectDegrees >
                angularSpreadWarningDegrees)
            require(minimumParallaxDegrees >= 0.0)
            require(strongParallaxDegrees >= minimumParallaxDegrees)
            require(saccadeComfortDegreesPerSecond > 0.0)
            require(saccadeHardDegreesPerSecond >
                saccadeComfortDegreesPerSecond)
            require(binocularRejectThreshold in 0.0..1.0)
            require(minimumEyeEvidence in 0.0..1.0)
            require(minimumFaceEvidence in 0.0..1.0)
            require(rejectFusedConfidence in 0.0..1.0)
            require(quarantineAfterBadFrames >= 1)
            require(recoveryFrames >= 1)
            require(recoveryEyeConfidence in 0.0..1.0)
            require(recoveryBinocularConfidence in 0.0..1.0)
            require(historyAlpha in 0.0..1.0)
            require(eyeQuarantineScale in 0.0..1.0)
            require(faceQuarantineScale in 0.0..1.0)
            require(ipdCalibrationWarmupSamples >= 2)
            require(minimumIpdCalibrationEyeConfidence in 0.0..1.0)
            require(minimumIpdCalibrationBinocularConfidence in 0.0..1.0)
            require(minimumIpdCalibrationParallaxDegrees >= 0.0)
            require(maximumIpdCalibrationAngularSpreadDegrees > 0.0)
            require(maximumIpdDeviationMeters > 0.0)
            require(ipdConsistencyWeight >= 0.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val faceGeometryConfidence: Double,
        val faceTemporalConfidence: Double,
        val denseMeshConfidence: Double,
        val faceShapeConfidence: Double = 0.85,
        val faceShapeAccepted: Boolean = true,
        val denseMeshObserved: Boolean = true,
        val facePointCount: Int = 468,
        val faceUncertaintyMeters: Double,
        val leftEyeConfidence: Double,
        val rightEyeConfidence: Double,
        val binocularConfidence: Double,
        val eyeTemporalConfidence: Double,
        val eyeTemporalAccepted: Boolean = true,
        val binocularAccepted: Boolean = true,
        val eyeUncertaintyMeters: Double,
        val angularSpreadDegrees: Double,
        val parallaxDegrees: Double,
        val saccadeVelocityDegreesPerSecond: Double,
        val gazeAccelerationDegreesPerSecond2: Double = 0.0,
        val blinkProbability: Double,
        val faceFresh: Boolean = true,
        val eyesFresh: Boolean = true,
        val faceOutOfDistributionRisk: Double = 0.0,
        val eyeOutOfDistributionRisk: Double = 0.0,
        val predictionOnlyFace: Boolean = false,
        val predictionOnlyEyes: Boolean = false,
        val faceFreshness: Double = 1.0,
        val faceReprojectionScore: Double = 1.0,
        val gazeStability: Double = 1.0,
        val gazeAgreement: Double = 1.0,
        val interpupillaryDistanceMeters: Double = 0.0,
        val lightingConfidence: Double = 1.0,
        val faceScaleConfidence: Double = 1.0,
    ) {
        init {
            require(timestampNanos > 0L)
            listOf(
                faceGeometryConfidence,
                faceTemporalConfidence,
                denseMeshConfidence,
                faceShapeConfidence,
                leftEyeConfidence,
                rightEyeConfidence,
                binocularConfidence,
                eyeTemporalConfidence,
                blinkProbability,
                faceOutOfDistributionRisk,
                eyeOutOfDistributionRisk,
                faceFreshness,
                faceReprojectionScore,
                gazeStability,
                gazeAgreement,
                lightingConfidence,
                faceScaleConfidence,
            ).forEach {
                require(it in 0.0..1.0)
            }
            require(faceUncertaintyMeters.isFinite() &&
                faceUncertaintyMeters >= 0.0)
            require(eyeUncertaintyMeters.isFinite() &&
                eyeUncertaintyMeters >= 0.0)
            require(angularSpreadDegrees.isFinite() &&
                angularSpreadDegrees >= 0.0)
            require(parallaxDegrees.isFinite() &&
                parallaxDegrees >= 0.0)
            require(saccadeVelocityDegreesPerSecond.isFinite() &&
                saccadeVelocityDegreesPerSecond >= 0.0)
            require(gazeAccelerationDegreesPerSecond2.isFinite() &&
                gazeAccelerationDegreesPerSecond2 >= 0.0)
            require(facePointCount >= 0)
            require(
                interpupillaryDistanceMeters.isFinite() &&
                    interpupillaryDistanceMeters >= 0.0
            )
        }
    }

    enum class FailureDomain {
        FACE_GEOMETRY,
        FACE_TEMPORAL,
        FACE_SURFACE,
        FACE_CALIBRATION,
        FACE_OOD,
        EYE_BINOCULAR,
        EYE_TEMPORAL,
        EYE_UNCERTAINTY,
        EYE_DYNAMICS,
        EYE_OOD,
        IPD_CALIBRATION,
        TEMPORAL_ORDERING,
        CAPTURE_DOMAIN,
    }

    data class FailureAttribution(
        val domain: FailureDomain,
        val severity: Double,
        val evidence: Double,
        val blocking: Boolean,
        val rationale: String,
    ) {
        init {
            require(severity in 0.0..1.0)
            require(evidence in 0.0..1.0)
            require(rationale.isNotBlank())
        }
    }

    enum class FusionMode {
        BALANCED,
        FACE_DOMINANT,
        EYE_DOMINANT,
        FACE_ONLY,
        EYE_ONLY,
        HOLD,
    }

    data class Result(
        val timestampNanos: Long,
        val faceConfidence: Double,
        val eyeConfidence: Double,
        val fusedConfidence: Double,
        val faceReliability: Double,
        val eyeReliability: Double,
        val faceEyeAgreement: Double,
        val eyeQuarantined: Boolean,
        val faceQuarantined: Boolean,
        val accepted: Boolean,
        val fusionMode: FusionMode = FusionMode.HOLD,
        val independentEvidenceSupport: Double = 0.0,
        val faceReliabilityTrend: Double = 0.0,
        val eyeReliabilityTrend: Double = 0.0,
        val selectedSources: List<String>,
        val warnings: List<String>,
        val recommendations: List<String>,
        val faceFreshnessScore: Double = 1.0,
        val faceReprojectionScore: Double = 1.0,
        val gazeStabilityScore: Double = 1.0,
        val gazeAgreementScore: Double = 1.0,
        val geometricIntegrityScore: Double = 1.0,
        val ipdConsistencyScore: Double = 1.0,
        val faceOutOfDistributionRisk: Double = 0.0,
        val eyeOutOfDistributionRisk: Double = 0.0,
        val faceShapeConfidence: Double = 1.0,
        val eyeDynamicConsistency: Double = 1.0,
        val faceHardFailure: Boolean = false,
        val eyeHardFailure: Boolean = false,
        val hardFailureReasons: List<String> = emptyList(),
        val failureAttribution: List<FailureAttribution> = emptyList(),
        val faceUncertaintyCalibrationMultiplier: Double = 1.0,
        val eyeUncertaintyCalibrationMultiplier: Double = 1.0,
        val faceUncertaintyCoverage: Double = 0.0,
        val eyeUncertaintyCoverage: Double = 0.0,
        val uncertaintyCalibrationConfidence: Double = 0.0,
        val uncertaintyCalibrationReady: Boolean = false,
        val captureDomainConfidence: Double = 1.0,
        val lightingConfidence: Double = 1.0,
        val faceScaleConfidence: Double = 1.0,
    ) {
        val dominantFailure: FailureAttribution?
            get() =
                failureAttribution.maxByOrNull {
                    it.severity
                }
    }

    private data class State(
        var faceReliability: Double = 0.72,
        var eyeReliability: Double = 0.72,
        var eyeBadFrames: Int = 0,
        var eyeGoodFrames: Int = 0,
        var faceBadFrames: Int = 0,
        var faceGoodFrames: Int = 0,
        var eyeQuarantined: Boolean = false,
        var faceQuarantined: Boolean = false,
        var lastTimestampNanos: Long = Long.MIN_VALUE,
        var ipdBaselineMeters: Double = 0.0,
        var ipdSamples: Int = 0,
        var previousFaceReliability: Double = 0.72,
        var previousEyeReliability: Double = 0.72,
        var faceReliabilityTrend: Double = 0.0,
        var eyeReliabilityTrend: Double = 0.0,
    )

    private val state = State()
    private val uncertaintyCalibrationEngine =
        FaceEyeUncertaintyCalibrationEngine()

    @Synchronized
    fun configure(newConfig: Config) {
        if (newConfig == config) return
        config = newConfig
        state.faceReliability =
            state.faceReliability.coerceIn(0.02, 1.0)
        state.eyeReliability =
            state.eyeReliability.coerceIn(0.02, 1.0)
    }

    fun configuration(): Config = config

    @Synchronized
    fun configureUncertaintyCalibration(
        newConfig: FaceEyeUncertaintyCalibrationEngine.Config,
    ) {
        uncertaintyCalibrationEngine.configure(newConfig)
    }

    fun uncertaintyCalibrationSnapshot():
        FaceEyeUncertaintyCalibrationEngine.Snapshot =
        uncertaintyCalibrationEngine.snapshot()

    fun reliabilitySnapshot(): Snapshot =
        Snapshot(
            faceReliability = state.faceReliability,
            eyeReliability = state.eyeReliability,
            eyeQuarantined = state.eyeQuarantined,
            faceQuarantined = state.faceQuarantined,
            faceBadFrames = state.faceBadFrames,
            eyeBadFrames = state.eyeBadFrames,
            ipdBaselineMeters = state.ipdBaselineMeters,
            ipdSamples = state.ipdSamples,
        )

    data class Snapshot(
        val faceReliability: Double,
        val eyeReliability: Double,
        val eyeQuarantined: Boolean,
        val faceQuarantined: Boolean,
        val faceBadFrames: Int,
        val eyeBadFrames: Int,
        val ipdBaselineMeters: Double,
        val ipdSamples: Int,
    )

    @Synchronized
    fun fuse(input: Input): Result {
        if (
            state.lastTimestampNanos != Long.MIN_VALUE &&
            input.timestampNanos <= state.lastTimestampNanos
        ) {
            val previousTimestamp =
                state.lastTimestampNanos
            val result =
                fuseMonotonicFallback(
                    timestampNanos = previousTimestamp,
                )
            return result
        }

        val uncertaintyCalibration =
            uncertaintyCalibrationEngine.evaluate(
                FaceEyeUncertaintyCalibrationEngine.Input(
                    timestampNanos = input.timestampNanos,
                    faceConfidence = input.faceGeometryConfidence,
                    faceTemporalConfidence = input.faceTemporalConfidence,
                    faceReprojectionScore = input.faceReprojectionScore,
                    faceShapeConfidence = input.faceShapeConfidence,
                    faceFreshness = input.faceFreshness,
                    faceUncertaintyMeters = input.faceUncertaintyMeters,
                    leftEyeConfidence = input.leftEyeConfidence,
                    rightEyeConfidence = input.rightEyeConfidence,
                    binocularConfidence = input.binocularConfidence,
                    eyeTemporalConfidence = input.eyeTemporalConfidence,
                    gazeAgreement = input.gazeAgreement,
                    angularSpreadDegrees = input.angularSpreadDegrees,
                    angularSpreadRejectDegrees =
                        config.angularSpreadRejectDegrees,
                    eyeUncertaintyMeters = input.eyeUncertaintyMeters,
                    ipdConsistencyScore = ipdConsistency(input),
                    faceOodRisk = input.faceOutOfDistributionRisk,
                    eyeOodRisk = input.eyeOutOfDistributionRisk,
                    faceAccepted =
                        input.faceShapeAccepted &&
                            input.faceFresh &&
                            input.faceReprojectionScore > 0.12,
                    eyeAccepted =
                        input.eyeTemporalAccepted &&
                            input.binocularAccepted,
                    predictionOnlyFace = input.predictionOnlyFace,
                    predictionOnlyEyes = input.predictionOnlyEyes,
                    faceFresh = input.faceFresh,
                    eyesFresh = input.eyesFresh,
                ),
            )

        val calibratedInput =
            input.copy(
                faceUncertaintyMeters =
                    uncertaintyCalibration.calibratedFaceUncertaintyMeters,
                eyeUncertaintyMeters =
                    uncertaintyCalibration.calibratedEyeUncertaintyMeters,
            )

        val previousFaceReliability =
            state.faceReliability.coerceIn(0.0, 1.0)
        val previousEyeReliability =
            state.eyeReliability.coerceIn(0.0, 1.0)

        val faceRaw =
            faceRawConfidence(calibratedInput)

        val eyeRaw =
            eyeRawConfidence(calibratedInput)

        val faceShapeScore =
            faceShapeQuality(calibratedInput)

        val eyeDynamicScore =
            eyeDynamicConsistency(calibratedInput)

        val faceReliability =
            temporalReliability(
                raw = (
                    faceRaw * 0.90 +
                        faceShapeScore * config.faceShapeWeight * 0.10
                    ).coerceIn(0.0, 1.0),
                prior = previousFaceReliability,
                quarantined = state.faceQuarantined,
                quarantineScale = config.faceQuarantineScale,
            )

        val eyeReliability =
            temporalReliability(
                raw = (
                    eyeRaw * 0.92 +
                        eyeDynamicScore *
                            config.gazeAccelerationWeight * 0.08
                    ).coerceIn(0.0, 1.0),
                prior = previousEyeReliability,
                quarantined = state.eyeQuarantined,
                quarantineScale = config.eyeQuarantineScale,
            )

        val faceEyeAgreement =
            faceEyeAgreement(
                input = calibratedInput,
            )

        val captureDomainConfidence =
            (
                calibratedInput.lightingConfidence * 0.58 +
                    calibratedInput.faceScaleConfidence * 0.42
                ).coerceIn(0.0, 1.0)

        val ipdConsistencyScore =
            ipdConsistency(input)
        val geometricIntegrityScore =
            geometricIntegrity(
                input = calibratedInput,
                ipdConsistencyScore = ipdConsistencyScore,
            )

        val faceConfidence =
            (
                faceReliability * 0.76 +
                    faceEyeAgreement * 0.10 +
                    calibratedInput.faceFreshness * config.faceFreshnessWeight * 0.70 +
                    calibratedInput.faceReprojectionScore *
                        config.faceReprojectionWeight * 0.50 +
                    geometricIntegrityScore * 0.11 +
                    ipdConsistencyScore * config.ipdConsistencyWeight
                ).coerceIn(0.0, 1.0)

        val rawEyeConfidence =
            (
                eyeReliability * 0.70 +
                    faceEyeAgreement * 0.07 +
                    binocularQuality(input) * 0.11 +
                    calibratedInput.gazeStability * config.gazeStabilityWeight * 0.60 +
                    calibratedInput.gazeAgreement * config.gazeAgreementWeight * 0.60 +
                    geometricIntegrityScore * 0.09 +
                    ipdConsistencyScore * config.ipdConsistencyWeight
                ).coerceIn(0.0, 1.0)

        val eyeConfidence =
            if (
                calibratedInput.binocularConfidence <
                    config.binocularRejectThreshold
            ) {
                minOf(rawEyeConfidence, 0.54)
            } else {
                rawEyeConfidence
            }

        val independentEvidenceSupport =
            (
                1.0 -
                    (1.0 - faceReliability) *
                    (1.0 - eyeReliability)
                ).coerceIn(0.0, 1.0)

        val faceReliabilityTrend =
            (
                faceReliability -
                    state.previousFaceReliability
                ).coerceIn(-1.0, 1.0)

        val eyeReliabilityTrend =
            (
                eyeReliability -
                    state.previousEyeReliability
                ).coerceIn(-1.0, 1.0)

        val trendPenalty =
            (
                maxOf(0.0, -faceReliabilityTrend) * 0.07 +
                    maxOf(0.0, -eyeReliabilityTrend) * 0.07
                ).coerceIn(0.0, 0.14)

        val fusedConfidence =
            (
                faceConfidence * 0.38 +
                    eyeConfidence * 0.36 +
                    faceEyeAgreement * 0.07 +
                    geometricIntegrityScore * 0.06 +
                    independentEvidenceSupport * 0.08 +
                    captureDomainConfidence * 0.06 +
                    ipdConsistencyScore * config.ipdConsistencyWeight -
                    trendPenalty
                ).coerceIn(0.0, 1.0)

        val eyeHardFailure =
            calibratedInput.binocularConfidence <
                config.binocularRejectThreshold ||
                calibratedInput.angularSpreadDegrees >=
                    config.angularSpreadRejectDegrees ||
                !calibratedInput.eyeTemporalAccepted ||
                !calibratedInput.binocularAccepted

        val faceHardFailure =
            !calibratedInput.faceFresh ||
                calibratedInput.faceFreshness <=
                    config.hardStaleFaceThreshold ||
                calibratedInput.faceReprojectionScore <= 0.12 ||
                !calibratedInput.faceShapeAccepted ||
                faceShapeScore <= config.faceShapeRejectThreshold ||
                calibratedInput.facePointCount < 14

        val meaningfulFace =
            faceConfidence >= config.minimumFaceEvidence &&
                !calibratedInput.predictionOnlyFace

        val meaningfulEyes =
            eyeConfidence >= config.minimumEyeEvidence &&
                !calibratedInput.predictionOnlyEyes &&
                calibratedInput.eyeTemporalAccepted &&
                calibratedInput.binocularAccepted &&
                !state.eyeQuarantined

        val blockingBinocularFailure =
            calibratedInput.binocularConfidence <
                config.binocularRejectThreshold ||
                calibratedInput.angularSpreadDegrees >=
                    config.angularSpreadRejectDegrees

        val severeFaceCaptureFailure =
            calibratedInput.faceFreshness <=
                config.hardStaleFaceThreshold ||
                calibratedInput.faceReprojectionScore <= 0.12

        val angularDisagreementFailure =
            calibratedInput.angularSpreadDegrees >=
                config.angularSpreadRejectDegrees

        val strictBinocularDisagreement =
            calibratedInput.binocularConfidence <
                config.binocularRejectThreshold &&
                calibratedInput.leftEyeConfidence >= 0.70 &&
                calibratedInput.rightEyeConfidence >= 0.70 &&
                calibratedInput.eyeTemporalAccepted &&
                calibratedInput.binocularAccepted

        val accepted =
            fusedConfidence >= config.rejectFusedConfidence &&
                (meaningfulFace || meaningfulEyes) &&
                (!eyeHardFailure || meaningfulFace) &&
                (!faceHardFailure || meaningfulEyes) &&
                !severeFaceCaptureFailure &&
                !blockingBinocularFailure &&
                !angularDisagreementFailure &&
                !strictBinocularDisagreement &&
                calibratedInput.faceOutOfDistributionRisk < 0.96 &&
                calibratedInput.eyeOutOfDistributionRisk < 0.96

        val fusionMode =
            selectFusionMode(
                faceConfidence = faceConfidence,
                eyeConfidence = eyeConfidence,
                independentEvidenceSupport =
                    independentEvidenceSupport,
                faceHardFailure = faceHardFailure,
                eyeHardFailure = eyeHardFailure,
                accepted = accepted,
            )

        val selectedSources =
            buildSelectedSources(
                input = calibratedInput,
                faceConfidence = faceConfidence,
                eyeConfidence = eyeConfidence,
            )

        val warnings =
            (
                buildWarnings(
                    input = calibratedInput,
                    faceConfidence = faceConfidence,
                    eyeConfidence = eyeConfidence,
                    fusedConfidence = fusedConfidence,
                    eyeHardFailure = eyeHardFailure,
                    faceHardFailure = faceHardFailure,
                ) +
                    outOfDistributionWarnings(calibratedInput) +
                    uncertaintyCalibration.warnings +
                    if (!uncertaintyCalibration.ready) {
                        listOf("face-eye-uncertainty-calibration-warmup")
                    } else {
                        buildList {
                            if (
                                calibratedInput.faceGeometryConfidence < 0.35 ||
                                    calibratedInput.faceTemporalConfidence < 0.35 ||
                                    calibratedInput.leftEyeConfidence < 0.35 ||
                                    calibratedInput.rightEyeConfidence < 0.35 ||
                                    calibratedInput.faceReprojectionScore < 0.35
                            ) {
                                add("face-eye-calibration-evidence-low-confidence")
                            }
                        }
                    }
                ).distinct()

        val recommendations =
            buildRecommendations(
                input = calibratedInput,
                faceConfidence = faceConfidence,
                eyeConfidence = eyeConfidence,
                fusedConfidence = fusedConfidence,
                eyeHardFailure = eyeHardFailure,
                faceHardFailure = faceHardFailure,
            )

        updateReliability(
            input = calibratedInput,
            faceRaw = faceRaw,
            eyeRaw = eyeRaw,
            eyeHardFailure = eyeHardFailure,
        )

        state.previousFaceReliability =
            faceReliability
        state.previousEyeReliability =
            eyeReliability
        state.faceReliabilityTrend =
            faceReliabilityTrend
        state.eyeReliabilityTrend =
            eyeReliabilityTrend

        state.lastTimestampNanos =
            calibratedInput.timestampNanos

        return Result(
            timestampNanos = calibratedInput.timestampNanos,
            faceConfidence = faceConfidence,
            eyeConfidence = eyeConfidence,
            fusedConfidence = fusedConfidence,
            faceReliability =
                state.faceReliability.coerceIn(0.0, 1.0),
            eyeReliability =
                state.eyeReliability.coerceIn(0.0, 1.0),
            faceEyeAgreement = faceEyeAgreement,
            eyeQuarantined = state.eyeQuarantined,
            faceQuarantined = state.faceQuarantined,
            accepted = accepted,
            fusionMode = fusionMode,
            independentEvidenceSupport =
                independentEvidenceSupport,
            faceReliabilityTrend =
                faceReliabilityTrend,
            eyeReliabilityTrend =
                eyeReliabilityTrend,
            selectedSources = selectedSources,
            warnings = warnings,
            recommendations = recommendations,
            faceUncertaintyCalibrationMultiplier =
                uncertaintyCalibration.face.multiplier,
            eyeUncertaintyCalibrationMultiplier =
                uncertaintyCalibration.eye.multiplier,
            faceUncertaintyCoverage =
                uncertaintyCalibration.face.empiricalCoverage,
            eyeUncertaintyCoverage =
                uncertaintyCalibration.eye.empiricalCoverage,
            uncertaintyCalibrationConfidence =
                uncertaintyCalibration.calibrationConfidence,
            uncertaintyCalibrationReady =
                uncertaintyCalibration.ready,
            captureDomainConfidence =
                captureDomainConfidence,
            lightingConfidence =
                calibratedInput.lightingConfidence,
            faceScaleConfidence =
                calibratedInput.faceScaleConfidence,
            faceFreshnessScore = calibratedInput.faceFreshness,
            faceReprojectionScore = calibratedInput.faceReprojectionScore,
            gazeStabilityScore = calibratedInput.gazeStability,
            gazeAgreementScore = calibratedInput.gazeAgreement,
            geometricIntegrityScore = geometricIntegrityScore,
            ipdConsistencyScore = ipdConsistencyScore,
            faceOutOfDistributionRisk =
                calibratedInput.faceOutOfDistributionRisk,
            eyeOutOfDistributionRisk =
                calibratedInput.eyeOutOfDistributionRisk,
            faceShapeConfidence = faceShapeScore,
            eyeDynamicConsistency = eyeDynamicScore,
            faceHardFailure = faceHardFailure,
            eyeHardFailure = eyeHardFailure,
            hardFailureReasons =
                hardFailureReasons(
                    input = input,
                    faceHardFailure = faceHardFailure,
                    eyeHardFailure = eyeHardFailure,
                ),
            failureAttribution =
                failureAttribution(
                    input = input,
                    faceConfidence = faceConfidence,
                    eyeConfidence = eyeConfidence,
                    fusedConfidence = fusedConfidence,
                    faceShapeScore = faceShapeScore,
                    eyeDynamicScore = eyeDynamicScore,
                    faceEyeAgreement = faceEyeAgreement,
                    ipdConsistencyScore = ipdConsistencyScore,
                    geometricIntegrityScore = geometricIntegrityScore,
                    faceHardFailure = faceHardFailure,
                    eyeHardFailure = eyeHardFailure,
                ),
        )
    }

    private fun fuseMonotonicFallback(
        timestampNanos: Long,
    ): Result =
        Result(
            timestampNanos = timestampNanos,
            faceConfidence = state.faceReliability,
            eyeConfidence = state.eyeReliability,
            fusedConfidence = (
                state.faceReliability * 0.50 +
                    state.eyeReliability * 0.50
                ).coerceIn(0.0, 1.0),
            faceReliability = state.faceReliability,
            eyeReliability = state.eyeReliability,
            faceEyeAgreement = (
                state.faceReliability +
                    state.eyeReliability
                ).coerceIn(0.0, 1.0) * 0.50,
            eyeQuarantined = state.eyeQuarantined,
            faceQuarantined = state.faceQuarantined,
            accepted =
                !state.eyeQuarantined &&
                    !state.faceQuarantined,
            selectedSources = emptyList(),
            warnings = listOf(
                "face-eye-duplicate-or-out-of-order-timestamp",
            ),
            recommendations = listOf(
                "Retain the latest valid face/eye state and discard duplicate telemetry before continuing.",
            ),
            faceFreshnessScore = 0.0,
            faceReprojectionScore = 0.0,
            gazeStabilityScore = 0.0,
            gazeAgreementScore = 0.0,
            geometricIntegrityScore = 0.0,
            ipdConsistencyScore = 0.0,
            faceOutOfDistributionRisk = 0.0,
            eyeOutOfDistributionRisk = 0.0,
            faceShapeConfidence = 0.0,
            eyeDynamicConsistency = 0.0,
            faceHardFailure = false,
            eyeHardFailure = false,
            hardFailureReasons = emptyList(),
            failureAttribution = listOf(
                FailureAttribution(
                    domain = FailureDomain.TEMPORAL_ORDERING,
                    severity = 1.0,
                    evidence = 1.0,
                    blocking = true,
                    rationale =
                        "Input timestamp is not newer than the latest accepted face/eye sample.",
                ),
            ),
        )

    @Synchronized
    fun reset() {
        state.faceReliability = 0.72
        state.eyeReliability = 0.72
        state.eyeBadFrames = 0
        state.eyeGoodFrames = 0
        state.faceBadFrames = 0
        state.faceGoodFrames = 0
        state.eyeQuarantined = false
        state.faceQuarantined = false
        state.lastTimestampNanos = Long.MIN_VALUE
        state.ipdBaselineMeters = 0.0
        state.ipdSamples = 0
        state.previousFaceReliability = 0.72
        state.previousEyeReliability = 0.72
        state.faceReliabilityTrend = 0.0
        state.eyeReliabilityTrend = 0.0
        uncertaintyCalibrationEngine.reset()
    }

    private fun failureAttribution(
        input: Input,
        faceConfidence: Double,
        eyeConfidence: Double,
        fusedConfidence: Double,
        faceShapeScore: Double,
        eyeDynamicScore: Double,
        faceEyeAgreement: Double,
        ipdConsistencyScore: Double,
        geometricIntegrityScore: Double,
        faceHardFailure: Boolean,
        eyeHardFailure: Boolean,
    ): List<FailureAttribution> {
        val items = mutableListOf<FailureAttribution>()

        fun add(
            domain: FailureDomain,
            severity: Double,
            evidence: Double,
            blocking: Boolean,
            rationale: String,
        ) {
            val normalizedSeverity =
                severity.coerceIn(0.0, 1.0)
            if (normalizedSeverity < 0.08) return
            items += FailureAttribution(
                domain = domain,
                severity = normalizedSeverity,
                evidence = evidence.coerceIn(0.0, 1.0),
                blocking = blocking,
                rationale = rationale,
            )
        }

        add(
            domain = FailureDomain.CAPTURE_DOMAIN,
            severity =
                1.0 -
                    (
                        input.lightingConfidence * 0.58 +
                            input.faceScaleConfidence * 0.42
                        ).coerceIn(0.0, 1.0),
            evidence =
                (
                    input.lightingConfidence * 0.58 +
                        input.faceScaleConfidence * 0.42
                    ).coerceIn(0.0, 1.0),
            blocking =
                input.lightingConfidence < 0.18 ||
                    input.faceScaleConfidence < 0.18,
            rationale =
                "Current lighting or subject scale places the face/eye stack outside its preferred capture domain.",
        )

        add(
            domain = FailureDomain.FACE_GEOMETRY,
            severity =
                (
                    1.0 -
                        (
                            input.faceGeometryConfidence * 0.62 +
                                input.faceReprojectionScore * 0.38
                            )
                    ).coerceIn(0.0, 1.0),
            evidence = faceConfidence,
            blocking = faceHardFailure,
            rationale =
                "Face landmark geometry or reprojection quality is limiting the fused state.",
        )

        add(
            domain = FailureDomain.FACE_TEMPORAL,
            severity =
                1.0 -
                    input.faceTemporalConfidence
                        .coerceIn(0.0, 1.0),
            evidence = input.faceTemporalConfidence,
            blocking = input.faceFreshness <=
                config.hardStaleFaceThreshold,
            rationale =
                "Face temporal continuity or freshness is weaker than the production target.",
        )

        add(
            domain = FailureDomain.FACE_SURFACE,
            severity =
                (
                    1.0 -
                        (
                            faceShapeScore * 0.56 +
                                input.denseMeshConfidence * 0.44
                            )
                    ).coerceIn(0.0, 1.0),
            evidence = faceShapeScore,
            blocking =
                !input.faceShapeAccepted ||
                    faceShapeScore <=
                    config.faceShapeRejectThreshold,
            rationale =
                "Dense face surface or shape alignment evidence is insufficient.",
        )

        add(
            domain = FailureDomain.FACE_OOD,
            severity =
                input.faceOutOfDistributionRisk
                    .coerceIn(0.0, 1.0),
            evidence =
                1.0 -
                    input.faceOutOfDistributionRisk
                        .coerceIn(0.0, 1.0),
            blocking = input.faceOutOfDistributionRisk >= 0.92,
            rationale =
                "Face evidence appears outside the calibrated operating distribution.",
        )

        add(
            domain = FailureDomain.EYE_BINOCULAR,
            severity =
                (
                    1.0 -
                        (
                            input.binocularConfidence * 0.58 +
                                input.gazeAgreement * 0.22 +
                                (
                                    1.0 -
                                        input.angularSpreadDegrees /
                                            config.angularSpreadRejectDegrees
                                    ).coerceIn(0.0, 1.0) * 0.20
                            )
                    ).coerceIn(0.0, 1.0),
            evidence = input.binocularConfidence,
            blocking = eyeHardFailure &&
                (
                    !input.binocularAccepted ||
                        input.binocularConfidence <
                        config.binocularRejectThreshold
                    ),
            rationale =
                "Binocular agreement, angular consistency, or geometry acceptance is limiting gaze reliability.",
        )

        add(
            domain = FailureDomain.EYE_TEMPORAL,
            severity =
                1.0 -
                    input.eyeTemporalConfidence
                        .coerceIn(0.0, 1.0),
            evidence = input.eyeTemporalConfidence,
            blocking = !input.eyeTemporalAccepted,
            rationale =
                "Eye temporal continuity is limiting confidence in the current gaze trajectory.",
        )

        add(
            domain = FailureDomain.EYE_UNCERTAINTY,
            severity =
                (
                    input.eyeUncertaintyMeters /
                        config.uncertaintyScaleMeters.coerceAtLeast(1e-6)
                    ).coerceIn(0.0, 1.0),
            evidence = eyeConfidence,
            blocking =
                input.eyeUncertaintyMeters >
                    config.uncertaintyScaleMeters * 2.0,
            rationale =
                "Triangulated eye position uncertainty exceeds the preferred calibrated range.",
        )

        add(
            domain = FailureDomain.EYE_DYNAMICS,
            severity =
                1.0 -
                    eyeDynamicScore.coerceIn(0.0, 1.0),
            evidence =
                (
                    input.saccadeVelocityDegreesPerSecond /
                        config.saccadeHardDegreesPerSecond
                    ).coerceIn(0.0, 1.0),
            blocking =
                input.saccadeVelocityDegreesPerSecond >=
                    config.saccadeHardDegreesPerSecond ||
                    input.gazeAccelerationDegreesPerSecond2 >=
                    config.gazeAccelerationHardDegreesPerSecond2,
            rationale =
                "Rapid gaze velocity or acceleration is reducing confidence in eye dynamics.",
        )

        add(
            domain = FailureDomain.EYE_OOD,
            severity =
                input.eyeOutOfDistributionRisk
                    .coerceIn(0.0, 1.0),
            evidence =
                1.0 -
                    input.eyeOutOfDistributionRisk
                        .coerceIn(0.0, 1.0),
            blocking = input.eyeOutOfDistributionRisk >= 0.92,
            rationale =
                "Eye evidence appears outside the calibrated operating distribution.",
        )

        add(
            domain = FailureDomain.IPD_CALIBRATION,
            severity =
                1.0 -
                    ipdConsistencyScore
                        .coerceIn(0.0, 1.0),
            evidence = ipdConsistencyScore,
            blocking =
                input.interpupillaryDistanceMeters > 0.0 &&
                    ipdConsistencyScore < 0.25,
            rationale =
                "Measured interpupillary scale disagrees with the learned session baseline.",
        )

        add(
            domain = FailureDomain.FACE_CALIBRATION,
            severity =
                (
                    1.0 -
                        geometricIntegrityScore
                    ).coerceIn(0.0, 1.0) *
                    (
                        1.0 -
                            faceEyeAgreement
                                .coerceIn(0.0, 1.0)
                        )
                    .coerceIn(0.0, 1.0),
            evidence = geometricIntegrityScore,
            blocking =
                geometricIntegrityScore < 0.28 &&
                    fusedConfidence < config.rejectFusedConfidence,
            rationale =
                "Face/eye geometric agreement is insufficient for trustworthy cross-channel fusion.",
        )

        return items
            .sortedWith(
                compareByDescending<FailureAttribution> {
                    it.severity
                }.thenBy {
                    it.domain.ordinal
                },
            )
            .take(8)
    }

    private fun faceShapeQuality(
        input: Input,
    ): Double {
        val acceptance =
            if (input.faceShapeAccepted) 1.0 else 0.18

        val pointCoverage =
            if (input.facePointCount <= 0) {
                0.0
            } else {
                (
                    input.facePointCount.toDouble() /
                        468.0
                    ).coerceIn(0.0, 1.0)
            }

        val denseEvidence =
            if (input.denseMeshObserved) {
                input.denseMeshConfidence.coerceIn(0.0, 1.0)
            } else {
                config.denseMeshMissingScore
            }

        return (
            input.faceShapeConfidence.coerceIn(0.0, 1.0) * 0.48 +
                pointCoverage * 0.16 +
                denseEvidence * 0.18 +
                acceptance * 0.18
            ).coerceIn(0.0, 1.0)
    }

    private fun eyeDynamicConsistency(
        input: Input,
    ): Double {
        val velocityScore =
            saccadeQuality(
                input.saccadeVelocityDegreesPerSecond,
            )

        val acceleration =
            input.gazeAccelerationDegreesPerSecond2

        val accelerationScore =
            when {
                acceleration <=
                    config.gazeAccelerationComfortDegreesPerSecond2 ->
                    1.0

                acceleration >=
                    config.gazeAccelerationHardDegreesPerSecond2 ->
                    0.38

                else -> {
                    val ratio =
                        (
                            acceleration -
                                config.gazeAccelerationComfortDegreesPerSecond2
                            ) /
                            (
                                config.gazeAccelerationHardDegreesPerSecond2 -
                                    config.gazeAccelerationComfortDegreesPerSecond2
                                ).coerceAtLeast(1.0)
                    1.0 - ratio * 0.62
                }
            }.coerceIn(0.38, 1.0)

        val acceptance =
            if (
                input.eyeTemporalAccepted &&
                input.binocularAccepted
            ) {
                1.0
            } else {
                0.30
            }

        return (
            velocityScore * 0.34 +
                accelerationScore * 0.36 +
                acceptance * 0.30
            ).coerceIn(0.0, 1.0)
    }

    private fun hardFailureReasons(
        input: Input,
        faceHardFailure: Boolean,
        eyeHardFailure: Boolean,
    ): List<String> =
        buildList {
            if (faceHardFailure) {
                if (!input.faceShapeAccepted) {
                    add("face-shape-rejected")
                }
                if (input.facePointCount < 14) {
                    add("face-point-coverage-insufficient")
                }
                if (input.faceFreshness <= config.hardStaleFaceThreshold) {
                    add("face-stale")
                }
                if (input.faceReprojectionScore <= 0.12) {
                    add("face-reprojection-hard-failure")
                }
                if (faceShapeQuality(input) <= config.faceShapeRejectThreshold) {
                    add("face-shape-quality-hard-failure")
                }
            }

            if (eyeHardFailure) {
                if (!input.eyeTemporalAccepted) {
                    add("eye-temporal-rejected")
                }
                if (!input.binocularAccepted) {
                    add("binocular-geometry-rejected")
                }
                if (
                    input.binocularConfidence <
                        config.binocularRejectThreshold
                ) {
                    add("binocular-confidence-below-hard-floor")
                }
                if (
                    input.angularSpreadDegrees >=
                        config.angularSpreadRejectDegrees
                ) {
                    add("binocular-angular-hard-failure")
                }
            }
        }.distinct()

    private fun faceRawConfidence(
        input: Input,
    ): Double {
        val uncertainty =
            exp(
                -input.faceUncertaintyMeters /
                    config.uncertaintyScaleMeters.coerceAtLeast(1e-6),
            ).coerceIn(0.0, 1.0)

        val freshness =
            if (input.faceFresh) {
                1.0
            } else {
                0.0
            }

        val ood =
            1.0 -
                input.faceOutOfDistributionRisk

        val prediction =
            if (input.predictionOnlyFace) {
                0.18
            } else {
                1.0
            }

        return weighted(
            listOf(
                input.faceGeometryConfidence to
                    config.faceGeometryWeight,
                input.faceTemporalConfidence to
                    config.faceTemporalWeight,
                input.denseMeshConfidence to
                    (
                        config.denseMeshWeight *
                            if (input.denseMeshObserved) 1.0 else 0.72
                        ),
                input.faceShapeConfidence to
                    config.faceShapeWeight,
                uncertainty to
                    config.uncertaintyWeight,
                freshness to
                    config.faceFreshnessWeight,
                input.faceReprojectionScore to
                    config.faceReprojectionWeight,
                input.lightingConfidence to
                    config.captureLightingWeight,
                input.faceScaleConfidence to
                    config.captureScaleWeight,
                ood to
                    0.08,
                prediction to
                    0.06,
            ),
        )
    }

    private fun eyeRawConfidence(
        input: Input,
    ): Double {
        val uncertainty =
            exp(
                -input.eyeUncertaintyMeters /
                    config.uncertaintyScaleMeters.coerceAtLeast(1e-6),
            ).coerceIn(0.0, 1.0)

        val angular =
            (
                1.0 -
                    input.angularSpreadDegrees /
                        config.angularSpreadRejectDegrees
                ).coerceIn(0.0, 1.0)

        val parallax =
            parallaxQuality(
                input.parallaxDegrees,
            )

        val saccade =
            saccadeQuality(
                input.saccadeVelocityDegreesPerSecond,
            )

        val blink =
            (
                1.0 -
                    input.blinkProbability * 0.35
                ).coerceIn(0.0, 1.0)

        val freshness =
            if (input.eyesFresh) {
                1.0
            } else {
                0.42
            }

        val ood =
            1.0 -
                input.eyeOutOfDistributionRisk

        val prediction =
            if (input.predictionOnlyEyes) {
                0.18
            } else {
                1.0
            }

        return weighted(
            listOf(
                input.leftEyeConfidence to config.leftEyeWeight,
                input.rightEyeConfidence to config.rightEyeWeight,
                input.binocularConfidence to config.binocularWeight,
                input.eyeTemporalConfidence to config.eyeTemporalWeight,
                uncertainty to config.uncertaintyWeight,
                angular to config.angularAgreementWeight,
                input.gazeStability to config.gazeStabilityWeight,
                input.gazeAgreement to config.gazeAgreementWeight,
                parallax to config.parallaxWeight,
                saccade to config.saccadeWeight,
                blink to config.blinkWeight,
                input.lightingConfidence to 0.04,
                freshness to 0.06,
                ood to 0.06,
                prediction to 0.04,
            ),
        ).let { score ->
            val hardGate =
                input.binocularConfidence <
                    config.binocularRejectThreshold ||
                    input.angularSpreadDegrees >=
                        config.angularSpreadRejectDegrees ||
                    !input.binocularAccepted
            if (hardGate) minOf(score, 0.52) else score
        }
    }

    private fun binocularQuality(
        input: Input,
    ): Double {
        val agreement =
            (
                input.binocularConfidence * 0.58 +
                    (
                        1.0 -
                            input.angularSpreadDegrees /
                                config.angularSpreadRejectDegrees
                        ).coerceIn(0.0, 1.0) * 0.24 +
                    parallaxQuality(input.parallaxDegrees) * 0.18
                ).coerceIn(0.0, 1.0)

        return agreement
    }

    private fun parallaxQuality(
        degrees: Double,
    ): Double {
        if (degrees <= 0.0) return 0.0
        if (degrees >= config.strongParallaxDegrees) return 1.0

        return (
            (
                degrees -
                    config.minimumParallaxDegrees
                ) /
                (
                    config.strongParallaxDegrees -
                        config.minimumParallaxDegrees
                    ).coerceAtLeast(1e-6)
            ).coerceIn(0.0, 1.0)
    }

    private fun saccadeQuality(
        degreesPerSecond: Double,
    ): Double {
        if (degreesPerSecond <=
            config.saccadeComfortDegreesPerSecond
        ) {
            return 1.0
        }

        if (
            degreesPerSecond >=
                config.saccadeHardDegreesPerSecond
        ) {
            return 0.44
        }

        val range =
            (
                config.saccadeHardDegreesPerSecond -
                    config.saccadeComfortDegreesPerSecond
                ).coerceAtLeast(1.0)

        return (
            1.0 -
                (
                    degreesPerSecond -
                        config.saccadeComfortDegreesPerSecond
                    ) / range * 0.56
            ).coerceIn(0.44, 1.0)
    }

    private fun faceEyeAgreement(
        input: Input,
    ): Double {
        val faceStability =
            (
                input.faceGeometryConfidence * 0.30 +
                    input.faceTemporalConfidence * 0.27 +
                    input.denseMeshConfidence * 0.20 +
                    input.faceFreshness * 0.13 +
                    input.faceReprojectionScore * 0.10 +
                    input.lightingConfidence * 0.08 +
                    input.faceScaleConfidence * 0.07
                ).coerceIn(0.0, 1.0)

        val eyeStability =
            (
                input.binocularConfidence * 0.34 +
                    input.eyeTemporalConfidence * 0.25 +
                    input.gazeStability * 0.13 +
                    input.gazeAgreement * 0.12 +
                    (
                        1.0 -
                            input.angularSpreadDegrees /
                                config.angularSpreadRejectDegrees
                        ).coerceIn(0.0, 1.0) * 0.16
                ).coerceIn(0.0, 1.0)

        return (
            1.0 -
                kotlin.math.abs(
                    faceStability -
                        eyeStability,
                )
            ).coerceIn(0.0, 1.0)
    }

    private fun temporalReliability(
        raw: Double,
        prior: Double,
        quarantined: Boolean,
        quarantineScale: Double,
    ): Double {
        val blended =
            raw * (1.0 - config.historyAlpha) +
                prior * config.historyAlpha

        return if (quarantined) {
            max(
                quarantineScale,
                blended * quarantineScale,
            )
        } else {
            blended.coerceIn(0.0, 1.0)
        }
    }

    private fun outOfDistributionWarnings(
        input: Input,
    ): List<String> =
        buildList {
            if (input.lightingConfidence < 0.55) {
                add("capture-domain-lighting-confidence-low")
            }
            if (input.faceScaleConfidence < 0.55) {
                add("capture-domain-face-scale-confidence-low")
            }
            if (input.faceOutOfDistributionRisk >= 0.72) {
                add("face-out-of-distribution-risk-elevated")
            }
            if (input.faceOutOfDistributionRisk >= 0.92) {
                add("face-out-of-distribution-risk-high")
            }
            if (input.eyeOutOfDistributionRisk >= 0.72) {
                add("eye-out-of-distribution-risk-elevated")
            }
            if (input.eyeOutOfDistributionRisk >= 0.92) {
                add("eye-out-of-distribution-risk-high")
            }
        }

    private fun selectFusionMode(
        faceConfidence: Double,
        eyeConfidence: Double,
        independentEvidenceSupport: Double,
        faceHardFailure: Boolean,
        eyeHardFailure: Boolean,
        accepted: Boolean,
    ): FusionMode =
        when {
            faceHardFailure &&
                !eyeHardFailure &&
                eyeConfidence >= config.minimumEyeEvidence ->
                FusionMode.EYE_ONLY

            eyeHardFailure &&
                !faceHardFailure ->
                FusionMode.FACE_ONLY

            !accepted ->
                FusionMode.HOLD

            faceHardFailure &&
                !eyeHardFailure &&
                eyeConfidence >= config.minimumEyeEvidence ->
                FusionMode.EYE_ONLY

            eyeHardFailure &&
                !faceHardFailure &&
                faceConfidence >= config.minimumFaceEvidence ->
                FusionMode.FACE_ONLY

            faceConfidence >=
                eyeConfidence * 1.14 ->
                FusionMode.FACE_DOMINANT

            eyeConfidence >=
                faceConfidence * 1.14 ->
                FusionMode.EYE_DOMINANT

            independentEvidenceSupport < 0.42 ->
                if (faceConfidence >= eyeConfidence) {
                    FusionMode.FACE_DOMINANT
                } else {
                    FusionMode.EYE_DOMINANT
                }

            else ->
                FusionMode.BALANCED
        }

    private fun buildSelectedSources(
        input: Input,
        faceConfidence: Double,
        eyeConfidence: Double,
    ): List<String> = buildList {
        if (
            input.faceGeometryConfidence >= 0.45 &&
            !state.faceQuarantined
        ) {
            add("FACE_GEOMETRY")
        }

        if (
            input.faceTemporalConfidence >= 0.45 &&
            !state.faceQuarantined
        ) {
            add("FACE_TEMPORAL")
        }

        if (
            input.denseMeshConfidence >= 0.50 &&
            !state.faceQuarantined
        ) {
            add("DENSE_FACE_MESH")
        }

        if (
            input.binocularConfidence >= 0.50 &&
            !state.eyeQuarantined
        ) {
            add("BINOCULAR_EYE")
        }

        if (
            input.eyeTemporalConfidence >= 0.50 &&
            !state.eyeQuarantined
        ) {
            add("EYE_TEMPORAL")
        }

        if (
            faceConfidence >= 0.72 &&
            eyeConfidence >= 0.72
        ) {
            add("FACE_EYE_AGREEMENT")
        }
    }

    private fun buildWarnings(
        input: Input,
        faceConfidence: Double,
        eyeConfidence: Double,
        fusedConfidence: Double,
        eyeHardFailure: Boolean,
        faceHardFailure: Boolean,
    ): List<String> = buildList {
        if (input.faceUncertaintyMeters >
            config.uncertaintyScaleMeters * 2.0
        ) {
            add("face-uncertainty-elevated")
        }

        if (input.eyeUncertaintyMeters >
            config.uncertaintyScaleMeters * 2.0
        ) {
            add("eye-uncertainty-elevated")
        }

        if (input.angularSpreadDegrees >=
            config.angularSpreadWarningDegrees
        ) {
            add("binocular-angular-disagreement")
        }

        if (
            input.parallaxDegrees <
                config.minimumParallaxDegrees &&
            eyeConfidence >= 0.55
        ) {
            add("eye-parallax-limited")
        }

        if (
            input.saccadeVelocityDegreesPerSecond >
                config.saccadeComfortDegreesPerSecond
        ) {
            add("rapid-saccade-regime")
        }

        if (
            faceConfidence < eyeConfidence * 0.80 &&
            (
                input.faceGeometryConfidence < 0.50 ||
                    input.faceTemporalConfidence < 0.50 ||
                    input.denseMeshConfidence < 0.50 ||
                    input.faceUncertaintyMeters >
                        config.uncertaintyScaleMeters * 2.0
            )
        ) {
            add("face-evidence-deprioritized")
        }

        if (eyeConfidence < faceConfidence * 0.64) {
            add("eye-evidence-deprioritized")
        }

        if (eyeHardFailure) {
            add("binocular-eye-hard-failure")
        }
        if (faceHardFailure) {
            add("face-geometry-hard-failure")
        }
        if (input.faceFreshness < config.staleFaceThreshold) {
            add("face-stale-evidence-downweighted")
        }
        if (input.faceReprojectionScore < 0.55) {
            add("face-reprojection-quality-limited")
        }
        if (input.gazeAgreement < 0.50) {
            add("gaze-agreement-limited")
        }
        if (input.gazeStability < 0.50) {
            add("gaze-stability-limited")
        }

        if (state.eyeQuarantined) {
            add("eye-evidence-quarantined")
        }

        if (state.faceQuarantined) {
            add("face-evidence-quarantined")
        }

        if (
            input.interpupillaryDistanceMeters > 0.0 &&
            state.ipdSamples >= config.ipdCalibrationWarmupSamples &&
            abs(
                input.interpupillaryDistanceMeters -
                    state.ipdBaselineMeters,
            ) >
                config.maximumIpdDeviationMeters * 0.60
        ) {
            add("binocular-gaze-ipd-session-drift")
            add("ipd-calibration-drift-elevated")
        }

        if (fusedConfidence <
            config.rejectFusedConfidence
        ) {
            add("face-eye-fused-confidence-low")
        }
    }.distinct()

    private fun buildRecommendations(
        input: Input,
        faceConfidence: Double,
        eyeConfidence: Double,
        fusedConfidence: Double,
        eyeHardFailure: Boolean,
        faceHardFailure: Boolean,
    ): List<String> = buildList {
        if (faceConfidence < eyeConfidence) {
            add(
                "Preserve binocular eye geometry while face reconstruction recovers.",
            )
        }

        if (eyeConfidence < faceConfidence) {
            add(
                "Preserve face geometry while binocular gaze evidence recovers.",
            )
        }

        if (eyeHardFailure) {
            add(
                "Hold eye-driven metric output until binocular agreement and parallax recover.",
            )
        }

        if (faceHardFailure) {
            add(
                "Hold face-driven metric output until fresh, low-reprojection face geometry returns.",
            )
        }

        if (
            input.parallaxDegrees <
                config.minimumParallaxDegrees
        ) {
            add(
                "Increase camera baseline or preserve additional synchronized views for eye triangulation.",
            )
        }

        if (
            input.interpupillaryDistanceMeters > 0.0 &&
            state.ipdSamples >= config.ipdCalibrationWarmupSamples &&
            abs(
                input.interpupillaryDistanceMeters -
                    state.ipdBaselineMeters,
            ) >
                config.maximumIpdDeviationMeters * 0.60
        ) {
            add("binocular-gaze-ipd-session-drift")
            add("ipd-calibration-drift-elevated")
        }

        if (
            input.angularSpreadDegrees >=
                config.angularSpreadWarningDegrees
        ) {
            add(
                "Re-check eye observations and calibration before committing a gaze trajectory.",
            )
        }

        if (
            fusedConfidence <
                config.rejectFusedConfidence
        ) {
            add(
                "Keep face/eye output in degraded state and prefer temporally adjacent evidence.",
            )
        }
    }.distinct().take(8)

    private fun updateReliability(
        input: Input,
        faceRaw: Double,
        eyeRaw: Double,
        eyeHardFailure: Boolean,
    ) {
        val faceBad =
            input.faceOutOfDistributionRisk >= 0.85 ||
                input.faceFreshness < config.hardStaleFaceThreshold ||
                input.faceReprojectionScore < 0.18 ||
                faceRaw < config.minimumFaceEvidence

        val eyeBad =
            eyeHardFailure ||
                input.eyeOutOfDistributionRisk >= 0.85 ||
                eyeRaw < config.minimumEyeEvidence

        if (input.predictionOnlyFace) {
            state.faceGoodFrames = 0
            state.faceBadFrames = 0
        } else if (faceBad) {
            state.faceBadFrames += 1
            state.faceGoodFrames = 0
            state.faceReliability =
                (
                    state.faceReliability *
                        (1.0 - config.historyAlpha * 0.72)
                    ).coerceAtLeast(0.02)
        } else {
            state.faceBadFrames =
                max(0, state.faceBadFrames - 1)
            state.faceGoodFrames += 1
            state.faceReliability =
                (
                    state.faceReliability *
                        (1.0 - config.historyAlpha) +
                        faceRaw *
                        config.historyAlpha
                    ).coerceIn(0.02, 1.0)
        }

        if (input.predictionOnlyEyes) {
            state.eyeGoodFrames = 0
            state.eyeBadFrames = 0
        } else if (eyeBad) {
            state.eyeBadFrames += 1
            state.eyeGoodFrames = 0
            state.eyeReliability =
                (
                    state.eyeReliability *
                        (1.0 - config.historyAlpha * 0.72)
                    ).coerceAtLeast(0.02)
        } else {
            state.eyeBadFrames =
                max(0, state.eyeBadFrames - 1)
            state.eyeGoodFrames += 1
            state.eyeReliability =
                (
                    state.eyeReliability *
                        (1.0 - config.historyAlpha) +
                        eyeRaw *
                        config.historyAlpha
                    ).coerceIn(0.02, 1.0)
        }

        if (
            state.eyeBadFrames >=
                config.quarantineAfterBadFrames
        ) {
            state.eyeQuarantined = true
            state.eyeGoodFrames = 0
        }

        if (
            state.faceBadFrames >=
                config.quarantineAfterBadFrames
        ) {
            state.faceQuarantined = true
            state.faceGoodFrames = 0
        }

        if (
            state.eyeQuarantined &&
            state.eyeGoodFrames >=
                config.recoveryFrames &&
            eyeRaw >=
                config.recoveryEyeConfidence &&
            input.binocularConfidence >=
                config.recoveryBinocularConfidence
        ) {
            state.eyeQuarantined = false
            state.eyeBadFrames = 0
            state.eyeGoodFrames = 0
        }

        if (
            state.faceQuarantined &&
            state.faceGoodFrames >=
                config.recoveryFrames &&
            faceRaw >=
                config.recoveryEyeConfidence
        ) {
            state.faceQuarantined = false
            state.faceBadFrames = 0
            state.faceGoodFrames = 0
        }
    }

    private fun geometricIntegrity(
        input: Input,
        ipdConsistencyScore: Double,
    ): Double {
        val reprojection =
            input.faceReprojectionScore.coerceIn(0.0, 1.0)
        val faceFreshness =
            input.faceFreshness.coerceIn(0.0, 1.0)
        val gazeAgreement =
            input.gazeAgreement.coerceIn(0.0, 1.0)
        val binocular =
            input.binocularConfidence.coerceIn(0.0, 1.0)
        val angular =
            (
                1.0 -
                    input.angularSpreadDegrees /
                        config.angularSpreadRejectDegrees
                ).coerceIn(0.0, 1.0)

        return (
            reprojection * 0.22 +
                faceFreshness * 0.16 +
                gazeAgreement * 0.20 +
                binocular * 0.18 +
                angular * 0.12 +
                ipdConsistencyScore * 0.12
            ).coerceIn(0.0, 1.0)
    }

    private fun ipdConsistency(
        input: Input,
    ): Double {
        val observed = input.interpupillaryDistanceMeters
        if (observed <= 0.0) {
            return 1.0
        }

        val calibrationEligible =
            input.leftEyeConfidence >=
                config.minimumIpdCalibrationEyeConfidence &&
                input.rightEyeConfidence >=
                    config.minimumIpdCalibrationEyeConfidence &&
                input.binocularConfidence >=
                    config.minimumIpdCalibrationBinocularConfidence &&
                input.parallaxDegrees >=
                    config.minimumIpdCalibrationParallaxDegrees &&
                input.angularSpreadDegrees <=
                    config.maximumIpdCalibrationAngularSpreadDegrees &&
                input.eyeUncertaintyMeters <=
                    config.uncertaintyScaleMeters * 1.35 &&
                input.eyeOutOfDistributionRisk < 0.30 &&
                !input.predictionOnlyEyes

        /*
         * Weak binocular frames remain useful for current-frame confidence, but they are never
         * permitted to define or rewrite the metric IPD baseline. This prevents a bad first frame
         * from permanently teaching the system an anatomically impossible scale.
         */
        if (state.ipdSamples < config.ipdCalibrationWarmupSamples) {
            if (!calibrationEligible) {
                return if (state.ipdSamples <= 0) {
                    1.0
                } else {
                    (
                        1.0 -
                            kotlin.math.abs(
                                observed -
                                    state.ipdBaselineMeters,
                            ) /
                            config.maximumIpdDeviationMeters
                        ).coerceIn(0.0, 1.0)
                }
            }

            val alpha =
                1.0 /
                    (state.ipdSamples + 1).toDouble()

            state.ipdBaselineMeters =
                if (state.ipdSamples == 0) {
                    observed
                } else {
                    state.ipdBaselineMeters * (1.0 - alpha) +
                        observed * alpha
                }

            state.ipdSamples += 1

            return 1.0
        }

        val deviation =
            kotlin.math.abs(
                observed -
                    state.ipdBaselineMeters,
            )

        val score =
            (
                1.0 -
                    deviation /
                        config.maximumIpdDeviationMeters
                ).coerceIn(0.0, 1.0)

        if (
            calibrationEligible &&
            score >= 0.72
        ) {
            val alpha =
                (
                    config.historyAlpha * 0.40
                ).coerceIn(0.01, 0.18)
            state.ipdBaselineMeters =
                state.ipdBaselineMeters * (1.0 - alpha) +
                    observed * alpha
            state.ipdSamples += 1
        }

        return score
    }

    private fun weighted(
        values: List<Pair<Double, Double>>,
    ): Double {
        val denominator =
            values.sumOf { it.second }
        if (denominator <= 1e-9) return 0.0

        return (
            values.sumOf {
                it.first.coerceIn(0.0, 1.0) *
                    it.second
            } / denominator
        ).coerceIn(0.0, 1.0)
    }
}

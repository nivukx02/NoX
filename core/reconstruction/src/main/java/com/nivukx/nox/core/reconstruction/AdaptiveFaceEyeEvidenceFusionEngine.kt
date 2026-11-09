package com.nivukx.nox.core.reconstruction

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Production face/eye reliability arbiter.
 *
 * This engine treats face shape, dense facial geometry, temporal face stability and binocular eye
 * geometry as separate evidence channels. It prevents an eye failure from poisoning otherwise
 * healthy head pose, and prevents stale face geometry from dominating reliable binocular gaze.
 */
class AdaptiveFaceEyeEvidenceFusionEngine(
    private val config: Config = Config(),
) {

    enum class State {
        UNKNOWN,
        STABLE,
        FACE_DEGRADED,
        EYE_DEGRADED,
        BOTH_DEGRADED,
        RAPID_GAZE,
        BILATERAL_MISMATCH,
    }

    data class Config(
        val minimumChannelConfidence: Double = 0.16,
        val staleFaceThreshold: Double = 0.45,
        val staleFaceHardThreshold: Double = 0.18,
        val eyeRapidGazeWarningDegreesPerSecond: Double = 520.0,
        val eyeRapidGazeHardDegreesPerSecond: Double = 980.0,
        val eyeFailureThreshold: Double = 0.34,
        val faceFailureThreshold: Double = 0.36,
        val warningFusionConfidence: Double = 0.58,
        val rejectFusionConfidence: Double = 0.38,
        val persistentDegradeFrames: Int = 3,
        val recoveryFrames: Int = 4,
        val historyAlpha: Double = 0.24,
        val eyeWeightFloor: Double = 0.04,
        val faceWeightFloor: Double = 0.08,
        val faceShapeFailureThreshold: Double = 0.30,
        val eyeDynamicWarningAccelerationDegreesPerSecond2: Double = 3200.0,
        val eyeDynamicHardAccelerationDegreesPerSecond2: Double = 9000.0,
    ) {
        init {
            require(minimumChannelConfidence in 0.0..1.0)
            require(staleFaceThreshold in 0.0..1.0)
            require(staleFaceHardThreshold in 0.0..staleFaceThreshold)
            require(eyeRapidGazeWarningDegreesPerSecond > 0.0)
            require(
                eyeRapidGazeHardDegreesPerSecond >
                    eyeRapidGazeWarningDegreesPerSecond,
            )
            require(eyeFailureThreshold in 0.0..1.0)
            require(faceFailureThreshold in 0.0..1.0)
            require(warningFusionConfidence in 0.0..1.0)
            require(rejectFusionConfidence in 0.0..warningFusionConfidence)
            require(persistentDegradeFrames >= 1)
            require(recoveryFrames >= 1)
            require(historyAlpha in 0.0..1.0)
            require(eyeWeightFloor in 0.0..1.0)
            require(faceWeightFloor in 0.0..1.0)
            require(faceShapeFailureThreshold in 0.0..1.0)
            require(eyeDynamicWarningAccelerationDegreesPerSecond2 > 0.0)
            require(
                eyeDynamicHardAccelerationDegreesPerSecond2 >
                    eyeDynamicWarningAccelerationDegreesPerSecond2
            )
        }
    }

    data class Input(
        val timestampNanos: Long,
        val faceGeometryConfidence: Double,
        val faceTemporalConfidence: Double,
        val denseFaceMeshConfidence: Double,
        val faceFreshness: Double,
        val faceReprojectionScore: Double,
        val leftEyeConfidence: Double,
        val rightEyeConfidence: Double,
        val eyeTemporalConfidence: Double,
        val binocularConfidence: Double,
        val gazeStability: Double,
        val gazeAgreement: Double,
        val saccadeVelocityDegreesPerSecond: Double,
        val eyeUncertaintyMeters: Double,
        val blinkProbability: Double,
        val faceShapeConfidence: Double = 0.85,
        val faceShapeAccepted: Boolean = true,
        val denseFaceMeshObserved: Boolean = true,
        val eyeTemporalAccepted: Boolean = true,
        val binocularAccepted: Boolean = true,
        val gazeAccelerationDegreesPerSecond2: Double = 0.0,
    ) {
        init {
            listOf(
                faceGeometryConfidence,
                faceTemporalConfidence,
                denseFaceMeshConfidence,
                faceFreshness,
                faceReprojectionScore,
                leftEyeConfidence,
                rightEyeConfidence,
                eyeTemporalConfidence,
                binocularConfidence,
                gazeStability,
                gazeAgreement,
                blinkProbability,
                faceShapeConfidence,
            ).forEach { require(it in 0.0..1.0) }
            require(saccadeVelocityDegreesPerSecond >= 0.0)
            require(eyeUncertaintyMeters.isFinite() && eyeUncertaintyMeters >= 0.0)
            require(gazeAccelerationDegreesPerSecond2 >= 0.0)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val state: State,
        val confidence: Double,
        val faceWeight: Double,
        val eyeWeight: Double,
        val faceReliability: Double,
        val eyeReliability: Double,
        val bilateralEyeSupport: Double,
        val faceSupport: Double,
        val rapidGazeRisk: Double,
        val gazeConsistency: Double,
        val eyeDegradedPersistently: Boolean,
        val faceDegradedPersistently: Boolean,
        val faceShapeReliability: Double = 1.0,
        val eyeDynamicConsistency: Double = 1.0,
        val faceHardFailure: Boolean = false,
        val eyeHardFailure: Boolean = false,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val accepted: Boolean
            get() =
                confidence >= 0.38 &&
                    state != State.BOTH_DEGRADED
    }

    private data class History(
        var timestampNanos: Long,
        var faceReliabilityEma: Double,
        var eyeReliabilityEma: Double,
        var eyeBadFrames: Int,
        var faceBadFrames: Int,
        var eyeGoodFrames: Int,
        var faceGoodFrames: Int,
        var state: State,
    )

    private var history: History? = null

    @Synchronized
    fun evaluate(input: Input): Result {
        val faceSupportBase =
            faceSupport(input)

        val faceShapeReliability =
            (
                input.faceShapeConfidence * 0.72 +
                    (
                        if (input.faceShapeAccepted) 1.0 else 0.18
                    ) * 0.28
                ).coerceIn(0.0, 1.0)

        val eyeDynamicConsistency =
            eyeDynamicConsistency(input)

        val faceHardFailure =
            !input.faceShapeAccepted ||
                faceShapeReliability <
                    config.faceShapeFailureThreshold ||
                input.faceFreshness <
                    config.staleFaceHardThreshold

        val eyeHardFailure =
            !input.eyeTemporalAccepted ||
                !input.binocularAccepted ||
                input.binocularConfidence <
                    config.minimumChannelConfidence ||
                eyeDynamicConsistency < 0.22

        val faceBase =
            (
                faceSupportBase * 0.88 +
                    faceShapeReliability * 0.12
                ).coerceIn(0.0, 1.0)

        val eyeBase =
            eyeSupport(input)

        val rapidGazeRisk =
            when {
                input.saccadeVelocityDegreesPerSecond <=
                    config.eyeRapidGazeWarningDegreesPerSecond ->
                    0.0

                input.saccadeVelocityDegreesPerSecond >=
                    config.eyeRapidGazeHardDegreesPerSecond ->
                    1.0

                else ->
                    (
                        input.saccadeVelocityDegreesPerSecond -
                            config.eyeRapidGazeWarningDegreesPerSecond
                        ) /
                        (
                            config.eyeRapidGazeHardDegreesPerSecond -
                                config.eyeRapidGazeWarningDegreesPerSecond
                        )
            }.coerceIn(0.0, 1.0)

        val previous = history
        if (
            previous != null &&
            input.timestampNanos <= previous.timestampNanos
        ) {
            history = null
        }

        val active =
            history ?: History(
                timestampNanos = input.timestampNanos,
                faceReliabilityEma = faceBase,
                eyeReliabilityEma = eyeBase,
                eyeBadFrames = 0,
                faceBadFrames = 0,
                eyeGoodFrames = 0,
                faceGoodFrames = 0,
                state = State.UNKNOWN,
            ).also {
                history = it
            }

        val faceReliability =
            (
                active.faceReliabilityEma * (1.0 - config.historyAlpha) +
                    faceBase * config.historyAlpha
                ).coerceIn(0.0, 1.0)

        val eyeReliability =
            (
                active.eyeReliabilityEma * (1.0 - config.historyAlpha) +
                    eyeBase * config.historyAlpha
                ).coerceIn(0.0, 1.0)

        val eyeBad =
            eyeBase <
                config.eyeFailureThreshold ||
                input.binocularConfidence <
                    config.minimumChannelConfidence ||
                input.gazeAgreement < 0.30 ||
                !input.eyeTemporalAccepted ||
                !input.binocularAccepted ||
                eyeDynamicConsistency < 0.22

        val faceBad =
            faceBase <
                config.faceFailureThreshold ||
                faceHardFailure

        if (eyeBad) {
            active.eyeBadFrames += 1
            active.eyeGoodFrames = 0
        } else {
            active.eyeBadFrames =
                max(0, active.eyeBadFrames - 1)
            active.eyeGoodFrames += 1
        }

        if (faceBad) {
            active.faceBadFrames += 1
            active.faceGoodFrames = 0
        } else {
            active.faceBadFrames =
                max(0, active.faceBadFrames - 1)
            active.faceGoodFrames += 1
        }

        val eyePersistent =
            active.eyeBadFrames >=
                config.persistentDegradeFrames

        val facePersistent =
            active.faceBadFrames >=
                config.persistentDegradeFrames

        if (
            active.eyeGoodFrames >= config.recoveryFrames &&
            eyeReliability >= config.eyeFailureThreshold
        ) {
            active.eyeBadFrames =
                max(0, active.eyeBadFrames - 1)
        }

        if (
            active.faceGoodFrames >= config.recoveryFrames &&
            faceReliability >= config.faceFailureThreshold
        ) {
            active.faceBadFrames =
                max(0, active.faceBadFrames - 1)
        }

        val eyeRapid =
            rapidGazeRisk >= 0.45

        val bilateralMismatch =
            input.binocularConfidence < 0.35 ||
                input.gazeAgreement < 0.38

        val faceMeshAvailabilityScale =
            if (input.denseFaceMeshObserved) {
                1.0
            } else {
                0.78
            }

        val faceFreshnessScale =
            if (
                input.faceFreshness <
                    config.staleFaceThreshold
            ) {
                0.48
            } else {
                1.0
            }

        val faceWeightRaw =
            faceReliability *
                (
                    0.46 +
                        input.faceFreshness * 0.22 +
                        input.denseFaceMeshConfidence * 0.13 +
                        faceShapeReliability * 0.08 +
                        input.faceReprojectionScore * 0.15
                    ).coerceAtLeast(0.0) *
                faceMeshAvailabilityScale *
                faceFreshnessScale

        val eyeWeightRaw =
            eyeReliability *
                (
                    0.34 +
                        input.eyeTemporalConfidence * 0.18 +
                        input.binocularConfidence * 0.20 +
                        input.gazeStability * 0.12 +
                        input.gazeAgreement * 0.10 +
                        bilateralEyeSupport(input) * 0.06 +
                        eyeDynamicConsistency * 0.10
                    ) *
                (
                    1.0 -
                        rapidGazeRisk * 0.34
                    ) *
                exp(
                    -input.eyeUncertaintyMeters / 0.045,
                ).coerceIn(0.08, 1.0)

        val faceWeight =
            (
                if (facePersistent) {
                    faceWeightRaw * 0.78
                } else {
                    faceWeightRaw
                }
                ).coerceAtLeast(config.faceWeightFloor)

        val eyeWeight =
            (
                if (eyePersistent) {
                    eyeWeightRaw * 0.58
                } else {
                    eyeWeightRaw
                }
                ).coerceAtLeast(config.eyeWeightFloor)

        val totalWeight =
            faceWeight + eyeWeight

        val normalizedFaceWeight =
            (
                faceWeight /
                    totalWeight.coerceAtLeast(1e-9)
                ).coerceIn(0.0, 1.0)

        val normalizedEyeWeight =
            (
                eyeWeight /
                    totalWeight.coerceAtLeast(1e-9)
                ).coerceIn(0.0, 1.0)

        val independentSupport =
            (
                1.0 -
                    (1.0 - faceReliability) *
                    (1.0 - eyeReliability)
                ).coerceIn(0.0, 1.0)

        val confidence =
            (
                faceReliability * 0.25 +
                    eyeReliability * 0.25 +
                    independentSupport * 0.20 +
                    faceShapeReliability * 0.08 +
                    input.faceGeometryConfidence * 0.07 +
                    input.faceTemporalConfidence * 0.05 +
                    input.binocularConfidence * 0.05 +
                    input.gazeStability * 0.05
                ).coerceIn(0.0, 1.0)

        val state =
            when {
                facePersistent && eyePersistent ->
                    State.BOTH_DEGRADED

                eyePersistent ->
                    State.EYE_DEGRADED

                facePersistent ->
                    State.FACE_DEGRADED

                eyeRapid &&
                    eyeReliability < faceReliability ->
                    State.RAPID_GAZE

                bilateralMismatch &&
                    eyeReliability < faceReliability ->
                    State.BILATERAL_MISMATCH

                confidence >= config.warningFusionConfidence ->
                    State.STABLE

                else ->
                    State.UNKNOWN
            }

        active.timestampNanos = input.timestampNanos
        active.faceReliabilityEma = faceReliability
        active.eyeReliabilityEma = eyeReliability
        active.state = state

        val warnings = buildWarnings(
            state = state,
            faceWeight = normalizedFaceWeight,
            eyeWeight = normalizedEyeWeight,
            faceReliability = faceReliability,
            eyeReliability = eyeReliability,
            faceFreshness = input.faceFreshness,
            rapidGazeRisk = rapidGazeRisk,
            bilateralMismatch = bilateralMismatch,
            eyePersistent = eyePersistent,
            facePersistent = facePersistent,
            blinkProbability = input.blinkProbability,
            gazeAccelerationDegreesPerSecond2 =
                input.gazeAccelerationDegreesPerSecond2,
            eyeTemporalAccepted = input.eyeTemporalAccepted,
            binocularAccepted = input.binocularAccepted,
            faceShapeAccepted = input.faceShapeAccepted,
            denseFaceMeshObserved = input.denseFaceMeshObserved,
        )

        val recommendations = buildRecommendations(
            state = state,
            faceReliability = faceReliability,
            eyeReliability = eyeReliability,
            faceFreshness = input.faceFreshness,
            eyeUncertaintyMeters = input.eyeUncertaintyMeters,
            rapidGazeRisk = rapidGazeRisk,
        )

        return Result(
            timestampNanos = input.timestampNanos,
            state = state,
            confidence = confidence,
            faceWeight = normalizedFaceWeight,
            eyeWeight = normalizedEyeWeight,
            faceReliability = faceReliability,
            eyeReliability = eyeReliability,
            bilateralEyeSupport = bilateralEyeSupport(input),
            faceSupport = faceBase,
            rapidGazeRisk = rapidGazeRisk,
            faceShapeReliability = faceShapeReliability,
            eyeDynamicConsistency = eyeDynamicConsistency,
            faceHardFailure = faceHardFailure,
            eyeHardFailure = eyeHardFailure,
            gazeConsistency = input.gazeAgreement,
            eyeDegradedPersistently = eyePersistent,
            faceDegradedPersistently = facePersistent,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    @Synchronized
    fun reset() {
        history = null
    }

    fun configuration(): Config = config

    private fun faceSupport(
        input: Input,
    ): Double =
        (
            input.faceGeometryConfidence * 0.32 +
                input.faceTemporalConfidence * 0.24 +
                input.denseFaceMeshConfidence * 0.20 +
                input.faceFreshness * 0.14 +
                input.faceReprojectionScore * 0.10
            ).coerceIn(0.0, 1.0)

    private fun eyeSupport(
        input: Input,
    ): Double =
        (
            input.leftEyeConfidence * 0.16 +
                input.rightEyeConfidence * 0.16 +
                input.eyeTemporalConfidence * 0.22 +
                input.binocularConfidence * 0.20 +
                input.gazeStability * 0.10 +
                input.gazeAgreement * 0.08 +
                bilateralEyeSupport(input) * 0.08
            ).coerceIn(0.0, 1.0)

    private fun bilateralEyeSupport(
        input: Input,
    ): Double =
        (
            min(
                input.leftEyeConfidence,
                input.rightEyeConfidence,
            ) * 0.46 +
                input.binocularConfidence * 0.36 +
                input.gazeAgreement * 0.18
            ).coerceIn(0.0, 1.0)

    private fun eyeDynamicConsistency(
        input: Input,
    ): Double {
        val velocity =
            when {
                input.saccadeVelocityDegreesPerSecond <=
                    config.eyeRapidGazeWarningDegreesPerSecond ->
                    1.0
                input.saccadeVelocityDegreesPerSecond >=
                    config.eyeRapidGazeHardDegreesPerSecond ->
                    0.46
                else ->
                    (
                        1.0 -
                            (
                                input.saccadeVelocityDegreesPerSecond -
                                    config.eyeRapidGazeWarningDegreesPerSecond
                                ) /
                                (
                                    config.eyeRapidGazeHardDegreesPerSecond -
                                        config.eyeRapidGazeWarningDegreesPerSecond
                                    )
                            ).coerceIn(0.0, 1.0) * 0.54
            }.coerceIn(0.46, 1.0)

        val acceleration =
            when {
                input.gazeAccelerationDegreesPerSecond2 <=
                    config.eyeDynamicWarningAccelerationDegreesPerSecond2 ->
                    1.0
                input.gazeAccelerationDegreesPerSecond2 >=
                    config.eyeDynamicHardAccelerationDegreesPerSecond2 ->
                    0.38
                else -> {
                    val normalized =
                        (
                            input.gazeAccelerationDegreesPerSecond2 -
                                config.eyeDynamicWarningAccelerationDegreesPerSecond2
                        ) /
                            (
                                config.eyeDynamicHardAccelerationDegreesPerSecond2 -
                                    config.eyeDynamicWarningAccelerationDegreesPerSecond2
                            )
                    (1.0 - normalized.coerceIn(0.0, 1.0) * 0.62)
                        .coerceIn(0.38, 1.0)
                }
            }

        val gate =
            if (
                input.eyeTemporalAccepted &&
                input.binocularAccepted
            ) {
                1.0
            } else {
                0.28
            }

        return (
            velocity * 0.36 +
                acceleration * 0.34 +
                input.gazeStability * 0.16 +
                gate * 0.14
            ).coerceIn(0.0, 1.0)
    }

    private fun buildWarnings(
        state: State,
        faceWeight: Double,
        eyeWeight: Double,
        faceReliability: Double,
        eyeReliability: Double,
        faceFreshness: Double,
        rapidGazeRisk: Double,
        bilateralMismatch: Boolean,
        eyePersistent: Boolean,
        facePersistent: Boolean,
        blinkProbability: Double,
        gazeAccelerationDegreesPerSecond2: Double,
        eyeTemporalAccepted: Boolean,
        binocularAccepted: Boolean,
        faceShapeAccepted: Boolean,
        denseFaceMeshObserved: Boolean,
    ): List<String> = buildList {
        if (faceFreshness < config.staleFaceThreshold) {
            add("face-stale-evidence-downweighted")
        }
        if (faceFreshness < config.staleFaceHardThreshold) {
            add("face-stale-hard-floor")
        }
        if (eyePersistent) {
            add("eye-evidence-persistently-degraded")
        }
        if (facePersistent) {
            add("face-evidence-persistently-degraded")
        }
        if (bilateralMismatch) {
            add("binocular-eye-evidence-mismatch")
        }
        if (rapidGazeRisk >= 0.45) {
            add("rapid-gaze-evidence-reweighted")
        }
        if (
            gazeAccelerationDegreesPerSecond2 >
                config.eyeDynamicWarningAccelerationDegreesPerSecond2
        ) {
            add("rapid-gaze-acceleration-evidence-reweighted")
        }
        if (!eyeTemporalAccepted) {
            add("eye-temporal-gate-rejected")
        }
        if (!binocularAccepted) {
            add("binocular-gate-rejected")
        }
        if (!faceShapeAccepted) {
            add("face-shape-gate-rejected")
        }
        if (!denseFaceMeshObserved) {
            add("dense-face-mesh-not-observed")
        }
        if (blinkProbability > 0.80) {
            add("blink-state-reduces-gaze-reliability")
        }
        if (eyeWeight < faceWeight * 0.42) {
            add("eye-contribution-limited-by-evidence-quality")
        }
        if (faceWeight < eyeWeight * 0.42) {
            add("face-contribution-limited-by-evidence-quality")
        }
        if (faceReliability < config.faceFailureThreshold) {
            add("face-geometry-confidence-below-production-floor")
        }
        if (eyeReliability < config.eyeFailureThreshold) {
            add("eye-geometry-confidence-below-production-floor")
        }
        when (state) {
            State.BOTH_DEGRADED ->
                add("face-eye-fusion-production-hold")

            State.BILATERAL_MISMATCH ->
                add("hold-gaze-target-commit-until-binocular-agreement-recovers")

            State.RAPID_GAZE ->
                add("preserve-saccade-dynamics-but-protect-head-pose-from-eye-outliers")

            else -> Unit
        }
    }.distinct()

    private fun buildRecommendations(
        state: State,
        faceReliability: Double,
        eyeReliability: Double,
        faceFreshness: Double,
        eyeUncertaintyMeters: Double,
        rapidGazeRisk: Double,
    ): List<String> = buildList {
        if (faceFreshness < config.staleFaceThreshold) {
            add(
                "reacquire face geometry before trusting expression or head-pose refinement",
            )
        }
        if (eyeReliability < config.eyeFailureThreshold) {
            add(
                "hold eye-derived gaze updates and keep head pose driven by face geometry",
            )
        }
        if (eyeUncertaintyMeters > 0.045) {
            add(
                "increase eye-camera baseline or reject high-uncertainty eye triangulation",
            )
        }
        if (rapidGazeRisk >= 0.45) {
            add(
                "retain the saccade event while lowering eye influence on persistent head state",
            )
        }
        if (state == State.BILATERAL_MISMATCH) {
            add(
                "verify inter-eye calibration and ensure both observations belong to the same actor",
            )
        }
        if (faceReliability < config.faceFailureThreshold) {
            add(
                "increase face-camera coverage or reduce dense-face refinement until geometry recovers",
            )
        }
        if (state == State.BOTH_DEGRADED) {
            add(
                "hold facial metric output and retain only downstream body continuity until face/eye evidence recovers",
            )
        }
    }.distinct().take(8)
}

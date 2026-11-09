
package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.EyeObservation
import com.nivukx.nox.core.model.EyeTrackingFrame
import com.nivukx.nox.core.model.FaceEyeFusionState
import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.PerceptionSource
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Fuses ML face head-pose evidence with the dedicated iris/gaze tracker.
 *
 * The goal is not to replace either detector. Face tracking provides stable identity and coarse
 * head orientation; eye tracking provides high-rate gaze/closure evidence. The fusion stage handles
 * freshness, blink-aware confidence, binocular agreement, vergence estimation, saccade velocity,
 * temporal hysteresis, and a single avatar-friendly contract.
 */
class FaceEyeFusionEngine(
    private var config: Config = Config(),
) {

    data class Config(
        val maximumStaleMillis: Double = 220.0,
        val minimumFaceConfidence: Double = 0.18,
        val minimumEyeConfidence: Double = 0.12,
        val binocularAgreementWeight: Double = 0.22,
        val temporalSmoothing: Double = 0.72,
        val blinkConfidencePenalty: Double = 0.70,
        val saccadeSoftLimitDegreesPerSecond: Double = 600.0,
        val vergenceMinMeters: Double = 0.15,
        val vergenceMaxMeters: Double = 8.0,
        val headGazeBlend: Double = 0.18,
        val captureTimingWeight: Double = 0.12,
        val maximumTrackAgeMillis: Double = 1_500.0,
        val maximumRetainedTracks: Int = 256,
        val aleatoricUncertaintyWeight: Double = 0.34,
        val temporalUncertaintyWeight: Double = 0.20,
        val binocularUncertaintyWeight: Double = 0.18,
        val innovationUncertaintyWeight: Double = 0.14,
        val staleEvidenceUncertaintyWeight: Double = 0.14,
        val hardUncertainty: Double = 0.78,
    ) {
        init {
            require(maximumStaleMillis > 0.0)
            require(minimumFaceConfidence in 0.0..1.0)
            require(minimumEyeConfidence in 0.0..1.0)
            require(binocularAgreementWeight in 0.0..1.0)
            require(temporalSmoothing in 0.0..1.0)
            require(blinkConfidencePenalty in 0.0..1.0)
            require(saccadeSoftLimitDegreesPerSecond > 0.0)
            require(vergenceMinMeters > 0.0)
            require(vergenceMaxMeters > vergenceMinMeters)
            require(headGazeBlend in 0.0..1.0)
            require(captureTimingWeight in 0.0..1.0)
            require(maximumTrackAgeMillis > 0.0)
            require(maximumRetainedTracks >= 32)
            listOf(
                aleatoricUncertaintyWeight,
                temporalUncertaintyWeight,
                binocularUncertaintyWeight,
                innovationUncertaintyWeight,
                staleEvidenceUncertaintyWeight,
            ).forEach { require(it >= 0.0) }
            require(
                (
                    aleatoricUncertaintyWeight +
                        temporalUncertaintyWeight +
                        binocularUncertaintyWeight +
                        innovationUncertaintyWeight +
                        staleEvidenceUncertaintyWeight
                    ) > 0.0,
            )
            require(hardUncertainty in 0.0..1.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val face: FaceFrame,
        val eyes: EyeTrackingFrame?,
    )

    @Synchronized
    fun configureCustomization(
        headPoseStrictness: Double,
        gazePriority: Double,
        eyeTemporalStrictness: Double,
        semanticMotionSensitivity: Double,
    ) {
        val head = headPoseStrictness.coerceIn(0.0, 1.0)
        val gaze = gazePriority.coerceIn(0.0, 1.0)
        val temporal = eyeTemporalStrictness.coerceIn(0.0, 1.0)
        val sensitivity = semanticMotionSensitivity.coerceIn(0.0, 1.0)

        config = config.copy(
            minimumFaceConfidence = (
                0.10 + head * 0.14
            ).coerceIn(0.10, 0.30),
            minimumEyeConfidence = (
                0.08 + gaze * 0.12
            ).coerceIn(0.08, 0.22),
            binocularAgreementWeight = (
                0.16 + gaze * 0.24
            ).coerceIn(0.10, 0.42),
            temporalSmoothing = (
                0.46 + temporal * 0.40
            ).coerceIn(0.35, 0.92),
            saccadeSoftLimitDegreesPerSecond = (
                420.0 + sensitivity * 420.0
            ).coerceIn(360.0, 900.0),
            headGazeBlend = (
                0.08 + gaze * 0.30
            ).coerceIn(0.04, 0.45),
        )
    }

    @Synchronized
    fun configuration(): Config = config

    data class UncertaintyDecomposition(
        val aleatoric: Double,
        val temporal: Double,
        val binocular: Double,
        val innovation: Double,
        val staleEvidence: Double,
        val modelMismatch: Double,
        val outOfDistributionRisk: Double,
        val total: Double,
        val trustworthy: Boolean,
    )

    data class Result(
        val state: FaceEyeFusionState,
        val binocularAgreement: Double,
        val eyeFresh: Boolean,
        val faceFresh: Boolean,
        val rejectionReasons: List<String>,
        val diagnostics: List<String>,
        val uncertainty: UncertaintyDecomposition,
    )

    private data class TrackState(
        var gaze: Vec3 = Vec3(0.0, 0.0, 1.0),
        var headRotation: Quat = Quat.IDENTITY,
        var lastGaze: Vec3 = Vec3(0.0, 0.0, 1.0),
        var lastTimestampNanos: Long = Long.MIN_VALUE,
        var gazeSpeedDegreesPerSecond: Double = 0.0,
        var gazeAccelerationDegreesPerSecond2: Double = 0.0,
        var stability: Double = 0.50,
        var confidenceEma: Double = 0.50,
        var disagreementEma: Double = 0.0,
        var innovationEmaDegrees: Double = 0.0,
        var uncertaintyEma: Double = 0.50,
    )

    private val tracks = mutableMapOf<Long, TrackState>()

    @Synchronized
    fun fuse(input: Input): Result {
        val faceAgeMillis = ageMillis(
            input.timestampNanos,
            input.face.timestampNanos,
        )

        val faceFresh = faceAgeMillis <= config.maximumStaleMillis
        val eye = input.eyes

        val eyeAgeMillis = eye?.let {
            ageMillis(
                input.timestampNanos,
                it.timestampNanos,
            )
        } ?: Double.POSITIVE_INFINITY

        val eyeFresh =
            eye != null &&
                eyeAgeMillis <= config.maximumStaleMillis

        val faceConfidence = faceConfidence(input.face)
        val left = eye?.left
        val right = eye?.right

        val leftConfidence = left?.confidence
            ?.coerceIn(0.0, 1.0)
            ?.takeIf { it >= config.minimumEyeConfidence }
            ?: 0.0

        val rightConfidence = right?.confidence
            ?.coerceIn(0.0, 1.0)
            ?.takeIf { it >= config.minimumEyeConfidence }
            ?: 0.0

        val binocularAgreement = binocularAgreement(left, right)
        val captureTimingQuality =
            FaceLandmarkEngine.temporalQualityScale(
                captureTimingConfidence =
                    input.face.captureTimingConfidence.coerceIn(0.0, 1.0),
                rollingShutterSkewNanos =
                    input.face.rollingShutterSkewNanos.coerceAtLeast(0L),
                frameDurationNanos =
                    input.face.frameDurationNanos.coerceAtLeast(0L),
                exposureTimeNanos =
                    input.face.exposureTimeNanos.coerceAtLeast(0L),
            ).coerceIn(0.05, 1.0)

        val rawGaze = when {
            left != null && right != null &&
                leftConfidence > 0.0 && rightConfidence > 0.0 -> {
                val weighted =
                    left.gazeDirection * leftConfidence +
                        right.gazeDirection * rightConfidence
                weighted.normalized()
            }
            left != null && leftConfidence > 0.0 ->
                left.gazeDirection.normalized()
            right != null && rightConfidence > 0.0 ->
                right.gazeDirection.normalized()
            eye != null ->
                eye.combinedGaze.normalized()
            else ->
                Vec3(0.0, 0.0, 1.0)
        }

        val trackId =
            input.face.trackingId?.toLong()
                ?: stableTrackId(input.face)

        pruneExpiredTracks(input.timestampNanos)
        enforceTrackCapacity()

        val state = tracks.getOrPut(trackId) {
            TrackState()
        }

        val dtSeconds =
            if (state.lastTimestampNanos == Long.MIN_VALUE) {
                1.0 / 60.0
            } else {
                (
                    input.timestampNanos -
                        state.lastTimestampNanos
                    ).coerceAtLeast(1L).toDouble() / 1e9
            }.coerceIn(1.0 / 240.0, 0.50)

        val alpha =
            config.temporalSmoothing *
                if (eyeFresh) 1.0 else 0.42

        val angularInnovation =
            Math.toDegrees(
                acos(
                    state.gaze
                        .normalized()
                        .dot(
                            rawGaze
                                .normalized(),
                        )
                        .coerceIn(-1.0, 1.0),
                ),
            )

        val predictedSaccadeAllowance =
            (
                state.gazeSpeedDegreesPerSecond *
                    dtSeconds *
                    1.35 +
                    0.5 *
                        state.gazeAccelerationDegreesPerSecond2 *
                        dtSeconds *
                        dtSeconds
                ).coerceIn(
                8.0,
                config.saccadeSoftLimitDegreesPerSecond * 1.55,
            )

        val adaptiveInnovationGate =
            (
                24.0 +
                    predictedSaccadeAllowance +
                    (1.0 - max(
                        leftConfidence,
                        rightConfidence,
                    )) * 48.0
                ).coerceIn(
                18.0,
                140.0,
            )

        val rejectInnovation =
            angularInnovation >
                adaptiveInnovationGate &&
                max(
                    leftConfidence,
                    rightConfidence,
                ) < 0.72 &&
                eyeFresh

        val acceptedGaze =
            if (rejectInnovation) {
                state.gaze
            } else {
                rawGaze
            }

        val smoothedGaze =
            state.gaze.slerpVector(
                acceptedGaze,
                alpha.coerceIn(0.04, 0.96),
            ).normalized()

        val headRotation = headQuaternion(
            yawDegrees = input.face.yawDegrees.toDouble(),
            rollDegrees = input.face.rollDegrees.toDouble(),
        )

        val blendedHead =
            state.headRotation.slerp(
                headRotation,
                if (faceFresh) 0.68 else 0.22,
            )

        val saccadeSpeed = angularVelocityDegrees(
            state.lastGaze,
            smoothedGaze,
            dtSeconds,
        )

        val blinkProbability = when {
            left == null && right == null -> 0.0
            else -> max(
                left?.let { blinkProbability(it) } ?: 0.0,
                right?.let { blinkProbability(it) } ?: 0.0,
            )
        }

        val blinkPenalty =
            1.0 -
                blinkProbability *
                config.blinkConfidencePenalty

        val faceScore =
            faceConfidence *
                (if (faceFresh) 1.0 else 0.46) *
                (
                    1.0 -
                        config.captureTimingWeight +
                        config.captureTimingWeight *
                            captureTimingQuality
                )

        val eyeScore = when {
            !eyeFresh -> 0.0
            leftConfidence <= 0.0 && rightConfidence <= 0.0 -> 0.0
            else -> (
                max(leftConfidence, rightConfidence) * 0.62 +
                    binocularAgreement * config.binocularAgreementWeight +
                    (eye?.fixationConfidence ?: 0.0) * 0.10 +
                    (1.0 - angularInnovation / 120.0).coerceIn(0.0, 1.0) * 0.08
                ).coerceIn(0.0, 1.0) *
                (
                    1.0 -
                        config.captureTimingWeight +
                        config.captureTimingWeight *
                            captureTimingQuality
                )
        }

        val instantaneousConfidence = (
            faceScore * 0.40 +
                eyeScore * 0.42 +
                binocularAgreement * 0.10 +
                state.stability * 0.08
            ) *
            blinkPenalty
            .coerceIn(0.0, 1.0)

        state.confidenceEma =
            state.confidenceEma * 0.76 +
                instantaneousConfidence * 0.24

        state.disagreementEma =
            state.disagreementEma * 0.78 +
                (1.0 - binocularAgreement) * 0.22

        state.innovationEmaDegrees =
            state.innovationEmaDegrees * 0.72 +
                angularInnovation * 0.28

        val confidence = (
            (
                state.confidenceEma * 0.72 +
                    instantaneousConfidence * 0.28
                ) *
                (
                    1.0 -
                        state.uncertaintyEma * 0.18
                    ).coerceIn(0.72, 1.0)
            ).coerceIn(0.0, 1.0)

        val vergence = estimateVergence(
            left = left,
            right = right,
            face = input.face,
        )

        val newStability = (
            state.stability * 0.76 +
                (1.0 - min(
                    1.0,
                    saccadeSpeed /
                        config.saccadeSoftLimitDegreesPerSecond,
                )) * 0.24
            ).coerceIn(0.0, 1.0)

        state.lastGaze = smoothedGaze
        state.gaze = smoothedGaze
        state.headRotation = blendedHead
        state.lastTimestampNanos = input.timestampNanos
        val measuredAngularAcceleration =
            (
                saccadeSpeed -
                    state.gazeSpeedDegreesPerSecond
            ) /
                dtSeconds.coerceAtLeast(1e-6)

        state.gazeSpeedDegreesPerSecond = saccadeSpeed
        state.gazeAccelerationDegreesPerSecond2 =
            state.gazeAccelerationDegreesPerSecond2 * 0.72 +
                measuredAngularAcceleration
                    .coerceIn(-18_000.0, 18_000.0) * 0.28
        state.stability = newStability

        val aleatoricUncertainty = (
            (
                1.0 -
                    (
                        faceConfidence * 0.46 +
                            max(
                                leftConfidence,
                                rightConfidence,
                            ) * 0.54
                        )
                    ).coerceIn(0.0, 1.0) +
                blinkProbability * 0.35
            ).coerceIn(0.0, 1.0)

        val temporalUncertainty =
            (1.0 - newStability).coerceIn(0.0, 1.0)

        val binocularUncertainty =
            (1.0 - binocularAgreement).coerceIn(0.0, 1.0)

        val innovationUncertainty =
            (
                angularInnovation /
                    adaptiveInnovationGate.coerceAtLeast(1.0)
                ).coerceIn(0.0, 1.0)

        val staleEvidenceUncertainty = (
            (if (faceFresh) 0.0 else 0.58) +
                (if (eyeFresh) 0.0 else 0.42)
            ).coerceIn(0.0, 1.0)

        val modelMismatch = (
            state.disagreementEma * 0.56 +
                (
                    state.innovationEmaDegrees / 90.0
                    ).coerceIn(0.0, 1.0) * 0.44
            ).coerceIn(0.0, 1.0)

        val totalWeight = (
            config.aleatoricUncertaintyWeight +
                config.temporalUncertaintyWeight +
                config.binocularUncertaintyWeight +
                config.innovationUncertaintyWeight +
                config.staleEvidenceUncertaintyWeight
            ).coerceAtLeast(1e-9)

        val rawUncertainty = (
            aleatoricUncertainty *
                config.aleatoricUncertaintyWeight +
                temporalUncertainty *
                config.temporalUncertaintyWeight +
                binocularUncertainty *
                config.binocularUncertaintyWeight +
                innovationUncertainty *
                config.innovationUncertaintyWeight +
                staleEvidenceUncertainty *
                config.staleEvidenceUncertaintyWeight
            ) / totalWeight

        val oodRisk = (
            innovationUncertainty * 0.44 +
                binocularUncertainty * 0.26 +
                staleEvidenceUncertainty * 0.18 +
                (
                    1.0 -
                        max(
                            faceConfidence,
                            max(
                                leftConfidence,
                                rightConfidence,
                            ),
                        )
                    ).coerceIn(0.0, 1.0) * 0.12
            ).coerceIn(0.0, 1.0)

        val totalUncertainty = (
            rawUncertainty * 0.76 +
                modelMismatch * 0.24
            ).coerceIn(0.0, 1.0)

        state.uncertaintyEma =
            state.uncertaintyEma * 0.72 +
                totalUncertainty * 0.28

        val uncertainty = UncertaintyDecomposition(
            aleatoric = aleatoricUncertainty,
            temporal = temporalUncertainty,
            binocular = binocularUncertainty,
            innovation = innovationUncertainty,
            staleEvidence = staleEvidenceUncertainty,
            modelMismatch = modelMismatch,
            outOfDistributionRisk = oodRisk,
            total = state.uncertaintyEma,
            trustworthy =
                state.uncertaintyEma <
                    config.hardUncertainty &&
                    oodRisk < 0.70,
        )

        val authoritativeConfidence =
            (
                confidence *
                    if (uncertainty.trustworthy) {
                        1.0
                    } else {
                        0.62
                    }
            ).coerceIn(0.0, 1.0)

        val reasons = buildList {
            if (!uncertainty.trustworthy) {
                add("gaze uncertainty exceeded authoritative threshold")
            }
            if (!faceFresh) add("face evidence stale")
            if (!eyeFresh) add("eye evidence stale or unavailable")
            if (faceScore < config.minimumFaceConfidence) {
                add("face confidence below head-pose threshold")
            }
            if (eyeScore < config.minimumEyeConfidence) {
                add("eye confidence below gaze threshold")
            }
            if (binocularAgreement < 0.45 && left != null && right != null) {
                add("left/right gaze vectors disagree")
            }
            if (blinkProbability > 0.82) {
                add("blink state suppressing gaze confidence")
            }
            if (angularInnovation > adaptiveInnovationGate) {
                add("gaze innovation exceeded adaptive angular gate")
            }
            if (rejectInnovation) {
                add("low-confidence gaze innovation rejected")
            }
            if (state.disagreementEma > 0.45) {
                add("persistent binocular disagreement")
            }
        }

        val diagnostics = listOf(
            "faceAgeMs=" + format(faceAgeMillis),
            "eyeAgeMs=" + format(eyeAgeMillis),
            "faceConfidence=" + format(faceScore),
            "eyeConfidence=" + format(eyeScore),
            "binocularAgreement=" + format(binocularAgreement),
            "gazeStability=" + format(newStability),
            "angularInnovationDeg=" + format(angularInnovation),
            "adaptiveInnovationGateDeg=" + format(adaptiveInnovationGate),
            "innovationRejected=" + rejectInnovation,
            "confidenceEma=" + format(state.confidenceEma),
            "authoritativeConfidence=" + format(authoritativeConfidence),
            "uncertaintyTrustworthy=" + uncertainty.trustworthy,
            "captureTimingQuality=" + format(captureTimingQuality),
            "saccadeDps=" + format(saccadeSpeed),
            "gazeAngularAccelerationDps2=" +
                format(
                    state.gazeAccelerationDegreesPerSecond2,
                ),
            "vergenceMeters=" + (
                vergence?.let(::format) ?: "unavailable"
            ),
            "blinkProbability=" + format(blinkProbability),
            "uncertaintyTotal=" + format(uncertainty.total),
            "uncertaintyAleatoric=" + format(uncertainty.aleatoric),
            "uncertaintyTemporal=" + format(uncertainty.temporal),
            "uncertaintyBinocular=" + format(uncertainty.binocular),
            "uncertaintyInnovation=" + format(uncertainty.innovation),
            "uncertaintyModelMismatch=" + format(uncertainty.modelMismatch),
            "oodRisk=" + format(uncertainty.outOfDistributionRisk),
        )

        return Result(
            state = FaceEyeFusionState(
                timestampNanos = input.timestampNanos,
                trackId = trackId,
                headRotation = blendedHead,
                headYawDegrees = input.face.yawDegrees.toDouble(),
                headRollDegrees = input.face.rollDegrees.toDouble(),
                combinedGazeCamera = smoothedGaze,
                leftGazeCamera = left?.gazeDirection?.normalized(),
                rightGazeCamera = right?.gazeDirection?.normalized(),
                leftOpenness = left?.openness ?: input.face.leftEyeOpenProbability,
                rightOpenness = right?.openness ?: input.face.rightEyeOpenProbability,
                blinkProbability = blinkProbability,
                gazeStability = newStability,
                saccadeVelocityDegreesPerSecond = saccadeSpeed,
                vergenceMeters = vergence,
                confidence = authoritativeConfidence,
                fresh = faceFresh || eyeFresh,
                source = if (eyeFresh) {
                    PerceptionSource.ML_KIT
                } else {
                    PerceptionSource.ML_KIT
                },
                diagnostics = diagnostics,
            ),
            binocularAgreement = binocularAgreement,
            eyeFresh = eyeFresh,
            faceFresh = faceFresh,
            rejectionReasons = reasons.distinct(),
            diagnostics = diagnostics,
            uncertainty = uncertainty,
        )
    }

    @Synchronized
    fun reset() {
        tracks.clear()
    }

    private fun pruneExpiredTracks(nowTimestampNanos: Long) {
        tracks.entries.removeIf { (_, state) ->
            state.lastTimestampNanos != Long.MIN_VALUE &&
                ageMillis(
                    nowTimestampNanos,
                    state.lastTimestampNanos,
                ) > config.maximumTrackAgeMillis
        }
    }

    private fun enforceTrackCapacity() {
        if (tracks.size < config.maximumRetainedTracks) return
        val overflow = tracks.size - config.maximumRetainedTracks + 1
        tracks.entries
            .sortedBy { entry -> entry.value.lastTimestampNanos }
            .take(overflow)
            .forEach { entry -> tracks.remove(entry.key) }
    }

    private fun faceConfidence(face: FaceFrame): Double =
        if (face.points.isEmpty()) {
            0.0
        } else {
            val geometryConfidence =
                face.points
                    .map { it.confidence.coerceIn(0.0, 1.0) }
                    .average()
                    .coerceIn(0.0, 1.0)

            min(
                geometryConfidence,
                face.captureTimingConfidence.coerceIn(0.0, 1.0),
            ).coerceIn(0.0, 1.0)
        }

    private fun binocularAgreement(
        left: EyeObservation?,
        right: EyeObservation?,
    ): Double {
        if (left == null || right == null) return 0.56

        val a = left.gazeDirection.normalized()
        val b = right.gazeDirection.normalized()
        val dot = (
            a.x * b.x +
                a.y * b.y +
                a.z * b.z
            ).coerceIn(-1.0, 1.0)

        return (
            1.0 -
                acos(dot) / Math.PI
            ).coerceIn(0.0, 1.0)
    }

    private fun estimateVergence(
        left: EyeObservation?,
        right: EyeObservation?,
        face: FaceFrame,
    ): Double? {
        if (left == null || right == null) return null

        val a = left.gazeDirection.normalized()
        val b = right.gazeDirection.normalized()

        val horizontal = abs(
            atan2(a.x, a.z) -
                atan2(b.x, b.z),
        )

        val leftEyeCenter = face.leftEyeCenter
        val rightEyeCenter = face.rightEyeCenter

        val interEyePixels =
            if (leftEyeCenter != null && rightEyeCenter != null) {
                leftEyeCenter.distanceTo(rightEyeCenter)
            } else {
                0.0
            }

        if (horizontal < 1e-4 || interEyePixels <= 0.0) {
            return null
        }

        // Without stereo calibration the estimate is a useful relative cue, not metric truth.
        // Keep it conservative and bounded, and expose its uncertainty through confidence.
        val normalizedBaseline = (
            interEyePixels /
                max(
                    face.boundsRight - face.boundsLeft,
                    1.0f,
                )
            ).toDouble().coerceIn(0.05, 0.50)

        val estimate = (
            normalizedBaseline /
                kotlin.math.tan(
                    horizontal.coerceIn(
                        1e-4,
                        Math.PI / 2.0,
                    ) / 2.0,
                )
            ).coerceIn(
                config.vergenceMinMeters,
                config.vergenceMaxMeters,
            )

        return estimate
    }

    private fun blinkProbability(
        eye: EyeObservation,
    ): Double =
        (
            if (eye.blink) 1.0 else 0.0
        ) * 0.70 +
            (1.0 - eye.openness.coerceIn(0.0, 1.0)) * 0.30

    private fun stableTrackId(face: FaceFrame): Long {
        val cx = ((face.boundsLeft + face.boundsRight) * 0.5f).toInt()
        val cy = ((face.boundsTop + face.boundsBottom) * 0.5f).toInt()
        val width = (
            face.boundsRight - face.boundsLeft
        ).coerceAtLeast(1f).toInt()

        return (
            cx.toLong() / 32L * 73856093L +
                cy.toLong() / 32L * 19349663L +
                width.toLong() / 32L * 83492791L
            ).coerceAtLeast(1L)
    }

    private fun ageMillis(
        nowNanos: Long,
        sampleNanos: Long,
    ): Double =
        (
            nowNanos -
                sampleNanos
            ).coerceAtLeast(0L) / 1e6

    private fun angularVelocityDegrees(
        previous: Vec3,
        current: Vec3,
        dtSeconds: Double,
    ): Double {
        val dot = (
            previous.normalized().x * current.normalized().x +
                previous.normalized().y * current.normalized().y +
                previous.normalized().z * current.normalized().z
            ).coerceIn(-1.0, 1.0)

        return Math.toDegrees(
            acos(dot),
        ) / dtSeconds.coerceAtLeast(1e-4)
    }

    private fun headQuaternion(
        yawDegrees: Double,
        rollDegrees: Double,
    ): Quat {
        val yaw = Math.toRadians(yawDegrees)
        val roll = Math.toRadians(rollDegrees)

        val halfYaw = yaw * 0.5
        val halfRoll = roll * 0.5

        val cy = kotlin.math.cos(halfYaw)
        val sy = sin(halfYaw)
        val cr = kotlin.math.cos(halfRoll)
        val sr = sin(halfRoll)

        return Quat(
            w = cy * cr,
            x = sr * cy,
            y = sy * cr,
            z = -sy * sr,
        ).normalized()
    }

    private fun Vec3.slerpVector(
        target: Vec3,
        alpha: Double,
    ): Vec3 {
        val a = normalized()
        val b = target.normalized()
        val dot = (
            a.x * b.x +
                a.y * b.y +
                a.z * b.z
            ).coerceIn(-1.0, 1.0)

        if (dot > 0.9995) {
            return lerp(b, alpha).normalized()
        }

        val theta = acos(dot)
        val sinTheta = kotlin.math.sin(theta).coerceAtLeast(1e-6)

        val scaleA = kotlin.math.sin((1.0 - alpha) * theta) / sinTheta
        val scaleB = kotlin.math.sin(alpha * theta) / sinTheta

        return (
            a * scaleA +
                b * scaleB
            ).normalized()
    }

    private fun format(value: Double): String =
        if (!value.isFinite()) {
            "nan"
        } else {
            "%.4f".format(java.util.Locale.US, value)
        }
}

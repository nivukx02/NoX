package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.EyeState
import com.nivukx.nox.core.model.FaceEyeFusionState
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Production face/eye consistency gate.
 *
 * Face landmarks and eye gaze are related but not interchangeable measurements. The engine checks
 * whether their geometry, temporal behavior and confidence agree before exposing a high-confidence
 * gaze/head state to downstream motion and avatar systems.
 *
 * The engine is deliberately conservative:
 * - head orientation is never overwritten by gaze;
 * - gaze disagreement reduces confidence instead of silently forcing alignment;
 * - blink events do not erase valid head motion;
 * - 3D face residuals can veto inflated eye confidence;
 * - temporal state is retained per tracked face;
 * - all outputs remain bounded and explainable.
 */
class FaceEyeConsistencyEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumExpectedGazeOffsetDegrees: Double = 42.0,
        val hardGazeOffsetDegrees: Double = 75.0,
        val maximumBlinkDisagreement: Double = 0.30,
        val maximumFace3DErrorPx: Double = 8.0,
        val targetFace3DErrorPx: Double = 2.5,
        val minimumFaceConfidence: Double = 0.20,
        val minimumEyeConfidence: Double = 0.15,
        val minimumTemporalStability: Double = 0.22,
        val temporalHalfLifeMillis: Double = 180.0,
        val smoothingResponsiveness: Double = 0.72,
        val gazeAccelerationPenalty: Double = 0.14,
        val maximumTrackGapSeconds: Double = 0.50,
        val maximumRetainedTracks: Int = 128,
    ) {
        init {
            require(maximumExpectedGazeOffsetDegrees > 0.0)
            require(hardGazeOffsetDegrees > maximumExpectedGazeOffsetDegrees)
            require(maximumBlinkDisagreement in 0.0..1.0)
            require(maximumFace3DErrorPx > targetFace3DErrorPx)
            require(targetFace3DErrorPx >= 0.0)
            require(minimumFaceConfidence in 0.0..1.0)
            require(minimumEyeConfidence in 0.0..1.0)
            require(minimumTemporalStability in 0.0..1.0)
            require(temporalHalfLifeMillis > 0.0)
            require(smoothingResponsiveness in 0.0..1.0)
            require(gazeAccelerationPenalty in 0.0..1.0)
            require(maximumTrackGapSeconds > 0.0)
            require(maximumRetainedTracks >= 16)
        }
    }

    data class Input(
        val face: FaceEyeFusionState?,
        val eyes: EyeState?,
        val face3DConfidence: Double = 1.0,
        val face3DErrorPx: Double = 0.0,
        val timestampNanos: Long,
        val fusionUncertainty:
            FaceEyeFusionEngine.UncertaintyDecomposition? = null,
    ) {
        init {
            require(face3DConfidence in 0.0..1.0)
            require(face3DErrorPx >= 0.0)
            require(timestampNanos >= 0L)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val available: Boolean,
        val confidence: Double,
        val faceConfidence: Double,
        val eyeConfidence: Double,
        val headGazeConsistency: Double,
        val blinkConsistency: Double,
        val temporalStability: Double,
        val face3DConsistency: Double,
        val uncertaintyTotal: Double,
        val uncertaintyOodRisk: Double,
        val uncertaintyTrustworthy: Boolean,
        val gazeAngularVelocityDegreesPerSecond: Double,
        val temporalHistoryReset: Boolean,
        val adjustedGazeDirection: Vec3?,
        val warnings: List<String>,
        val diagnostics: List<String>,
    ) {
        val reliable: Boolean
            get() =
                available &&
                    confidence >= 0.55 &&
                    warnings.none { it.startsWith("fatal:") }
    }

    private data class TrackState(
        var timestampNanos: Long,
        var gazeDirection: Vec3,
        var angularVelocityDegreesPerSecond: Double,
        var stability: Double,
    )

    private val history = HashMap<Long, TrackState>()

    @Synchronized
    fun evaluate(input: Input): Result {
        val face = input.face
        if (face == null) {
            prune(input.timestampNanos)
            return Result(
                timestampNanos = input.timestampNanos,
                available = false,
                confidence = 0.0,
                faceConfidence = 0.0,
                eyeConfidence = 0.0,
                headGazeConsistency = 0.0,
                blinkConsistency = 0.0,
                temporalStability = 0.0,
                face3DConsistency = 0.0,
                uncertaintyTotal = 1.0,
                uncertaintyOodRisk = 1.0,
                uncertaintyTrustworthy = false,
                gazeAngularVelocityDegreesPerSecond = 0.0,
                temporalHistoryReset = false,
                adjustedGazeDirection = null,
                warnings = listOf("face state unavailable"),
                diagnostics = emptyList(),
            )
        }

        val trackKey = face.trackId
        val eye = input.eyes

        val headForward = face.headRotation
            .rotate(Vec3(0.0, 0.0, 1.0))
            .normalized()

        val rawGaze = (
            face.combinedGazeCamera
                .let { if (it.norm() < 1e-6) headForward else it.normalized() }
        )

        val headGazeAngle = angleDegrees(
            headForward,
            rawGaze,
        )

        val headConsistency = when {
            headGazeAngle <= config.maximumExpectedGazeOffsetDegrees ->
                1.0

            headGazeAngle >= config.hardGazeOffsetDegrees ->
                0.0

            else ->
                (
                    1.0 -
                        (
                            headGazeAngle -
                                config.maximumExpectedGazeOffsetDegrees
                            ) /
                            (
                                config.hardGazeOffsetDegrees -
                                    config.maximumExpectedGazeOffsetDegrees
                            )
                    ).coerceIn(0.0, 1.0)
        }

        val blinkConsistency = blinkConsistency(
            face = face,
            eyes = eye,
        )

        val face3dConsistency = face3DConsistency(
            confidence = input.face3DConfidence,
            errorPx = input.face3DErrorPx,
        )

        val uncertaintyTotal =
            input.fusionUncertainty?.total ?: 0.0
        val uncertaintyOodRisk =
            input.fusionUncertainty
                ?.outOfDistributionRisk
                ?: 0.0
        val uncertaintyTrustworthy =
            input.fusionUncertainty
                ?.trustworthy
                ?: true

        val currentEyeConfidence = eye?.confidence
            ?.coerceIn(0.0, 1.0)
            ?: 0.0

        val currentFaceConfidence =
            face.confidence.coerceIn(0.0, 1.0)

        val temporal = updateTemporal(
            trackId = trackKey,
            timestampNanos = input.timestampNanos,
            gaze = rawGaze,
        )

        val eyeAvailabilityScore =
            if (eye == null) {
                0.58
            } else {
                (
                    currentEyeConfidence * 0.62 +
                        eye.gazeStability * 0.24 +
                        (
                            1.0 -
                                eye.blinkProbability * 0.35
                            ).coerceIn(0.0, 1.0) * 0.14
                    ).coerceIn(0.0, 1.0)
            }

        val blinkPenalty =
            if (eye?.bothEyesOpen == false) {
                0.88
            } else {
                1.0
            }

        val motionPenalty = (
            1.0 -
                min(
                    1.0,
                    temporal.angularVelocityDegreesPerSecond /
                        900.0,
                ) *
                config.gazeAccelerationPenalty
            ).coerceIn(0.70, 1.0)

        val geometricAgreementPenalty =
            (0.55 + 0.45 * headConsistency)
                .coerceIn(0.55, 1.0)
        val face3dEvidencePenalty =
            (0.55 + 0.45 * face3dConsistency)
                .coerceIn(0.55, 1.0)

        val confidence = (
            currentFaceConfidence * 0.24 +
                currentEyeConfidence * 0.22 +
                headConsistency * 0.16 +
                blinkConsistency * 0.10 +
                temporal.stability * 0.12 +
                face3dConsistency * 0.16
            ).coerceIn(0.0, 1.0) *
            geometricAgreementPenalty *
            face3dEvidencePenalty *
            blinkPenalty *
            motionPenalty *
            (
                1.0 -
                    uncertaintyTotal * 0.22
                ).coerceIn(0.74, 1.0)

        val warnings = buildList {
            if (currentFaceConfidence <
                config.minimumFaceConfidence
            ) {
                add("face confidence below production target")
            }

            if (eye != null &&
                currentEyeConfidence <
                config.minimumEyeConfidence
            ) {
                add("eye confidence below production target")
            }

            if (headGazeAngle >
                config.maximumExpectedGazeOffsetDegrees
            ) {
                add(
                    "gaze/head angular disagreement " +
                        "%.1f".format(headGazeAngle) +
                        " degrees",
                )
            }

            if (blinkConsistency <
                1.0 -
                    config.maximumBlinkDisagreement
            ) {
                add("face and eye blink states disagree")
            }

            if (face3dConsistency < 0.55) {
                add("3D face reconstruction confidence reduced")
            }

            if (temporal.stability <
                config.minimumTemporalStability
            ) {
                add("gaze temporal stability below target")
            }

            if (headGazeAngle >=
                config.hardGazeOffsetDegrees
            ) {
                add("fatal: face and eye geometry disagree materially")
            }
            if (uncertaintyTotal >= 0.78) {
                add("face-eye fusion uncertainty above production threshold")
            }
            if (uncertaintyOodRisk >= 0.70) {
                add("face-eye out-of-distribution risk elevated")
            }
            if (!uncertaintyTrustworthy) {
                add("face-eye uncertainty contract is not trustworthy")
            }
        }

        val diagnostics = listOf(
            "trackId=$trackKey",
            "headGazeAngleDegrees=" +
                "%.3f".format(headGazeAngle),
            "headConsistency=" +
                "%.4f".format(headConsistency),
            "blinkConsistency=" +
                "%.4f".format(blinkConsistency),
            "face3DConsistency=" +
                "%.4f".format(face3dConsistency),
            "temporalStability=" +
                "%.4f".format(temporal.stability),
            "gazeAngularVelocity=" +
                "%.3f".format(
                    temporal.angularVelocityDegreesPerSecond,
                ),
            "confidence=" +
                "%.4f".format(confidence),
            "uncertaintyTotal=" +
                "%.4f".format(uncertaintyTotal),
            "uncertaintyOodRisk=" +
                "%.4f".format(uncertaintyOodRisk),
            "uncertaintyTrustworthy=" +
                uncertaintyTrustworthy,
        )

        return Result(
            timestampNanos = input.timestampNanos,
            available = true,
            confidence = confidence.coerceIn(0.0, 1.0),
            faceConfidence = currentFaceConfidence,
            eyeConfidence = currentEyeConfidence,
            headGazeConsistency = headConsistency,
            blinkConsistency = blinkConsistency,
            temporalStability = temporal.stability,
            face3DConsistency = face3dConsistency,
            uncertaintyTotal = uncertaintyTotal,
            uncertaintyOodRisk = uncertaintyOodRisk,
            uncertaintyTrustworthy = uncertaintyTrustworthy,
            gazeAngularVelocityDegreesPerSecond =
                temporal.angularVelocityDegreesPerSecond,
            temporalHistoryReset =
                temporal.historyReset,
            adjustedGazeDirection =
                temporal.gazeDirection,
            warnings = warnings.distinct(),
            diagnostics = diagnostics,
        )
    }

    @Synchronized
    fun reset() {
        history.clear()
    }

    private data class TemporalResult(
        val gazeDirection: Vec3,
        val angularVelocityDegreesPerSecond: Double,
        val stability: Double,
        val historyReset: Boolean,
    )

    private fun updateTemporal(
        trackId: Long,
        timestampNanos: Long,
        gaze: Vec3,
    ): TemporalResult {
        val current =
            gaze.normalized()

        val previous =
            history[trackId]

        if (previous == null) {
            history[trackId] =
                TrackState(
                    timestampNanos = timestampNanos,
                    gazeDirection = current,
                    angularVelocityDegreesPerSecond = 0.0,
                    stability = 0.72,
                )

            prune(timestampNanos)

            return TemporalResult(
                gazeDirection = current,
                angularVelocityDegreesPerSecond = 0.0,
                stability = 0.72,
                historyReset = true,
            )
        }

        val gapSeconds =
            (
                timestampNanos -
                    previous.timestampNanos
            ).coerceAtLeast(0L) /
                1e9

        if (
            gapSeconds >
                config.maximumTrackGapSeconds
        ) {
            history.remove(trackId)
            history[trackId] =
                TrackState(
                    timestampNanos = timestampNanos,
                    gazeDirection = current,
                    angularVelocityDegreesPerSecond = 0.0,
                    stability = 0.72,
                )
            prune(timestampNanos)

            return TemporalResult(
                gazeDirection = current,
                angularVelocityDegreesPerSecond = 0.0,
                stability = 0.72,
                historyReset = true,
            )
        }

        val dt = (
            (timestampNanos - previous.timestampNanos)
                .coerceAtLeast(1L)
                .toDouble() /
                1e9
        ).coerceIn(
            1.0 / 240.0,
            0.50,
        )

        val angularDelta =
            angleDegrees(
                previous.gazeDirection,
                current,
            )

        val angularVelocity =
            angularDelta /
                dt.coerceAtLeast(
                    1e-4,
                )

        val expected =
            previous.gazeDirection

        val responsive =
            config.smoothingResponsiveness

        val filtered =
            expected.lerpDirection(
                current,
                (
                    responsive *
                        (
                            0.35 +
                                previous.stability * 0.45
                            )
                    ).coerceIn(
                        0.10,
                        0.92,
                    ),
            )

        val velocityScore =
            (
                1.0 -
                    angularVelocity /
                        720.0
                ).coerceIn(0.0, 1.0)

        val directionScore =
            (
                1.0 -
                    angularDelta /
                        28.0
                ).coerceIn(0.0, 1.0)

        val stability =
            (
                previous.stability * 0.58 +
                    velocityScore * 0.20 +
                    directionScore * 0.22
                ).coerceIn(0.0, 1.0)

        previous.timestampNanos =
            timestampNanos
        previous.gazeDirection =
            filtered
        previous.angularVelocityDegreesPerSecond =
            angularVelocity
        previous.stability =
            stability

        prune(timestampNanos)

        return TemporalResult(
            gazeDirection = filtered,
            angularVelocityDegreesPerSecond =
                angularVelocity,
            stability = stability,
            historyReset = false,
        )
    }

    private fun blinkConsistency(
        face: FaceEyeFusionState,
        eyes: EyeState?,
    ): Double {
        if (eyes == null) return 0.64

        val faceClosure =
            (
                face.leftOpenness +
                    face.rightOpenness
                ) * 0.5

        val eyeClosure =
            (
                eyes.leftBlinkProbability +
                    eyes.rightBlinkProbability
                ) * 0.5

        val normalizedFaceClosure =
            (
                1.0 -
                    faceClosure.coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)

        val disagreement =
            abs(
                normalizedFaceClosure -
                    eyeClosure,
            )

        return (
            1.0 -
                disagreement
            ).coerceIn(0.0, 1.0)
    }

    private fun face3DConsistency(
        confidence: Double,
        errorPx: Double,
    ): Double {
        val reprojectionScore =
            if (!errorPx.isFinite()) {
                0.0
            } else {
                (
                    1.0 -
                        (
                            errorPx -
                                config.targetFace3DErrorPx
                            ) /
                            (
                                config.maximumFace3DErrorPx -
                                    config.targetFace3DErrorPx
                            )
                    ).coerceIn(0.0, 1.0)
            }

        return (
            confidence.coerceIn(0.0, 1.0) * 0.60 +
                reprojectionScore * 0.40
            ).coerceIn(0.0, 1.0)
    }

    private fun angleDegrees(
        a: Vec3,
        b: Vec3,
    ): Double {
        val na = a.normalized()
        val nb = b.normalized()

        if (na.norm() < 1e-6 ||
            nb.norm() < 1e-6
        ) {
            return 180.0
        }

        return Math.toDegrees(
            acos(
                na.dot(nb).coerceIn(
                    -1.0,
                    1.0,
                ),
            ),
        )
    }

    private fun Vec3.lerpDirection(
        target: Vec3,
        alpha: Double,
    ): Vec3 =
        (
            normalized() *
                (1.0 - alpha) +
                target.normalized() *
                alpha
            ).normalized()

    private fun prune(nowNanos: Long) {
        val cutoff =
            nowNanos -
                (
                    config.maximumTrackGapSeconds *
                        2.5 *
                        1_000_000_000.0
                    ).toLong()

        history.entries.removeIf {
            it.value.timestampNanos < cutoff
        }

        if (history.size <=
            config.maximumRetainedTracks
        ) {
            return
        }

        val overflow =
            history.size -
                config.maximumRetainedTracks

        history.entries
            .sortedBy {
                it.value.timestampNanos
            }
            .take(overflow)
            .forEach {
                history.remove(it.key)
            }
    }
}

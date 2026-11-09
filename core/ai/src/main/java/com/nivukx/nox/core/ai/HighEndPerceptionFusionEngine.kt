
package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.CameraTelemetryState
import com.nivukx.nox.core.model.EyeState
import com.nivukx.nox.core.model.FaceExpressionState
import com.nivukx.nox.core.model.FusionEvidence
import com.nivukx.nox.core.model.HandState
import com.nivukx.nox.core.model.LightingState
import com.nivukx.nox.core.model.LightingStabilityBand
import com.nivukx.nox.core.model.PerceptionSource
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Production-oriented evidence fusion for the NoX RGB motion pipeline.
 *
 * The engine deliberately does not perform model inference itself. Instead it turns asynchronous
 * body, face, eye, hand, depth, topology, lighting and camera evidence into a consistent decision
 * surface with freshness decay, disagreement detection, micro-motion preservation, and diagnostics.
 *
 * Design goals:
 * - no single detector result can silently dominate the pipeline;
 * - stale evidence decays instead of being treated as current truth;
 * - camera motion and poor lighting reduce confidence without deleting motion;
 * - face and eye state can be used to refine head orientation without replacing body geometry;
 * - hand state contributes articulation evidence without forcing a fixed human-only topology;
 * - all thresholds are explicit and tunable.
 */
class HighEndPerceptionFusionEngine(
    private val config: Config = Config(),
) {

    private val adaptiveReliability =
        AdaptiveEvidenceReliability(
            AdaptiveEvidenceReliability.Config(
                baseReliability = 0.74,
                minimumReliability = 0.18,
                learningRate = 0.14,
                recoveryRate = 0.050,
                freshnessHalfLifeMillis = config.staleHalfLifeMillis,
                residualScale = 1.50,
                disagreementScale = 2.40,
                consensusWeight = 0.32,
                residualWeight = 0.20,
                freshnessWeight = 0.18,
                reportedConfidenceWeight = 0.18,
                stabilityWeight = 0.12,
            ),
        )

    data class Config(
        val bodyWeight: Double = 0.30,
        val faceWeight: Double = 0.10,
        val eyeWeight: Double = 0.08,
        val handWeight: Double = 0.08,
        val depthWeight: Double = 0.14,
        val topologyWeight: Double = 0.10,
        val temporalWeight: Double = 0.10,
        val calibrationWeight: Double = 0.06,
        val lightingWeight: Double = 0.02,
        val cameraWeight: Double = 0.02,
        val staleHalfLifeMillis: Double = 220.0,
        val disagreementThreshold: Double = 0.42,
        val cameraRiskScale: Double = 0.18,
        val lightingRiskScale: Double = 0.12,
        val microMotionFloorMetersPerSecond: Double = 0.004,
        val microMotionCeilingMetersPerSecond: Double = 0.30,
        val gazeHeadBlend: Double = 0.24,
    ) {
        init {
            val total = bodyWeight + faceWeight + eyeWeight + handWeight +
                depthWeight + topologyWeight + temporalWeight + calibrationWeight +
                lightingWeight + cameraWeight
            require(total > 0.0)
            require(staleHalfLifeMillis > 0.0)
            require(disagreementThreshold in 0.0..1.0)
            require(cameraRiskScale in 0.0..1.0)
            require(lightingRiskScale in 0.0..1.0)
            require(microMotionFloorMetersPerSecond >= 0.0)
            require(microMotionCeilingMetersPerSecond > microMotionFloorMetersPerSecond)
            require(gazeHeadBlend in 0.0..1.0)
        }
    }

    data class BodyEvidence(
        val confidence: Double,
        val visibleNodes: Int,
        val totalNodes: Int,
        val velocityMetersPerSecond: Vec3 = Vec3(0.0, 0.0, 0.0),
        val accelerationMetersPerSecond2: Vec3 = Vec3(0.0, 0.0, 0.0),
        val freshnessMillis: Double = 0.0,
        val source: PerceptionSource = PerceptionSource.ML_KIT,
    )

    data class DepthEvidence(
        val confidence: Double,
        val temporalStability: Double,
        val metric: Boolean,
        val meanDepthMeters: Double,
        val freshnessMillis: Double = 0.0,
        val source: PerceptionSource = PerceptionSource.ONNX,
    )

    data class TopologyEvidence(
        val confidence: Double,
        val stability: Double,
        val edgeConstraintResidual: Double,
        val nodeCount: Int,
        val freshnessMillis: Double = 0.0,
    )

    data class TemporalEvidence(
        val stability: Double,
        val innovation: Double,
        val velocityContinuity: Double,
        val freshnessMillis: Double = 0.0,
    )

    data class CalibrationEvidence(
        val confidence: Double,
        val reprojectionErrorPx: Double,
        val scaleConfidence: Double,
    )

    data class Input(
        val timestampNanos: Long,
        val body: BodyEvidence,
        val depth: DepthEvidence?,
        val topology: TopologyEvidence?,
        val temporal: TemporalEvidence?,
        val calibration: CalibrationEvidence,
        val lighting: LightingState?,
        val camera: CameraTelemetryState?,
        val face: FaceExpressionState?,
        val eyes: EyeState?,
        val hands: List<HandState>,
    )

    data class MotionSignals(
        val translationalSpeedMetersPerSecond: Double,
        val accelerationMetersPerSecond2: Double,
        val microMotion: Double,
        val motionEnergy: Double,
        val directionChange: Double,
    )

    data class Result(
        val timestampNanos: Long,
        val confidence: Double,
        val evidence: FusionEvidence,
        val reliability: AdaptiveEvidenceReliability.Snapshot? = null,
        val motionSignals: MotionSignals,
        val face: FaceExpressionState?,
        val eyes: EyeState?,
        val hands: List<HandState>,
        val lighting: LightingState?,
        val camera: CameraTelemetryState?,
        val cameraMotionPenalty: Double,
        val lightingPenalty: Double,
        val disagreement: Double,
        val headOrientationBlend: Double,
        val warnings: List<String>,
        val diagnostics: List<String>,
    ) {
        val reliable: Boolean
            get() = confidence >= 0.55 && disagreement < 0.55

        val preserveMicroMotion: Boolean
            get() =
                motionSignals.microMotion > 0.0 &&
                    confidence >= 0.35 &&
                    cameraMotionPenalty < 0.80
    }

    @Synchronized
    fun fuse(input: Input): Result {
        val body = sanitize(input.body.confidence)
        val visible = if (input.body.totalNodes <= 0) 0.0 else {
            input.body.visibleNodes.toDouble() /
                input.body.totalNodes.toDouble()
        }.coerceIn(0.0, 1.0)

        val bodyFresh = freshnessWeight(input.body.freshnessMillis)
        val depthFresh = input.depth?.let { freshnessWeight(it.freshnessMillis) } ?: 0.0
        val topologyFresh = input.topology?.let { freshnessWeight(it.freshnessMillis) } ?: 0.0
        val temporalFresh = input.temporal?.let { freshnessWeight(it.freshnessMillis) } ?: 0.0

        val depth = input.depth?.let {
            sanitize(it.confidence) * depthFresh
        } ?: 0.0

        val topology = input.topology?.let {
            (
                sanitize(it.confidence) * 0.58 +
                    sanitize(it.stability) * 0.32 +
                    residualConfidence(it.edgeConstraintResidual) * 0.10
                ) * topologyFresh
        } ?: 0.0

        val temporal = input.temporal?.let {
            (
                sanitize(it.stability) * 0.56 +
                    sanitize(it.velocityContinuity) * 0.29 +
                    (1.0 - sanitize(it.innovation)) * 0.15
                ) * temporalFresh
        } ?: 0.0

        val calibration = (
            sanitize(input.calibration.confidence) * 0.60 +
                sanitize(input.calibration.scaleConfidence) * 0.26 +
                residualConfidence(input.calibration.reprojectionErrorPx) * 0.14
            ).coerceIn(0.0, 1.0)

        val face = input.face?.let {
            sanitize(it.confidence) * if (it.fresh) 1.0 else 0.60
        } ?: 0.0

        val eyes = input.eyes?.let {
            sanitize(it.confidence) * if (it.fresh) 1.0 else 0.55
        } ?: 0.0

        val hands = if (input.hands.isEmpty()) {
            0.0
        } else {
            input.hands
                .map {
                    sanitize(it.confidence) *
                        if (it.fresh) 1.0 else 0.55
                }
                .average()
                .coerceIn(0.0, 1.0)
        }

        val lighting = lightingEvidence(input.lighting)
        val camera = cameraEvidence(input.camera)

        val evidence = FusionEvidence(
            timestampNanos = input.timestampNanos,
            body = (body * visible * bodyFresh).coerceIn(0.0, 1.0),
            face = face,
            eyes = eyes,
            hands = hands,
            depth = depth,
            topology = topology,
            temporal = temporal,
            calibration = calibration,
            lighting = lighting,
            camera = camera,
        )

        val raw = evidence.aggregate()

        val provisionalConsensus = (
            evidence.body * 0.30 +
                evidence.depth * 0.16 +
                evidence.topology * 0.14 +
                evidence.temporal * 0.14 +
                evidence.calibration * 0.12 +
                evidence.face * 0.05 +
                evidence.eyes * 0.04 +
                evidence.hands * 0.05
            ).coerceIn(0.0, 1.0)

        val reliabilityObservations = mutableListOf<AdaptiveEvidenceReliability.Observation>()

        fun observe(
            source: PerceptionSource,
            confidence: Double,
            ageMillis: Double = 0.0,
            residual: Double = 0.0,
            disagreement: Double = 0.0,
            stability: Double = 1.0,
            accepted: Boolean = true,
        ) {
            reliabilityObservations +=
                AdaptiveEvidenceReliability.Observation(
                    source = source,
                    timestampNanos = input.timestampNanos,
                    reportedConfidence = confidence,
                    ageMillis = ageMillis,
                    residual = residual,
                    disagreement = disagreement,
                    stability = stability,
                    accepted = accepted,
                )
        }

        observe(
            source = input.body.source,
            confidence = evidence.body,
            ageMillis = input.body.freshnessMillis,
            residual = input.temporal?.innovation ?: 0.0,
            disagreement = abs(evidence.body - evidence.temporal),
            stability = input.temporal?.stability ?: 0.70,
            accepted = input.body.visibleNodes > 0,
        )

        input.depth?.let {
            observe(
                source = it.source,
                confidence = evidence.depth,
                ageMillis = it.freshnessMillis,
                residual = abs(it.meanDepthMeters).coerceAtMost(4.0),
                disagreement = abs(evidence.body - evidence.depth),
                stability = it.temporalStability,
            )
        }

        input.topology?.let {
            observe(
                source = PerceptionSource.TEMPORAL_PREDICTION,
                confidence = evidence.topology,
                ageMillis = it.freshnessMillis,
                residual = it.edgeConstraintResidual,
                disagreement = abs(evidence.body - evidence.topology),
                stability = it.stability,
            )
        }

        input.temporal?.let {
            observe(
                source = PerceptionSource.TEMPORAL_PREDICTION,
                confidence = evidence.temporal,
                ageMillis = it.freshnessMillis,
                residual = it.innovation,
                disagreement = abs(evidence.body - evidence.temporal),
                stability = it.stability,
            )
        }

        observe(
            source = PerceptionSource.USER_CALIBRATION,
            confidence = evidence.calibration,
            residual = input.calibration.reprojectionErrorPx,
            disagreement = abs(evidence.calibration - evidence.geometryLikeCalibrationScore()),
            stability = input.calibration.scaleConfidence,
        )

        input.face?.let {
            observe(
                source = it.source,
                confidence = evidence.face,
                ageMillis = if (it.fresh) 0.0 else config.staleHalfLifeMillis * 1.6,
                disagreement = abs(evidence.face - evidence.eyes),
                stability = it.expressionEnergy.coerceIn(0.0, 1.0),
            )
        }

        input.eyes?.let {
            observe(
                source = it.source,
                confidence = evidence.eyes,
                ageMillis = if (it.fresh) 0.0 else config.staleHalfLifeMillis * 1.8,
                disagreement = abs(evidence.eyes - evidence.face),
                stability = it.gazeStability,
            )
        }

        if (input.hands.isNotEmpty()) {
            observe(
                source = PerceptionSource.MEDIAPIPE,
                confidence = evidence.hands,
                disagreement = abs(evidence.hands - evidence.body),
                stability = input.hands.map {
                    it.articulationEnergy
                }.average().coerceIn(0.0, 1.0),
            )
        }

        input.lighting?.let {
            observe(
                source = it.source,
                confidence = evidence.lighting,
                disagreement = it.dynamicRangePressure,
                stability = when (it.stability) {
                    LightingStabilityBand.STABLE -> 1.0
                    LightingStabilityBand.CHANGING -> 0.82
                    LightingStabilityBand.CHALLENGING -> 0.60
                    LightingStabilityBand.UNKNOWN -> 0.42
                },
            )
        }

        input.camera?.let {
            observe(
                source = PerceptionSource.SENSOR,
                confidence = evidence.camera,
                residual = it.cameraMotionRisk,
                disagreement = it.cameraMotionRisk,
                stability = 1.0 - it.cameraMotionRisk,
            )
        }

        reliabilityObservations.forEach {
            adaptiveReliability.observe(
                observation = it,
                consensusConfidence = provisionalConsensus,
            )
        }

        adaptiveReliability.prune(input.timestampNanos)

        val reliabilitySnapshot =
            adaptiveReliability.snapshot(input.timestampNanos)

        val reliabilityScore =
            reliabilitySnapshot.effectiveConsensus

        val cameraPenalty = input.camera
            ?.let { cameraPenalty(it.cameraMotionRisk) }
            ?: 0.0

        val lightingPenalty = input.lighting
            ?.let {
                (
                    it.highlightRisk * 0.35 +
                        it.shadowRisk * 0.35 +
                        it.flickerRisk * 0.30
                    ).coerceIn(0.0, 1.0)
            }
            ?: 0.0

        val disagreement = estimateDisagreement(
            body = body,
            depth = depth,
            topology = topology,
            temporal = temporal,
            face = face,
            eyes = eyes,
            cameraPenalty = cameraPenalty,
        )

        val confidence = (
            raw *
                (0.82 + reliabilityScore * 0.18) *
                (1.0 - cameraPenalty * config.cameraRiskScale) *
                (1.0 - lightingPenalty * config.lightingRiskScale) *
                (1.0 - disagreement * 0.16)
            ).coerceIn(0.0, 1.0)

        val motionSignals = motionSignals(input.body, input.temporal)

        val warnings = buildList {
            if (body < 0.40) add("body perception weak")
            if (depth < 0.35) add("depth evidence weak")
            if (topology < 0.35) add("topology evidence weak")
            if (temporal < 0.40) add("temporal continuity weak")
            if (cameraPenalty > 0.55) add("camera motion is materially affecting geometry")
            if (lightingPenalty > 0.50) add("lighting conditions are reducing perception quality")
            if (disagreement > config.disagreementThreshold) add("perception sources disagree")
            if (input.eyes?.bothEyesOpen == false) add("blink state may reduce eye-gaze confidence")
            if (input.hands.size > 2) add("more than two hand streams supplied; verify association")
            if (input.calibration.reprojectionErrorPx > 6.0) {
                add("calibration residual is above the live tracking target")
            }
            reliabilitySnapshot.warnings
                .filter { !it.startsWith("dominant evidence source:") }
                .take(3)
                .forEach { add(it) }
        }

        val diagnostics = buildDiagnostics(
            input = input,
            evidence = evidence,
            confidence = confidence,
            disagreement = disagreement,
            cameraPenalty = cameraPenalty,
            lightingPenalty = lightingPenalty,
        )

        return Result(
            timestampNanos = input.timestampNanos,
            confidence = confidence,
            evidence = evidence,
            reliability = reliabilitySnapshot,
            motionSignals = motionSignals,
            face = refineFace(input.face, input.eyes, eyes),
            eyes = input.eyes,
            hands = input.hands.sortedByDescending { it.confidence },
            lighting = input.lighting,
            camera = input.camera,
            cameraMotionPenalty = cameraPenalty,
            lightingPenalty = lightingPenalty,
            disagreement = disagreement,
            headOrientationBlend = if (input.eyes == null) 0.0 else {
                (eyes * config.gazeHeadBlend).coerceIn(0.0, 1.0)
            },
            warnings = warnings.distinct(),
            diagnostics = diagnostics,
        )
    }

    private fun freshnessWeight(ageMillis: Double): Double {
        if (!ageMillis.isFinite()) return 0.0
        if (ageMillis <= 0.0) return 1.0
        return kotlin.math.exp(
            -kotlin.math.ln(2.0) *
                ageMillis /
                config.staleHalfLifeMillis,
        ).coerceIn(0.0, 1.0)
    }

    private fun residualConfidence(residual: Double): Double {
        if (!residual.isFinite()) return 0.0
        return kotlin.math.exp(-residual.coerceAtLeast(0.0) / 4.0)
            .coerceIn(0.0, 1.0)
    }

    private fun lightingEvidence(value: LightingState?): Double {
        if (value == null) return 0.35
        val dynamic = 1.0 - (
            value.highlightRisk * 0.32 +
                value.shadowRisk * 0.32 +
                value.flickerRisk * 0.22 +
                value.dynamicRangePressure * 0.14
            ).coerceIn(0.0, 1.0)
        val stability = when (value.stability) {
            LightingStabilityBand.STABLE -> 1.0
            LightingStabilityBand.CHANGING -> 0.82
            LightingStabilityBand.CHALLENGING -> 0.62
            LightingStabilityBand.UNKNOWN -> 0.48
        }
        return (
            dynamic * 0.65 +
                stability * 0.20 +
                sanitize(value.confidence) * 0.15
            ).coerceIn(0.0, 1.0)
    }

    private fun cameraEvidence(value: CameraTelemetryState?): Double {
        if (value == null) return 0.50
        val focus = sanitize(value.focusConfidence)
        val motion = 1.0 - sanitize(value.cameraMotionRisk)
        val exposure = exposureHealth(value)
        return (
            focus * 0.34 +
                motion * 0.46 +
                exposure * 0.20
            ).coerceIn(0.0, 1.0)
    }

    private fun exposureHealth(value: CameraTelemetryState): Double {
        if (value.exposureTimeNanos <= 0L || value.frameDurationNanos <= 0L) {
            return 0.45
        }
        val duty = (
            value.exposureTimeNanos.toDouble() /
                value.frameDurationNanos.toDouble()
            ).coerceIn(0.0, 1.0)
        val motionBlurRisk = if (duty > 0.75) {
            ((duty - 0.75) / 0.25).coerceIn(0.0, 1.0)
        } else {
            0.0
        }
        return (1.0 - motionBlurRisk * 0.55).coerceIn(0.0, 1.0)
    }

    private fun estimateDisagreement(
        body: Double,
        depth: Double,
        topology: Double,
        temporal: Double,
        face: Double,
        eyes: Double,
        cameraPenalty: Double,
    ): Double {
        val bodyDepthGap = abs(body - depth)
        val topologyGap = abs(body - topology)
        val temporalGap = abs(body - temporal)
        val headGap = if (face <= 0.0 || eyes <= 0.0) 0.0 else abs(face - eyes)
        return (
            bodyDepthGap * 0.34 +
                topologyGap * 0.24 +
                temporalGap * 0.20 +
                headGap * 0.10 +
                cameraPenalty * 0.12
            ).coerceIn(0.0, 1.0)
    }

    private fun motionSignals(
        body: BodyEvidence,
        temporal: TemporalEvidence?,
    ): MotionSignals {
        val velocity = magnitude(body.velocityMetersPerSecond)
        val acceleration = magnitude(body.accelerationMetersPerSecond2)
        val microRange = (
            config.microMotionCeilingMetersPerSecond -
                config.microMotionFloorMetersPerSecond
            ).coerceAtLeast(1e-6)
        val microMotion = when {
            velocity < config.microMotionFloorMetersPerSecond -> 0.0
            velocity >= config.microMotionCeilingMetersPerSecond -> 1.0
            else -> (
                velocity - config.microMotionFloorMetersPerSecond
            ) / microRange
        }.coerceIn(0.0, 1.0)

        val continuity = temporal?.velocityContinuity ?: 0.5
        val innovation = temporal?.innovation ?: 0.5
        val energy = (
            velocity * 1.8 +
                acceleration * 0.09 +
                microMotion * 0.45 +
                continuity * 0.25 -
                innovation * 0.18
            ).coerceIn(0.0, 10.0) / 10.0

        val directionChange = temporal?.let {
            (
                1.0 -
                    it.velocityContinuity
                ).coerceIn(0.0, 1.0)
        } ?: 0.5

        return MotionSignals(
            translationalSpeedMetersPerSecond = velocity,
            accelerationMetersPerSecond2 = acceleration,
            microMotion = microMotion,
            motionEnergy = energy,
            directionChange = directionChange,
        )
    }

    private fun refineFace(
        face: FaceExpressionState?,
        eyes: EyeState?,
        eyeConfidence: Double,
    ): FaceExpressionState? {
        if (face == null) return null
        if (eyes == null || eyeConfidence <= 0.15) return face

        val closure = (
            eyes.leftBlinkProbability +
                eyes.rightBlinkProbability
            ) * 0.5

        val adjustedClosure = (
            face.leftEyeClosure * 0.55 +
                closure * 0.45
            ).coerceIn(0.0, 1.0)

        return face.copy(
            leftEyeClosure = adjustedClosure,
            rightEyeClosure = adjustedClosure,
            confidence = (
                face.confidence * 0.76 +
                    eyeConfidence * 0.24
                ).coerceIn(0.0, 1.0),
        )
    }

    private fun buildDiagnostics(
        input: Input,
        evidence: FusionEvidence,
        confidence: Double,
        disagreement: Double,
        cameraPenalty: Double,
        lightingPenalty: Double,
    ): List<String> {
        val output = mutableListOf<String>()
        output += "aggregateConfidence=" + format(confidence)
        output += "body=" + format(evidence.body)
        output += "depth=" + format(evidence.depth)
        output += "topology=" + format(evidence.topology)
        output += "temporal=" + format(evidence.temporal)
        output += "calibration=" + format(evidence.calibration)
        output += "cameraPenalty=" + format(cameraPenalty)
        output += "lightingPenalty=" + format(lightingPenalty)
        output += "disagreement=" + format(disagreement)

        input.depth?.let {
            output += "depthMetric=" + it.metric
            output += "depthMeanMeters=" + format(it.meanDepthMeters)
        }

        input.camera?.let {
            output += "iso=" + it.sensitivityIso
            output += "focusConfidence=" + format(it.focusConfidence)
            output += "cameraMotionRisk=" + format(it.cameraMotionRisk)
        }

        input.lighting?.let {
            output += "lux=" + format(it.lux)
            output += "cct=" + format(it.correlatedColorTemperatureKelvin)
            output += "flickerRisk=" + format(it.flickerRisk)
        }

        input.eyes?.let {
            output += "gazeConfidence=" + format(it.confidence)
            output += "blink=" + format(it.blinkProbability)
            output += "saccadeDps=" + format(it.saccadeVelocityDegPerSecond)
        }

        output += "hands=" + input.hands.size
        return output
    }

    private fun FusionEvidence.geometryLikeCalibrationScore(): Double =
        (
            calibration * 0.72 +
                topology * 0.16 +
                temporal * 0.12
            ).coerceIn(0.0, 1.0)

    private fun cameraPenalty(risk: Double): Double =
        sanitize(risk)

    private fun magnitude(value: Vec3): Double =
        sqrt(
            value.x * value.x +
                value.y * value.y +
                value.z * value.z,
        )

    private fun sanitize(value: Double): Double =
        if (value.isFinite()) value.coerceIn(0.0, 1.0) else 0.0

    private fun format(value: Double): String =
        if (!value.isFinite()) "nan"
        else "%.4f".format(java.util.Locale.US, value)
}

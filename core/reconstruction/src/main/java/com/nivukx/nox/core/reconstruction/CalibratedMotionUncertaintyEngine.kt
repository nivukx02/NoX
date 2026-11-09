package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec3
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * Empirical calibrated motion-uncertainty envelopes for production reconstruction.
 *
 * This stage preserves uncertainty instead of collapsing every failure mode into one confidence
 * scalar. It tracks per-joint innovation residuals, derives a finite-sample target-coverage
 * envelope, expands that envelope under dynamic/occlusion/synchronization stress, and exposes an
 * OOD-risk signal. Formal conformal coverage guarantees require exchangeability/calibration
 * assumptions that are intentionally not claimed by this implementation.
 */
class CalibratedMotionUncertaintyEngine(
    private var config: Config = Config(),
) {
    data class Config(
        val targetCoverage: Double = 0.95,
        val minimumCalibrationSamples: Int = 8,
        val maximumResidualSamples: Int = 96,
        val minimumRadiusMeters: Double = 0.004,
        val maximumRadiusMeters: Double = 0.60,
        val baseInflation: Double = 1.08,
        val dynamicStressInflation: Double = 0.16,
        val occlusionInflation: Double = 0.24,
        val synchronizationInflation: Double = 0.18,
        val cameraMotionInflation: Double = 0.12,
        val accelerationInflation: Double = 0.025,
        val maximumTrackedJointSpeedMetersPerSecond: Double = 24.0,
        val oodResidualMultiplier: Double = 2.80,
        val highRiskThreshold: Double = 0.74,
        val holdRiskThreshold: Double = 0.58,
        val temporalAlpha: Double = 0.18,
        val confidenceAlpha: Double = 0.20,
        val resetAfterGapSeconds: Double = 1.20,
    ) {
        init {
            require(targetCoverage in 0.80..0.999)
            require(minimumCalibrationSamples >= 2)
            require(maximumResidualSamples >= minimumCalibrationSamples)
            require(minimumRadiusMeters > 0.0)
            require(maximumRadiusMeters > minimumRadiusMeters)
            require(baseInflation >= 1.0)
            require(dynamicStressInflation >= 0.0)
            require(occlusionInflation >= 0.0)
            require(synchronizationInflation >= 0.0)
            require(cameraMotionInflation >= 0.0)
            require(accelerationInflation >= 0.0)
            require(maximumTrackedJointSpeedMetersPerSecond > 0.0)
            require(oodResidualMultiplier > 1.0)
            require(highRiskThreshold in 0.0..1.0)
            require(holdRiskThreshold in 0.0..1.0)
            require(temporalAlpha in 0.01..1.0)
            require(confidenceAlpha in 0.01..1.0)
            require(resetAfterGapSeconds > 0.20)
        }
    }

    data class JointEnvelope(
        val landmarkId: Int,
        val radiusMeters: Double,
        val empiricalCoverage: Double,
        val calibrationSamples: Int,
        val innovationMeters: Double,
        val normalizedInnovation: Double,
        val outOfDistributionRisk: Double,
        val measurementUncertaintyMeters: Double,
        val predictionUncertaintyMeters: Double,
        val confidence: Double,
        val predictionOnly: Boolean,
        val calibrationReady: Boolean,
        val warning: String?,
    )

    data class Result(
        val timestampNanos: Long,
        val targetCoverage: Double,
        val calibrationReady: Boolean,
        val measuredJointCount: Int,
        val envelopeJointCount: Int,
        val predictionOnlyJointCount: Int,
        val meanRadiusMeters: Double,
        val p95RadiusMeters: Double,
        val empiricalCoverage: Double,
        val uncertaintyConfidence: Double,
        val outOfDistributionRisk: Double,
        val dynamicInflation: Double,
        val calibrationProgress: Double,
        val policy: Policy,
        val warnings: List<String>,
        val recommendations: List<String>,
        val envelopes: List<JointEnvelope>,
    ) {
        enum class Policy {
            NORMAL,
            CAUTION,
            HOLD,
            REJECT,
        }
    }

    private data class JointState(
        var timestampNanos: Long,
        var position: Vec3,
        var velocity: Vec3,
        var acceleration: Vec3,
        var uncertaintyMeters: Double,
        var confidence: Double,
        var residuals: ArrayDeque<Double> = ArrayDeque(),
        var smoothedRisk: Double = 0.0,
        var smoothedCoverage: Double = 0.0,
    )

    private val states = LinkedHashMap<Int, JointState>()
    private var lastTimestampNanos = Long.MIN_VALUE

    @Synchronized
    fun configure(value: Config) {
        config = value
    }

    @Synchronized
    fun configuration(): Config = config

    @Synchronized
    fun reset() {
        states.clear()
        lastTimestampNanos = Long.MIN_VALUE
    }

    @Synchronized
    fun evaluate(
        skeleton: SkeletonFrame,
        motionScore: Double = 0.0,
        dynamicStress: Double = 0.0,
        occlusionRisk: Double = 0.0,
        cameraMotionRisk: Double = 0.0,
        synchronizationConfidence: Double = 1.0,
    ): Result {
        val timestamp = skeleton.timestampNanos
        require(timestamp > 0L)

        if (
            lastTimestampNanos != Long.MIN_VALUE &&
            timestamp <= lastTimestampNanos
        ) {
            reset()
        }

        val gapSeconds =
            if (lastTimestampNanos == Long.MIN_VALUE) {
                1.0 / 30.0
            } else {
                (timestamp - lastTimestampNanos).coerceAtLeast(1L) / 1e9
            }

        if (gapSeconds > config.resetAfterGapSeconds) {
            states.values.forEach {
                it.residuals.clear()
                it.smoothedRisk = 0.0
                it.smoothedCoverage = 0.0
            }
        }

        val motion = motionScore.coerceIn(0.0, 1.0)
        val stress = dynamicStress.coerceIn(0.0, 1.0)
        val occlusion = occlusionRisk.coerceIn(0.0, 1.0)
        val camera = cameraMotionRisk.coerceIn(0.0, 1.0)
        val synchronization = synchronizationConfidence.coerceIn(0.0, 1.0)

        val dynamicInflation = (
            config.baseInflation +
                motion * 0.05 +
                stress * config.dynamicStressInflation +
                occlusion * config.occlusionInflation +
                camera * config.cameraMotionInflation +
                (1.0 - synchronization) * config.synchronizationInflation
            ).coerceAtLeast(1.0)

        val envelopes = ArrayList<JointEnvelope>(skeleton.joints.size)

        for (rawJoint in skeleton.joints) {
            val joint = sanitize(rawJoint)
            val predictionOnly = joint.observationCount <= 0
            val state = states[joint.landmarkId]

            if (state == null) {
                val initial = JointState(
                    timestampNanos = timestamp,
                    position = joint.position,
                    velocity = joint.velocity,
                    acceleration = ZERO,
                    uncertaintyMeters = sanitizeUncertainty(
                        joint.positionUncertaintyMeters,
                    ),
                    confidence = joint.confidence.coerceIn(0.0, 1.0),
                )
                states[joint.landmarkId] = initial
                envelopes += JointEnvelope(
                    landmarkId = joint.landmarkId,
                    radiusMeters = initial.uncertaintyMeters,
                    empiricalCoverage = 0.0,
                    calibrationSamples = 0,
                    innovationMeters = 0.0,
                    normalizedInnovation = 0.0,
                    outOfDistributionRisk = 0.0,
                    measurementUncertaintyMeters = initial.uncertaintyMeters,
                    predictionUncertaintyMeters = initial.uncertaintyMeters,
                    confidence = initial.confidence,
                    predictionOnly = predictionOnly,
                    calibrationReady = false,
                    warning = "joint-" + joint.landmarkId + "-calibration-warmup",
                )
                continue
            }

            val dt = (
                timestamp - state.timestampNanos
            ).coerceAtLeast(1L) / 1e9

            val predictionDt = dt.coerceIn(
                1.0 / 240.0,
                0.20,
            )

            val prediction =
                state.position +
                    state.velocity * predictionDt +
                    state.acceleration * (0.5 * predictionDt * predictionDt)

            val innovation = prediction.distanceTo(joint.position)
            val measurementUncertainty =
                sanitizeUncertainty(joint.positionUncertaintyMeters)

            // Score the current observation against the prior calibration history.
            // The current residual is deliberately not inserted before scoring; otherwise an
            // extreme innovation would widen the envelope that is supposed to detect it.
            val quantile = finiteSampleQuantile(
                state.residuals,
                config.targetCoverage,
            )

            val predictionUncertainty = (
                state.uncertaintyMeters +
                    state.velocity.norm() * predictionDt * 0.18 +
                    state.acceleration.norm() *
                    predictionDt *
                    predictionDt *
                    config.accelerationInflation
                ).coerceIn(
                config.minimumRadiusMeters,
                config.maximumRadiusMeters,
            )

            val calibratedBase = max(
                measurementUncertainty,
                max(
                    quantile,
                    predictionUncertainty,
                ),
            )

            val radius = (
                calibratedBase * dynamicInflation
            ).coerceIn(
                config.minimumRadiusMeters,
                config.maximumRadiusMeters,
            )

            val normalizedInnovation =
                innovation / radius.coerceAtLeast(config.minimumRadiusMeters)

            val instantaneousRisk = (
                normalizedInnovation /
                    config.oodResidualMultiplier
                ).coerceIn(0.0, 1.0)

            val calibrationReadyBeforeUpdate =
                state.residuals.size >= config.minimumCalibrationSamples

            val acceptedForCalibration =
                !predictionOnly &&
                    joint.confidence >= 0.05 &&
                    innovation.isFinite() &&
                    (
                        !calibrationReadyBeforeUpdate ||
                            normalizedInnovation <= 1.50
                        )

            if (acceptedForCalibration) {
                if (state.residuals.size >= config.maximumResidualSamples) {
                    state.residuals.removeFirst()
                }
                state.residuals.addLast(innovation.coerceAtLeast(0.0))
            }

            val coverage = empiricalCoverage(
                state.residuals,
                radius,
            )

            state.smoothedRisk =
                state.smoothedRisk * (1.0 - config.temporalAlpha) +
                    instantaneousRisk * config.temporalAlpha

            state.smoothedCoverage =
                state.smoothedCoverage * (1.0 - config.confidenceAlpha) +
                    coverage * config.confidenceAlpha

            val calibrationReady =
                state.residuals.size >= config.minimumCalibrationSamples

            val warning =
                when {
                    state.smoothedRisk >= config.highRiskThreshold ||
                        instantaneousRisk >= config.highRiskThreshold ->
                        "joint-" + joint.landmarkId + "-ood-risk-high"
                    !calibrationReady ->
                        "joint-" + joint.landmarkId + "-calibration-warmup"
                    normalizedInnovation > 0.85 ||
                        instantaneousRisk >= config.highRiskThreshold * 0.35 ->
                        "joint-" + joint.landmarkId + "-outside-calibrated-envelope"
                    else -> null
                }

            envelopes += JointEnvelope(
                landmarkId = joint.landmarkId,
                radiusMeters = radius,
                empiricalCoverage = state.smoothedCoverage.coerceIn(0.0, 1.0),
                calibrationSamples = state.residuals.size,
                innovationMeters = innovation,
                normalizedInnovation = normalizedInnovation,
                outOfDistributionRisk =
                    state.smoothedRisk.coerceIn(0.0, 1.0),
                measurementUncertaintyMeters = measurementUncertainty,
                predictionUncertaintyMeters = predictionUncertainty,
                confidence = combinedConfidence(
                    joint = joint,
                    coverage = state.smoothedCoverage,
                    risk = state.smoothedRisk,
                ),
                predictionOnly = predictionOnly,
                calibrationReady = calibrationReady,
                warning = warning,
            )

            updateState(
                state = state,
                joint = joint,
                timestampNanos = timestamp,
                residual = innovation,
            )
        }

        lastTimestampNanos = timestamp

        return aggregate(
            skeleton = skeleton,
            envelopes = envelopes,
            timestampNanos = timestamp,
            dynamicInflation = dynamicInflation,
        )
    }

    private fun updateState(
        state: JointState,
        joint: ReconstructionJoint,
        timestampNanos: Long,
        residual: Double,
    ) {
        if (
            !residual.isFinite() ||
            residual > config.maximumRadiusMeters * 1.50
        ) {
            state.timestampNanos = timestampNanos
            return
        }

        val dt = (
            timestampNanos - state.timestampNanos
        ).coerceAtLeast(1L) / 1e9

        val measuredVelocity =
            (joint.position - state.position) /
                dt.coerceAtLeast(1.0 / 240.0)

        val velocityAlpha =
            (0.22 + joint.confidence * 0.54)
                .coerceIn(0.18, 0.82)

        val newVelocity =
            state.velocity.lerp(
                measuredVelocity,
                velocityAlpha,
            )

        val measuredAcceleration =
            (newVelocity - state.velocity) /
                dt.coerceAtLeast(1.0 / 240.0)

        state.acceleration =
            state.acceleration.lerp(
                measuredAcceleration,
                (0.14 + joint.confidence * 0.36)
                    .coerceIn(0.10, 0.55),
            )

        val speed = newVelocity.norm()
        state.velocity =
            if (speed.isFinite() && speed > 14.0) {
                newVelocity * (14.0 / speed)
            } else {
                newVelocity
            }

        state.position =
            state.position.lerp(
                joint.position,
                velocityAlpha,
            )

        state.uncertaintyMeters =
            if (
                residual.isFinite() &&
                residual <= config.maximumRadiusMeters * 1.50
            ) {
                (
                    joint.positionUncertaintyMeters * 0.60 +
                        residual.coerceAtLeast(0.0) * 0.40
                    ).coerceIn(
                    config.minimumRadiusMeters,
                    config.maximumRadiusMeters,
                )
            } else {
                state.uncertaintyMeters
            }

        state.confidence =
            state.confidence * 0.76 +
                joint.confidence.coerceIn(0.0, 1.0) * 0.24

        state.timestampNanos = timestampNanos
    }

    private fun aggregate(
        skeleton: SkeletonFrame,
        envelopes: List<JointEnvelope>,
        timestampNanos: Long,
        dynamicInflation: Double,
    ): Result {
        if (envelopes.isEmpty()) {
            return Result(
                timestampNanos = timestampNanos,
                targetCoverage = config.targetCoverage,
                calibrationReady = false,
                measuredJointCount = 0,
                envelopeJointCount = 0,
                predictionOnlyJointCount = 0,
                meanRadiusMeters = 0.0,
                p95RadiusMeters = 0.0,
                empiricalCoverage = 0.0,
                uncertaintyConfidence = 0.0,
                outOfDistributionRisk = 1.0,
                dynamicInflation = dynamicInflation,
                calibrationProgress = 0.0,
                policy = Result.Policy.REJECT,
                warnings = listOf("no-joints-for-uncertainty-analysis"),
                recommendations = listOf(
                    "Require measured 3D joints before evaluating uncertainty.",
                ),
                envelopes = emptyList(),
            )
        }

        val radii = envelopes.map { it.radiusMeters }.filter(Double::isFinite)
        val coverages = envelopes.map { it.empiricalCoverage }
        val risks = envelopes.map { it.outOfDistributionRisk }
        val readyRatio =
            envelopes.count { it.calibrationReady }.toDouble() /
                envelopes.size.coerceAtLeast(1)

        val coverage = coverages.averageOrZero()
        val oodRisk = risks.maxOrNull()?.coerceIn(0.0, 1.0) ?: 1.0
        val confidence = (
            (1.0 - oodRisk) * 0.46 +
                coverage * 0.34 +
                readyRatio * 0.20
            ).coerceIn(0.0, 1.0)

        val policy = when {
            oodRisk >= config.highRiskThreshold ->
                Result.Policy.REJECT
            oodRisk >= config.holdRiskThreshold ->
                Result.Policy.HOLD
            readyRatio < 0.35 ->
                Result.Policy.CAUTION
            else ->
                Result.Policy.NORMAL
        }

        val warnings = buildList {
            if (readyRatio < 0.50) {
                add("motion-uncertainty-calibration-warmup")
            }
            if (
                envelopes.any {
                    it.warning?.contains("ood-risk") == true ||
                        it.warning?.contains("outside-calibrated-envelope") == true
                }
            ) {
                add("motion-out-of-distribution-evidence-detected")
            }
            if (coverage < config.targetCoverage * 0.90) {
                add("empirical-uncertainty-coverage-below-target")
            }
            if (oodRisk >= config.holdRiskThreshold) {
                add("motion-out-of-distribution-risk-elevated")
            }
            if (radii.averageOrZero() > config.maximumRadiusMeters * 0.65) {
                add("uncertainty-envelope-wide")
            }
            if (dynamicInflation >= 1.35) {
                add("dynamic-regime-inflating-prediction-envelope")
            }
        }.distinct()

        val recommendations = buildList {
            if (readyRatio < 0.50) {
                add("Accumulate adjacent measured frames before trusting the calibrated envelope.")
            }
            if (coverage < config.targetCoverage * 0.90) {
                add("Recalibrate residual history or strengthen measured multi-view evidence before export.")
            }
            if (oodRisk >= config.holdRiskThreshold) {
                add("Hold aggressive extrapolation and reacquire observed joints.")
            }
            if (dynamicInflation >= 1.35) {
                add("Prioritize inference frequency during the high-dynamics interval.")
            }
            if (confidence >= 0.82) {
                add("Use joint envelopes as weights for offline refinement and export validation.")
            }
        }.distinct()

        return Result(
            timestampNanos = timestampNanos,
            targetCoverage = config.targetCoverage,
            calibrationReady = readyRatio >= 0.75,
            measuredJointCount = envelopes.count { !it.predictionOnly },
            envelopeJointCount = envelopes.size,
            predictionOnlyJointCount = envelopes.count { it.predictionOnly },
            meanRadiusMeters = radii.averageOrZero(),
            p95RadiusMeters = quantileFromSorted(radii, 0.95),
            empiricalCoverage = coverage,
            uncertaintyConfidence = confidence,
            outOfDistributionRisk = oodRisk,
            dynamicInflation = dynamicInflation,
            calibrationProgress = readyRatio.coerceIn(0.0, 1.0),
            policy = policy,
            warnings = warnings,
            recommendations = recommendations,
            envelopes = envelopes,
        )
    }

    private fun combinedConfidence(
        joint: ReconstructionJoint,
        coverage: Double,
        risk: Double,
    ): Double {
        val coverageEvidence =
            if (coverage <= 0.0) 0.35 else coverage

        return (
            joint.confidence.coerceIn(0.0, 1.0) * 0.52 +
                coverageEvidence.coerceIn(0.0, 1.0) * 0.26 +
                (1.0 - risk.coerceIn(0.0, 1.0)) * 0.22
            ).coerceIn(0.0, 1.0)
    }

    private fun finiteSampleQuantile(
        values: Collection<Double>,
        coverage: Double,
    ): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.filter(Double::isFinite).sorted()
        if (sorted.isEmpty()) return 0.0

        val rawIndex =
            ceil((sorted.size + 1) * coverage).toInt() - 1

        return sorted[
            rawIndex.coerceIn(
                0,
                sorted.lastIndex,
            )
        ]
    }

    private fun empiricalCoverage(
        values: Collection<Double>,
        radius: Double,
    ): Double {
        val valid = values.filter(Double::isFinite)
        if (valid.isEmpty()) return 0.0
        return valid.count { it <= radius }.toDouble() /
            valid.size.toDouble()
    }

    private fun quantileFromSorted(
        values: List<Double>,
        probability: Double,
    ): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val index =
            ceil(
                (sorted.size - 1) * probability,
            ).toInt().coerceIn(
                0,
                sorted.lastIndex,
            )
        return sorted[index]
    }

    private fun sanitize(
        joint: ReconstructionJoint,
    ): ReconstructionJoint =
        joint.copy(
            position = sanitizeVec3(joint.position),
            velocity = sanitizeVec3(joint.velocity),
            confidence = joint.confidence.coerceIn(0.0, 1.0),
            reprojectionErrorPx =
                joint.reprojectionErrorPx.takeIf {
                    it.isFinite() && it >= 0.0
                } ?: Double.POSITIVE_INFINITY,
            observationCount = joint.observationCount.coerceAtLeast(0),
            positionUncertaintyMeters =
                sanitizeUncertainty(
                    joint.positionUncertaintyMeters,
                ),
        )

    private fun sanitizeVec3(value: Vec3): Vec3 =
        Vec3(
            if (value.x.isFinite()) value.x else 0.0,
            if (value.y.isFinite()) value.y else 0.0,
            if (value.z.isFinite()) value.z else 0.0,
        )

    private fun sanitizeUncertainty(value: Double): Double =
        if (value.isFinite() && value > 0.0) {
            value.coerceIn(
                config.minimumRadiusMeters,
                config.maximumRadiusMeters,
            )
        } else {
            config.minimumRadiusMeters
        }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private companion object {
        val ZERO = Vec3(0.0, 0.0, 0.0)
    }
}

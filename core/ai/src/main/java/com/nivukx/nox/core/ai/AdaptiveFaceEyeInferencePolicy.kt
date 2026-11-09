package com.nivukx.nox.core.ai

import kotlin.math.max
import kotlin.math.min

/**
 * Evidence-driven scheduler for expensive face-mesh and eye inference.
 *
 * The policy separates temporal freshness from compute demand. High-value facial frames are allowed
 * to consume more inference budget when the face occupies useful image area, confidence falls,
 * gaze becomes unstable, motion accelerates, or occlusion rises. Stable close-to-perfect tracks
 * are allowed to coast, while stale or unavailable eye state is never hidden behind a quality-saving
 * decision.
 */
class AdaptiveFaceEyeInferencePolicy(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumEyeIntervalNanos: Long = 16_000_000L,
        val maximumEyeIntervalNanos: Long = 140_000_000L,
        val minimumFaceMeshIntervalNanos: Long = 24_000_000L,
        val maximumFaceMeshIntervalNanos: Long = 260_000_000L,
        val eyePriorityThreshold: Double = 0.48,
        val faceMeshPriorityThreshold: Double = 0.46,
        val forcedEyeStaleNanos: Long = 110_000_000L,
        val forcedFaceMeshStaleNanos: Long = 220_000_000L,
        val lightingPriorityWeight: Double = 0.10,
        val uncertaintyPriorityWeight: Double = 0.10,
    ) {
        init {
            require(minimumEyeIntervalNanos > 0L)
            require(maximumEyeIntervalNanos >= minimumEyeIntervalNanos)
            require(minimumFaceMeshIntervalNanos > 0L)
            require(maximumFaceMeshIntervalNanos >= minimumFaceMeshIntervalNanos)
            require(eyePriorityThreshold in 0.0..1.0)
            require(faceMeshPriorityThreshold in 0.0..1.0)
            require(forcedEyeStaleNanos > 0L)
            require(forcedFaceMeshStaleNanos > 0L)
            require(lightingPriorityWeight >= 0.0)
            require(uncertaintyPriorityWeight >= 0.0)
        }
    }

    data class Evidence(
        val faceAvailable: Boolean,
        val eyeAvailable: Boolean,
        val faceAreaRatio: Double,
        val faceConfidence: Double,
        val eyeConfidence: Double,
        val fixationStability: Double,
        val motionEnergy: Double,
        val occlusionRisk: Double,
        val faceMeshAgeNanos: Long,
        val eyeAgeNanos: Long,
        val thermalHeadroom: Double,
        val gpuHeadroom: Double,
        val faceDetail: Double,
        val eyeDetail: Double,
        val lightingConfidence: Double = 1.0,
        val faceSharpness: Double = 1.0,
        val eyeSharpness: Double = 1.0,
        val eyeVisibility: Double = 1.0,
        val faceUncertainty: Double = 0.0,
        val eyeUncertainty: Double = 0.0,
    ) {
        init {
            require(faceAreaRatio.isFinite() && faceAreaRatio >= 0.0)
            listOf(
                faceConfidence,
                eyeConfidence,
                fixationStability,
                motionEnergy,
                occlusionRisk,
                thermalHeadroom,
                gpuHeadroom,
                faceDetail,
                eyeDetail,
                lightingConfidence,
                faceSharpness,
                eyeSharpness,
                eyeVisibility,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            require(faceMeshAgeNanos >= 0L || faceMeshAgeNanos == Long.MAX_VALUE)
            require(eyeAgeNanos >= 0L || eyeAgeNanos == Long.MAX_VALUE)
            require(faceUncertainty >= 0.0)
            require(eyeUncertainty >= 0.0)
        }
    }

    data class Decision(
        val runFaceMesh: Boolean,
        val runEyes: Boolean,
        val faceMeshPriority: Double,
        val eyePriority: Double,
        val faceMeshIntervalNanos: Long,
        val eyeIntervalNanos: Long,
        val computePressure: Double,
        val reason: String,
    )

    fun decide(evidence: Evidence): Decision {
        if (!evidence.faceAvailable) {
            return Decision(
                runFaceMesh = false,
                runEyes = false,
                faceMeshPriority = 0.0,
                eyePriority = 0.0,
                faceMeshIntervalNanos = config.maximumFaceMeshIntervalNanos,
                eyeIntervalNanos = config.maximumEyeIntervalNanos,
                computePressure = computePressure(evidence),
                reason = "no-face",
            )
        }

        val computePressure = computePressure(evidence)

        val normalizedFaceScale =
            (evidence.faceAreaRatio / 0.12)
                .coerceIn(0.0, 1.0)

        val faceOcclusion = evidence.occlusionRisk.coerceIn(0.0, 1.0)
        val motion = evidence.motionEnergy.coerceIn(0.0, 1.0)
        val lightingStress =
            (1.0 - evidence.lightingConfidence)
                .coerceIn(0.0, 1.0)
        val faceUncertaintyStress =
            (evidence.faceUncertainty / 0.10)
                .coerceIn(0.0, 1.0)
        val eyeUncertaintyStress =
            (evidence.eyeUncertainty / 0.05)
                .coerceIn(0.0, 1.0)
        val faceSharpnessStress =
            (1.0 - evidence.faceSharpness)
                .coerceIn(0.0, 1.0)
        val eyeSharpnessStress =
            (1.0 - evidence.eyeSharpness)
                .coerceIn(0.0, 1.0)
        val eyeVisibilityStress =
            (1.0 - evidence.eyeVisibility)
                .coerceIn(0.0, 1.0)

        val facePriority = (
            normalizedFaceScale * 0.23 +
                (1.0 - evidence.faceConfidence) * 0.20 +
                faceOcclusion * 0.17 +
                motion * 0.15 +
                (1.0 - evidence.fixationStability) * 0.08 +
                evidence.faceDetail * 0.15 +
                faceSharpnessStress * 0.07 +
                lightingStress * config.lightingPriorityWeight +
                faceUncertaintyStress * config.uncertaintyPriorityWeight
            ).coerceIn(0.0, 1.0)

        val eyePriority = (
            normalizedFaceScale * 0.10 +
                (1.0 - evidence.eyeConfidence) * 0.22 +
                (1.0 - evidence.fixationStability) * 0.24 +
                motion * 0.17 +
                faceOcclusion * 0.12 +
                evidence.eyeDetail * 0.12 +
                eyeSharpnessStress * 0.08 +
                eyeVisibilityStress * 0.08 +
                lightingStress * config.lightingPriorityWeight +
                eyeUncertaintyStress * config.uncertaintyPriorityWeight
            ).coerceIn(0.0, 1.0)

        val eyeUrgency = when {
            !evidence.eyeAvailable -> 1.0
            evidence.eyeAgeNanos >= config.forcedEyeStaleNanos -> 1.0
            else -> eyePriority
        }

        val faceUrgency = when {
            evidence.faceMeshAgeNanos >= config.forcedFaceMeshStaleNanos -> 1.0
            else -> facePriority
        }

        val thermalFactor = evidence.thermalHeadroom.coerceIn(0.10, 1.0)
        val gpuFactor = evidence.gpuHeadroom.coerceIn(0.10, 1.0)
        val headroom = min(thermalFactor, gpuFactor)

        val eyeBaseInterval = interpolateInterval(
            minimum = config.minimumEyeIntervalNanos,
            maximum = config.maximumEyeIntervalNanos,
            priority = eyeUrgency,
        )

        val faceBaseInterval = interpolateInterval(
            minimum = config.minimumFaceMeshIntervalNanos,
            maximum = config.maximumFaceMeshIntervalNanos,
            priority = faceUrgency,
        )

        val pressureStretch = 1.0 + computePressure * 1.65
        val eyeInterval = (
            eyeBaseInterval * pressureStretch
        ).toLong().coerceIn(
            config.minimumEyeIntervalNanos,
            config.maximumEyeIntervalNanos,
        )
        val faceInterval = (
            faceBaseInterval * pressureStretch
        ).toLong().coerceIn(
            config.minimumFaceMeshIntervalNanos,
            config.maximumFaceMeshIntervalNanos,
        )

        val eyeDue = evidence.eyeAgeNanos == Long.MAX_VALUE ||
            evidence.eyeAgeNanos >= eyeInterval
        val faceDue = evidence.faceMeshAgeNanos == Long.MAX_VALUE ||
            evidence.faceMeshAgeNanos >= faceInterval

        val runEyes =
            eyeDue ||
                eyeUrgency >= config.eyePriorityThreshold ||
                !evidence.eyeAvailable

        val closeEnoughForMesh = normalizedFaceScale >= 0.06
        val runFaceMesh =
            closeEnoughForMesh &&
                (
                    faceDue ||
                        faceUrgency >= config.faceMeshPriorityThreshold
                    )

        val thermalCritical =
            evidence.thermalHeadroom < 0.36 ||
                evidence.gpuHeadroom < 0.40

        val reason = when {
            !evidence.eyeAvailable -> "eye-acquisition"
            thermalCritical && eyeUrgency >= 0.72 -> "thermal-pressure-eye-priority"
            thermalCritical -> "thermal-pressure"
            faceUrgency >= 0.78 && runFaceMesh -> "high-value-face"
            eyeUrgency >= 0.72 -> "gaze-priority"
            lightingStress >= 0.55 -> "lighting-recovery"
            faceUncertaintyStress >= 0.60 ||
                eyeUncertaintyStress >= 0.60 -> "uncertainty-recovery"
            motion >= 0.78 -> "motion-priority"
            faceOcclusion >= 0.72 -> "occlusion-priority"
            runEyes || runFaceMesh -> "adaptive-due"
            else -> "stable-coast"
        }

        return Decision(
            runFaceMesh = runFaceMesh,
            runEyes = runEyes,
            faceMeshPriority = facePriority,
            eyePriority = eyePriority,
            faceMeshIntervalNanos = faceInterval,
            eyeIntervalNanos = eyeInterval,
            computePressure = computePressure,
            reason = reason,
        )
    }

    private fun computePressure(evidence: Evidence): Double = (
        (1.0 - evidence.thermalHeadroom) * 0.42 +
            (1.0 - evidence.gpuHeadroom) * 0.28 +
            evidence.motionEnergy * 0.08 +
            (1.0 - evidence.fixationStability) * 0.07 +
            (1.0 - evidence.eyeConfidence) * 0.05 +
            (1.0 - evidence.faceConfidence) * 0.05 +
            evidence.occlusionRisk * 0.05 +
            lightingStress(evidence) * 0.08 +
            (
                (1.0 - evidence.faceSharpness).coerceIn(0.0, 1.0) * 0.04 +
                    (1.0 - evidence.eyeSharpness).coerceIn(0.0, 1.0) * 0.05 +
                    (1.0 - evidence.eyeVisibility).coerceIn(0.0, 1.0) * 0.05
                ) +
            max(
                (
                    evidence.faceUncertainty / 0.10
                ).coerceIn(0.0, 1.0),
                (
                    evidence.eyeUncertainty / 0.05
                ).coerceIn(0.0, 1.0),
            ) * 0.07
        ).coerceIn(0.0, 1.0)

    private fun lightingStress(
        evidence: Evidence,
    ): Double =
        (
            1.0 -
                evidence.lightingConfidence
                    .coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)

    private fun interpolateInterval(
        minimum: Long,
        maximum: Long,
        priority: Double,
    ): Long {
        val factor = priority.coerceIn(0.0, 1.0)
        return (
            maximum.toDouble() +
                (minimum.toDouble() - maximum.toDouble()) * factor
            ).toLong().coerceIn(minimum, maximum)
    }
}

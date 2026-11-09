package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.Vec3
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.ln

/**
 * Metric camera-placement information scorer for distributed multi-actor capture.
 *
 * It combines measured camera pose, an actor world anchor and existing camera authority evidence.
 * The result is designed for deterministic view allocation rather than decorative confidence
 * inflation. Missing geometry is represented as an explicit degraded state.
 */
class CameraPlacementInformationEngine(
    private val config: Config = Config.production(),
) {
    data class Config(
        val preferredActorDistanceMeters: Double = 5.0,
        val distanceScaleMeters: Double = 6.0,
        val strongFacingCosine: Double = 0.68,
        val minimumFacingCosine: Double = -0.20,
        val minimumUsefulNoveltyDegrees: Double = 8.0,
        val preferredNoveltyDegrees: Double = 24.0,
        val strongNoveltyDegrees: Double = 55.0,
        val facingWeight: Double = 0.38,
        val distanceWeight: Double = 0.18,
        val noveltyWeight: Double = 0.28,
        val baselineWeight: Double = 0.16,
        val opticalIntegrityWeight: Double = 0.12,
        val fallbackGeometryScore: Double = 0.42,
        val minimumDistanceMeters: Double = 0.15,
        val maximumDistanceMeters: Double = 250.0,
    ) {
        init {
            require(preferredActorDistanceMeters > 0.0)
            require(distanceScaleMeters > 0.0)
            require(strongFacingCosine in -1.0..1.0)
            require(minimumFacingCosine < strongFacingCosine)
            require(minimumUsefulNoveltyDegrees > 0.0)
            require(preferredNoveltyDegrees >= minimumUsefulNoveltyDegrees)
            require(strongNoveltyDegrees >= preferredNoveltyDegrees)
            require(facingWeight >= 0.0)
            require(distanceWeight >= 0.0)
            require(noveltyWeight >= 0.0)
            require(baselineWeight >= 0.0)
            require(opticalIntegrityWeight >= 0.0)
            require(
                facingWeight +
                    distanceWeight +
                    noveltyWeight +
                    baselineWeight +
                    opticalIntegrityWeight > 0.0,
            )
            require(fallbackGeometryScore in 0.0..1.0)
            require(minimumDistanceMeters > 0.0)
            require(maximumDistanceMeters > minimumDistanceMeters)
        }

        companion object {
            fun production(): Config =
                Config()
        }
    }

    data class ActorAnchor(
        val actorId: Int,
        val positionMeters: Vec3,
        val priority: Double = 0.5,
        val baseVisibility: Double = 1.0,
        val occlusionRisk: Double = 0.0,
    ) {
        init {
            require(actorId >= 0)
            require(finite(positionMeters))
            require(priority in 0.0..1.0)
            require(baseVisibility in 0.0..1.0)
            require(occlusionRisk in 0.0..1.0)
        }
    }

    data class CameraAnchor(
        val cameraId: String,
        val positionMeters: Vec3,
        val forward: Vec3,
        val authorityScore: Double,
        val sector: Int,
        val calibrationConfidence: Double,
        val synchronizationConfidence: Double,
        val networkHeadroom: Double,
        val thermalHeadroom: Double,
        val motionBlurRisk: Double? = null,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(finite(positionMeters))
            require(finite(forward))
            require(authorityScore in 0.0..1.0)
            require(sector >= 0)
            require(calibrationConfidence in 0.0..1.0)
            require(synchronizationConfidence in 0.0..1.0)
            require(networkHeadroom in 0.0..1.0)
            require(thermalHeadroom in 0.0..1.0)
            motionBlurRisk?.let {
                require(
                    it.isFinite() &&
                        it in 0.0..1.0,
                )
            }
        }
    }

    data class Score(
        val cameraId: String,
        val actorId: Int,
        val geometryAvailable: Boolean,
        val distanceMeters: Double,
        val distanceScore: Double,
        val facingScore: Double,
        val baseVisibilityScore: Double,
        val actorVisibilityScore: Double,
        val occlusionPenalty: Double,
        val authorityScore: Double,
        val calibrationScore: Double,
        val synchronizationScore: Double,
        val resourceScore: Double,
        val opticalIntegrityScore: Double,
        val noveltyScore: Double,
        val noveltyDegrees: Double,
        val baselineScore: Double,
        val combinedGeometryScore: Double,
        val finalScore: Double,
        val diagnostics: List<String>,
    )

    fun evaluate(
        actor: ActorAnchor,
        camera: CameraAnchor,
        selectedCameras: Collection<CameraAnchor> = emptyList(),
    ): Score {
        val target = actor.positionMeters - camera.positionMeters
        val distance = target.norm()

        if (
            !distance.isFinite() ||
            distance < config.minimumDistanceMeters ||
            distance > config.maximumDistanceMeters
        ) {
            return fallback(
                actor,
                camera,
                selectedCameras,
                "ACTOR_CAMERA_DISTANCE_UNUSABLE",
            )
        }

        val targetDirection = target.normalized()
        val forward = camera.forward.normalized()
        if (
            targetDirection.norm() < 1e-9 ||
            forward.norm() < 1e-9
        ) {
            return fallback(
                actor,
                camera,
                selectedCameras,
                "CAMERA_FORWARD_OR_TARGET_DEGENERATE",
            )
        }

        val cosine =
            forward.dot(targetDirection)
                .coerceIn(-1.0, 1.0)
        val facing =
            facingScore(cosine)
        val distanceQuality =
            distanceScore(distance)
        val baseVisibility =
            actor.baseVisibility
        val occlusion =
            actor.occlusionRisk

        val metricVisibility =
            (
                0.58 * facing +
                    0.27 * distanceQuality +
                    0.15 * (1.0 - occlusion)
            ).coerceIn(0.0, 1.0)

        val selected =
            selectedCameras.filter {
                it.cameraId != camera.cameraId
            }

        val noveltyDegrees =
            angularNovelty(
                actor.positionMeters,
                camera.positionMeters,
                selected,
            )

        val novelty =
            noveltyScore(
                noveltyDegrees,
                selected.isNotEmpty(),
            )

        val baseline =
            baselineConditioning(
                actor.positionMeters,
                camera.positionMeters,
                selected,
            )

        val geometry =
            (
                config.facingWeight * facing +
                    config.distanceWeight * distanceQuality +
                    config.noveltyWeight * novelty +
                    config.baselineWeight * baseline
            ).div(
                config.facingWeight +
                    config.distanceWeight +
                    config.noveltyWeight +
                    config.baselineWeight,
            ).coerceIn(0.0, 1.0)

        val calibration =
            camera.calibrationConfidence
        val synchronization =
            camera.synchronizationConfidence
        val resources =
            (
                0.50 * camera.networkHeadroom +
                    0.50 * camera.thermalHeadroom
            ).coerceIn(0.0, 1.0)
        val opticalIntegrity =
            camera.motionBlurRisk
                ?.let {
                    (1.0 - it).coerceIn(0.0, 1.0)
                }
                ?: 1.0

        val finalScore =
            (
                0.27 * camera.authorityScore +
                    0.22 * baseVisibility +
                    0.19 * metricVisibility +
                    0.17 * geometry +
                    0.08 * calibration +
                    0.05 * synchronization +
                    0.02 * resources +
                    config.opticalIntegrityWeight *
                        opticalIntegrity
            ).coerceIn(0.0, 1.0)

        val diagnostics =
            buildList {
                if (cosine < config.minimumFacingCosine) {
                    add("CAMERA_FACES_AWAY_FROM_ACTOR")
                }
                if (facing < 0.35) {
                    add("LOW_OPTICAL_AXIS_ALIGNMENT")
                }
                if (distanceQuality < 0.35) {
                    add("DISTANCE_CONDITIONING_WEAK")
                }
                if (
                    selected.isNotEmpty() &&
                        noveltyDegrees <
                            config.minimumUsefulNoveltyDegrees
                ) {
                    add("VIEW_ANGULAR_NOVELTY_LOW")
                }
                if (
                    selected.size >= 2 &&
                        baseline < 0.25
                ) {
                    add("BASELINE_CONDITIONING_WEAK")
                }
                if (occlusion > 0.60) {
                    add("ACTOR_OCCLUSION_RISK_HIGH")
                }
                if (calibration < 0.60) {
                    add("CALIBRATION_CONFIDENCE_LOW")
                }
                if (synchronization < 0.60) {
                    add("SYNCHRONIZATION_CONFIDENCE_LOW")
                }
                camera.motionBlurRisk?.let { risk ->
                    if (risk > 0.35) {
                        add("MOTION_BLUR_RISK_HIGH")
                    } else if (risk > 0.18) {
                        add("MOTION_BLUR_RISK_ELEVATED")
                    }
                }
            }

        return Score(
            cameraId = camera.cameraId,
            actorId = actor.actorId,
            geometryAvailable = true,
            distanceMeters = distance,
            distanceScore = distanceQuality,
            facingScore = facing,
            baseVisibilityScore = baseVisibility,
            actorVisibilityScore = metricVisibility,
            occlusionPenalty = occlusion,
            authorityScore = camera.authorityScore,
            calibrationScore = calibration,
            synchronizationScore = synchronization,
            resourceScore = resources,
            opticalIntegrityScore = opticalIntegrity,
            noveltyScore = novelty,
            noveltyDegrees = noveltyDegrees,
            baselineScore = baseline,
            combinedGeometryScore = geometry,
            finalScore = finalScore,
            diagnostics = diagnostics.distinct(),
        )
    }

    private fun fallback(
        actor: ActorAnchor,
        camera: CameraAnchor,
        selectedCameras: Collection<CameraAnchor>,
        reason: String,
    ): Score {
        val noveltyDegrees =
            angularNovelty(
                actor.positionMeters,
                camera.positionMeters,
                selectedCameras,
            )
        val novelty =
            noveltyScore(
                noveltyDegrees,
                selectedCameras.isNotEmpty(),
            )
        return Score(
            cameraId = camera.cameraId,
            actorId = actor.actorId,
            geometryAvailable = false,
            distanceMeters = Double.POSITIVE_INFINITY,
            distanceScore = config.fallbackGeometryScore,
            facingScore = config.fallbackGeometryScore,
            baseVisibilityScore = actor.baseVisibility,
            actorVisibilityScore = actor.baseVisibility,
            occlusionPenalty = actor.occlusionRisk,
            authorityScore = camera.authorityScore,
            calibrationScore = camera.calibrationConfidence,
            synchronizationScore = camera.synchronizationConfidence,
            resourceScore =
                (
                    camera.networkHeadroom +
                        camera.thermalHeadroom
                ) * 0.5,
            opticalIntegrityScore =
                camera.motionBlurRisk?.let {
                    (1.0 - it).coerceIn(0.0, 1.0)
                } ?: 1.0,
            noveltyScore = novelty,
            noveltyDegrees = noveltyDegrees,
            baselineScore = 0.0,
            combinedGeometryScore = config.fallbackGeometryScore,
            finalScore =
                (
                    0.52 * camera.authorityScore +
                        0.30 * actor.baseVisibility +
                        0.10 * novelty +
                        0.08 * camera.calibrationConfidence
                ).coerceIn(0.0, 1.0),
            diagnostics = listOf(reason),
        )
    }

    private fun facingScore(cosine: Double): Double {
        val span =
            (1.0 - config.minimumFacingCosine)
                .coerceAtLeast(1e-6)
        val normalized =
            (
                (cosine - config.minimumFacingCosine) /
                    span
            ).coerceIn(0.0, 1.0)
        return if (cosine >= config.strongFacingCosine) {
            0.75 + 0.25 * normalized
        } else {
            normalized * 0.75
        }.coerceIn(0.0, 1.0)
    }

    private fun distanceScore(distance: Double): Double {
        val ratio =
            distance /
                config.preferredActorDistanceMeters
        if (!ratio.isFinite() || ratio <= 0.0) return 0.0
        return exp(
            -abs(ln(ratio)) /
                ln(1.0 + config.distanceScaleMeters),
        ).coerceIn(0.0, 1.0)
    }

    private fun noveltyScore(
        degrees: Double,
        hasSelection: Boolean,
    ): Double {
        if (!hasSelection) return 1.0
        if (!degrees.isFinite()) return 0.0
        if (degrees < config.minimumUsefulNoveltyDegrees) return 0.0

        val radians =
            Math.toRadians(
                degrees.coerceIn(0.0, 180.0),
            )
        val triangulationConditioning =
            abs(kotlin.math.sin(radians))

        val informationRamp =
            (
                degrees /
                    config.preferredNoveltyDegrees
            ).coerceIn(0.0, 1.0)

        return (
            0.75 * triangulationConditioning +
                0.25 * informationRamp
        ).coerceIn(0.0, 1.0)
    }

    private fun angularNovelty(
        actorPosition: Vec3,
        candidatePosition: Vec3,
        selected: Collection<CameraAnchor>,
    ): Double {
        if (selected.isEmpty()) {
            return config.preferredNoveltyDegrees
        }

        val candidateRay =
            (
                candidatePosition -
                    actorPosition
            ).normalized()

        if (candidateRay.norm() < 1e-9) return 0.0

        return selected.mapNotNull { camera ->
            val ray =
                (
                    camera.positionMeters -
                        actorPosition
                ).normalized()
            if (ray.norm() < 1e-9) {
                null
            } else {
                angleDegrees(candidateRay, ray)
            }
        }.minOrNull() ?: 0.0
    }

    private fun baselineConditioning(
        actorPosition: Vec3,
        candidatePosition: Vec3,
        selected: Collection<CameraAnchor>,
    ): Double {
        if (selected.isEmpty()) return 0.5

        val candidateDistance =
            candidatePosition.distanceTo(actorPosition)
                .coerceAtLeast(config.minimumDistanceMeters)

        val baselines =
            selected
                .map {
                    it.positionMeters.distanceTo(candidatePosition)
                }
                .filter {
                    it.isFinite() && it > 1e-6
                }

        if (baselines.isEmpty()) return 0.0

        return (
            baselines
                .map {
                    (
                        it /
                            candidateDistance
                    ).coerceIn(0.0, 4.0)
                }
                .average() /
                2.0
        ).coerceIn(0.0, 1.0)
    }

    private fun angleDegrees(
        a: Vec3,
        b: Vec3,
    ): Double {
        val denominator =
            (
                a.norm() *
                    b.norm()
            ).coerceAtLeast(1e-12)

        return (
            acos(
                (
                    a.dot(b) /
                        denominator
                ).coerceIn(-1.0, 1.0),
            ) *
                180.0 /
                PI
        ).coerceIn(0.0, 180.0)
    }

    private fun finite(value: Vec3): Boolean =
        value.x.isFinite() &&
            value.y.isFinite() &&
            value.z.isFinite()
}

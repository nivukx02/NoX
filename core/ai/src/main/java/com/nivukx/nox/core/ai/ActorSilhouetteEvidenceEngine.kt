package com.nivukx.nox.core.ai

import kotlin.math.max
import kotlin.math.min

/**
 * Converts a semantic foreground mask into conservative per-track support evidence.
 *
 * The segmentation model is not an instance-segmentation model, so this engine never calls its
 * output an actor mask. It samples each tracked actor's box, center and semantic body joints,
 * then estimates foreground support and overlap ambiguity. Low support is evidence of occlusion,
 * truncation, detector drift, or background contamination, not evidence that another actor exists.
 */
class ActorSilhouetteEvidenceEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val foregroundThreshold: Double = 0.45,
        val strongThreshold: Double = 0.72,
        val minimumJointSamples: Int = 4,
        val maximumJointSamples: Int = 24,
        val maximumActors: Int = 64,
        val overlapAmbiguityIou: Double = 0.35,
        val hardAmbiguityIou: Double = 0.60,
    ) {
        init {
            require(foregroundThreshold in 0.0..1.0)
            require(strongThreshold >= foregroundThreshold)
            require(minimumJointSamples >= 1)
            require(maximumJointSamples >= minimumJointSamples)
            require(maximumActors in 1..64)
            require(overlapAmbiguityIou in 0.0..1.0)
            require(hardAmbiguityIou >= overlapAmbiguityIou)
        }
    }

    data class Evidence(
        val actorId: Long,
        val foregroundSupport: Double,
        val strongForegroundSupport: Double,
        val jointSupport: Double,
        val poseMaskSupport: Double,
        val poseMaskQuality: Double,
        val boundarySupport: Double,
        val overlapAmbiguity: Double,
        val visibilityAgreement: Double,
        val reliability: Double,
        val samplesEvaluated: Int,
        val warnings: List<String>,
    ) {
        val usable: Boolean
            get() =
                reliability >= 0.20 &&
                    samplesEvaluated >= 1
    }

    @Synchronized
    fun evaluate(
        tracks: List<MultiActorPoseAssociationEngine.Result>,
        segmentation: PersonSegmentationEngine.Result?,
        poseSegmentationEvidence:
            Map<Long, PoseMaskEvidenceEngine.Result> =
            emptyMap(),
    ): Map<Long, Evidence> {
        if (tracks.isEmpty() || segmentation == null) {
            return emptyMap()
        }

        val boundedTracks =
            tracks
                .asSequence()
                .filter { it.actorId > 0L }
                .take(config.maximumActors)
                .toList()

        return boundedTracks.associate { actor ->
            actor.actorId to evaluateActor(
                actor = actor,
                allActors = boundedTracks,
                segmentation = segmentation,
                poseSegmentationEvidence =
                    poseSegmentationEvidence[actor.actorId],
            )
        }
    }

    private fun evaluateActor(
        actor: MultiActorPoseAssociationEngine.Result,
        allActors: List<MultiActorPoseAssociationEngine.Result>,
        segmentation: PersonSegmentationEngine.Result,
        poseSegmentationEvidence:
            PoseMaskEvidenceEngine.Result?,
    ): Evidence {
        val samples =
            ArrayList<Sample>(
                config.maximumJointSamples + 12,
            )

        samples += Sample(
            x = actor.center.x,
            y = actor.center.y,
            semanticWeight = 0.75,
        )

        val bounds = actor.bounds
        samples += listOf(
            Sample(bounds.left, bounds.top, 0.55),
            Sample(bounds.right, bounds.top, 0.55),
            Sample(bounds.left, bounds.bottom, 0.55),
            Sample(bounds.right, bounds.bottom, 0.55),
            Sample(
                (bounds.left + bounds.right) * 0.5,
                bounds.top,
                0.65,
            ),
            Sample(
                (bounds.left + bounds.right) * 0.5,
                bounds.bottom,
                0.65,
            ),
            Sample(
                bounds.left,
                (bounds.top + bounds.bottom) * 0.5,
                0.55,
            ),
            Sample(
                bounds.right,
                (bounds.top + bounds.bottom) * 0.5,
                0.55,
            ),
        )

        actor.joints
            .asSequence()
            .filter {
                it.x.isFinite() &&
                    it.y.isFinite() &&
                    it.confidence.isFinite()
            }
            .sortedByDescending {
                it.confidence
            }
            .take(config.maximumJointSamples)
            .forEach { joint ->
                samples +=
                    Sample(
                        x = joint.x,
                        y = joint.y,
                        semanticWeight =
                            (
                                0.55 +
                                    joint.confidence
                                        .coerceIn(0.0, 1.0) *
                                    0.45
                            ),
                    )
            }

        val deduplicated =
            samples
                .filter {
                    it.x.isFinite() &&
                        it.y.isFinite()
                }
                .map {
                    it.copy(
                        x = it.x.coerceIn(0.0, 1.0),
                        y = it.y.coerceIn(0.0, 1.0),
                    )
                }
                .distinctBy {
                    (
                        (it.x * 2048.0)
                            .toInt()
                            .coerceIn(0, 2048)
                    ) to
                        (
                            (it.y * 2048.0)
                                .toInt()
                                .coerceIn(0, 2048)
                        )
                }

        val evaluated =
            if (deduplicated.size >= config.minimumJointSamples) {
                deduplicated
            } else {
                deduplicated.take(config.maximumJointSamples)
            }

        if (evaluated.isEmpty()) {
            return Evidence(
                actorId = actor.actorId,
                foregroundSupport = 0.0,
                strongForegroundSupport = 0.0,
                jointSupport = 0.0,
                poseMaskSupport =
                    poseSegmentationEvidence?.support ?: 0.0,
                poseMaskQuality =
                    poseSegmentationEvidence?.quality ?: 0.0,
                boundarySupport = segmentation.boundaryStability,
                overlapAmbiguity = overlapAmbiguity(
                    actor,
                    allActors,
                ),
                visibilityAgreement = 0.0,
                reliability = 0.0,
                samplesEvaluated = 0,
                warnings = listOf(
                    "SILHOUETTE_NO_SAMPLES",
                ),
            )
        }

        var weightedSupport = 0.0
        var weightSum = 0.0
        var strongSupport = 0.0
        var strongWeight = 0.0
        var jointSupport = 0.0
        var jointWeight = 0.0

        evaluated.forEach { sample ->
            val support =
                segmentation.silhouetteSupport(
                    sample.x,
                    sample.y,
                ).coerceIn(0.0, 1.0)

            weightedSupport +=
                support *
                    sample.semanticWeight
            weightSum += sample.semanticWeight

            if (
                support >=
                    config.strongThreshold
            ) {
                strongSupport +=
                    sample.semanticWeight
            }
            strongWeight += sample.semanticWeight

            val isJoint =
                sample.semanticWeight >=
                    0.80
            if (isJoint) {
                jointSupport +=
                    support *
                        sample.semanticWeight
                jointWeight +=
                    sample.semanticWeight
            }
        }

        val foregroundSupport =
            (
                weightedSupport /
                    weightSum.coerceAtLeast(1e-9)
                )
                .coerceIn(0.0, 1.0)

        val strongForegroundSupport =
            (
                strongSupport /
                    strongWeight.coerceAtLeast(1e-9)
                )
                .coerceIn(0.0, 1.0)

        val measuredJointSupport =
            if (jointWeight <= 1e-9) {
                foregroundSupport
            } else {
                (
                    jointSupport /
                        jointWeight
                ).coerceIn(0.0, 1.0)
            }

        val ambiguity =
            overlapAmbiguity(
                actor,
                allActors,
            )

        val visibilityAgreement =
            (
                foregroundSupport * 0.62 +
                    measuredJointSupport * 0.22 +
                    actor.confidence
                        .coerceIn(0.0, 1.0) *
                        0.16
                ).coerceIn(0.0, 1.0)

        val boundarySupport =
            segmentation.boundaryStability
                .coerceIn(0.0, 1.0)

        val poseMaskSupport =
            poseSegmentationEvidence
                ?.support
                ?.coerceIn(0.0, 1.0)
                ?: 0.0

        val poseMaskQuality =
            poseSegmentationEvidence
                ?.quality
                ?.coerceIn(0.0, 1.0)
                ?: 0.0

        val reliability =
            (
                foregroundSupport * 0.28 +
                    strongForegroundSupport * 0.10 +
                    measuredJointSupport * 0.18 +
                    poseMaskSupport * 0.16 +
                    poseMaskQuality * 0.08 +
                    boundarySupport * 0.12 +
                    visibilityAgreement * 0.08 +
                    (1.0 - ambiguity) * 0.08
                ).coerceIn(0.0, 1.0)

        val warnings =
            buildList {
                if (foregroundSupport < 0.25) {
                    add("ACTOR_SILHOUETTE_SUPPORT_LOW")
                }
                if (strongForegroundSupport < 0.25) {
                    add("ACTOR_STRONG_SILHOUETTE_SUPPORT_LOW")
                }
                if (boundarySupport < 0.45) {
                    add("SILHOUETTE_BOUNDARY_UNSTABLE")
                }
                if (
                    poseSegmentationEvidence != null &&
                    poseMaskQuality < 0.40
                ) {
                    add("POSE_MASK_QUALITY_DEGRADED")
                }
                if (
                    ambiguity >=
                        config.overlapAmbiguityIou
                ) {
                    add("ACTOR_SILHOUETTE_OVERLAP_AMBIGUITY")
                }
                if (
                    ambiguity >=
                        config.hardAmbiguityIou
                ) {
                    add("ACTOR_SILHOUETTE_HARD_AMBIGUITY")
                }
                if (
                    actor.missedFrames > 0
                ) {
                    add("ACTOR_SILHOUETTE_TRACK_RECOVERY")
                }
            }.distinct()

        return Evidence(
            actorId = actor.actorId,
            foregroundSupport = foregroundSupport,
            strongForegroundSupport = strongForegroundSupport,
            jointSupport = measuredJointSupport,
            poseMaskSupport = poseMaskSupport,
            poseMaskQuality = poseMaskQuality,
            boundarySupport = boundarySupport,
            overlapAmbiguity = ambiguity,
            visibilityAgreement = visibilityAgreement,
            reliability = reliability,
            samplesEvaluated = evaluated.size,
            warnings = warnings,
        )
    }

    private fun overlapAmbiguity(
        actor: MultiActorPoseAssociationEngine.Result,
        allActors: List<MultiActorPoseAssociationEngine.Result>,
    ): Double {
        val actorArea =
            actor.bounds.area
                .coerceAtLeast(1e-9)

        val maximumIou =
            allActors
                .asSequence()
                .filter {
                    it.actorId != actor.actorId
                }
                .map { other ->
                    intersectionOverUnion(
                        actor.bounds,
                        other.bounds,
                        actorArea,
                        other.bounds.area
                            .coerceAtLeast(1e-9),
                    )
                }
                .maxOrNull()
                ?: 0.0

        return if (
            maximumIou <=
                config.overlapAmbiguityIou
        ) {
            0.0
        } else {
            (
                (
                    maximumIou -
                        config.overlapAmbiguityIou
                    ) /
                    (
                        config.hardAmbiguityIou -
                            config.overlapAmbiguityIou
                    ).coerceAtLeast(1e-6)
                )
                    .coerceIn(0.0, 1.0)
        }
    }

    private fun intersectionOverUnion(
        a: MultiActorPoseAssociationEngine.Bounds,
        b: MultiActorPoseAssociationEngine.Bounds,
        areaA: Double,
        areaB: Double,
    ): Double {
        val left =
            max(
                a.left,
                b.left,
            )
        val top =
            max(
                a.top,
                b.top,
            )
        val right =
            min(
                a.right,
                b.right,
            )
        val bottom =
            min(
                a.bottom,
                b.bottom,
            )

        val intersection =
            (
                right - left
            ).coerceAtLeast(0.0) *
                (
                    bottom - top
                ).coerceAtLeast(0.0)

        return (
            intersection /
                (
                    areaA +
                        areaB -
                        intersection
                ).coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)
    }

    private data class Sample(
        val x: Double,
        val y: Double,
        val semanticWeight: Double,
    )
}

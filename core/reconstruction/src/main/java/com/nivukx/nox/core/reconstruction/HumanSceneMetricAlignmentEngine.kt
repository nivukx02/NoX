package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Metric human-scene alignment stage.
 *
 * Explicitly solves for scale and translation from anthropometric and contact evidence while
 * retaining gravity residuals and robust scene-contact residuals as production quality evidence.
 */
class HumanSceneMetricAlignmentEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val minimumConfidence: Double = 0.30,
        val minimumScaleRatio: Double = 0.40,
        val maximumScaleRatio: Double = 2.50,
        val maximumTranslationMeters: Double = 5.0,
        val contactHuberMeters: Double = 0.045,
        val contactOutlierMeters: Double = 0.20,
        val gravityLimitDegrees: Double = 4.0,
        val minimumEvidence: Int = 3,
        val maximumLandmarks: Int = 4096,
        val maximumSceneAnchors: Int = 4096,
        val maximumContacts: Int = 1024,
    ) {
        init {
            require(minimumConfidence in 0.0..1.0)
            require(minimumScaleRatio > 0.0)
            require(maximumScaleRatio > minimumScaleRatio)
            require(maximumTranslationMeters > 0.0)
            require(contactHuberMeters > 0.0)
            require(contactOutlierMeters > contactHuberMeters)
            require(gravityLimitDegrees > 0.0)
            require(minimumEvidence > 0)
        }
    }

    data class HumanLandmark(
        val id: Int,
        val position: Vec3,
        val confidence: Double,
        val contactCandidate: Boolean = false,
    ) {
        init {
            require(id >= 0)
            requireFinite(position)
            require(confidence in 0.0..1.0)
        }
    }

    data class SceneAnchor(
        val id: Int,
        val position: Vec3,
        val normal: Vec3,
        val confidence: Double,
        val static: Boolean = true,
    ) {
        init {
            require(id >= 0)
            requireFinite(position)
            requireFinite(normal)
            require(confidence in 0.0..1.0)
        }
    }

    data class ScaleEvidence(
        val segmentId: String,
        val observedLengthMeters: Double,
        val referenceLengthMeters: Double,
        val confidence: Double,
    ) {
        init {
            require(segmentId.isNotBlank())
            require(observedLengthMeters > 0.0 && observedLengthMeters.isFinite())
            require(referenceLengthMeters > 0.0 && referenceLengthMeters.isFinite())
            require(confidence in 0.0..1.0)
        }
        val ratio: Double get() = observedLengthMeters / referenceLengthMeters
    }

    data class GravityEvidence(
        val direction: Vec3,
        val confidence: Double,
    ) {
        init {
            requireFinite(direction)
            require(confidence in 0.0..1.0)
        }
    }

    data class ContactConstraint(
        val landmarkId: Int,
        val sceneAnchorId: Int,
        val confidence: Double,
        val expectedOffsetMeters: Double = 0.0,
    ) {
        init {
            require(landmarkId >= 0)
            require(sceneAnchorId >= 0)
            require(confidence in 0.0..1.0)
            require(expectedOffsetMeters.isFinite())
        }
    }

    data class Input(
        val humanLandmarks: List<HumanLandmark>,
        val sceneAnchors: List<SceneAnchor>,
        val scaleEvidence: List<ScaleEvidence>,
        val gravityEvidence: List<GravityEvidence> = emptyList(),
        val contacts: List<ContactConstraint> = emptyList(),
        val worldUp: Vec3 = Vec3(0.0, 1.0, 0.0),
    ) {
        init {
            require(humanLandmarks.isNotEmpty())
            requireFinite(worldUp)
        }
    }

    data class Result(
        val metricScale: Double?,
        val scaleConfidence: Double,
        val translation: Vec3,
        val translationMagnitudeMeters: Double,
        val gravityResidualDegrees: Double?,
        val gravityConfidence: Double,
        val contactResidualMeters: Double?,
        val contactInlierRatio: Double,
        val humanSceneResidualMeters: Double?,
        val worldConsistency: Double,
        val accepted: Boolean,
        val warnings: List<String>,
    )

    private data class ContactPair(
        val human: HumanLandmark,
        val scene: SceneAnchor,
        val confidence: Double,
        val expectedOffsetMeters: Double,
    )

    fun align(input: Input): Result {
        val human = input.humanLandmarks
            .filter { it.confidence >= config.minimumConfidence }
            .sortedByDescending { it.confidence }
            .take(config.maximumLandmarks)

        val scene = input.sceneAnchors
            .filter {
                it.static &&
                    it.confidence >= config.minimumConfidence
            }
            .sortedByDescending { it.confidence }
            .take(config.maximumSceneAnchors)

        val scale = estimateScale(input.scaleEvidence)
        val gravity = estimateGravity(input.gravityEvidence, input.worldUp)
        val pairs = resolvePairs(human, scene, input.contacts)
        val scaleValue = scale?.first ?: 1.0
        val translation = estimateTranslation(pairs, scaleValue)
        val contactResidual = contactResidual(pairs, translation, scaleValue)
        val inliers = contactInliers(pairs, translation, scaleValue)
        val sceneResidual = sceneResidual(pairs, translation, scaleValue)
        val residualScore = sceneResidual?.let {
            (1.0 - it / config.contactOutlierMeters).coerceIn(0.0, 1.0)
        } ?: 0.0
        val worldConsistency = (
            (scale?.second ?: 0.0) * 0.34 +
                (gravity?.second ?: 0.0) * 0.18 +
                inliers * 0.28 +
                residualScore * 0.20
            ).coerceIn(0.0, 1.0)

        val translationMagnitude = magnitude(translation)
        val warnings = buildList {
            if (scale == null) add("Metric scale evidence is unavailable.")
            else if (scale.second < 0.70) add("Metric scale evidence is inconsistent.")
            if (gravity == null) add("Gravity evidence is unavailable.")
            else if (gravity.first > config.gravityLimitDegrees) {
                add("World-up residual exceeds the target envelope.")
            }
            if (pairs.size < config.minimumEvidence) {
                add("Too few human-scene contact correspondences are available.")
            }
            if (contactResidual != null && contactResidual > config.contactHuberMeters) {
                add("Contact residual exceeds the robust transition.")
            }
            if (sceneResidual != null && sceneResidual > config.contactOutlierMeters) {
                add("Human-scene residual exceeds the accepted outlier envelope.")
            }
            if (translationMagnitude > config.maximumTranslationMeters) {
                add("World translation exceeds the bounded alignment envelope.")
            }
            if (worldConsistency < 0.70) {
                add("Global human-scene consistency is below the authoritative threshold.")
            }
        }

        val accepted =
            input.scaleEvidence.size >= config.minimumEvidence &&
                scale?.first?.let {
                    it in config.minimumScaleRatio..config.maximumScaleRatio
                } == true &&
                worldConsistency >= 0.70 &&
                translationMagnitude <= config.maximumTranslationMeters &&
                inliers >= 0.50

        return Result(
            metricScale = scale?.first,
            scaleConfidence = scale?.second ?: 0.0,
            translation = translation,
            translationMagnitudeMeters = translationMagnitude,
            gravityResidualDegrees = gravity?.first,
            gravityConfidence = gravity?.second ?: 0.0,
            contactResidualMeters = contactResidual,
            contactInlierRatio = inliers,
            humanSceneResidualMeters = sceneResidual,
            worldConsistency = worldConsistency,
            accepted = accepted,
            warnings = warnings,
        )
    }

    private fun estimateScale(
        evidence: List<ScaleEvidence>,
    ): Pair<Double, Double>? {
        val values = evidence
            .filter { it.confidence >= config.minimumConfidence }
            .map {
                it.ratio.coerceIn(
                    config.minimumScaleRatio,
                    config.maximumScaleRatio,
                ) to it.confidence
            }
        if (values.isEmpty()) return null
        val center = weightedMedian(values)
        val total = values.sumOf { it.second }.coerceAtLeast(1e-9)
        val deviation = values.sumOf {
            abs(it.first - center) * it.second
        } / total
        val consistency = (
            1.0 - deviation / max(center * 0.25, 1e-6)
            ).coerceIn(0.0, 1.0)
        return center to (values.map { it.second }.average() * consistency)
            .coerceIn(0.0, 1.0)
    }

    private fun estimateGravity(
        evidence: List<GravityEvidence>,
        worldUp: Vec3,
    ): Pair<Double, Double>? {
        if (evidence.isEmpty()) return null
        val target = normalize(worldUp) ?: Vec3(0.0, 1.0, 0.0)
        val valid = evidence.mapNotNull {
            val d = normalize(it.direction) ?: return@mapNotNull null
            d to it.confidence
        }
        if (valid.isEmpty()) return null
        val average = normalize(
            valid.fold(Vec3(0.0, 0.0, 0.0)) { acc, item ->
                acc + item.first * item.second
            },
        ) ?: return null
        val residual = angleDegrees(average, target)
        val confidence = (
            valid.map { it.second }.average() *
                (1.0 - residual / config.gravityLimitDegrees).coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)
        return residual to confidence
    }

    private fun resolvePairs(
        human: List<HumanLandmark>,
        scene: List<SceneAnchor>,
        contacts: List<ContactConstraint>,
    ): List<ContactPair> {
        val h = human.associateBy { it.id }
        val s = scene.associateBy { it.id }
        return contacts
            .sortedByDescending { it.confidence }
            .take(config.maximumContacts)
            .mapNotNull { c ->
                val humanPoint = h[c.landmarkId] ?: return@mapNotNull null
                val scenePoint = s[c.sceneAnchorId] ?: return@mapNotNull null
                ContactPair(
                    human = humanPoint,
                    scene = scenePoint,
                    confidence = (
                        humanPoint.confidence *
                            scenePoint.confidence *
                            c.confidence
                        ).coerceIn(0.0, 1.0),
                    expectedOffsetMeters = c.expectedOffsetMeters,
                )
            }
    }

    private fun estimateTranslation(
        pairs: List<ContactPair>,
        scale: Double,
    ): Vec3 {
        if (pairs.isEmpty()) return Vec3(0.0, 0.0, 0.0)
        var total = 0.0
        var humanSum = Vec3(0.0, 0.0, 0.0)
        var sceneSum = Vec3(0.0, 0.0, 0.0)
        for (pair in pairs) {
            val w = pair.confidence
            humanSum += pair.human.position * (w * scale)
            sceneSum += (
                pair.scene.position +
                    Vec3(0.0, pair.expectedOffsetMeters, 0.0)
                ) * w
            total += w
        }
        if (total <= 1e-9) return Vec3(0.0, 0.0, 0.0)
        return sceneSum / total - humanSum / total
    }

    private fun contactResidual(
        pairs: List<ContactPair>,
        translation: Vec3,
        scale: Double,
    ): Double? {
        if (pairs.isEmpty()) return null
        var weighted = 0.0
        var total = 0.0
        for (pair in pairs) {
            val d = distance(
                pair.human.position * scale + translation,
                pair.scene.position + Vec3(0.0, pair.expectedOffsetMeters, 0.0),
            )
            weighted += huber(d, config.contactHuberMeters) * pair.confidence
            total += pair.confidence
        }
        return weighted / total.coerceAtLeast(1e-9)
    }

    private fun sceneResidual(
        pairs: List<ContactPair>,
        translation: Vec3,
        scale: Double,
    ): Double? {
        if (pairs.isEmpty()) return null
        var weighted = 0.0
        var total = 0.0
        for (pair in pairs) {
            val d = distance(
                pair.human.position * scale + translation,
                pair.scene.position + Vec3(0.0, pair.expectedOffsetMeters, 0.0),
            )
            weighted += d * pair.confidence
            total += pair.confidence
        }
        return weighted / total.coerceAtLeast(1e-9)
    }

    private fun contactInliers(
        pairs: List<ContactPair>,
        translation: Vec3,
        scale: Double,
    ): Double {
        if (pairs.isEmpty()) return 0.0
        val count = pairs.count {
            val d = distance(
                it.human.position * scale + translation,
                it.scene.position + Vec3(0.0, it.expectedOffsetMeters, 0.0),
            )
            d <= config.contactOutlierMeters
        }
        return count.toDouble() / pairs.size
    }

    private fun weightedMedian(values: List<Pair<Double, Double>>): Double {
        val sorted = values.sortedBy { it.first }
        val target = sorted.sumOf { it.second } * 0.5
        var running = 0.0
        for (v in sorted) {
            running += v.second
            if (running >= target) return v.first
        }
        return sorted.last().first
    }

    private fun huber(value: Double, delta: Double): Double =
        if (value <= delta) 0.5 * value * value / delta
        else value - 0.5 * delta

    private fun normalize(v: Vec3): Vec3? {
        val length = magnitude(v)
        if (!length.isFinite() || length <= 1e-9) return null
        return v / length
    }

    private fun angleDegrees(a: Vec3, b: Vec3): Double =
        Math.toDegrees(
            kotlin.math.acos(
                (a.x * b.x + a.y * b.y + a.z * b.z)
                    .coerceIn(-1.0, 1.0),
            ),
        )

    private fun distance(a: Vec3, b: Vec3): Double = magnitude(a - b)

    private fun magnitude(v: Vec3): Double =
        sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    private companion object {
        fun requireFinite(v: Vec3) {
            require(v.x.isFinite() && v.y.isFinite() && v.z.isFinite())
        }
    }
}

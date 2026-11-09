package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Provenance-rich fusion layer for dense markerless landmarks.
 *
 * A landmark observation carries uncertainty, visibility, contact, occlusion, identity and
 * camera provenance. Those signals are preserved independently and only combined through bounded
 * weights. The output is evidence for calibrated geometry, never a fabricated metric guarantee.
 */
class DenseLandmarkEvidenceFusionEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val temporalToleranceNanos: Long = 4_000_000L,
        val minimumVisibility: Double = 0.40,
        val minimumObservationConfidence: Double = 0.35,
        val maximumPixelUncertainty: Double = 96.0,
        val robustScalePixels: Double = 6.0,
        val minimumDistinctViews: Int = 2,
        val preferredDistinctViews: Int = 4,
        val minimumConsensus: Double = 0.42,
        val contactBoost: Double = 0.08,
        val occlusionPenalty: Double = 0.32,
        val identityPenalty: Double = 0.45,
        val duplicateCameraPenalty: Double = 0.38,
        val maximumSymmetricRayDistanceMeters: Double = 0.08,
        val rayConsistencyWeight: Double = 0.18,
        val minimumCrossViewRayConsistency: Double = 0.45,
        val minimumRayAngleDegrees: Double = 0.25,
    )

    data class Observation(
        val landmarkId: Int,
        val deviceId: DeviceId,
        val xPx: Double,
        val yPx: Double,
        val timestampNanos: Long,
        val pixelUncertainty: Double,
        val detectorConfidence: Double,
        val visibilityConfidence: Double,
        val contactConfidence: Double = 0.0,
        val occlusionRisk: Double = 0.0,
        val identityConfidence: Double = 1.0,
        val semanticMaskId: Int? = null,
        val frameWidth: Int,
        val frameHeight: Int,
        val motionMagnitudePixelsPerSecond: Double = 0.0,
        val worldRayOriginMeters: Vec3? = null,
        val worldRayDirection: Vec3? = null,
    ) {
        init {
            require(landmarkId >= 0)
            require(xPx.isFinite() && yPx.isFinite())
            require(timestampNanos >= 0L)
            require(pixelUncertainty.isFinite() && pixelUncertainty >= 0.0)
            listOf(
                detectorConfidence,
                visibilityConfidence,
                contactConfidence,
                occlusionRisk,
                identityConfidence,
            ).forEach { require(it in 0.0..1.0) }
            require(frameWidth > 0 && frameHeight > 0)
            require(motionMagnitudePixelsPerSecond.isFinite())
            worldRayOriginMeters?.let { require(finite(it)) }
            worldRayDirection?.let {
                require(finite(it))
                require(it.norm() > 1e-9)
            }
            require(
                (worldRayOriginMeters == null) ==
                    (worldRayDirection == null),
            )
        }
    }

    data class WeightedObservation(
        val observation: Observation,
        val temporalWeight: Double,
        val visibilityWeight: Double,
        val uncertaintyWeight: Double,
        val contactWeight: Double,
        val occlusionWeight: Double,
        val identityWeight: Double,
        val uniquenessWeight: Double,
        val robustWeight: Double,
        val finalWeight: Double,
        val warnings: List<String>,
    )

    data class LandmarkFusion(
        val landmarkId: Int,
        val timestampNanos: Long,
        val observations: List<WeightedObservation>,
        val distinctCameras: Int,
        val effectiveViewCount: Double,
        val fusedXNormalized: Double,
        val fusedYNormalized: Double,
        val pixelScatter: Double,
        val consensusConfidence: Double,
        val visibilityConfidence: Double,
        val contactConfidence: Double,
        val occlusionRisk: Double,
        val semanticMaskAgreement: Double,
        val crossViewRayConsistency: Double = 0.50,
        val meanSymmetricRayDistanceMeters: Double = Double.POSITIVE_INFINITY,
        val admissibleForGeometry: Boolean,
        val blockers: List<String>,
        val warnings: List<String>,
    )

    data class SessionFusion(
        val timestampNanos: Long,
        val landmarks: List<LandmarkFusion>,
        val geometryReadyCount: Int,
        val geometryBlockedCount: Int,
        val meanConsensus: Double,
        val meanEffectiveViews: Double,
        val warnings: List<String>,
    )

    fun fuse(
        observations: List<Observation>,
        targetTimestampNanos: Long? = null,
    ): SessionFusion {
        if (observations.isEmpty()) {
            return SessionFusion(
                timestampNanos = targetTimestampNanos ?: 0L,
                landmarks = emptyList(),
                geometryReadyCount = 0,
                geometryBlockedCount = 0,
                meanConsensus = 0.0,
                meanEffectiveViews = 0.0,
                warnings = listOf("NO_LANDMARK_OBSERVATIONS"),
            )
        }

        val timestamp = targetTimestampNanos
            ?: observations.map { it.timestampNanos }.sorted()[observations.size / 2]

        val groups = observations
            .filter {
                abs(it.timestampNanos - timestamp) <= config.temporalToleranceNanos
            }
            .groupBy { it.landmarkId }

        val landmarks = groups
            .toSortedMap()
            .map { (id, candidates) ->
                fuseLandmark(id, timestamp, candidates)
            }

        return SessionFusion(
            timestampNanos = timestamp,
            landmarks = landmarks,
            geometryReadyCount = landmarks.count { it.admissibleForGeometry },
            geometryBlockedCount = landmarks.count { !it.admissibleForGeometry },
            meanConsensus = landmarks.map { it.consensusConfidence }.averageOrZero(),
            meanEffectiveViews = landmarks.map { it.effectiveViewCount }.averageOrZero(),
            warnings = landmarks.flatMap { it.warnings }.distinct().take(32),
        )
    }

    fun scoreObservation(
        observation: Observation,
        referenceTimestampNanos: Long,
        occupiedCameras: Set<DeviceId> = emptySet(),
    ): WeightedObservation =
        weightObservation(
            observation,
            referenceTimestampNanos,
            occupiedCameras,
            robustWeight = 1.0,
        )

    private fun fuseLandmark(
        landmarkId: Int,
        timestamp: Long,
        candidates: List<Observation>,
    ): LandmarkFusion {
        val first = candidates
            .sortedByDescending(::priority)
            .take(24)
            .map { observation ->
                weightObservation(
                    observation,
                    timestamp,
                    emptySet(),
                    robustWeight = 1.0,
                )
            }
            .filter { it.finalWeight > 0.0 }

        val coarse = centroid(first)
        val refined = first.map { weighted ->
            val robust = robustKernel(pixelDistance(weighted.observation, coarse))
            val final = multiply(
                weighted.temporalWeight,
                weighted.visibilityWeight,
                weighted.uncertaintyWeight,
                weighted.contactWeight,
                weighted.occlusionWeight,
                weighted.identityWeight,
                weighted.uniquenessWeight,
                robust,
                weighted.observation.detectorConfidence,
            )
            weighted.copy(
                robustWeight = robust,
                finalWeight = final,
            )
        }
            .filter { it.finalWeight >= 0.05 }
            .sortedByDescending { it.finalWeight }
            .take(24)

        val fused = centroid(refined)
        val scatter = scatter(refined, fused)
        val cameras = refined.map { it.observation.deviceId }.distinct()
        val effectiveViews = effectiveViews(refined)
        val visibility = weightedAverage(refined) { it.observation.visibilityConfidence }
        val contact = weightedAverage(refined) { it.observation.contactConfidence }
        val occlusion = weightedAverage(refined) { it.observation.occlusionRisk }
        val maskAgreement = maskAgreement(refined)
        val rayGeometry = crossViewRayGeometry(refined)
        val consensus = (
            min(effectiveViews / config.preferredDistinctViews.toDouble(), 1.0) * 0.25 +
                robustScatterScore(scatter) * 0.25 +
                visibility * 0.20 +
                maskAgreement * 0.08 +
                rayGeometry.consistency * config.rayConsistencyWeight +
                weightedAverage(refined) {
                    it.finalWeight
                } * (0.12 - 0.04 * config.rayConsistencyWeight)
            ).coerceIn(0.0, 1.0)

        val blockers = buildList {
            if (cameras.size < config.minimumDistinctViews) add("INSUFFICIENT_DISTINCT_CAMERA_VIEWS")
            if (effectiveViews < config.minimumDistinctViews) add("INSUFFICIENT_EFFECTIVE_VIEWS")
            if (visibility < config.minimumVisibility) add("VISIBILITY_BELOW_GEOMETRIC_GATE")
            if (consensus < config.minimumConsensus) add("LANDMARK_CONSENSUS_BELOW_GATE")
            if (occlusion > 0.70) add("OCCLUSION_RISK_TOO_HIGH")
            if (rayGeometry.available && rayGeometry.consistency < config.minimumCrossViewRayConsistency) {
                add("CROSS_VIEW_RAY_CONSISTENCY_LOW")
            }
        }.distinct()

        val warnings = buildList {
            if (effectiveViews < config.preferredDistinctViews) add("PREFERRED_VIEW_DIVERSITY_NOT_REACHED")
            if (scatter > config.robustScalePixels) add("PIXEL_SCATTER_HIGH")
            if (maskAgreement < 0.70) add("SEMANTIC_MASK_AGREEMENT_LOW")
            if (contact > 0.70) add("CONTACT_CONTEXT_ACTIVE")
        }.distinct()

        return LandmarkFusion(
            landmarkId = landmarkId,
            timestampNanos = timestamp,
            observations = refined,
            distinctCameras = cameras.size,
            effectiveViewCount = effectiveViews,
            fusedXNormalized = fused.first,
            fusedYNormalized = fused.second,
            pixelScatter = scatter,
            consensusConfidence = consensus,
            visibilityConfidence = visibility,
            contactConfidence = contact,
            occlusionRisk = occlusion,
            semanticMaskAgreement = maskAgreement,
            crossViewRayConsistency = rayGeometry.consistency,
            meanSymmetricRayDistanceMeters = rayGeometry.meanDistanceMeters,
            admissibleForGeometry = blockers.isEmpty(),
            blockers = blockers,
            warnings = warnings,
        )
    }

    private fun weightObservation(
        observation: Observation,
        referenceTimestampNanos: Long,
        occupiedCameras: Set<DeviceId>,
        robustWeight: Double,
    ): WeightedObservation {
        val temporalWeight = exp(
            -abs(
                observation.timestampNanos - referenceTimestampNanos,
            ).toDouble() / config.temporalToleranceNanos,
        ).coerceIn(0.0, 1.0)

        val visibilityWeight = observation.visibilityConfidence
        val uncertaintyWeight = (
            1.0 -
                observation.pixelUncertainty /
                    config.maximumPixelUncertainty
            ).coerceIn(0.0, 1.0)
        val contactWeight = 1.0 +
            config.contactBoost * observation.contactConfidence
        val occlusionWeight = (
            1.0 -
                config.occlusionPenalty * observation.occlusionRisk
            ).coerceIn(0.0, 1.0)
        val identityWeight = (
            (1.0 - config.identityPenalty) +
                config.identityPenalty * observation.identityConfidence
            ).coerceIn(0.0, 1.0)
        val uniquenessWeight = if (observation.deviceId in occupiedCameras) {
            config.duplicateCameraPenalty
        } else {
            1.0
        }

        val warnings = buildList {
            if (temporalWeight < 0.50) add("TEMPORAL_MISMATCH")
            if (uncertaintyWeight < 0.50) add("PIXEL_UNCERTAINTY_HIGH")
            if (visibilityWeight < config.minimumVisibility) add("VISIBILITY_LOW")
            if (observation.occlusionRisk > 0.60) add("OCCLUSION_HIGH")
            if (observation.identityConfidence < 0.70) add("IDENTITY_AMBIGUOUS")
            if (observation.semanticMaskId == null) add("SEMANTIC_MASK_MISSING")
        }

        val final = multiply(
            observation.detectorConfidence,
            temporalWeight,
            visibilityWeight,
            uncertaintyWeight,
            contactWeight,
            occlusionWeight,
            identityWeight,
            uniquenessWeight,
            robustWeight,
        )

        return WeightedObservation(
            observation = observation,
            temporalWeight = temporalWeight,
            visibilityWeight = visibilityWeight,
            uncertaintyWeight = uncertaintyWeight,
            contactWeight = contactWeight,
            occlusionWeight = occlusionWeight,
            identityWeight = identityWeight,
            uniquenessWeight = uniquenessWeight,
            robustWeight = robustWeight,
            finalWeight = final,
            warnings = warnings.distinct(),
        )
    }

    private fun centroid(values: List<WeightedObservation>): Pair<Double, Double> {
        if (values.isEmpty()) return 0.0 to 0.0
        var x = 0.0
        var y = 0.0
        var weight = 0.0
        values.forEach { value ->
            val w = value.finalWeight.coerceAtLeast(0.0)
            x += value.observation.xPx / value.observation.frameWidth * w
            y += value.observation.yPx / value.observation.frameHeight * w
            weight += w
        }
        if (weight <= 1e-12) return 0.0 to 0.0
        return (x / weight).coerceIn(0.0, 1.0) to
            (y / weight).coerceIn(0.0, 1.0)
    }

    private fun pixelDistance(
        observation: Observation,
        center: Pair<Double, Double>,
    ): Double {
        val dx = observation.xPx / observation.frameWidth - center.first
        val dy = observation.yPx / observation.frameHeight - center.second
        return sqrt(
            (dx * observation.frameWidth) * (dx * observation.frameWidth) +
                (dy * observation.frameHeight) * (dy * observation.frameHeight),
        )
    }

    private fun scatter(
        values: List<WeightedObservation>,
        center: Pair<Double, Double>,
    ): Double {
        if (values.isEmpty()) return Double.POSITIVE_INFINITY
        var weighted = 0.0
        var total = 0.0
        values.forEach {
            weighted += pixelDistance(it.observation, center) * it.finalWeight
            total += it.finalWeight
        }
        return if (total <= 1e-12) Double.POSITIVE_INFINITY else weighted / total
    }

    private data class RayGeometryScore(
        val available: Boolean,
        val consistency: Double,
        val meanDistanceMeters: Double,
    )

    private fun crossViewRayGeometry(
        values: List<WeightedObservation>,
    ): RayGeometryScore {
        data class Ray(
            val origin: Vec3,
            val direction: Vec3,
            val weight: Double,
        )

        val rays = values.mapNotNull { weighted ->
            val origin = weighted.observation.worldRayOriginMeters
            val direction = weighted.observation.worldRayDirection
            if (origin == null || direction == null) {
                null
            } else {
                Ray(
                    origin = origin,
                    direction = direction.normalized(),
                    weight = weighted.finalWeight,
                )
            }
        }

        if (rays.size < 2) {
            return RayGeometryScore(
                available = false,
                consistency = 0.50,
                meanDistanceMeters = Double.POSITIVE_INFINITY,
            )
        }

        var weightedDistance = 0.0
        var totalWeight = 0.0
        var pairCount = 0

        for (i in 0 until rays.lastIndex) {
            for (j in i + 1 until rays.size) {
                val first = rays[i]
                val second = rays[j]
                val distance = symmetricRayDistance(
                    first.origin,
                    first.direction,
                    second.origin,
                    second.direction,
                )
                if (!distance.isFinite()) continue

                val angle = Math.toDegrees(
                    kotlin.math.acos(
                        first.direction.dot(second.direction)
                            .coerceIn(-1.0, 1.0),
                    ),
                )
                if (!angle.isFinite() || angle < config.minimumRayAngleDegrees) {
                    continue
                }

                val weight =
                    (first.weight * second.weight)
                        .coerceAtLeast(1e-6)

                weightedDistance += distance * weight
                totalWeight += weight
                pairCount++
            }
        }

        if (pairCount == 0 || totalWeight <= 1e-9) {
            return RayGeometryScore(
                available = true,
                consistency = 0.0,
                meanDistanceMeters = Double.POSITIVE_INFINITY,
            )
        }

        val meanDistance = weightedDistance / totalWeight
        val consistency = (
            1.0 -
                meanDistance /
                config.maximumSymmetricRayDistanceMeters
        ).coerceIn(0.0, 1.0)

        return RayGeometryScore(
            available = true,
            consistency = consistency,
            meanDistanceMeters = meanDistance,
        )
    }

    private fun symmetricRayDistance(
        originA: Vec3,
        directionA: Vec3,
        originB: Vec3,
        directionB: Vec3,
    ): Double {
        val a = directionA.normalized()
        val b = directionB.normalized()
        val w0 = originA - originB
        val ab = a.dot(b)
        val denominator = 1.0 - ab * ab

        if (!denominator.isFinite() || denominator < 1e-8) {
            val cross = w0.cross(a)
            return cross.norm()
        }

        val d = a.dot(w0)
        val e = b.dot(w0)
        val s = (ab * e - d) / denominator
        val t = (e - ab * d) / denominator
        val closestA = originA + a * s
        val closestB = originB + b * t
        return closestA.distanceTo(closestB)
    }

    private fun robustKernel(distancePx: Double): Double {
        val u = distancePx / max(config.robustScalePixels, 1e-6)
        if (!u.isFinite()) return 0.0
        if (u <= 1.0) {
            val t = 1.0 - u * u
            return t * t
        }
        return 1.0 / (1.0 + u * u)
    }

    private fun robustScatterScore(scatter: Double): Double {
        if (!scatter.isFinite()) return 0.0
        return exp(
            -scatter / max(config.robustScalePixels, 1e-6),
        ).coerceIn(0.0, 1.0)
    }

    private fun effectiveViews(values: List<WeightedObservation>): Double {
        if (values.isEmpty()) return 0.0
        val perCamera = values
            .groupBy { it.observation.deviceId }
            .mapValues { (_, list) -> list.sumOf { it.finalWeight } }
            .values
        if (perCamera.isEmpty()) return 0.0
        return perCamera.sum() / max(perCamera.maxOrNull() ?: 1.0, 1e-9)
    }

    private fun maskAgreement(values: List<WeightedObservation>): Double {
        val labels = values.mapNotNull { it.observation.semanticMaskId }
        if (labels.isEmpty()) return 0.50
        val counts = labels.groupingBy { it }.eachCount()
        return (counts.values.maxOrNull()?.toDouble() ?: 0.0) / labels.size
    }

    private fun priority(observation: Observation): Double = (
        observation.detectorConfidence * 0.34 +
            observation.visibilityConfidence * 0.24 +
            observation.identityConfidence * 0.16 +
            (1.0 - observation.occlusionRisk) * 0.14 +
            (1.0 - (
                observation.pixelUncertainty /
                    config.maximumPixelUncertainty
                ).coerceIn(0.0, 1.0)) * 0.12
        ).coerceIn(0.0, 1.0)

    private fun multiply(vararg values: Double): Double {
        if (values.isEmpty()) return 0.0
        var result = 1.0
        values.forEach { result *= it.coerceIn(0.0, 1.0) }
        return result.coerceIn(0.0, 1.0)
    }

    private fun weightedAverage(
        values: List<WeightedObservation>,
        selector: (WeightedObservation) -> Double,
    ): Double {
        if (values.isEmpty()) return 0.0
        var numerator = 0.0
        var denominator = 0.0
        values.forEach {
            numerator += selector(it).coerceIn(0.0, 1.0) * it.finalWeight
            denominator += it.finalWeight
        }
        return if (denominator <= 1e-12) 0.0 else (numerator / denominator).coerceIn(0.0, 1.0)
    }

    private fun List<Double>.averageOrZero() = if (isEmpty()) 0.0 else average()

private fun finite(value: Vec3): Boolean =
    value.x.isFinite() && value.y.isFinite() && value.z.isFinite()
}

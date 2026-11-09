package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Vec3
import kotlin.math.max
import kotlin.math.min

/**
 * Practical camera-placement advisor for arbitrary multi-camera layouts.
 *
 * This layer does not claim that any camera position is universally optimal. It scores candidate
 * locations using baseline, angular diversity, sector coverage, subject visibility, calibration,
 * synchronization and information gain through CaptureRigObservabilityEngine.
 */
class FrontierRigPlacementAdvisor(
    private val observability: CaptureRigObservabilityEngine = CaptureRigObservabilityEngine(),
    private val maximumCandidates: Int = 256,
) {
    data class SceneTarget(
        val actorId: Int,
        val positionMeters: Vec3,
        val importance: Double = 1.0,
    ) {
        init {
            require(actorId >= 0)
            require(positionMeters.x.isFinite() && positionMeters.y.isFinite() && positionMeters.z.isFinite())
            require(importance.isFinite() && importance in 0.0..1.0)
        }
    }

    data class Candidate(
        val cameraId: String,
        val positionMeters: Vec3,
        val forwardVector: Vec3,
        val calibrationConfidence: Double = 1.0,
        val synchronizationConfidence: Double = 1.0,
        val transportHeadroom: Double = 1.0,
        val thermalHeadroom: Double = 1.0,
        /** Optional optical motion-blur risk for the proposed capture node. */
        val motionBlurRisk: Double? = null,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(positionMeters.x.isFinite() && positionMeters.y.isFinite() && positionMeters.z.isFinite())
            require(forwardVector.x.isFinite() && forwardVector.y.isFinite() && forwardVector.z.isFinite())
            require(forwardVector.norm() > 1e-9)
            listOf(calibrationConfidence, synchronizationConfidence, transportHeadroom, thermalHeadroom)
                .forEach { require(it in 0.0..1.0) }
            require(
                motionBlurRisk == null ||
                    (
                        motionBlurRisk.isFinite() &&
                            motionBlurRisk in 0.0..1.0
                    ),
            )
        }
    }

    data class Recommendation(
        val selected: List<Candidate>,
        val rejected: List<Candidate>,
        val objectiveScore: Double,
        val observabilityScore: Double,
        val meanSubjectCoverage: Double,
        val meanBaselineToTargetMeters: Double,
        val sectorCount: Int,
        val warnings: List<String>,
    )

    fun recommend(
        targets: List<SceneTarget>,
        candidates: List<Candidate>,
        requestedCount: Int,
    ): Recommendation {
        require(requestedCount >= 1)
        if (targets.isEmpty() || candidates.isEmpty()) {
            return Recommendation(
                selected = emptyList(),
                rejected = candidates,
                objectiveScore = 0.0,
                observabilityScore = 0.0,
                meanSubjectCoverage = 0.0,
                meanBaselineToTargetMeters = 0.0,
                sectorCount = 0,
                warnings = listOf("placement-target-or-candidate-set-empty"),
            )
        }

        val bounded = candidates.distinctBy { it.cameraId }.take(maximumCandidates)
        val targetCenter = weightedTargetCenter(targets)
        val evidence = bounded.map { candidate ->
            val visibility = subjectVisibility(candidate, targets)
            val sector = sectorFor(candidate.positionMeters, targetCenter)
            CaptureRigObservabilityEngine.CameraEvidence(
                cameraId = candidate.cameraId,
                role = DeviceRole.CAMERA,
                positionMeters = candidate.positionMeters,
                forwardVector = candidate.forwardVector.normalized(),
                calibrationConfidence = candidate.calibrationConfidence,
                synchronizationConfidence = candidate.synchronizationConfidence,
                trackingConfidence = visibility,
                thermalHeadroom = candidate.thermalHeadroom,
                transportHeadroom = candidate.transportHeadroom,
                active = true,
                sector = sector,
                trackedObjectCoverage = visibility,
                surfaceInformationGain = informationGain(candidate, targets),
                occlusionCoverage = visibility,
                dynamicParallaxScore = angularTargetSpreadScore(candidate, targets),
                motionBlurRisk = candidate.motionBlurRisk,
            )
        }

        val selection = observability.select(
            cameras = evidence,
            requestedCount = min(requestedCount, bounded.size),
        )
        val selectedIds = selection.selected.map { it.cameraId }.toSet()
        val selected = bounded.filter { it.cameraId in selectedIds }
        val targetCoverage = selected.map { subjectVisibility(it, targets) }.averageOrZero()
        val baseline = selected.map { it.positionMeters.distanceTo(targetCenter) }.averageOrZero()
        val sectors = selection.selected.mapNotNull { it.sector }.toSet()
        val objective = (
            selection.objectiveScore * 0.65 +
                targetCoverage * 0.20 +
                baselineQuality(baseline) * 0.15
            ).coerceIn(0.0, 1.0)

        return Recommendation(
            selected = selected,
            rejected = bounded.filterNot { it.cameraId in selectedIds },
            objectiveScore = objective,
            observabilityScore = observability.analyze(evidence).observabilityScore,
            meanSubjectCoverage = targetCoverage,
            meanBaselineToTargetMeters = baseline,
            sectorCount = sectors.size,
            warnings = buildList {
                addAll(selection.warnings)
                if (targetCoverage < 0.65) add("placement-subject-coverage-low")
                if (baseline < 0.30) add("placement-baseline-short")
                if (sectors.size < min(4, requestedCount)) add("placement-sector-diversity-low")
            }.distinct(),
        )
    }

    private fun weightedTargetCenter(targets: List<SceneTarget>): Vec3 {
        var weighted = Vec3(0.0, 0.0, 0.0)
        var weight = 0.0
        targets.forEach {
            weighted += it.positionMeters * it.importance
            weight += it.importance
        }
        return if (weight <= 1e-9) targets.first().positionMeters else weighted / weight
    }

    private fun subjectVisibility(candidate: Candidate, targets: List<SceneTarget>): Double {
        val forward = candidate.forwardVector.normalized()
        val totalWeight = targets.sumOf { it.importance }.coerceAtLeast(1e-9)
        return (
            targets.sumOf { target ->
                val direction = (target.positionMeters - candidate.positionMeters).normalized()
                val alignment = forward.dot(direction).coerceIn(-1.0, 1.0)
                target.importance * ((alignment + 1.0) * 0.5).coerceIn(0.0, 1.0)
            } / totalWeight
            ).coerceIn(0.0, 1.0)
    }

    private fun informationGain(candidate: Candidate, targets: List<SceneTarget>): Double {
        val distanceScore = targets.map {
            1.0 / (1.0 + candidate.positionMeters.distanceTo(it.positionMeters))
        }.averageOrZero()
        return (
            distanceScore * 0.45 +
                candidate.calibrationConfidence * 0.30 +
                candidate.synchronizationConfidence * 0.25
            ).coerceIn(0.0, 1.0)
    }

    private fun angularTargetSpreadScore(
        candidate: Candidate,
        targets: List<SceneTarget>,
    ): Double {
        if (targets.size < 2) return 0.50
        val directions = targets.map {
            (it.positionMeters - candidate.positionMeters).normalized()
        }
        var total = 0.0
        var count = 0
        for (i in directions.indices) {
            for (j in i + 1 until directions.size) {
                val angle = Math.toDegrees(
                    kotlin.math.acos(
                        directions[i].dot(directions[j]).coerceIn(-1.0, 1.0)
                    )
                )
                total += (angle / 90.0).coerceIn(0.0, 1.0)
                count++
            }
        }
        return if (count == 0) 0.50 else (total / count.toDouble()).coerceIn(0.0,1.0)
    }

    private fun sectorFor(
        position: Vec3,
        targetCenter: Vec3,
    ): CaptureRigObservabilityEngine.Sector {
        val delta = position - targetCenter
        val angle = Math.toDegrees(kotlin.math.atan2(delta.x, delta.z))
        return when {
            angle >= -22.5 && angle < 22.5 -> CaptureRigObservabilityEngine.Sector.FRONT
            angle >= 22.5 && angle < 67.5 -> CaptureRigObservabilityEngine.Sector.FRONT_RIGHT
            angle >= 67.5 && angle < 112.5 -> CaptureRigObservabilityEngine.Sector.RIGHT
            angle >= 112.5 && angle < 157.5 -> CaptureRigObservabilityEngine.Sector.REAR_RIGHT
            angle >= 157.5 || angle < -157.5 -> CaptureRigObservabilityEngine.Sector.REAR
            angle >= -157.5 && angle < -112.5 -> CaptureRigObservabilityEngine.Sector.REAR_LEFT
            angle >= -112.5 && angle < -67.5 -> CaptureRigObservabilityEngine.Sector.LEFT
            else -> CaptureRigObservabilityEngine.Sector.FRONT_LEFT
        }
    }
}

private fun Iterable<Double>.averageOrZero(): Double {
    val values = toList()
    return if (values.isEmpty()) 0.0 else values.average()
}

private fun baselineQuality(distanceMeters: Double): Double =
    (1.0 - kotlin.math.abs(distanceMeters - 1.5) / max(1.5, distanceMeters)).coerceIn(0.0, 1.0)

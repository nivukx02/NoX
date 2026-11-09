package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.ObservationProvenance
import com.nivukx.nox.core.model.UncertaintyEnvelope
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.model.WorldSpaceProposal
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** Geometry-first world-space proposal engine with a learned-proposal injection point.
 * Learned proposals are hypotheses only; calibrated ray geometry and uncertainty remain authoritative.
 */
class WorldSpaceMotionProposalEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val maxObservations: Int = 48,
        val minimumGeometricConfidence: Double = 0.12,
        val maximumRayPairDistanceMeters: Double = 2.0,
        val maximumLearnedDisplacementMeters: Double = 0.75,
        val learnedContributionWeight: Double = 0.18,
        val minimumViewDiversity: Double = 0.08,
        val robustIterations: Int = 4,
        val huberResidualMeters: Double = 0.015,
        val maximumConditionEstimate: Double = 1e10,
    ) {
        init {
            require(maxObservations >= 2)
            require(minimumGeometricConfidence in 0.0..1.0)
            require(maximumRayPairDistanceMeters > 0.0)
            require(maximumLearnedDisplacementMeters > 0.0)
            require(learnedContributionWeight in 0.0..1.0)
            require(minimumViewDiversity in 0.0..1.0)
            require(robustIterations in 1..8)
            require(huberResidualMeters > 0.0)
            require(maximumConditionEstimate > 1.0)
        }
    }

    data class RayObservation(
        val actorId: Int,
        val pointId: Int,
        val cameraId: String,
        val timestampNanos: Long,
        val rayOriginMeters: Vec3,
        val rayDirection: Vec3,
        val pixelConfidence: Double,
        val timingConfidence: Double,
        val geometricUncertaintyMeters: Double,
        val viewWeight: Double = 1.0,
        val calibrationFingerprint: String = "unknown",
        val timingFingerprint: String = "unknown",
        val modelFingerprint: String = "unknown",
    ) {
        init {
            require(actorId >= 0)
            require(pointId >= 0)
            require(cameraId.isNotBlank())
            require(timestampNanos >= 0L)
            require(rayOriginMeters.finite())
            require(rayDirection.finite())
            require(rayDirection.norm() > 1e-9)
            require(pixelConfidence in 0.0..1.0)
            require(timingConfidence in 0.0..1.0)
            require(geometricUncertaintyMeters.isFinite() && geometricUncertaintyMeters > 0.0)
            require(viewWeight.isFinite() && viewWeight >= 0.0)
        }

        val direction: Vec3
            get() = rayDirection.normalized()

        val weight: Double
            get() = max(1e-9, pixelConfidence * timingConfidence * viewWeight / geometricUncertaintyMeters)
    }

    data class LearnedProposal(
        val positionMeters: Vec3,
        val confidence: Double,
        val uncertaintyMeters: Double,
    ) {
        init {
            require(positionMeters.finite())
            require(confidence in 0.0..1.0)
            require(uncertaintyMeters.isFinite() && uncertaintyMeters > 0.0)
        }
    }

    data class Result(
        val proposal: WorldSpaceProposal?,
        val usedObservationCount: Int,
        val rejectedObservationCount: Int,
        val viewDiversity: Double,
        val conditionEstimate: Double,
        val residualMeters: Double,
        val warnings: List<String>,
    )

    @Synchronized
    fun propose(
        timestampNanos: Long,
        observations: List<RayObservation>,
        learnedProposal: LearnedProposal? = null,
    ): Result {
        require(timestampNanos >= 0L)
        if (observations.isEmpty()) {
            return Result(null, 0, 0, 0.0, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, listOf("no-world-space-observations"))
        }

        val valid = observations.asSequence()
            .filter { it.timestampNanos == timestampNanos }
            .filter { it.pixelConfidence >= config.minimumGeometricConfidence }
            .filter { it.rayDirection.norm() > 1e-9 }
            .distinctBy { it.cameraId }
            .take(config.maxObservations)
            .toList()
        val rejected = observations.size - valid.size
        if (valid.size < 2) {
            return Result(null, valid.size, rejected, viewDiversity(valid), Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, listOf("insufficient-independent-camera-evidence"))
        }

        val geometry = solveRobustLeastSquares(valid)
        if (geometry == null) {
            return Result(null, valid.size, rejected, viewDiversity(valid), Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, listOf("ill-conditioned-world-space-ray-system"))
        }

        val basePosition = geometry.first
        val effectiveViewDiversity = viewDiversity(valid)
        val refinedPosition = fuseLearnedPrior(basePosition, valid, learnedProposal)
        val reprojectionResidual = meanRayDistance(refinedPosition, valid)
        val sigma = max(
            1e-6,
            geometry.second / sqrt(valid.sumOf { it.weight }).coerceAtLeast(1.0),
        )
        val learnedSupport = learnedProposal?.let {
            (
                it.confidence *
                    config.learnedContributionWeight *
                    (
                        1.0 /
                            (1.0 + it.uncertaintyMeters / 0.010)
                    ).coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)
        } ?: 0.0
        val geometricSupport =
            (1.0 / (1.0 + reprojectionResidual / max(config.maximumRayPairDistanceMeters * 0.1, 1e-4)))
                .coerceIn(0.0, 1.0)
        val timingSupport = valid.map { it.timingConfidence }.average().coerceIn(0.0, 1.0)
        val confidence =
            (geometricSupport * 0.50 + timingSupport * 0.25 + effectiveViewDiversity * 0.20 + learnedSupport * 0.05)
                .coerceIn(0.0, 1.0)
        val uncertainty = sigma +
            valid.map { it.geometricUncertaintyMeters }.average() * 0.25
        val origin = when {
            learnedProposal == null -> EvidenceOrigin.GEOMETRICALLY_CONSTRAINED
            learnedSupport > 0.08 -> EvidenceOrigin.LEARNED_PRIOR
            else -> EvidenceOrigin.GEOMETRICALLY_CONSTRAINED
        }
        val first = valid.first()
        val provenance = valid.map { obs ->
            ObservationProvenance(
                cameraId = obs.cameraId,
                calibrationFingerprint = obs.calibrationFingerprint,
                timingFingerprint = obs.timingFingerprint,
                modelFingerprint = obs.modelFingerprint,
                origin = EvidenceOrigin.OBSERVED,
                captureTimestampNanos = obs.timestampNanos,
                sequenceNumber = obs.timestampNanos,
                freshnessNanos = 0L,
            )
        }
        val proposal = WorldSpaceProposal(
            actorId = first.actorId,
            pointId = first.pointId,
            timestampNanos = timestampNanos,
            positionMeters = refinedPosition,
            uncertainty = UncertaintyEnvelope(
                xxMeters2 = uncertainty * uncertainty,
                yyMeters2 = uncertainty * uncertainty,
                zzMeters2 = uncertainty * uncertainty,
                confidence = confidence,
            ),
            confidence = confidence,
            geometricSupport = geometricSupport,
            temporalSupport = timingSupport,
            learnedSupport = learnedSupport,
            independentCameraCount = valid.size,
            viewDiversity = effectiveViewDiversity,
            origin = origin,
            provenance = provenance,
        )
        val warnings = buildList {
            if (effectiveViewDiversity < config.minimumViewDiversity) add("weak-camera-view-diversity")
            if (reprojectionResidual > config.maximumRayPairDistanceMeters * 0.5) add("large-ray-consensus-residual")
            if (learnedProposal != null && learnedDistance(basePosition, learnedProposal.positionMeters) > config.maximumLearnedDisplacementMeters) add("learned-proposal-displacement-clamped")
        }
        return Result(proposal, valid.size, rejected, effectiveViewDiversity, geometry.third, reprojectionResidual, warnings)
    }

    private fun solveRobustLeastSquares(
        observations: List<RayObservation>,
    ): Triple<Vec3, Double, Double>? {
        var multipliers =
            DoubleArray(observations.size) { 1.0 }

        var solution =
            solveLeastSquares(
                observations = observations,
                robustMultipliers = multipliers,
            )
                ?: return null

        repeat(config.robustIterations) {
            val residuals =
                observations.mapIndexed { index, observation ->
                    val residual =
                        sqrt(
                            squaredDistanceToRay(
                                solution.first,
                                observation,
                            )
                                .coerceAtLeast(0.0),
                        )
                    residual to index
                }

            val median =
                residuals
                    .map { it.first }
                    .sorted()
                    .let { values ->
                        if (values.isEmpty()) 0.0
                        else values[values.size / 2]
                    }

            val delta =
                max(
                    config.huberResidualMeters,
                    median * 1.4826,
                )

            multipliers =
                DoubleArray(observations.size) { index ->
                    val residual =
                        residuals[index].first
                    if (!residual.isFinite()) {
                        0.02
                    } else if (residual <= delta) {
                        1.0
                    } else {
                        (delta / residual)
                            .coerceIn(0.02, 1.0)
                    }
                }

            solution =
                solveLeastSquares(
                    observations = observations,
                    robustMultipliers = multipliers,
                )
                    ?: return@repeat
        }

        if (
            !solution.third.isFinite() ||
            solution.third > config.maximumConditionEstimate
        ) {
            return null
        }

        return solution
    }

    private fun solveLeastSquares(
        observations: List<RayObservation>,
        robustMultipliers: DoubleArray =
            DoubleArray(observations.size) { 1.0 },
    ): Triple<Vec3, Double, Double>? {
        val a = DoubleArray(9)
        val b = DoubleArray(3)
        for ((index, obs) in observations.withIndex()) {
            val d = obs.direction
            val multiplier =
                robustMultipliers.getOrElse(index) { 1.0 }
                    .coerceIn(0.0, 1.0)
            val w = obs.weight * multiplier
            val px = 1.0 - d.x * d.x
            val py = 1.0 - d.y * d.y
            val pz = 1.0 - d.z * d.z
            val pxy = -d.x * d.y
            val pxz = -d.x * d.z
            val pyz = -d.y * d.z
            val o = obs.rayOriginMeters
            a[0] += w * px; a[1] += w * pxy; a[2] += w * pxz
            a[3] += w * pxy; a[4] += w * py; a[5] += w * pyz
            a[6] += w * pxz; a[7] += w * pyz; a[8] += w * pz
            b[0] += w * (px * o.x + pxy * o.y + pxz * o.z)
            b[1] += w * (pxy * o.x + py * o.y + pyz * o.z)
            b[2] += w * (pxz * o.x + pyz * o.y + pz * o.z)
        }
        val matrix = Array(3) { row -> DoubleArray(4) { col -> if (col < 3) a[row * 3 + col] else b[row] } }
        val pivotScale = matrix.flatMap { it.take(3).asList() }.maxOfOrNull { abs(it) } ?: return null
        if (pivotScale < 1e-12) return null
        if (robustMultipliers.size != observations.size) return null
        var condition = 1.0
        for (column in 0 until 3) {
            var pivot = column
            for (row in column + 1 until 3) if (abs(matrix[row][column]) > abs(matrix[pivot][column])) pivot = row
            val pivotValue = matrix[pivot][column]
            if (abs(pivotValue) < pivotScale * 1e-10) return null
            condition = max(condition, pivotScale / abs(pivotValue))
            if (pivot != column) { val temp = matrix[pivot]; matrix[pivot] = matrix[column]; matrix[column] = temp }
            val divisor = matrix[column][column]
            for (k in column until 4) matrix[column][k] /= divisor
            for (row in 0 until 3) {
                if (row == column) continue
                val factor = matrix[row][column]
                if (abs(factor) < 1e-14) continue
                for (k in column until 4) matrix[row][k] -= factor * matrix[column][k]
            }
        }
        val point = Vec3(matrix[0][3], matrix[1][3], matrix[2][3])
        if (!point.finite()) return null
        val residual = sqrt(observations.sumOf { it.weight * squaredDistanceToRay(point, it) } / observations.sumOf { it.weight }.coerceAtLeast(1e-9))
        return Triple(point, residual, condition)
    }

    private fun fuseLearnedPrior(base: Vec3, observations: List<RayObservation>, learned: LearnedProposal?): Vec3 {
        if (learned == null || learned.confidence <= 0.0) return base
        val distance = learnedDistance(base, learned.positionMeters)
        val cap = min(distance, config.maximumLearnedDisplacementMeters)
        if (distance < 1e-9) return base
        val direction = (learned.positionMeters - base) / distance
        val alpha = (learned.confidence * config.learnedContributionWeight *
            observations.map { it.timingConfidence }.average()).coerceIn(0.0, 1.0)
        return base + direction * (cap * alpha)
    }

    private fun meanRayDistance(point: Vec3, observations: List<RayObservation>): Double =
        sqrt(observations.map { it.weight * squaredDistanceToRay(point, it) }.sum() / observations.sumOf { it.weight }.coerceAtLeast(1e-9))

    private fun squaredDistanceToRay(point: Vec3, observation: RayObservation): Double {
        val delta = point - observation.rayOriginMeters
        val projection = delta.dot(observation.direction)
        val residual = delta - observation.direction * projection
        return residual.dot(residual)
    }

    private fun learnedDistance(a: Vec3, b: Vec3): Double = a.distanceTo(b)

    private fun viewDiversity(observations: List<RayObservation>): Double {
        if (observations.size < 2) return 0.0
        var total = 0.0
        var count = 0
        for (i in observations.indices) for (j in i + 1 until observations.size) {
            val dot = abs(observations[i].direction.dot(observations[j].direction)).coerceIn(0.0, 1.0)
            total += 1.0 - dot
            count++
        }
        return (total / max(count, 1)).coerceIn(0.0, 1.0)
    }
}

private fun Vec3.finite(): Boolean = x.isFinite() && y.isFinite() && z.isFinite()
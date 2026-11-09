package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.model.WorldSpaceProposal
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Geometry-first sparse-view human reconstruction with explicit evidence provenance.
 *
 * This engine does not treat a learned completion as equivalent to observed geometry. It performs
 * deterministic robust fusion across independent view hypotheses, optional temporal support, surface
 * quality cues and a bounded learned prior. Every output retains the evidence state used to create it.
 */
class SparseViewHumanReconstructionEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val minimumViewsForGeometry: Int = 2,
        val preferredViews: Int = 4,
        val maximumRegions: Int = 16_384,
        val maximumViewEvidencePerRegion: Int = 32,
        val learnedPriorWeight: Double = 0.18,
        val maximumPriorDisplacementMeters: Double = 0.25,
        val minimumObservedRatio: Double = 0.55,
        val minimumGeometryScore: Double = 0.45,
        val preferredGeometryScore: Double = 0.72,
        val maximumAcceptedReprojectionPixels: Double = 8.0,
        val preferredReprojectionPixels: Double = 2.0,
        val maximumUncertaintyMeters: Double = 0.15,
        val minimumSurfaceConsistency: Double = 0.30,
        val minimumTemporalConfidence: Double = 0.20,
        val robustScaleMeters: Double = 0.035,
        val minimumEvidenceConfidence: Double = 0.10,
    ) {
        init {
            require(minimumViewsForGeometry >= 2)
            require(preferredViews >= minimumViewsForGeometry)
            require(maximumRegions > 0)
            require(maximumViewEvidencePerRegion > 0)
            require(learnedPriorWeight in 0.0..1.0)
            require(maximumPriorDisplacementMeters > 0.0)
            require(minimumObservedRatio in 0.0..1.0)
            require(minimumGeometryScore in 0.0..1.0)
            require(preferredGeometryScore >= minimumGeometryScore)
            require(maximumAcceptedReprojectionPixels > 0.0)
            require(
                preferredReprojectionPixels > 0.0 &&
                    preferredReprojectionPixels <=
                    maximumAcceptedReprojectionPixels
            )
            require(maximumUncertaintyMeters > 0.0)
            require(minimumSurfaceConsistency in 0.0..1.0)
            require(minimumTemporalConfidence in 0.0..1.0)
            require(robustScaleMeters > 0.0)
            require(minimumEvidenceConfidence in 0.0..1.0)
        }
    }

    data class ViewEvidence(
        val cameraId: String,
        val positionMeters: Vec3,
        val confidence: Double = 1.0,
        val reprojectionResidualPixels: Double = 0.0,
        val visibility: Double = 1.0,
        val photometricConsistency: Double = 1.0,
        val normalConsistency: Double = 1.0,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(finite(positionMeters))
            require(confidence.isFinite() && confidence in 0.0..1.0)
            require(
                reprojectionResidualPixels.isFinite() &&
                    reprojectionResidualPixels >= 0.0
            )
            require(visibility.isFinite() && visibility in 0.0..1.0)
            require(
                photometricConsistency.isFinite() &&
                    photometricConsistency in 0.0..1.0
            )
            require(
                normalConsistency.isFinite() &&
                    normalConsistency in 0.0..1.0
            )
        }
    }

    data class RegionEvidence(
        val actorId: Int,
        val regionId: Int,
        val timestampNanos: Long,
        val worldPositionMeters: Vec3,
        val independentViews: Int,
        val viewDiversity: Double,
        val reprojectionResidualPixels: Double,
        val observedRatio: Double,
        val learnedPrior: Vec3? = null,
        val viewEvidence: List<ViewEvidence> = emptyList(),
        val temporalPositionMeters: Vec3? = null,
        val temporalConfidence: Double = 0.0,
        val uncertaintyMeters: Double = 0.0,
        val surfaceConsistency: Double = 1.0,
    ) {
        init {
            require(actorId >= 0 && regionId >= 0)
            require(timestampNanos >= 0L)
            require(finite(worldPositionMeters))
            require(independentViews >= 0)
            require(viewDiversity in 0.0..1.0)
            require(
                reprojectionResidualPixels.isFinite() &&
                    reprojectionResidualPixels >= 0.0
            )
            require(observedRatio in 0.0..1.0)
            learnedPrior?.let { require(finite(it)) }
            require(viewEvidence.size <= 64)
            temporalPositionMeters?.let { require(finite(it)) }
            require(
                temporalConfidence.isFinite() &&
                    temporalConfidence in 0.0..1.0
            )
            require(
                uncertaintyMeters.isFinite() &&
                    uncertaintyMeters >= 0.0
            )
            require(surfaceConsistency.isFinite() && surfaceConsistency in 0.0..1.0)
        }
    }

    enum class State {
        OBSERVED,
        CONSTRAINED,
        PROPAGATED,
        LEARNED,
        INSUFFICIENT_EVIDENCE,
    }

    data class RegionResult(
        val actorId: Int,
        val regionId: Int,
        val positionMeters: Vec3,
        val state: State,
        val confidence: Double,
        val measuredRatio: Double,
        val inferredRatio: Double,
        val warnings: List<String>,
        val independentViews: Int = 0,
        val viewDiversity: Double = 0.0,
        val reprojectionResidualPixels: Double = Double.POSITIVE_INFINITY,
        val uncertaintyMeters: Double = Double.POSITIVE_INFINITY,
        val surfaceConsistency: Double = 0.0,
        val evidenceOrigin: EvidenceOrigin = EvidenceOrigin.LEARNED_PRIOR,
        val observedViewCount: Int = 0,
        val temporalContribution: Double = 0.0,
        val learnedContribution: Double = 0.0,
    )

    data class FusionDiagnostics(
        val inputRegionCount: Int,
        val outputRegionCount: Int,
        val duplicateRegionCount: Int,
        val rejectedRegionCount: Int,
        val learnedRegionCount: Int,
        val observedRegionCount: Int,
        val constrainedRegionCount: Int,
        val propagatedRegionCount: Int,
        val meanConfidence: Double,
        val meanUncertaintyMeters: Double,
    )

    data class ReconstructionBatch(
        val regions: List<RegionResult>,
        val diagnostics: FusionDiagnostics,
    )

    @Volatile
    private var closed = false

    fun reconstruct(
        timestampNanos: Long,
        regions: List<RegionEvidence>,
    ): List<RegionResult> {
        return reconstructDetailed(timestampNanos, regions).regions
    }

    fun reconstructDetailed(
        timestampNanos: Long,
        regions: List<RegionEvidence>,
    ): ReconstructionBatch {
        check(!closed) { "SparseViewHumanReconstructionEngine is closed" }
        require(timestampNanos >= 0L)

        val candidates = regions
            .asSequence()
            .filter { it.timestampNanos == timestampNanos }
            .filter { finite(it.worldPositionMeters) }
            .sortedWith(
                compareByDescending<RegionEvidence> { it.observedRatio }
                    .thenByDescending { it.independentViews }
                    .thenBy { it.actorId }
                    .thenBy { it.regionId }
            )
            .take(config.maximumRegions)
            .toList()

        val duplicateGroups = candidates
            .groupBy { it.actorId to it.regionId }
        val duplicateRegionCount =
            (candidates.size - duplicateGroups.size).coerceAtLeast(0)

        val output = duplicateGroups
            .entries
            .sortedWith(
                compareBy<Pair<Pair<Int, Int>, List<RegionEvidence>>> { it.key.first }
                    .thenBy { it.key.second }
            )
            .mapNotNull { (_, group) ->
                fuseRegion(timestampNanos, group)
            }

        val meanConfidence =
            output.takeIf { it.isNotEmpty() }?.map { it.confidence }?.average()
                ?: 0.0
        val finiteUncertainties = output
            .map { it.uncertaintyMeters }
            .filter { it.isFinite() }

        return ReconstructionBatch(
            regions = output,
            diagnostics = FusionDiagnostics(
                inputRegionCount = regions.count { it.timestampNanos == timestampNanos },
                outputRegionCount = output.size,
                duplicateRegionCount = duplicateRegionCount,
                rejectedRegionCount =
                    (regions.count { it.timestampNanos == timestampNanos } - output.size)
                        .coerceAtLeast(0),
                learnedRegionCount = output.count { it.state == State.LEARNED },
                observedRegionCount = output.count { it.state == State.OBSERVED },
                constrainedRegionCount = output.count { it.state == State.CONSTRAINED },
                propagatedRegionCount = output.count { it.state == State.PROPAGATED },
                meanConfidence = meanConfidence,
                meanUncertaintyMeters =
                    finiteUncertainties.takeIf { it.isNotEmpty() }?.average()
                        ?: Double.POSITIVE_INFINITY,
            ),
        )
    }

    fun fromProposal(
        proposal: WorldSpaceProposal,
        regionId: Int,
        viewCount: Int,
        reprojectionPixels: Double,
        observedRatio: Double,
    ): RegionResult =
        reconstruct(
            proposal.timestampNanos,
            listOf(
                RegionEvidence(
                    actorId = proposal.actorId,
                    regionId = regionId,
                    timestampNanos = proposal.timestampNanos,
                    worldPositionMeters = proposal.positionMeters,
                    independentViews = viewCount,
                    viewDiversity = proposal.viewDiversity,
                    reprojectionResidualPixels = reprojectionPixels,
                    observedRatio = observedRatio,
                ),
            ),
        ).first()

    fun close() {
        closed = true
    }

    private fun fuseRegion(
        timestampNanos: Long,
        evidenceGroup: List<RegionEvidence>,
    ): RegionResult? {
        if (evidenceGroup.isEmpty()) return null

        val first = evidenceGroup.first()
        val warnings = LinkedHashSet<String>()

        val normalizedViews = evidenceGroup
            .flatMap { it.viewEvidence }
            .filter { it.confidence >= config.minimumEvidenceConfidence }
            .sortedWith(
                compareBy<ViewEvidence> { it.cameraId }
                    .thenByDescending { it.confidence }
            )
            .distinctBy { it.cameraId }
            .take(config.maximumViewEvidencePerRegion)

        val viewPositions = if (normalizedViews.isNotEmpty()) {
            normalizedViews.map { it.positionMeters }
        } else {
            evidenceGroup.map { it.worldPositionMeters }
        }

        val viewCount = max(
            first.independentViews,
            normalizedViews.size,
        )
        val diversity = evidenceGroup
            .map { it.viewDiversity }
            .plus(normalizedViews.map { viewDiversityFromEvidence(it) })
            .maxOrNull()
            ?.coerceIn(0.0, 1.0)
            ?: 0.0

        val robustObservedPosition = robustCentroid(viewPositions)
        val measuredResidual = medianFinite(
            evidenceGroup.map {
                it.reprojectionResidualPixels
            },
        )

        val observedRatio = evidenceGroup
            .map { it.observedRatio }
            .weightedMean(
                evidenceGroup.map {
                    (
                        0.5 +
                            0.25 * it.viewDiversity +
                            0.25 * it.surfaceConsistency
                        ).coerceAtLeast(0.01)
                },
            )
            .coerceIn(0.0, 1.0)

        val surfaceConsistency = evidenceGroup
            .map { it.surfaceConsistency }
            .average()
            .coerceIn(0.0, 1.0)

        val uncertainty = estimateUncertainty(
            evidenceGroup = evidenceGroup,
            positions = viewPositions,
            viewDiversity = diversity,
        )

        var position = robustObservedPosition
        val geometryScore = geometryScore(
            independentViews = viewCount,
            viewDiversity = diversity,
            reprojectionResidualPixels = measuredResidual,
            uncertaintyMeters = uncertainty,
            surfaceConsistency = surfaceConsistency,
        )

        var learnedContribution = 0.0
        val learnedPrior = evidenceGroup
            .mapNotNull { it.learnedPrior }
            .takeIf { it.isNotEmpty() }
            ?.let(::robustCentroid)

        val hasSufficientGeometry =
            viewCount >= config.minimumViewsForGeometry &&
                geometryScore >= config.minimumGeometryScore &&
                uncertainty <= config.maximumUncertaintyMeters &&
                measuredResidual <= config.maximumAcceptedReprojectionPixels &&
                surfaceConsistency >= config.minimumSurfaceConsistency

        if (learnedPrior != null && hasSufficientGeometry) {
            val delta = learnedPrior - position
            val distance = delta.norm()
            if (distance.isFinite() && distance > 1e-9) {
                val bounded = min(
                    distance,
                    config.maximumPriorDisplacementMeters,
                )
                val trust =
                    config.learnedPriorWeight *
                        (1.0 - observedRatio) *
                        (
                            0.5 +
                                0.5 * (1.0 - geometryScore)
                        )
                position += delta * (
                    bounded / distance
                ) * trust
                learnedContribution = trust.coerceIn(0.0, 1.0)
            }
        } else if (learnedPrior != null && !hasSufficientGeometry) {
            warnings += "learned-prior-blocked-by-geometry"
        }

        val temporal = evidenceGroup
            .asSequence()
            .filter {
                it.temporalPositionMeters != null &&
                    it.temporalConfidence >=
                    config.minimumTemporalConfidence
            }
            .maxByOrNull { it.temporalConfidence }

        var temporalContribution = 0.0
        if (temporal != null && !hasSufficientGeometry) {
            val temporalPosition = temporal.temporalPositionMeters
            if (temporalPosition != null) {
                val delta = temporalPosition - position
                val distance = delta.norm()
                if (distance.isFinite() && distance > 1e-9) {
                    val temporalWeight =
                        (
                            temporal.temporalConfidence *
                                config.minimumTemporalConfidence
                            ).coerceIn(0.0, 0.35)
                    position += clampNorm(
                        delta * temporalWeight,
                        config.maximumPriorDisplacementMeters * 0.5,
                    )
                    temporalContribution = temporalWeight
                }
            }
        }

        val state = when {
            viewCount < config.minimumViewsForGeometry -> {
                warnings += "insufficient-independent-views"
                if (
                    temporalContribution <= 0.0 &&
                    learnedContribution <= 0.0
                ) {
                    State.INSUFFICIENT_EVIDENCE
                } else {
                    if (temporalContribution > 0.0) {
                        State.PROPAGATED
                    } else {
                        State.LEARNED
                    }
                }
            }

            !uncertainty.isFinite() ||
                uncertainty > config.maximumUncertaintyMeters -> {
                warnings += "geometry-uncertainty-too-high"
                State.INSUFFICIENT_EVIDENCE
            }

            measuredResidual >
                config.maximumAcceptedReprojectionPixels -> {
                warnings += "reprojection-residual-too-high"
                State.INSUFFICIENT_EVIDENCE
            }

            surfaceConsistency <
                config.minimumSurfaceConsistency -> {
                warnings += "surface-consistency-low"
                State.CONSTRAINED
            }

            observedRatio >= config.minimumObservedRatio &&
                geometryScore >= config.preferredGeometryScore -> {
                State.OBSERVED
            }

            hasSufficientGeometry -> {
                State.CONSTRAINED
            }

            temporalContribution > 0.0 -> {
                State.PROPAGATED
            }

            learnedContribution > 0.0 -> {
                State.LEARNED
            }

            else -> {
                State.INSUFFICIENT_EVIDENCE
            }
        }

        val measuredRatio = (
            observedRatio *
                (
                    0.70 +
                        0.20 * geometryScore +
                        0.10 * surfaceConsistency
                    )
            ).coerceIn(0.0, 1.0)

        val inferredRatio = (
            1.0 -
                measuredRatio +
                learnedContribution * 0.25 +
                temporalContribution * 0.25
            ).coerceIn(0.0, 1.0)

        val confidence = (
            0.28 * geometryScore +
                0.22 * observedRatio +
                0.18 * diversity +
                0.14 * surfaceConsistency +
                0.10 * residualQuality(measuredResidual) +
                0.08 * uncertaintyQuality(uncertainty)
            ).coerceIn(0.0, 1.0) * stateConfidenceMultiplier(state)

        if (
            timestampNanos < first.timestampNanos ||
            first.timestampNanos < 0L
        ) {
            warnings += "timestamp-coherence-failure"
        }

        val origin = when (state) {
            State.OBSERVED -> EvidenceOrigin.OBSERVED
            State.CONSTRAINED -> EvidenceOrigin.GEOMETRICALLY_CONSTRAINED
            State.PROPAGATED -> EvidenceOrigin.TEMPORALLY_PROPAGATED
            State.LEARNED -> EvidenceOrigin.LEARNED_PRIOR
            State.INSUFFICIENT_EVIDENCE -> EvidenceOrigin.LEARNED_PRIOR
        }

        return RegionResult(
            actorId = first.actorId,
            regionId = first.regionId,
            positionMeters = position,
            state = state,
            confidence = confidence.coerceIn(0.0, 1.0),
            measuredRatio = measuredRatio,
            inferredRatio = inferredRatio,
            warnings = warnings.toList(),
            independentViews = viewCount,
            viewDiversity = diversity,
            reprojectionResidualPixels = measuredResidual,
            uncertaintyMeters = uncertainty,
            surfaceConsistency = surfaceConsistency,
            evidenceOrigin = origin,
            observedViewCount = normalizedViews.size,
            temporalContribution = temporalContribution,
            learnedContribution = learnedContribution,
        )
    }

    private fun geometryScore(
        independentViews: Int,
        viewDiversity: Double,
        reprojectionResidualPixels: Double,
        uncertaintyMeters: Double,
        surfaceConsistency: Double,
    ): Double {
        val viewScore = (
            independentViews.toDouble() /
                config.preferredViews.coerceAtLeast(1)
            ).coerceIn(0.0, 1.0)
        val residualScore = residualQuality(reprojectionResidualPixels)
        val uncertaintyScore = uncertaintyQuality(uncertaintyMeters)

        return (
            0.30 * viewScore +
                0.22 * viewDiversity +
                0.22 * residualScore +
                0.16 * uncertaintyScore +
                0.10 * surfaceConsistency
            ).coerceIn(0.0, 1.0)
    }

    private fun estimateUncertainty(
        evidenceGroup: List<RegionEvidence>,
        positions: List<Vec3>,
        viewDiversity: Double,
    ): Double {
        val declared = evidenceGroup
            .map { it.uncertaintyMeters }
            .filter { it.isFinite() && it >= 0.0 }
            .maxOrNull()
            ?: 0.0

        if (positions.size < 2) {
            return max(
                declared,
                config.maximumUncertaintyMeters *
                    (
                        1.0 -
                            0.35 * evidenceGroup
                                .first()
                                .viewDiversity
                        )
                    )
        }

        val centroid = robustCentroid(positions)
        val spread = sqrt(
            positions
                .map { it.distanceTo(centroid) }
                .map { it * it }
                .average()
                .coerceAtLeast(0.0),
        )

        val views = evidenceGroup
            .maxOfOrNull { it.independentViews }
            ?.coerceAtLeast(1)
            ?: 1

        val geometryReduction =
            (sqrt(views.toDouble()) * (0.55 + 0.45 * viewDiversity))
                .coerceAtLeast(1.0)

        return max(
            declared,
            spread / geometryReduction,
        )
    }

    private fun residualQuality(value: Double): Double {
        if (!value.isFinite() || value < 0.0) return 0.0
        if (value <= config.preferredReprojectionPixels) return 1.0
        return (
            1.0 -
                (
                    value -
                        config.preferredReprojectionPixels
                    ) /
                    (
                        config.maximumAcceptedReprojectionPixels -
                            config.preferredReprojectionPixels
                        )
            ).coerceIn(0.0, 1.0)
    }

    private fun uncertaintyQuality(value: Double): Double {
        if (!value.isFinite() || value < 0.0) return 0.0
        return (
            1.0 -
                value /
                config.maximumUncertaintyMeters
            ).coerceIn(0.0, 1.0)
    }

    private fun stateConfidenceMultiplier(state: State): Double =
        when (state) {
            State.OBSERVED -> 1.0
            State.CONSTRAINED -> 0.86
            State.PROPAGATED -> 0.62
            State.LEARNED -> 0.48
            State.INSUFFICIENT_EVIDENCE -> 0.08
        }

    private fun robustCentroid(
        positions: List<Vec3>,
    ): Vec3 {
        if (positions.isEmpty()) {
            return Vec3(0.0, 0.0, 0.0)
        }

        var estimate = positions.reduce(Vec3::plus) / positions.size.toDouble()

        repeat(3) {
            var sum = Vec3(0.0, 0.0, 0.0)
            var weightSum = 0.0

            positions.forEach { position ->
                val error = position.distanceTo(estimate)
                if (!error.isFinite()) return@forEach
                val normalized = error / config.robustScaleMeters
                val weight =
                    if (normalized <= 1.0) {
                        1.0
                    } else {
                        1.0 / (
                            1.0 +
                                normalized *
                                normalized
                            )
                    }
                sum += position * weight
                weightSum += weight
            }

            if (weightSum > 1e-9) {
                estimate = sum / weightSum
            }
        }

        return estimate
    }

    private fun medianFinite(values: List<Double>): Double {
        val sorted = values
            .filter { it.isFinite() }
            .sorted()
        if (sorted.isEmpty()) return Double.POSITIVE_INFINITY
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }

    private fun viewDiversityFromEvidence(
        evidence: ViewEvidence,
    ): Double = (
        0.35 * evidence.visibility +
            0.35 * evidence.photometricConsistency +
            0.30 * evidence.normalConsistency
        ).coerceIn(0.0, 1.0)

    private fun clampNorm(
        value: Vec3,
        maximum: Double,
    ): Vec3 {
        val length = value.norm()
        if (!length.isFinite() || length <= maximum || length < 1e-12) {
            return value
        }
        return value * (maximum / length)
    }

    private fun finite(value: Vec3): Boolean =
        value.x.isFinite() &&
            value.y.isFinite() &&
            value.z.isFinite()
}

private fun List<Double>.weightedMean(weights: List<Double>): Double {
    if (isEmpty() || weights.size != size) return 0.0
    var weighted = 0.0
    var total = 0.0
    forEachIndexed { index, value ->
        val weight = weights[index]
        if (value.isFinite() && weight.isFinite() && weight > 0.0) {
            weighted += value * weight
            total += weight
        }
    }
    return if (total <= 1e-9) 0.0 else weighted / total
}

private fun Vec3.distanceTo(other: Vec3): Double =
    sqrt(
        (x - other.x) * (x - other.x) +
            (y - other.y) * (y - other.y) +
            (z - other.z) * (z - other.z),
    )

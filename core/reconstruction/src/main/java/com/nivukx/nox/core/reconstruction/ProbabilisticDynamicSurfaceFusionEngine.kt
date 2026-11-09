package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.FrontierReconstructionProvenance
import com.nivukx.nox.core.model.MeasurementAuthority
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Probabilistic dynamic surface state used as a reconstruction-side companion to native Gaussian
 * surface fusion and TSDF/mesh stages.
 *
 * Each patch stores uncertainty, temporal deformation, surface orientation, and provenance. The
 * implementation intentionally avoids pretending that a completed surface is directly observed.
 */
class ProbabilisticDynamicSurfaceFusionEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        val cellSizeMeters: Double = 0.025,
        val maximumPatches: Int = 750_000,
        val maximumHistoryFrames: Int = 12,
        val minimumConfidence: Double = 0.08,
        val robustScaleMeters: Double = 0.030,
        val maximumUncertaintyMeters: Double = 0.060,
        val temporalHalfLifeSeconds: Double = 0.70,
        val velocityDecay: Double = 0.88,
        val topologyDecay: Double = 0.985,
        val maximumVelocityMetersPerSecond: Double = 15.0,
    ) {
        init {
            // Configuration comes from operator settings as well as direct engine construction.
            // Reject infinities and NaN before they can poison cell keys, temporal decay or weights.
            require(cellSizeMeters.isFinite() && cellSizeMeters > 0.0)
            require(maximumPatches >= 1024)
            require(maximumHistoryFrames >= 2)
            require(minimumConfidence.isFinite() && minimumConfidence in 0.0..1.0)
            require(robustScaleMeters.isFinite() && robustScaleMeters > 0.0)
            require(maximumUncertaintyMeters.isFinite() && maximumUncertaintyMeters > 0.0)
            require(temporalHalfLifeSeconds.isFinite() && temporalHalfLifeSeconds > 0.0)
            require(velocityDecay.isFinite() && velocityDecay in 0.0..1.0)
            require(topologyDecay.isFinite() && topologyDecay in 0.0..1.0)
            require(maximumVelocityMetersPerSecond.isFinite() && maximumVelocityMetersPerSecond > 0.0)
        }
    }

    data class SurfaceObservation(
        val patchId: Long,
        val timestampNanos: Long,
        val positionMeters: Vec3,
        val normal: Vec3,
        val confidence: Double,
        val visibility: Double,
        val temporalConsistency: Double,
        val photometricConsistency: Double,
        val motionBoundaryConfidence: Double,
        val cameraCount: Int,
        val uncertaintyMeters: Double,
        val authority: MeasurementAuthority = MeasurementAuthority.OBSERVED,
    ) {
        init {
            require(patchId >= 0L)
            require(timestampNanos > 0L)
            require(positionMeters.finite())
            require(normal.finite())
            require(confidence in 0.0..1.0)
            require(visibility in 0.0..1.0)
            require(temporalConsistency in 0.0..1.0)
            require(photometricConsistency in 0.0..1.0)
            require(motionBoundaryConfidence in 0.0..1.0)
            require(cameraCount >= 0)
            require(uncertaintyMeters.isFinite() && uncertaintyMeters >= 0.0)
        }

        val evidenceWeight: Double
            get() =
                (
                    0.34 * confidence +
                        0.18 * visibility +
                        0.18 * temporalConsistency +
                        0.14 * photometricConsistency +
                        0.08 * motionBoundaryConfidence +
                        0.08 * (cameraCount.toDouble() / 8.0).coerceIn(0.0, 1.0)
                    )
                    .coerceIn(0.001, 1.0)
    }

    data class PatchState(
        val patchId: Long,
        val positionMeters: Vec3,
        val normal: Vec3,
        val velocityMetersPerSecond: Vec3,
        val uncertaintyMeters: Double,
        val confidence: Double,
        val observationCount: Long,
        val geometricObservationCount: Long,
        val predictedFrameCount: Long,
        val topologyStability: Double,
        val timestampNanos: Long,
        val authority: MeasurementAuthority,
        val observedRatio: Double,
        val geometricRatio: Double,
        val temporalRatio: Double,
        val learnedRatio: Double,
        val completedRatio: Double,
    ) {
        val directlyMeasuredRatio: Double
            get() =
                (observedRatio + geometricRatio)
                    .coerceIn(0.0, 1.0)

        val provenance: FrontierReconstructionProvenance
            get() =
                FrontierReconstructionProvenance(
                    observedRatio = observedRatio,
                    geometricRatio = geometricRatio,
                    temporalRatio = temporalRatio,
                    learnedRatio = learnedRatio,
                    completedRatio = completedRatio,
                    meanUncertaintyMeters = uncertaintyMeters,
                    p95UncertaintyMeters = uncertaintyMeters * 1.96,
                    topologyContinuity = topologyStability,
                    temporalStability =
                        (
                            0.68 * confidence +
                                0.32 * topologyStability
                            ).coerceIn(0.0, 1.0),
                    viewDiversity =
                        (observationCount.toDouble() / 8.0)
                            .coerceIn(0.0, 1.0),
                )
    }

    data class Stats(
        val patchCount: Int,
        val observedPatchCount: Int,
        val geometricPatchCount: Int,
        val predictedPatchCount: Int,
        val meanUncertaintyMeters: Double,
        val p95UncertaintyMeters: Double,
        val topologyStability: Double,
        val directlyMeasuredRatio: Double,
        val dynamicMotionMagnitudeMetersPerSecond: Double,
    )

    private data class MutablePatch(
        val patchId: Long,
        var position: Vec3,
        var normal: Vec3,
        var velocity: Vec3,
        var uncertainty: Double,
        var confidence: Double,
        var observationCount: Long,
        var geometricObservationCount: Long,
        var predictedFrameCount: Long,
        var topologyStability: Double,
        var timestampNanos: Long,
        var authority: MeasurementAuthority,
        var observedEvidence: Double,
        var geometricEvidence: Double,
        var temporalEvidence: Double,
        var learnedEvidence: Double,
        var completedEvidence: Double,
        var previousPosition: Vec3,
    )

    private val patches = LinkedHashMap<Long, MutablePatch>()
    private val spatialIndex = HashMap<CellKey, LinkedHashSet<Long>>()

    @Synchronized
    fun integrate(observation: SurfaceObservation): Boolean {
        val existing =
            patches[observation.patchId]

        if (existing == null) {
            if (patches.size >= config.maximumPatches) {
                evictWeakest(observation.timestampNanos)
            }
            if (patches.size >= config.maximumPatches) {
                return false
            }

            val patch =
                MutablePatch(
                    patchId = observation.patchId,
                    position = observation.positionMeters,
                    normal = observation.normal.normalizedSafe(),
                    velocity = Vec3(0.0, 0.0, 0.0),
                    uncertainty =
                        observation.uncertaintyMeters
                            .coerceAtMost(config.maximumUncertaintyMeters),
                    confidence = observation.confidence,
                    observationCount = 1L,
                    geometricObservationCount =
                        if (observation.authority == MeasurementAuthority.GEOMETRIC) 1L else 0L,
                    predictedFrameCount = 0L,
                    topologyStability = observation.temporalConsistency,
                    timestampNanos = observation.timestampNanos,
                    authority = observation.authority,
                    observedEvidence =
                        if (observation.authority == MeasurementAuthority.OBSERVED) 1.0 else 0.0,
                    geometricEvidence =
                        if (observation.authority == MeasurementAuthority.GEOMETRIC) 1.0 else 0.0,
                    temporalEvidence = 0.0,
                    learnedEvidence =
                        if (observation.authority == MeasurementAuthority.LEARNED) 1.0 else 0.0,
                    completedEvidence =
                        if (observation.authority == MeasurementAuthority.COMPLETED) 1.0 else 0.0,
                    previousPosition = observation.positionMeters,
                )
            patches[observation.patchId] = patch
            spatialIndex
                .getOrPut(cellFor(patch.position)) { LinkedHashSet() }
                .add(observation.patchId)
            return true
        }

        if (observation.timestampNanos < existing.timestampNanos) {
            return false
        }

        val dt =
            (
                observation.timestampNanos -
                    existing.timestampNanos
                ).toDouble() / 1_000_000_000.0

        val predicted =
            if (dt > 0.0) {
                existing.position +
                    existing.velocity *
                        dt.coerceAtMost(0.25)
            } else {
                existing.position
            }

        val innovation =
            observation.positionMeters - predicted

        val sigma =
            max(
                config.robustScaleMeters,
                existing.uncertainty +
                    observation.uncertaintyMeters,
            )

        val normalizedInnovation =
            innovation.norm() / max(sigma, 1e-6)

        val robustWeight =
            if (normalizedInnovation <= 1.0) {
                1.0 - 0.50 * normalizedInnovation * normalizedInnovation
            } else {
                0.50 / max(normalizedInnovation, 1.0)
            }

        val weight =
            (
                observation.evidenceWeight *
                    robustWeight.coerceIn(0.04, 1.0)
                ).coerceIn(0.001, 1.0)

        val previousCell = cellFor(existing.position)

        val predictionAlpha =
            (
                weight /
                    (weight + existing.confidence + 1e-6)
                ).coerceIn(0.05, 0.82)

        existing.previousPosition = existing.position
        existing.position =
            existing.position.lerp(
                observation.positionMeters,
                predictionAlpha,
            )

        val measuredVelocity =
            if (dt > 1e-6) {
                (
                    observation.positionMeters -
                        existing.previousPosition
                    ) /
                    dt.coerceAtMost(0.25)
            } else {
                Vec3(0.0, 0.0, 0.0)
            }

        existing.velocity =
            (
                existing.velocity * config.velocityDecay +
                    measuredVelocity * (
                        1.0 - config.velocityDecay
                    )
                )
                .clampedMagnitude(config.maximumVelocityMetersPerSecond)

        existing.normal =
            existing.normal
                .lerp(
                    observation.normal.normalizedSafe(),
                    predictionAlpha,
                )
                .normalizedSafe()

        val posteriorUncertainty =
            sqrt(
                (
                    existing.uncertainty *
                        existing.uncertainty *
                        (1.0 - predictionAlpha) +
                    observation.uncertaintyMeters *
                        observation.uncertaintyMeters *
                        predictionAlpha
                    )
                    .coerceAtLeast(1e-12)
            )

        existing.uncertainty =
            (
                posteriorUncertainty +
                    abs(innovation.norm()) * 0.08
                )
                .coerceIn(
                    1e-6,
                    config.maximumUncertaintyMeters,
                )

        existing.confidence =
            (
                existing.confidence * (1.0 - predictionAlpha * 0.44) +
                    observation.confidence *
                        predictionAlpha *
                        robustWeight.coerceIn(0.0, 1.0) *
                        0.56
                )
                .coerceIn(0.0, 1.0)

        existing.topologyStability =
            (
                existing.topologyStability * 0.70 +
                    observation.temporalConsistency * 0.20 +
                    observation.visibility * 0.10
                )
                .coerceIn(0.0, 1.0)

        existing.observationCount += 1L
        if (observation.authority == MeasurementAuthority.GEOMETRIC) {
            existing.geometricObservationCount += 1L
        }

        existing.observedEvidence +=
            if (observation.authority == MeasurementAuthority.OBSERVED) 1.0 else 0.0
        existing.geometricEvidence +=
            if (observation.authority == MeasurementAuthority.GEOMETRIC) 1.0 else 0.0
        existing.temporalEvidence +=
            if (observation.authority == MeasurementAuthority.TEMPORAL) 1.0 else 0.0
        existing.learnedEvidence +=
            if (observation.authority == MeasurementAuthority.LEARNED) 1.0 else 0.0
        existing.completedEvidence +=
            if (observation.authority == MeasurementAuthority.COMPLETED) 1.0 else 0.0

        val totalEvidence =
            (
                existing.observedEvidence +
                    existing.geometricEvidence +
                    existing.temporalEvidence +
                    existing.learnedEvidence +
                    existing.completedEvidence
                )
                .coerceAtLeast(1.0)

        existing.authority =
            when {
                existing.geometricEvidence / totalEvidence >= 0.55 ->
                    MeasurementAuthority.GEOMETRIC
                existing.observedEvidence / totalEvidence >= 0.50 ->
                    MeasurementAuthority.OBSERVED
                existing.temporalEvidence / totalEvidence >= 0.45 ->
                    MeasurementAuthority.TEMPORAL
                existing.learnedEvidence / totalEvidence >= 0.45 ->
                    MeasurementAuthority.LEARNED
                else ->
                    MeasurementAuthority.COMPLETED
            }

        existing.timestampNanos =
            observation.timestampNanos

        val newCell = cellFor(existing.position)
        if (previousCell != newCell) {
            spatialIndex[previousCell]?.remove(existing.patchId)
            spatialIndex[previousCell]?.takeIf { it.isEmpty() }?.let {
                spatialIndex.remove(previousCell)
            }
            spatialIndex
                .getOrPut(newCell) { LinkedHashSet() }
                .add(existing.patchId)
        }

        return weight >= config.minimumConfidence
    }

    @Synchronized
    fun predict(timestampNanos: Long): Int {
        if (timestampNanos <= 0L) return 0

        var predicted = 0

        patches.values.forEach { patch ->
            if (timestampNanos <= patch.timestampNanos) return@forEach

            val dt =
                (
                    timestampNanos -
                        patch.timestampNanos
                    ).toDouble() / 1_000_000_000.0

            val boundedDt =
                dt.coerceIn(0.0, 0.25)

            val oldCell = cellFor(patch.position)

            patch.position +=
                patch.velocity * boundedDt

            val decayFrames =
                (
                    boundedDt /
                        (1.0 / 60.0)
                ).coerceIn(0.0, 30.0)

            val confidenceDecay =
                exp(
                    -0.6931471805599453 *
                        boundedDt /
                        config.temporalHalfLifeSeconds
                )

            patch.velocity *=
                config.velocityDecay
                    .let {
                        val exponent =
                            decayFrames.coerceIn(0.0, 30.0)
                        it.powSafe(exponent)
                    }

            patch.confidence =
                (
                    patch.confidence *
                        confidenceDecay
                    )
                    .coerceIn(0.0, 1.0)

            patch.uncertainty =
                (
                    patch.uncertainty +
                        0.0004 *
                            boundedDt +
                        patch.velocity.norm() *
                            0.00015 *
                            boundedDt
                    )
                    .coerceIn(
                        1e-6,
                        config.maximumUncertaintyMeters,
                    )

            patch.topologyStability =
                (
                    patch.topologyStability *
                        config.topologyDecay
                            .powSafe(decayFrames)
                    )
                    .coerceIn(0.0, 1.0)

            patch.predictedFrameCount += 1L
            patch.timestampNanos = timestampNanos
            patch.temporalEvidence += 1.0
            predicted += 1

            val newCell = cellFor(patch.position)
            if (oldCell != newCell) {
                spatialIndex[oldCell]?.remove(patch.patchId)
                spatialIndex
                    .getOrPut(newCell) { LinkedHashSet() }
                    .add(patch.patchId)
            }
        }

        prune(timestampNanos)
        return predicted
    }

    @Synchronized
    fun snapshot(): List<PatchState> =
        patches.values
            .map { patch ->
                val total =
                    (
                        patch.observedEvidence +
                            patch.geometricEvidence +
                            patch.temporalEvidence +
                            patch.learnedEvidence +
                            patch.completedEvidence
                        )
                        .coerceAtLeast(1.0)

                PatchState(
                    patchId = patch.patchId,
                    positionMeters = patch.position,
                    normal = patch.normal,
                    velocityMetersPerSecond = patch.velocity,
                    uncertaintyMeters = patch.uncertainty,
                    confidence = patch.confidence,
                    observationCount = patch.observationCount,
                    geometricObservationCount =
                        patch.geometricObservationCount,
                    predictedFrameCount = patch.predictedFrameCount,
                    topologyStability = patch.topologyStability,
                    timestampNanos = patch.timestampNanos,
                    authority = patch.authority,
                    observedRatio =
                        patch.observedEvidence / total,
                    geometricRatio =
                        patch.geometricEvidence / total,
                    temporalRatio =
                        patch.temporalEvidence / total,
                    learnedRatio =
                        patch.learnedEvidence / total,
                    completedRatio =
                        patch.completedEvidence / total,
                )
            }

    @Synchronized
    fun stats(): Stats {
        if (patches.isEmpty()) {
            return Stats(
                patchCount = 0,
                observedPatchCount = 0,
                geometricPatchCount = 0,
                predictedPatchCount = 0,
                meanUncertaintyMeters = 0.0,
                p95UncertaintyMeters = 0.0,
                topologyStability = 0.0,
                directlyMeasuredRatio = 0.0,
                dynamicMotionMagnitudeMetersPerSecond = 0.0,
            )
        }

        val snapshot = snapshot()
        val uncertainties =
            snapshot.map { it.uncertaintyMeters }.sorted()

        return Stats(
            patchCount = snapshot.size,
            observedPatchCount =
                snapshot.count {
                    it.observedRatio >= 0.35
                },
            geometricPatchCount =
                snapshot.count {
                    it.geometricRatio >= 0.35
                },
            predictedPatchCount =
                snapshot.count {
                    it.predictedFrameCount > 0L
                },
            meanUncertaintyMeters =
                uncertainties.average(),
            p95UncertaintyMeters =
                percentile(uncertainties, 0.95),
            topologyStability =
                snapshot.map { it.topologyStability }.average(),
            directlyMeasuredRatio =
                snapshot
                    .map { it.provenance.directlyMeasuredRatio }
                    .average(),
            dynamicMotionMagnitudeMetersPerSecond =
                snapshot
                    .map { it.velocityMetersPerSecond.norm() }
                    .average(),
        )
    }

    @Synchronized
    fun prune(timestampNanos: Long) {
        if (patches.isEmpty()) return

        val staleHorizonNanos =
            (
                config.temporalHalfLifeSeconds *
                    8.0 *
                    1_000_000_000.0
                )
                .toLong()
                .coerceAtLeast(1_000_000_000L)

        val toRemove =
            patches.values
                .filter { patch ->
                    val age =
                        (
                            timestampNanos -
                                patch.timestampNanos
                            ).coerceAtLeast(0L)
                    age > staleHorizonNanos ||
                        patch.confidence < config.minimumConfidence * 0.18
                }
                .map { it.patchId }

        toRemove.forEach(::removePatch)

        if (patches.size <= config.maximumPatches) return

        val overflow =
            patches.size -
                config.maximumPatches

        patches.values
            .asSequence()
            .sortedBy {
                patchUtility(it)
            }
            .take(overflow)
            .map { it.patchId }
            .toList()
            .forEach(::removePatch)
    }

    fun reset() {
        synchronized(this) {
            patches.clear()
            spatialIndex.clear()
        }
    }

    override fun close() {
        reset()
    }

    private fun evictWeakest(timestampNanos: Long) {
        prune(timestampNanos)
        if (patches.size < config.maximumPatches) return

        patches.values
            .asSequence()
            .sortedBy { patchUtility(it) }
            // Evict a small batch rather than sorting the entire patch map for every incoming
            // unique ID at saturation. At the default 750k cap this amortizes a global ranking
            // pass across thousands of incoming observations while keeping the same hard bound.
            .take(
                max(
                    1,
                    patches.size / 100,
                ),
            )
            .map { it.patchId }
            .toList()
            .forEach(::removePatch)
    }

    private fun patchUtility(
        patch: MutablePatch,
    ): Double =
        (
            0.38 * patch.confidence +
                0.22 * patch.topologyStability +
                0.16 *
                    (1.0 - patch.uncertainty /
                        config.maximumUncertaintyMeters)
                        .coerceIn(0.0, 1.0) +
                0.10 *
                    (patch.observationCount.toDouble() /
                        16.0)
                        .coerceIn(0.0, 1.0) +
                0.08 *
                    (patch.geometricEvidence /
                        (
                            patch.observedEvidence +
                                patch.geometricEvidence +
                                patch.temporalEvidence +
                                patch.learnedEvidence +
                                patch.completedEvidence +
                                1e-6
                        ))
                        .coerceIn(0.0, 1.0) +
                0.06 *
                    (1.0 -
                        patch.predictedFrameCount.toDouble() /
                            patch.observationCount
                                .coerceAtLeast(1L)
                                .toDouble()
                    ).coerceIn(0.0, 1.0)
            )
            .coerceIn(0.0, 1.0)

    private data class CellKey(
        val x: Long,
        val y: Long,
        val z: Long,
    )

    private fun cellFor(position: Vec3): CellKey =
        CellKey(
            x = kotlin.math.floor(position.x / config.cellSizeMeters).toLong(),
            y = kotlin.math.floor(position.y / config.cellSizeMeters).toLong(),
            z = kotlin.math.floor(position.z / config.cellSizeMeters).toLong(),
        )

    private fun removePatch(id: Long) {
        val patch =
            patches.remove(id)
            ?: return

        spatialIndex[cellFor(patch.position)]?.remove(id)
        spatialIndex[cellFor(patch.position)]?.takeIf { it.isEmpty() }?.let {
            spatialIndex.remove(cellFor(patch.position))
        }
    }

    private fun percentile(
        values: List<Double>,
        p: Double,
    ): Double {
        if (values.isEmpty()) return 0.0
        val clamped = p.coerceIn(0.0, 1.0)
        val index =
            clamped *
                (values.size - 1)
        val lower =
            index
                .toInt()
                .coerceIn(0, values.lastIndex)
        val upper =
            (lower + 1)
                .coerceAtMost(values.lastIndex)
        val alpha = index - lower
        return (
            values[lower] * (1.0 - alpha) +
                values[upper] * alpha
            )
    }

    private fun Vec3.finite(): Boolean =
        x.isFinite() && y.isFinite() && z.isFinite()

    private fun Vec3.normalizedSafe(): Vec3 =
        if (norm() < 1e-9) {
            Vec3(0.0, 0.0, 1.0)
        } else {
            normalized()
        }

    private fun Vec3.clampedMagnitude(maximum: Double): Vec3 {
        val magnitude = norm()
        if (!magnitude.isFinite() || magnitude <= maximum) return this
        return this * (maximum / magnitude)
    }

    private fun Vec3.lerp(
        target: Vec3,
        alpha: Double,
    ): Vec3 =
        this * (1.0 - alpha) +
            target * alpha

    private fun Double.powSafe(exponent: Double): Double =
        kotlin.math.exp(
            kotlin.math.ln(
                coerceAtLeast(1e-9),
            ) *
                exponent,
        )
}


package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Calibration-stage temporal bundle optimizer.
 *
 * Camera geometry stays fixed in this stage. Each non-reference camera gets a bounded temporal
 * offset estimated by minimizing robust reprojection error against an already reconstructed dynamic
 * trajectory. Offsets are returned as evidence and are never written into authoritative clocks.
 */
class SpatiotemporalBundleAdjustmentEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumOffsetNanos: Long = 80_000_000L,
        val searchStepNanos: Long = 1_000_000L,
        val coarseSearchStepNanos: Long = 4_000_000L,
        val coarseRefinementRadiusSteps: Int = 2,
        val minimumObservations: Int = 16,
        val minimumConfidence: Double = 0.40,
        val huberDeltaPx: Double = 3.0,
        val interpolationGapLimitNanos: Long = 50_000_000L,
        val minimumJointConsensusConfidence: Double = 0.70,
        val maximumJointOffsetSpreadNanos: Long = 8_000_000L,
        val minimumJointImprovementRatio: Double = 0.60,
        val dynamicAlignmentEnabled: Boolean = true,
        val dynamicAlignmentMaximumSamples: Int = 96,
        val dynamicAlignmentBand: Int = 12,
        val dynamicAlignmentMinimumMotionPixels: Double = 0.45,
        val dynamicAlignmentMaximumNormalizedCost: Double = 0.42,
    ) {
        init {
            require(maximumOffsetNanos > 0L)
            require(searchStepNanos > 0L && searchStepNanos <= maximumOffsetNanos)
            require(
                coarseSearchStepNanos >= searchStepNanos &&
                    coarseSearchStepNanos <= maximumOffsetNanos,
            )
            require(coarseRefinementRadiusSteps in 1..8)
            require(minimumObservations >= 4)
            require(minimumConfidence in 0.0..1.0)
            require(huberDeltaPx > 0.0)
            require(interpolationGapLimitNanos > 0L)
            require(minimumJointConsensusConfidence in 0.0..1.0)
            require(dynamicAlignmentMaximumSamples in 16..192)
            require(dynamicAlignmentBand in 2..48)
            require(dynamicAlignmentMinimumMotionPixels > 0.0)
            require(dynamicAlignmentMaximumNormalizedCost > 0.0)
            require(maximumJointOffsetSpreadNanos > 0L)
            require(minimumJointImprovementRatio in 0.0..1.0)
        }
    }

    data class Camera(
        val id: String,
        val pose: Pose,
        val intrinsics: CameraIntrinsics,
        val fixed: Boolean = false,
    )

    data class TrajectorySample(
        val pointId: Int,
        val timestampNanos: Long,
        val position: Vec3,
    ) {
        init {
            require(pointId >= 0)
            require(timestampNanos >= 0L)
            require(
                position.x.isFinite() &&
                    position.y.isFinite() &&
                    position.z.isFinite()
            )
        }
    }

    data class DynamicObservation(
        val cameraId: String,
        val pointId: Int,
        val timestampNanos: Long,
        val pixel: Vec2,
        val confidence: Double = 1.0,
        val uncertainty: Double = 0.0,
        val visibility: Double = 1.0,
        val temporalStability: Double = 1.0,
        val crossViewAgreement: Double = 1.0,
        val contactProbability: Double = 0.0,
        val rollingShutterFraction: Double = 0.0,
        val temporalAlignmentConfidence: Double = 1.0,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(pointId >= 0)
            require(timestampNanos >= 0L)
            require(pixel.x.isFinite() && pixel.y.isFinite())
            require(confidence in 0.0..1.0)
            require(uncertainty in 0.0..1.0)
            require(visibility in 0.0..1.0)
            require(temporalStability in 0.0..1.0)
            require(crossViewAgreement in 0.0..1.0)
            require(contactProbability in 0.0..1.0)
            require(
                rollingShutterFraction.isFinite() &&
                    rollingShutterFraction in 0.0..1.0,
            )
            require(temporalAlignmentConfidence in 0.0..1.0)
        }
    }

    data class CameraOffsetResult(
        val cameraId: String,
        val offsetNanos: Double,
        val initialRmsPx: Double,
        val finalRmsPx: Double,
        val observationsUsed: Int,
        val confidence: Double,
        val accepted: Boolean,
        val peakSharpness: Double,
    )

    data class Result(
        val cameraOffsetsNanos: Map<String, Double>,
        val cameraResults: List<CameraOffsetResult>,
        val initialRmsPx: Double,
        val finalRmsPx: Double,
        val confidence: Double,
        val accepted: Boolean,
        val converged: Boolean,
        val warnings: List<String>,
        val jointTemporalConsensusConfidence: Double = 0.0,
        val jointTemporalOffsetSpreadNanos: Double = 0.0,
        val jointTemporalImprovementRatio: Double = 0.0,
    )

    private data class Score(
        val cost: Double,
        val rmsPx: Double,
        val samples: Int,
    )

    fun optimize(
        cameras: List<Camera>,
        trajectories: List<TrajectorySample>,
        observations: List<DynamicObservation>,
        referenceCameraId: String,
    ): Result {
        val distinctCameras = cameras.distinctBy { it.id }
        require(distinctCameras.any { it.id == referenceCameraId }) {
            "referenceCameraId must be present in cameras"
        }

        if (
            distinctCameras.size < 2 ||
            trajectories.isEmpty() ||
            observations.isEmpty()
        ) {
            return rejected(
                distinctCameras.map { it.id },
                "insufficient-spatiotemporal-input",
            )
        }

        val cameraMap = distinctCameras.associateBy { it.id }
        val trajectoryMap =
            trajectories
                .groupBy { it.pointId }
                .mapValues { (_, value) ->
                    value.sortedBy { it.timestampNanos }
                }

        val offsets = distinctCameras
            .associate { it.id to 0.0 }
            .toMutableMap()

        val initialScores = scoreAll(
            cameraMap,
            trajectoryMap,
            observations,
            offsets,
            referenceCameraId,
        )

        val cameraResults = ArrayList<CameraOffsetResult>()


        distinctCameras
            .filter { it.id != referenceCameraId }
            .forEach { camera ->
                val cameraObservations =
                    observations.filter {
                        it.cameraId == camera.id
                    }

                if (
                    camera.fixed ||
                    cameraObservations.size <
                    config.minimumObservations
                ) {
                    cameraResults += CameraOffsetResult(
                        cameraId = camera.id,
                        offsetNanos = 0.0,
                        initialRmsPx =
                            initialScores[camera.id]?.rmsPx
                                ?: Double.POSITIVE_INFINITY,
                        finalRmsPx =
                            initialScores[camera.id]?.rmsPx
                                ?: Double.POSITIVE_INFINITY,
                        observationsUsed =
                            initialScores[camera.id]?.samples
                                ?: 0,
                        confidence = if (camera.fixed) 1.0 else 0.0,
                        accepted = camera.fixed,
                        peakSharpness = if (camera.fixed) 1.0 else 0.0,
                    )
                    return@forEach
                }

                val dynamicCandidate =
                    if (config.dynamicAlignmentEnabled) {
                        dynamicAlignmentOffset(
                            camera = camera,
                            trajectories = trajectoryMap,
                            observations = cameraObservations,
                        )
                    } else {
                        null
                    }

                val best =
                    searchOffset(
                        camera = camera,
                        trajectories = trajectoryMap,
                        observations = cameraObservations,
                        seedOffsetsNanos =
                            dynamicCandidate
                                ?.let { listOf(it) }
                                .orEmpty(),
                    )

                if (best == null) {
                    cameraResults += CameraOffsetResult(
                        cameraId = camera.id,
                        offsetNanos = 0.0,
                        initialRmsPx =
                            initialScores[camera.id]?.rmsPx
                                ?: Double.POSITIVE_INFINITY,
                        finalRmsPx =
                            initialScores[camera.id]?.rmsPx
                                ?: Double.POSITIVE_INFINITY,
                        observationsUsed =
                            initialScores[camera.id]?.samples
                                ?: 0,
                        confidence = 0.0,
                        accepted = false,
                        peakSharpness = 0.0,
                    )
                    return@forEach
                }

                offsets[camera.id] = best.offsetNanos
                cameraResults += makeCameraResult(
                    camera = camera,
                    best = best,
                    initial = initialScores[camera.id],
                )
            }

        val finalScores = scoreAll(
            cameraMap,
            trajectoryMap,
            observations,
            offsets,
            referenceCameraId,
        )

        val jointConsensus =
            jointConsensus(
                cameraResults = cameraResults,
            )

        val initialRms = aggregateRms(initialScores.values.map { it.rmsPx })
        val finalRms = aggregateRms(finalScores.values.map { it.rmsPx })
        val confidence =
            if (cameraResults.isEmpty()) {
                0.0
            } else {
                cameraResults.map { it.confidence }.average()
            }.coerceIn(0.0, 1.0)

        val accepted =
            cameraResults.any { it.accepted } &&
                initialRms.isFinite() &&
                finalRms.isFinite() &&
                finalRms < initialRms &&
                confidence >= config.minimumConfidence &&
                jointConsensus.confidence >=
                    config.minimumJointConsensusConfidence &&
                jointConsensus.improvementRatio >=
                    config.minimumJointImprovementRatio

        val warnings = buildList {
            if (cameraResults.none { it.accepted }) {
                add("no-camera-temporal-offset-was-accepted")
            }
            if (
                observations.any {
                    it.uncertainty > 0.60 ||
                        it.visibility < 0.35 ||
                        it.crossViewAgreement < 0.35
                }
            ) {
                add("dense-observation-quality-reduced")
            }
            if (
                finalRms.isFinite() &&
                initialRms.isFinite() &&
                finalRms >= initialRms
            ) {
                add("temporal-offsets-did-not-improve-rig-error")
            }
            if (
                jointConsensus.confidence <
                    config.minimumJointConsensusConfidence
            ) {
                add(
                    "joint-temporal-consensus-below-reference-threshold",
                )
            }
            if (
                jointConsensus.offsetSpreadNanos >
                    config.maximumJointOffsetSpreadNanos
            ) {
                add("joint-temporal-offset-spread-high")
            }
            if (
                jointConsensus.improvementRatio <
                    config.minimumJointImprovementRatio
            ) {
                add("joint-temporal-improvement-ratio-low")
            }
        }

        return Result(
            cameraOffsetsNanos = offsets,
            cameraResults = cameraResults,
            initialRmsPx = initialRms,
            finalRmsPx = finalRms,
            confidence = confidence,
            accepted = accepted,
            converged =
                cameraResults.isNotEmpty() &&
                    cameraResults.all { it.accepted },
            warnings = warnings,
            jointTemporalConsensusConfidence =
                jointConsensus.confidence,
            jointTemporalOffsetSpreadNanos =
                jointConsensus.offsetSpreadNanos,
            jointTemporalImprovementRatio =
                jointConsensus.improvementRatio,
        )
    }

    private data class JointConsensus(
        val confidence: Double,
        val offsetSpreadNanos: Double,
        val improvementRatio: Double,
    )

    private fun jointConsensus(
        cameraResults: List<CameraOffsetResult>,
    ): JointConsensus {
        val accepted =
            cameraResults.filter { it.accepted }
        if (accepted.isEmpty()) {
            return JointConsensus(
                confidence = 0.0,
                offsetSpreadNanos = Double.POSITIVE_INFINITY,
                improvementRatio = 0.0,
            )
        }

        val offsets =
            accepted.map { it.offsetNanos }
        val medianOffset =
            median(offsets)
        val maxDeviation =
            offsets.maxOfOrNull {
                abs(it - medianOffset)
            } ?: 0.0

        val acceptedRatio =
            accepted.size.toDouble() /
                cameraResults.size.coerceAtLeast(1).toDouble()

        val improvementRatio =
            accepted.count {
                it.finalRmsPx.isFinite() &&
                    it.initialRmsPx.isFinite() &&
                    it.finalRmsPx < it.initialRmsPx
            }.toDouble() /
                cameraResults.size.coerceAtLeast(1).toDouble()

        val spreadScore =
            (
                1.0 -
                    maxDeviation /
                        config.maximumJointOffsetSpreadNanos
                            .toDouble()
            ).coerceIn(0.0, 1.0)

        val confidence =
            (
                acceptedRatio * 0.35 +
                    improvementRatio * 0.35 +
                    spreadScore * 0.30
                ).coerceIn(0.0, 1.0)

        return JointConsensus(
            confidence = confidence,
            offsetSpreadNanos = maxDeviation,
            improvementRatio = improvementRatio,
        )
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }

    private data class DynamicMotionSample(
        val timestampNanos: Long,
        val magnitude: Double,
    )

    private data class DynamicAlignment(
        val offsetNanos: Double,
        val normalizedCost: Double,
        val pathLength: Int,
    )

    private fun dynamicAlignmentOffset(
        camera: Camera,
        trajectories: Map<Int, List<TrajectorySample>>,
        observations: List<DynamicObservation>,
    ): Double? {
        val pointGroups =
            observations
                .groupBy { it.pointId }
                .entries
                .sortedByDescending { it.value.size }
                .take(12)

        val alignments =
            pointGroups.mapNotNull { (pointId, pointObservations) ->
                val trajectory =
                    trajectories[pointId]
                        ?: return@mapNotNull null

                val projected =
                    trajectory.mapNotNull { sample ->
                        project(sample.position, camera)?.let { pixel ->
                            sample.timestampNanos to pixel
                        }
                    }.sortedBy { it.first }.take(
                        config.dynamicAlignmentMaximumSamples,
                    )

                val measured =
                    pointObservations.sortedBy { it.timestampNanos }
                        .take(config.dynamicAlignmentMaximumSamples)

                if (projected.size < 5 || measured.size < 5) {
                    return@mapNotNull null
                }

                val projectedMotion = projected.zipWithNext().map { (a,b) ->
                    DynamicMotionSample(
                        timestampNanos = a.first + (b.first-a.first)/2L,
                        magnitude = a.second.distanceTo(b.second),
                    )
                }.filter { it.magnitude >= config.dynamicAlignmentMinimumMotionPixels }

                val measuredMotion = measured.zipWithNext().map { (a,b) ->
                    DynamicMotionSample(
                        timestampNanos = a.timestampNanos + (b.timestampNanos-a.timestampNanos)/2L,
                        magnitude = a.pixel.distanceTo(b.pixel),
                    )
                }.filter { it.magnitude >= config.dynamicAlignmentMinimumMotionPixels }

                if (projectedMotion.size < 4 || measuredMotion.size < 4) {
                    return@mapNotNull null
                }

                dynamicTimeWarp(measuredMotion, projectedMotion)
            }

        val trusted =
            alignments.filter {
                it.normalizedCost <= config.dynamicAlignmentMaximumNormalizedCost
            }
        if (trusted.isEmpty()) return null

        val offsets = trusted.map { it.offsetNanos }.sorted()
        val middle = offsets.size / 2
        return if (offsets.size % 2 == 0) {
            (offsets[middle - 1] + offsets[middle]) * 0.5
        } else {
            offsets[middle]
        }
    }

    private fun dynamicTimeWarp(
        observed: List<DynamicMotionSample>,
        trajectory: List<DynamicMotionSample>,
    ): DynamicAlignment? {
        val rows = observed.size
        val columns = trajectory.size
        if (rows == 0 || columns == 0) return null

        val infinity = Double.POSITIVE_INFINITY
        val cost = Array(rows + 1) { DoubleArray(columns + 1) { infinity } }
        val direction = Array(rows + 1) { ByteArray(columns + 1) }
        cost[0][0] = 0.0
        val band = config.dynamicAlignmentBand.coerceAtLeast(1)

        for (row in 1..rows) {
            val minColumn = max(1, row - band)
            val maxColumn = min(columns, row + band)
            for (column in minColumn..maxColumn) {
                val a = observed[row - 1].magnitude
                val b = trajectory[column - 1].magnitude
                val scale = max(1.0, max(a,b))
                val local = abs(a-b) / scale
                val match = cost[row-1][column-1]
                val insertion = cost[row-1][column]
                val deletion = cost[row][column-1]
                val best = minOf(match,insertion,deletion)
                if (!best.isFinite()) continue
                cost[row][column] = best + local
                direction[row][column] = when(best) {
                    match -> 0
                    insertion -> 1
                    else -> 2
                }
            }
        }

        val terminal = cost[rows][columns]
        if (!terminal.isFinite()) return null

        var row = rows
        var column = columns
        val offsets = ArrayList<Double>(rows + columns)
        while (row > 0 || column > 0) {
            when (direction[row][column].toInt()) {
                0 -> {
                    offsets += (
                        observed[row-1].timestampNanos -
                            trajectory[column-1].timestampNanos
                        ).toDouble()
                    row--
                    column--
                }
                1 -> row--
                2 -> column--
                else -> return null
            }
        }

        if (offsets.size < 4) return null
        val sorted = offsets.sorted()
        val middle = sorted.size / 2
        val median = if (sorted.size % 2 == 0) {
            (sorted[middle-1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
        return DynamicAlignment(
            offsetNanos = median,
            normalizedCost = terminal / max(1.0, offsets.size.toDouble()),
            pathLength = offsets.size,
        )
    }
    private fun searchOffset(
        camera: Camera,
        trajectories: Map<Int, List<TrajectorySample>>,
        observations: List<DynamicObservation>,
        seedOffsetsNanos: List<Double> = emptyList(),
    ): OffsetScore? {
        fun evaluateCandidate(
            offsetNanos: Double,
        ): OffsetScore? {
            val score =
                scoreCamera(
                    camera,
                    trajectories,
                    observations,
                    offsetNanos,
                )
            return if (score.samples >= config.minimumObservations) {
                OffsetScore(
                    offsetNanos = offsetNanos,
                    score = score,
                )
            } else {
                null
            }
        }

        var best: OffsetScore? = null
        var coarseOffset = -config.maximumOffsetNanos.toDouble()

        while (coarseOffset <= config.maximumOffsetNanos.toDouble() + 0.5) {
            val candidate = evaluateCandidate(coarseOffset)
            if (candidate != null && (best == null || candidate.score.cost < best!!.score.cost)) {
                best = candidate
            }
            coarseOffset += config.coarseSearchStepNanos.toDouble()
        }

        val coarseBest = best ?: return null
        var fineBest = coarseBest
        val radius = config.coarseRefinementRadiusSteps
        val anchors = (
            listOf(coarseBest.offsetNanos) +
                seedOffsetsNanos.filter { it.isFinite() }
        ).map {
            it.coerceIn(
                -config.maximumOffsetNanos.toDouble(),
                config.maximumOffsetNanos.toDouble(),
            )
        }.distinct()

        anchors.forEach { anchor ->
            for (stepIndex in -radius..radius) {
                val offset =
                    anchor +
                        stepIndex *
                            config.searchStepNanos.toDouble()
                if (
                    offset < -config.maximumOffsetNanos.toDouble() ||
                    offset > config.maximumOffsetNanos.toDouble()
                ) continue
                val candidate = evaluateCandidate(offset) ?: continue
                if (candidate.score.cost < fineBest.score.cost) {
                    fineBest = candidate
                }
            }
        }

        val step = config.searchStepNanos.toDouble()
        val center = fineBest.offsetNanos
        if (center <= -config.maximumOffsetNanos || center >= config.maximumOffsetNanos) return fineBest

        val left = evaluateCandidate(center - step) ?: return fineBest
        val right = evaluateCandidate(center + step) ?: return fineBest
        val denominator = left.score.cost - 2.0 * fineBest.score.cost + right.score.cost

        if (!denominator.isFinite() || abs(denominator) < 1e-9) return fineBest

        val fraction =
            (0.5 * (left.score.cost - right.score.cost) / denominator)
                .coerceIn(-0.5, 0.5)

        val refinedOffset =
            (center + fraction * step).coerceIn(
                -config.maximumOffsetNanos.toDouble(),
                config.maximumOffsetNanos.toDouble(),
            )

        val refined = evaluateCandidate(refinedOffset)
        return if (refined != null && refined.score.cost <= fineBest.score.cost) refined else fineBest
    }
    private data class OffsetScore(
        val offsetNanos: Double,
        val score: Score,
    )

    private fun scoreAll(
        cameras: Map<String, Camera>,
        trajectories: Map<Int, List<TrajectorySample>>,
        observations: List<DynamicObservation>,
        offsets: Map<String, Double>,
        referenceCameraId: String,
    ): Map<String, Score> =
        cameras.values
            .filter { it.id != referenceCameraId }
            .associate { camera ->
                camera.id to scoreCamera(
                    camera,
                    trajectories,
                    observations.filter {
                        it.cameraId == camera.id
                    },
                    offsets[camera.id] ?: 0.0,
                )
            }

    private fun scoreCamera(
        camera: Camera,
        trajectories: Map<Int, List<TrajectorySample>>,
        observations: List<DynamicObservation>,
        offsetNanos: Double,
    ): Score {
        var weightedSquaredError = 0.0
        var weightSum = 0.0
        var cost = 0.0
        var samples = 0

        observations.forEach { observation ->
            val trajectory =
                trajectories[observation.pointId]
                    ?: return@forEach

            val point =
                interpolate(
                    trajectory,
                    observation.timestampNanos +
                        offsetNanos,
                ) ?: return@forEach

            val projection =
                project(point, camera)
                    ?: return@forEach

            val dx =
                observation.pixel.x -
                    projection.x
            val dy =
                observation.pixel.y -
                    projection.y
            val error =
                sqrt(dx * dx + dy * dy)

            val sourceWeight =
                effectiveObservationWeight(observation)
            val robustWeight =
                huberWeight(error)
            val weight =
                sourceWeight * robustWeight

            weightedSquaredError +=
                error * error * weight
            weightSum += weight
            cost +=
                huberCost(error) *
                    sourceWeight
            samples += 1
        }

        if (
            samples == 0 ||
            weightSum <= 1e-9
        ) {
            return Score(
                cost = Double.POSITIVE_INFINITY,
                rmsPx = Double.POSITIVE_INFINITY,
                samples = samples,
            )
        }

        return Score(
            cost = cost / samples,
            rmsPx =
                sqrt(
                    weightedSquaredError /
                        weightSum,
                ),
            samples = samples,
        )
    }

    private fun makeCameraResult(
        camera: Camera,
        best: OffsetScore,
        initial: Score?,
    ): CameraOffsetResult {
        val initialRms =
            initial?.rmsPx
                ?: Double.POSITIVE_INFINITY

        val improvement =
            if (initialRms.isFinite()) {
                (initialRms - best.score.rmsPx)
                    .coerceAtLeast(0.0)
            } else {
                0.0
            }

        val relativeImprovement =
            improvement /
                max(initialRms, 1e-6)

        val sampleConfidence =
            (
                best.score.samples.toDouble() /
                    max(
                        config.minimumObservations,
                        best.score.samples,
                    )
                ).coerceIn(0.0, 1.0)

        val peakSharpness =
            (
                1.0 -
                    best.score.cost /
                    max(
                        initial?.cost
                            ?: best.score.cost,
                        1e-6,
                    )
                ).coerceIn(0.0, 1.0)

        val confidence =
            (
                0.46 * relativeImprovement +
                    0.34 * sampleConfidence +
                    0.20 * peakSharpness
                ).coerceIn(0.0, 1.0)

        return CameraOffsetResult(
            cameraId = camera.id,
            offsetNanos = best.offsetNanos,
            initialRmsPx = initialRms,
            finalRmsPx = best.score.rmsPx,
            observationsUsed = best.score.samples,
            confidence = confidence,
            accepted =
                confidence >= config.minimumConfidence &&
                    best.score.rmsPx.isFinite() &&
                    (
                        !initialRms.isFinite() ||
                            best.score.rmsPx < initialRms
                    ),
            peakSharpness = peakSharpness,
        )
    }

    private fun interpolate(
        trajectory: List<TrajectorySample>,
        timestampNanos: Double,
    ): Vec3? {
        if (trajectory.size < 2) return null
        if (
            timestampNanos <
                trajectory.first().timestampNanos ||
            timestampNanos >
                trajectory.last().timestampNanos
        ) {
            return null
        }

        var low = 0
        var high = trajectory.lastIndex

        while (low <= high) {
            val mid =
                (low + high) ushr 1
            val timestamp =
                trajectory[mid].timestampNanos

            when {
                timestamp < timestampNanos -> {
                    low = mid + 1
                }
                timestamp > timestampNanos -> {
                    high = mid - 1
                }
                else -> return trajectory[mid].position
            }
        }

        val rightIndex = low
        val leftIndex = low - 1
        if (
            leftIndex < 0 ||
            rightIndex > trajectory.lastIndex
        ) {
            return null
        }

        val left = trajectory[leftIndex]
        val right = trajectory[rightIndex]
        val span =
            (
                right.timestampNanos -
                    left.timestampNanos
                ).toDouble()

        if (
            span <= 0.0 ||
            span > config.interpolationGapLimitNanos
        ) {
            return null
        }

        val alpha =
            (
                timestampNanos -
                    left.timestampNanos
                ) / span

        return left.position.lerp(
            right.position,
            alpha.coerceIn(0.0, 1.0),
        )
    }

    private fun project(
        point: Vec3,
        camera: Camera,
    ): Vec2? {
        val cameraPoint =
            camera.pose.rotation
                .conjugate()
                .normalized()
                .rotate(
                    point -
                        camera.pose.position,
                )

        if (
            !cameraPoint.isFinite() ||
            cameraPoint.z <= 1e-8
        ) {
            return null
        }

        var x =
            cameraPoint.x /
                cameraPoint.z
        var y =
            cameraPoint.y /
                cameraPoint.z

        val r2 = x * x + y * y
        val k = camera.intrinsics
        val radial =
            1.0 +
                k.k1 * r2 +
                k.k2 * r2 * r2 +
                k.k3 * r2 * r2 * r2
        val dx =
            2.0 * k.p1 * x * y +
                k.p2 * (r2 + 2.0 * x * x)
        val dy =
            k.p1 * (r2 + 2.0 * y * y) +
                2.0 * k.p2 * x * y

        x = x * radial + dx
        y = y * radial + dy

        return Vec2(
            x * k.fx + k.cx,
            y * k.fy + k.cy,
        )
    }

    private fun effectiveObservationWeight(
        observation: DynamicObservation,
    ): Double {
        val uncertaintyWeight =
            (1.0 - 0.72 * observation.uncertainty)
                .coerceIn(0.28, 1.0)
        val visibilityWeight =
            (0.35 + 0.65 * observation.visibility)
                .coerceIn(0.35, 1.0)
        val temporalWeight =
            (0.35 + 0.65 * observation.temporalStability)
                .coerceIn(0.35, 1.0)
        val crossViewWeight =
            (0.35 + 0.65 * observation.crossViewAgreement)
                .coerceIn(0.35, 1.0)
        val temporalAlignmentWeight =
            observation.temporalAlignmentConfidence
                .coerceIn(0.0, 1.0)
        val rollingShutterWeight =
            (1.0 - 0.50 * abs(observation.rollingShutterFraction))
                .coerceIn(0.25, 1.0)
        val contactSupport =
            (1.0 + 0.06 * observation.contactProbability)
                .coerceIn(1.0, 1.06)

        return (
            observation.confidence.coerceIn(0.02, 1.0) *
                uncertaintyWeight *
                visibilityWeight *
                temporalWeight *
                crossViewWeight *
                temporalAlignmentWeight *
                rollingShutterWeight *
                contactSupport
            ).coerceIn(0.02, 1.0)
    }

    private fun huberWeight(
        errorPx: Double,
    ): Double {
        val safeError = errorPx.takeIf { it.isFinite() }?.coerceAtLeast(0.0) ?: Double.POSITIVE_INFINITY
        val delta = config.huberDeltaPx.coerceAtLeast(1e-6)
        val huber = if (safeError <= delta) 1.0 else delta / safeError.coerceAtLeast(delta)
        val normalized = safeError / (delta * 1.75)
        val cauchy = 1.0 / (1.0 + normalized * normalized)
        val tukeyRadius = safeError / (delta * 4.5)
        val tukey = if (tukeyRadius < 1.0) {
            val oneMinus = 1.0 - tukeyRadius * tukeyRadius
            oneMinus * oneMinus
        } else {
            0.0
        }
        return (huber * cauchy * max(0.01, tukey)).coerceIn(0.0, 1.0)
    }

    private fun huberCost(
        errorPx: Double,
    ): Double {
        val delta = config.huberDeltaPx.coerceAtLeast(1e-6)
        val safeError = errorPx.takeIf { it.isFinite() }?.coerceAtLeast(0.0) ?: 1e9
        val huber = if (safeError <= delta) {
            0.5 * safeError * safeError
        } else {
            delta * (safeError - 0.5 * delta)
        }
        val normalized = safeError / (delta * 2.0)
        val cauchy = delta * delta * ln(1.0 + normalized * normalized)
        return (0.72 * huber + 0.28 * cauchy).coerceAtLeast(0.0)
    }

    private fun rejected(
        cameraIds: List<String>,
        warning: String,
    ): Result =
        Result(
            cameraOffsetsNanos =
                cameraIds.associateWith { 0.0 },
            cameraResults = emptyList(),
            initialRmsPx = Double.POSITIVE_INFINITY,
            finalRmsPx = Double.POSITIVE_INFINITY,
            confidence = 0.0,
            accepted = false,
            converged = false,
            warnings = listOf(warning),
        )
    private fun aggregateRms(scores: List<Double>): Double =
        if (scores.isEmpty()) Double.POSITIVE_INFINITY
        else sqrt(scores.filter { it.isFinite() }.map { it * it }.average().takeIf { it.isFinite() } ?: Double.POSITIVE_INFINITY)

}

package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec2
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Production temporal-fusion stage for asynchronous multi-camera capture.
 *
 * It does not invent missing geometry. Instead it:
 *  - preserves source timestamps,
 *  - models camera phase offsets explicitly,
 *  - fuses observations inside a bounded temporal window,
 *  - rejects stale/outlier evidence,
 *  - predicts only inside a configured horizon,
 *  - reports the evidence used to create every fused sample.
 *
 * The design follows the sparse-interleaved multi-view direction demonstrated in recent
 * multi-view pose research: camera views can contribute at different temporal phases rather
 * than waiting for a perfectly synchronized frame barrier.
 */
class InterleavedSpatiotemporalFusionEngine(
    initialConfig: Config = Config(),
) {
    @Volatile
    private var config: Config = initialConfig

    @Synchronized
    fun configure(newConfig: Config) {
        config = newConfig
    }

    fun configuration(): Config = config
    data class Config(
        val enabled: Boolean = true,
        val maximumFramesPerCamera: Int = 48,
        // Supports the full distributed evidence envelope; reconstruction stages may select a smaller authoritative subset.
        val maximumCameras: Int = 128,
        val maximumJoints: Int = 512,
        val maximumTemporalWindowNanos: Long = 45_000_000L,
        val maximumPredictionNanos: Long = 18_000_000L,
        val maximumSampleSkewNanos: Long = 25_000_000L,
        val minimumConfidence: Double = 0.20,
        val outlierDistancePixels: Double = 24.0,
        val huberTransitionPixels: Double = 6.0,
        val maximumVelocityPixelsPerSecond: Double = 12_000.0,
        val minimumSamplesForAuthoritativeOutput: Int = 2,
        val maximumEpipolarPeersPerSample: Int = 8,
        val maximumEpipolarTemporalDeltaNanos: Long = 4_000_000L,
    ) {
        init {
            require(maximumFramesPerCamera >= 2)
            require(maximumCameras >= 1)
            require(maximumJoints >= 1)
            require(maximumTemporalWindowNanos > 0L)
            require(maximumPredictionNanos >= 0L)
            require(maximumPredictionNanos <= maximumTemporalWindowNanos)
            require(maximumSampleSkewNanos > 0L)
            require(minimumConfidence in 0.0..1.0)
            require(outlierDistancePixels > 0.0)
            require(huberTransitionPixels > 0.0)
            require(maximumVelocityPixelsPerSecond > 0.0)
            require(minimumSamplesForAuthoritativeOutput >= 2)
            require(maximumEpipolarPeersPerSample in 1..16)
            require(maximumEpipolarTemporalDeltaNanos >= 0L)
        }
    }

    data class JointObservation(
        val jointId: Int,
        val position: Vec2,
        val confidence: Double,
        val visibility: Double = 1.0,
    ) {
        init {
            require(jointId >= 0)
            requireFinite(position)
            require(confidence in 0.0..1.0)
            require(visibility in 0.0..1.0)
        }

        val effectiveConfidence: Double
            get() = (confidence * visibility).coerceIn(0.0, 1.0)
    }

    data class CameraFrame(
        val cameraId: String,
        val timestampNanos: Long,
        val sequence: Long,
        val phaseOffsetNanos: Long = 0L,
        val cameraPose: Pose? = null,
        val intrinsics: CameraIntrinsics? = null,
        val observations: List<JointObservation>,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(timestampNanos >= 0L)
            require(sequence >= 0L)
            require(
                (cameraPose == null && intrinsics == null) ||
                    (cameraPose != null && intrinsics != null),
            )
            require(observations.isNotEmpty())
        }

        val effectiveTimestampNanos: Long
            get() = Math.addExact(timestampNanos, phaseOffsetNanos)
    }

    data class FusedJoint(
        val jointId: Int,
        val timestampNanos: Long,
        val position: Vec2,
        val confidence: Double,
        val uncertaintyPixels: Double,
        val supportingCameras: Int,
        val contributingSamples: Int,
        val temporalResidualPixels: Double,
        val predicted: Boolean,
    )

    data class FusedFrame(
        val timestampNanos: Long,
        val joints: List<FusedJoint>,
        val effectiveInputRateHz: Double,
        val cameraCount: Int,
        val temporalConsistency: Double,
        val spatialAgreement: Double,
        val uncertaintyScore: Double,
        val authoritative: Boolean,
        val warnings: List<String>,
    )

    data class Evidence(
        val cameraId: String,
        val timestampNanos: Long,
        val temporalDistanceNanos: Long,
        val weight: Double,
        val outlier: Boolean,
    )

    data class DetailedJoint(
        val fused: FusedJoint,
        val evidence: List<Evidence>,
    )

    data class DetailedFrame(
        val frame: FusedFrame,
        val joints: List<DetailedJoint>,
    )

    fun fuse(
        targetTimestampNanos: Long,
        frames: List<CameraFrame>,
    ): FusedFrame {
        return fuseDetailed(targetTimestampNanos, frames).frame
    }

    fun fuseDetailed(
        targetTimestampNanos: Long,
        frames: List<CameraFrame>,
    ): DetailedFrame {
        require(targetTimestampNanos >= 0L)

        if (!config.enabled) {
            return disabledFrame(targetTimestampNanos)
        }

        val prepared = sanitize(frames)
        if (prepared.isEmpty()) {
            return emptyFrame(targetTimestampNanos, "No usable camera frames.")
        }

        val cameraCount = prepared.map { it.cameraId }.distinct().size
        val jointIds = prepared.asSequence()
            .flatMap { it.observations.asSequence() }
            .map { it.jointId }
            .distinct()
            .sorted()
            .take(config.maximumJoints)
            .toList()

        val detailed = jointIds.mapNotNull { jointId ->
            fuseJoint(targetTimestampNanos, jointId, prepared)
        }

        val temporalConsistency =
            weightedMean(
                detailed.map {
                    it.fused.temporalResidualPixels to
                        it.fused.confidence
                },
            ).let {
                (1.0 - it / config.outlierDistancePixels)
                    .coerceIn(0.0, 1.0)
            }

        val spatialAgreement =
            weightedMean(
                detailed.map {
                    it.fused.uncertaintyPixels to
                        it.fused.confidence
                },
            ).let {
                (1.0 - it / config.outlierDistancePixels)
                    .coerceIn(0.0, 1.0)
            }

        val uncertaintyScore =
            weightedMean(
                detailed.map {
                    it.fused.uncertaintyPixels to
                        it.fused.confidence
                },
            ).let {
                (1.0 - it / config.outlierDistancePixels)
                    .coerceIn(0.0, 1.0)
            }

        val effectiveRate = effectiveInputRateHz(prepared)
        val warnings = buildList {
            if (cameraCount < 2) {
                add("Interleaved fusion has only one camera source.")
            }
            if (effectiveRate <= 0.0) {
                add("Effective input rate is unavailable.")
            }
            if (detailed.isEmpty()) {
                add("No joint has sufficient temporally valid evidence.")
            }
            if (temporalConsistency < 0.60) {
                add("Temporal disagreement exceeds the preferred envelope.")
            }
            if (spatialAgreement < 0.60) {
                add("Cross-view spatial disagreement exceeds the preferred envelope.")
            }
            if (detailed.any { it.fused.predicted }) {
                add("At least one joint uses bounded temporal prediction.")
            }
        }

        val authoritative =
            detailed.isNotEmpty() &&
                detailed.all {
                    it.fused.contributingSamples >=
                        config.minimumSamplesForAuthoritativeOutput &&
                        it.fused.confidence >= config.minimumConfidence &&
                        it.fused.uncertaintyPixels <
                        config.outlierDistancePixels
                } &&
                temporalConsistency >= 0.60 &&
                spatialAgreement >= 0.60

        return DetailedFrame(
            frame = FusedFrame(
                timestampNanos = targetTimestampNanos,
                joints = detailed.map { it.fused },
                effectiveInputRateHz = effectiveRate,
                cameraCount = cameraCount,
                temporalConsistency = temporalConsistency,
                spatialAgreement = spatialAgreement,
                uncertaintyScore = uncertaintyScore,
                authoritative = authoritative,
                warnings = warnings,
            ),
            joints = detailed,
        )
    }

    private fun fuseJoint(
        targetTimestampNanos: Long,
        jointId: Int,
        frames: List<CameraFrame>,
    ): DetailedJoint? {
        val samples = frames.asSequence()
            .mapNotNull { frame ->
                val observation = frame.observations
                    .firstOrNull { it.jointId == jointId }
                    ?: return@mapNotNull null

                val timestamp = frame.effectiveTimestampNanos
                val distance = abs(timestamp - targetTimestampNanos)
                if (distance > config.maximumTemporalWindowNanos) {
                    return@mapNotNull null
                }

                val confidence = observation.effectiveConfidence
                if (confidence < config.minimumConfidence) {
                    return@mapNotNull null
                }

                Sample(
                    cameraId = frame.cameraId,
                    timestampNanos = timestamp,
                    position = observation.position,
                    confidence = confidence,
                    temporalDistanceNanos = distance,
                    cameraPose = frame.cameraPose,
                    intrinsics = frame.intrinsics,
                )
            }
            .sortedWith(
                compareBy<Sample> { it.temporalDistanceNanos }
                    .thenByDescending { it.confidence }
                    .thenBy { it.cameraId },
            )
            .toList()

        if (samples.isEmpty()) return null

        val extrapolatingBeyondLatestObservation =
            samples.maxOfOrNull { it.timestampNanos }
                ?.let { targetTimestampNanos > it }
                ?: false

        val temporalEstimate =
            if (
                samples.size >= 3 &&
                    !extrapolatingBeyondLatestObservation
            ) {
                TemporalEstimate(
                    timestampNanos = targetTimestampNanos,
                    position = weightedTemporalPosition(samples),
                    predicted = false,
                )
            } else {
                estimateTemporalPosition(
                    targetTimestampNanos,
                    jointId,
                    frames,
                    samples,
                )
            }

        val selectedSamples =
            samples
                .sortedWith(
                    compareBy<Sample> { it.temporalDistanceNanos }
                        .thenByDescending { it.confidence }
                        .thenBy { it.cameraId },
                )
                .take(config.maximumCameras)

        val epipolarAdjustedSamples =
            selectedSamples.map { sample ->
                sample.copy(
                    confidence =
                        (
                            sample.confidence *
                                epipolarSupport(
                                    sample = sample,
                                    peers = selectedSamples,
                                )
                        ).coerceIn(0.0, 1.0),
                )
            }

        val residuals = epipolarAdjustedSamples.map {
            distance(it.position, temporalEstimate.position)
        }

        val robustScale = robustScale(residuals)
        val accepted =
            epipolarAdjustedSamples.mapIndexedNotNull { index, sample ->
            val residual = residuals[index]
            val threshold = max(
                config.huberTransitionPixels,
                robustScale * 3.0,
            )
            if (residual > config.outlierDistancePixels) {
                null
            } else {
                val robustWeight =
                    sample.confidence *
                        temporalWeight(sample.temporalDistanceNanos) *
                        huberWeight(residual, threshold)
                WeightedSample(sample, robustWeight, residual)
            }
        }

        if (accepted.isEmpty()) return null

        val fusedPosition = weightedPosition(accepted)
        val uncertainty = weightedUncertainty(accepted, fusedPosition)
        val supportingCameras = accepted.map { it.sample.cameraId }.distinct().size
        val meanResidual = weightedMean(
            accepted.map { it.residual to it.weight },
        )
        val confidence = (
            accepted.sumOf { it.weight } /
                accepted.size.toDouble()
            ).coerceIn(0.0, 1.0)

        val predicted =
            temporalEstimate.predicted &&
                abs(
                    temporalEstimate.timestampNanos -
                        targetTimestampNanos,
                ) <= config.maximumPredictionNanos

        return DetailedJoint(
            fused = FusedJoint(
                jointId = jointId,
                timestampNanos = targetTimestampNanos,
                position = fusedPosition,
                confidence = confidence,
                uncertaintyPixels = uncertainty,
                supportingCameras = supportingCameras,
                contributingSamples = accepted.size,
                temporalResidualPixels = meanResidual,
                predicted = predicted,
            ),
            evidence = accepted.map {
                Evidence(
                    cameraId = it.sample.cameraId,
                    timestampNanos = it.sample.timestampNanos,
                    temporalDistanceNanos =
                        it.sample.temporalDistanceNanos,
                    weight = it.weight,
                    outlier = it.residual >
                        config.huberTransitionPixels,
                )
            },
        )
    }

    private fun medianPosition(
        positions: List<Vec2>,
    ): Vec2 {
        if (positions.isEmpty()) return Vec2(0.0, 0.0)
        val xs = positions.map { it.x }.sorted()
        val ys = positions.map { it.y }.sorted()
        val mid = positions.size / 2
        val x =
            if (positions.size % 2 == 0) {
                (xs[mid - 1] + xs[mid]) * 0.5
            } else {
                xs[mid]
            }
        val y =
            if (positions.size % 2 == 0) {
                (ys[mid - 1] + ys[mid]) * 0.5
            } else {
                ys[mid]
            }
        return Vec2(x, y)
    }

    private fun estimateTemporalPosition(
        targetTimestampNanos: Long,
        jointId: Int,
        frames: List<CameraFrame>,
        samples: List<Sample>,
    ): TemporalEstimate {
        val before = samples
            .filter { it.timestampNanos <= targetTimestampNanos }
            .maxByOrNull { it.timestampNanos }
        val after = samples
            .filter { it.timestampNanos >= targetTimestampNanos }
            .minByOrNull { it.timestampNanos }

        val exactTimestampSamples =
            samples.filter {
                it.timestampNanos == targetTimestampNanos
            }

        if (exactTimestampSamples.size >= 2) {
            val xs =
                exactTimestampSamples.map { it.position.x }.sorted()
            val ys =
                exactTimestampSamples.map { it.position.y }.sorted()
            val medianIndex = (xs.size - 1) / 2
            return TemporalEstimate(
                timestampNanos = targetTimestampNanos,
                position = Vec2(
                    xs[medianIndex],
                    ys[medianIndex],
                ),
                predicted = false,
            )
        }

        if (before != null && after != null &&
            before.timestampNanos != after.timestampNanos
        ) {
            val interpolationDt =
                after.timestampNanos - before.timestampNanos
            val interpolationSpeed =
                if (interpolationDt <= 0L) {
                    Double.POSITIVE_INFINITY
                } else {
                    magnitude(
                        (after.position - before.position) *
                            (1_000_000_000.0 / interpolationDt.toDouble()),
                    )
                }
            if (
                !interpolationSpeed.isFinite() ||
                interpolationSpeed > config.maximumVelocityPixelsPerSecond
            ) {
                return TemporalEstimate(
                    timestampNanos = before.timestampNanos,
                    position = before.position,
                    predicted = false,
                )
            }

            val alpha =
                (
                    targetTimestampNanos - before.timestampNanos
                ).toDouble() /
                    (
                        after.timestampNanos -
                            before.timestampNanos
                    ).toDouble()

            return TemporalEstimate(
                timestampNanos = targetTimestampNanos,
                position = lerp(
                    before.position,
                    after.position,
                    alpha.coerceIn(0.0, 1.0),
                ),
                predicted = false,
            )
        }

        val latest = before ?: after ?: return TemporalEstimate(
            timestampNanos = targetTimestampNanos,
            position = Vec2(0.0, 0.0),
            predicted = false,
        )

        if (targetTimestampNanos <= latest.timestampNanos) {
            return TemporalEstimate(
                timestampNanos = latest.timestampNanos,
                position = latest.position,
                predicted = false,
            )
        }

        val history = frames
            .asSequence()
            .flatMap { frame ->
                frame.observations.asSequence()
                    .filter { it.jointId == jointId }
                    .map {
                        TimedPosition(
                            frame.effectiveTimestampNanos,
                            it.position,
                            it.effectiveConfidence,
                        )
                    }
            }
            .filter { it.confidence >= config.minimumConfidence }
            .sortedByDescending { it.timestampNanos }
            .take(4)
            .toList()

        if (history.size < 2) {
            return TemporalEstimate(
                timestampNanos = latest.timestampNanos,
                position = latest.position,
                predicted = false,
            )
        }

        val newest = history[0]
        val previous = history[1]
        val dt = newest.timestampNanos - previous.timestampNanos
        if (dt <= 0L) {
            return TemporalEstimate(
                timestampNanos = newest.timestampNanos,
                position = newest.position,
                predicted = false,
            )
        }

        val velocity = (
            newest.position - previous.position
            ) * (
                1_000_000_000.0 / dt.toDouble()
            )

        val speed = magnitude(velocity)
        if (!speed.isFinite() ||
            speed > config.maximumVelocityPixelsPerSecond
        ) {
            return TemporalEstimate(
                timestampNanos = newest.timestampNanos,
                position = newest.position,
                predicted = false,
            )
        }

        val predictionNanos =
            (
                targetTimestampNanos -
                    newest.timestampNanos
            ).coerceIn(0L, config.maximumPredictionNanos)

        return TemporalEstimate(
            timestampNanos =
                newest.timestampNanos + predictionNanos,
            position =
                newest.position +
                    velocity * (
                        predictionNanos /
                            1_000_000_000.0
                        ),
            predicted = predictionNanos > 0L,
        )
    }

    private fun sanitize(
        frames: List<CameraFrame>,
    ): List<CameraFrame> {
        val selectedCameras = frames
            .map { it.cameraId }
            .distinct()
            .sorted()
            .take(config.maximumCameras)
            .toSet()

        return frames.asSequence()
            .filter { it.cameraId in selectedCameras }
            .sortedWith(
                compareBy<CameraFrame> { it.effectiveTimestampNanos }
                    .thenBy { it.cameraId }
                    .thenBy { it.sequence },
            )
            .groupBy { it.cameraId }
            .values
            .flatMap { cameraFrames ->
                cameraFrames.takeLast(config.maximumFramesPerCamera)
            }
            .filter { it.observations.any { o ->
                o.effectiveConfidence >= config.minimumConfidence
            } }
            .toList()
    }

    private fun effectiveInputRateHz(
        frames: List<CameraFrame>,
    ): Double {
        val timestamps = frames
            .map { it.effectiveTimestampNanos }
            .distinct()
            .sorted()
        if (timestamps.size < 2) return 0.0
        val span = timestamps.last() - timestamps.first()
        if (span <= 0L) return 0.0
        return (
            (timestamps.size - 1).toDouble() /
                (span / 1_000_000_000.0)
            ).coerceAtMost(1000.0)
    }

    private val epipolarEngine =
        SymmetricEpipolarConsistencyEngine()

    private fun epipolarSupport(
        sample: Sample,
        peers: List<Sample>,
    ): Double {
        val pose = sample.cameraPose ?: return 1.0
        val intrinsics = sample.intrinsics ?: return 1.0

        val pairScores =
            peers
                .asSequence()
                .filter {
                    it.cameraId != sample.cameraId
                }
                .filter {
                    abs(
                        it.timestampNanos -
                            sample.timestampNanos,
                    ) <=
                        config.maximumEpipolarTemporalDeltaNanos
                }
                .sortedBy {
                    abs(
                        it.timestampNanos -
                            sample.timestampNanos,
                    )
                }
                .take(config.maximumEpipolarPeersPerSample)
                .mapNotNull { peer ->
                    val peerPose = peer.cameraPose ?: return@mapNotNull null
                    val peerIntrinsics =
                        peer.intrinsics
                            ?: return@mapNotNull null

                    runCatching {
                        epipolarEngine.evaluate(
                            left =
                                SymmetricEpipolarConsistencyEngine.Observation(
                                    cameraPose = pose,
                                    intrinsics = intrinsics,
                                    u = sample.position.x,
                                    v = sample.position.y,
                                    confidence = sample.confidence,
                                ),
                            right =
                                SymmetricEpipolarConsistencyEngine.Observation(
                                    cameraPose = peerPose,
                                    intrinsics = peerIntrinsics,
                                    u = peer.position.x,
                                    v = peer.position.y,
                                    confidence = peer.confidence,
                                ),
                        )
                    }.getOrNull()?.let {
                        if (it.accepted) {
                            it.score
                        } else {
                            0.0
                        }
                    }
                }
                .toList()

        if (pairScores.isEmpty()) return 1.0

        return (
            0.60 +
                0.40 * pairScores.average()
            ).coerceIn(0.20, 1.0)
    }

    private fun weightedTemporalPosition(
        samples: List<Sample>,
    ): Vec2 {
        if (samples.isEmpty()) {
            return Vec2(0.0, 0.0)
        }

        fun weightedMedian(values: List<Pair<Double, Double>>): Double {
            val ordered = values
                .filter { it.first.isFinite() && it.second.isFinite() && it.second > 0.0 }
                .sortedBy { it.first }
            if (ordered.isEmpty()) {
                return 0.0
            }
            val total = ordered.sumOf { it.second }
            var cumulative = 0.0
            for ((value, weight) in ordered) {
                cumulative += weight
                if (cumulative >= total * 0.5) {
                    return value
                }
            }
            return ordered.last().first
        }

        return Vec2(
            weightedMedian(
                samples.map {
                    it.position.x to it.confidence
                },
            ),
            weightedMedian(
                samples.map {
                    it.position.y to it.confidence
                },
            ),
        )
    }

    private fun weightedPosition(
        samples: List<WeightedSample>,
    ): Vec2 {
        var x = 0.0
        var y = 0.0
        var weight = 0.0
        for (sample in samples) {
            x += sample.sample.position.x * sample.weight
            y += sample.sample.position.y * sample.weight
            weight += sample.weight
        }
        if (weight <= 1e-12) return samples.first().sample.position
        return Vec2(x / weight, y / weight)
    }

    private fun weightedUncertainty(
        samples: List<WeightedSample>,
        center: Vec2,
    ): Double {
        var variance = 0.0
        var weight = 0.0
        for (sample in samples) {
            variance +=
                sample.weight *
                    distance(sample.sample.position, center).let {
                        it * it
                    }
            weight += sample.weight
        }
        if (weight <= 1e-12) return Double.POSITIVE_INFINITY
        return sqrt(variance / weight)
    }

    private fun temporalWeight(distanceNanos: Long): Double {
        val base =
            (
                1.0 -
                    distanceNanos.toDouble() /
                        config.maximumTemporalWindowNanos.toDouble()
                ).coerceIn(0.0, 1.0)
        return if (
            distanceNanos <= config.maximumSampleSkewNanos
        ) {
            base
        } else {
            base * 0.25
        }
    }

    private fun huberWeight(
        residual: Double,
        transition: Double,
    ): Double {
        if (residual <= transition) return 1.0
        return transition / residual.coerceAtLeast(transition)
    }

    private fun robustScale(residuals: List<Double>): Double {
        if (residuals.isEmpty()) return 0.0
        val sorted = residuals.sorted()
        val median = percentile(sorted, 0.5)
        val deviations = sorted.map { abs(it - median) }.sorted()
        return max(1e-6, percentile(deviations, 0.5) * 1.4826)
    }

    private fun weightedMean(
        values: List<Pair<Double, Double>>,
    ): Double {
        var numerator = 0.0
        var denominator = 0.0
        for ((value, weight) in values) {
            if (!value.isFinite() || !weight.isFinite() || weight <= 0.0) {
                continue
            }
            numerator += value * weight
            denominator += weight
        }
        return if (denominator <= 1e-12) {
            Double.POSITIVE_INFINITY
        } else {
            numerator / denominator
        }
    }

    private fun percentile(
        sorted: List<Double>,
        probability: Double,
    ): Double {
        if (sorted.isEmpty()) return Double.POSITIVE_INFINITY
        if (sorted.size == 1) return sorted[0]
        val position =
            probability.coerceIn(0.0, 1.0) *
                (sorted.size - 1)
        val lower = position.toInt()
        val upper = min(lower + 1, sorted.lastIndex)
        return sorted[lower] +
            (sorted[upper] - sorted[lower]) *
            (position - lower)
    }

    private fun distance(a: Vec2, b: Vec2): Double {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun magnitude(v: Vec2): Double =
        sqrt(v.x * v.x + v.y * v.y)

    private fun lerp(a: Vec2, b: Vec2, alpha: Double): Vec2 =
        Vec2(
            a.x + (b.x - a.x) * alpha,
            a.y + (b.y - a.y) * alpha,
        )

    private data class Sample(
        val cameraId: String,
        val timestampNanos: Long,
        val position: Vec2,
        val confidence: Double,
        val temporalDistanceNanos: Long,
        val cameraPose: Pose? = null,
        val intrinsics: CameraIntrinsics? = null,
    )

    private data class WeightedSample(
        val sample: Sample,
        val weight: Double,
        val residual: Double,
    )

    private data class TimedPosition(
        val timestampNanos: Long,
        val position: Vec2,
        val confidence: Double,
    )

    private data class TemporalEstimate(
        val timestampNanos: Long,
        val position: Vec2,
        val predicted: Boolean,
    )

    private fun disabledFrame(
        timestampNanos: Long,
    ): DetailedFrame =
        DetailedFrame(
            frame = FusedFrame(
                timestampNanos = timestampNanos,
                joints = emptyList(),
                effectiveInputRateHz = 0.0,
                cameraCount = 0,
                temporalConsistency = 1.0,
                spatialAgreement = 1.0,
                uncertaintyScore = 1.0,
                authoritative = true,
                warnings = listOf("Interleaved temporal fusion disabled by capture policy."),
            ),
            joints = emptyList(),
        )

    private fun emptyFrame(
        timestampNanos: Long,
        warning: String,
    ): DetailedFrame =
        DetailedFrame(
            frame = FusedFrame(
                timestampNanos = timestampNanos,
                joints = emptyList(),
                effectiveInputRateHz = 0.0,
                cameraCount = 0,
                temporalConsistency = 0.0,
                spatialAgreement = 0.0,
                uncertaintyScore = 0.0,
                authoritative = false,
                warnings = listOf(warning),
            ),
            joints = emptyList(),
        )

    private companion object {
        fun requireFinite(position: Vec2) {
            require(
                position.x.isFinite() &&
                    position.y.isFinite(),
            ) {
                "position must be finite"
            }
        }
    }
}

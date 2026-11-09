package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Production-grade scheduler for sparse/interleaved multi-camera temporal sampling.
 *
 * The engine does not hallucinate frames. It only accepts timestamped observations and derives
 * executable windows from measured camera samples. A window is publishable only when temporal
 * coverage, camera diversity and sample skew stay inside the configured contract.
 *
 * The design is intentionally bounded:
 * - one bounded history deque per camera;
 * - one bounded candidate window;
 * - deterministic oldest-first eviction;
 * - no unbounded coroutine/job creation;
 * - no synthetic timestamps.
 *
 * It is suitable for large federated rigs because it performs planning over metadata rather than
 * retaining image buffers.
 */
class InterleavedTemporalSamplingEngine(
    initialConfig: Config = Config.production(),
) : AutoCloseable {

    data class Config(
        val maximumCameras: Int = 128,
        val historyPerCamera: Int = 8,
        val targetCameraCountPerWindow: Int = 4,
        val nominalCameraFrameIntervalMillis: Double = 16.6667,
        val maximumWindowMillis: Double = 45.0,
        val predictionHorizonMillis: Double = 18.0,
        val maximumSampleSkewMillis: Double = 25.0,
        val maximumTimingResidualMillis: Double = 6.0,
        val minimumCoverageRatio: Double = 0.72,
        val minimumCameraDiversity: Double = 0.65,
        val minimumTemporalHealth: Double = 0.62,
        val maximumTimestampGapMultiplier: Double = 3.0,
    ) {
        init {
            require(maximumCameras in 2..128)
            require(historyPerCamera >= 2)
            require(targetCameraCountPerWindow in 2..32)
            require(nominalCameraFrameIntervalMillis > 0.0)
            require(maximumWindowMillis >= nominalCameraFrameIntervalMillis)
            require(predictionHorizonMillis >= 0.0)
            require(maximumSampleSkewMillis > 0.0)
            require(maximumTimingResidualMillis >= 0.0)
            require(minimumCoverageRatio in 0.0..1.0)
            require(minimumCameraDiversity in 0.0..1.0)
            require(minimumTemporalHealth in 0.0..1.0)
            require(maximumTimestampGapMultiplier >= 1.0)
        }

        companion object {
            fun production(): Config = Config()

            fun lowLatency(): Config =
                Config(
                    maximumWindowMillis = 32.0,
                    predictionHorizonMillis = 10.0,
                    maximumSampleSkewMillis = 14.0,
                    minimumCoverageRatio = 0.80,
                    minimumTemporalHealth = 0.72,
                )
        }
    }

    data class CameraSample(
        val cameraId: String,
        val timestampNanos: Long,
        val sequence: Long,
        val sector: Int = 0,
        val confidence: Double,
        val synchronized: Boolean = true,
        val timingResidualMillis: Double = 0.0,
        val geometricQuality: Double = 1.0,
        val outlierScore: Double = 0.0,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(timestampNanos >= 0L)
            require(sequence >= 0L)
            require(sector >= 0)
            require(confidence in 0.0..1.0)
            require(timingResidualMillis.isFinite() && timingResidualMillis >= 0.0)
            require(geometricQuality in 0.0..1.0)
            require(outlierScore in 0.0..1.0)
        }

        val effectiveQuality: Double
            get() =
                (
                    0.38 * confidence +
                        0.24 * if (synchronized) 1.0 else 0.0 +
                        0.22 * geometricQuality +
                        0.16 * (1.0 - outlierScore)
                ).coerceIn(0.0, 1.0)
    }

    data class WindowObservation(
        val cameraId: String,
        val timestampNanos: Long,
        val deltaFromTargetMillis: Double,
        val quality: Double,
        val timingResidualMillis: Double,
        val predictedMillis: Double,
    )

    data class WindowPlan(
        val targetTimestampNanos: Long,
        val observations: List<WindowObservation>,
        val contributingCameraIds: List<String>,
        val cameraCount: Int,
        val coverageRatio: Double,
        val cameraDiversity: Double,
        val temporalHealth: Double,
        val maximumAbsoluteSkewMillis: Double,
        val meanAbsoluteSkewMillis: Double,
        val outlierRatio: Double,
        val effectiveSamplingMultiplier: Double,
        val publishable: Boolean,
        val warnings: List<String>,
    ) {
        val effectiveOutputIntervalMillis: Double
            get() =
                if (effectiveSamplingMultiplier <= 0.0) {
                    Double.POSITIVE_INFINITY
                } else {
                    1_000.0 /
                        (
                            effectiveSamplingMultiplier *
                                1_000.0 /
                                16.6667
                        )
                }
    }

    data class RuntimeSummary(
        val latestTimestampNanos: Long,
        val acceptedSamples: Long,
        val droppedSamples: Long,
        val activeCameras: Int,
        val latestPlan: WindowPlan?,
        val coverageRatio: Double,
        val temporalHealth: Double,
        val effectiveSamplingMultiplier: Double,
        val publishableWindows: Long,
        val rejectedWindows: Long,
        val warnings: List<String>,
    )

    private val config = initialConfig
    private val history =
        LinkedHashMap<String, ArrayDeque<CameraSample>>()
    private var acceptedSamples = 0L
    private var droppedSamples = 0L
    private var publishableWindows = 0L
    private var rejectedWindows = 0L
    private var latestTimestampNanos = 0L
    private var latestPlan: WindowPlan? = null
    private var closed = false

    @Synchronized
    fun ingest(sample: CameraSample): WindowPlan? {
        check(!closed) { "InterleavedTemporalSamplingEngine is closed" }
        latestTimestampNanos =
            max(
                latestTimestampNanos,
                sample.timestampNanos,
            )

        val queue =
            history.getOrPut(sample.cameraId) {
                if (history.size >= config.maximumCameras) {
                    val victim =
                        history.entries
                            .minByOrNull { entry ->
                                entry.value.lastOrNull()?.timestampNanos
                                    ?: Long.MIN_VALUE
                            }
                            ?.key
                    if (victim != null) history.remove(victim)
                }
                ArrayDeque()
            }

        if (
            queue.lastOrNull()?.let {
                sample.timestampNanos <= it.timestampNanos
            } == true
        ) {
            droppedSamples++
            return latestPlan
        }

        queue.addLast(sample)
        while (queue.size > config.historyPerCamera) {
            queue.removeFirst()
        }

        acceptedSamples++
        val plan =
            buildPlan(sample.timestampNanos)
        latestPlan = plan

        if (plan.publishable) {
            publishableWindows++
        } else {
            rejectedWindows++
        }

        return plan
    }

    @Synchronized
    fun latestPlan(): WindowPlan? = latestPlan

    @Synchronized
    fun summary(): RuntimeSummary {
        val plan = latestPlan
        return RuntimeSummary(
            latestTimestampNanos = latestTimestampNanos,
            acceptedSamples = acceptedSamples,
            droppedSamples = droppedSamples,
            activeCameras = history.size,
            latestPlan = plan,
            coverageRatio = plan?.coverageRatio ?: 0.0,
            temporalHealth = plan?.temporalHealth ?: 0.0,
            effectiveSamplingMultiplier =
                plan?.effectiveSamplingMultiplier ?: 1.0,
            publishableWindows = publishableWindows,
            rejectedWindows = rejectedWindows,
            warnings = plan?.warnings ?: emptyList(),
        )
    }

    @Synchronized
    fun reset() {
        check(!closed) { "InterleavedTemporalSamplingEngine is closed" }
        history.clear()
        acceptedSamples = 0L
        droppedSamples = 0L
        publishableWindows = 0L
        rejectedWindows = 0L
        latestTimestampNanos = 0L
        latestPlan = null
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        history.clear()
        latestPlan = null
    }

    private fun buildPlan(targetTimestampNanos: Long): WindowPlan {
        val maxWindowNanos =
            (
                config.maximumWindowMillis *
                    1_000_000.0
            ).toLong()

        val targetSamples =
            history.values
                .mapNotNull { queue ->
                    selectClosest(
                        queue = queue,
                        targetTimestampNanos = targetTimestampNanos,
                        maximumWindowNanos = maxWindowNanos,
                    )
                }
                .sortedByDescending {
                    it.quality
                }

        val bounded =
            targetSamples
                .distinctBy { it.sample.cameraId }
                .take(config.targetCameraCountPerWindow)

        val contributingIds =
            bounded.map { it.sample.cameraId }

        val observations =
            bounded.map { selection ->
                WindowObservation(
                    cameraId = selection.sample.cameraId,
                    timestampNanos =
                        selection.sample.timestampNanos,
                    deltaFromTargetMillis =
                        (
                            selection.sample.timestampNanos -
                                targetTimestampNanos
                        ).toDouble() /
                            1_000_000.0,
                    quality = selection.sample.effectiveQuality,
                    timingResidualMillis =
                        selection.sample.timingResidualMillis,
                    predictedMillis =
                        abs(
                            selection.sample.timestampNanos -
                                targetTimestampNanos,
                        ).toDouble() /
                            1_000_000.0,
                )
            }

        val coverageRatio =
            coverageRatio(
                cameraCount = bounded.size,
                targetTimestampNanos = targetTimestampNanos,
            )

        val cameraDiversity =
            cameraDiversity(bounded)

        val maximumSkew =
            observations
                .maxOfOrNull {
                    abs(it.deltaFromTargetMillis)
                } ?: Double.POSITIVE_INFINITY

        val meanSkew =
            observations
                .map {
                    abs(it.deltaFromTargetMillis)
                }
                .averageOrZero()

        val outlierRatio =
            if (observations.isEmpty()) {
                1.0
            } else {
                observations.count {
                    abs(it.deltaFromTargetMillis) >
                        config.maximumSampleSkewMillis ||
                        it.timingResidualMillis >
                        config.maximumTimingResidualMillis ||
                        it.quality < 0.35
                }.toDouble() /
                    observations.size.toDouble()
            }

        val skewHealth =
            (
                1.0 -
                    maximumSkew /
                        config.maximumSampleSkewMillis
            ).coerceIn(0.0, 1.0)

        val outlierHealth =
            (1.0 - outlierRatio).coerceIn(0.0, 1.0)

        val qualityHealth =
            observations
                .map { it.quality }
                .averageOrZero()

        val temporalHealth =
            (
                0.30 * coverageRatio +
                    0.24 * cameraDiversity +
                    0.22 * skewHealth +
                    0.14 * outlierHealth +
                    0.10 * qualityHealth
            ).coerceIn(0.0, 1.0)

        val rawMultiplier =
            if (bounded.isEmpty()) {
                1.0
            } else {
                min(
                    bounded.size.toDouble(),
                    (
                        1.0 +
                            (
                                bounded.size -
                                    1
                            ) * coverageRatio *
                                cameraDiversity
                        )
                    .coerceIn(1.0, config.maximumCameras.toDouble()),
                )
            }

        val healthMultiplier =
            (
                0.55 +
                    0.45 * temporalHealth
            ).coerceIn(0.55, 1.0)

        val effectiveMultiplier =
            (
                rawMultiplier *
                    healthMultiplier
            ).coerceIn(1.0, config.maximumCameras.toDouble())

        val warnings =
            buildList {
                if (bounded.size < config.targetCameraCountPerWindow) {
                    add("INTERLEAVED_CAMERA_COVERAGE_BELOW_TARGET")
                }
                if (coverageRatio < config.minimumCoverageRatio) {
                    add("INTERLEAVED_TEMPORAL_COVERAGE_LOW")
                }
                if (cameraDiversity < config.minimumCameraDiversity) {
                    add("INTERLEAVED_CAMERA_DIVERSITY_LOW")
                }
                if (
                    maximumSkew >
                        config.maximumSampleSkewMillis
                ) {
                    add("INTERLEAVED_SAMPLE_SKEW_EXCEEDED")
                }
                if (outlierRatio > 0.20) {
                    add("INTERLEAVED_OUTLIER_RATE_HIGH")
                }
                if (
                    temporalHealth <
                        config.minimumTemporalHealth
                ) {
                    add("INTERLEAVED_TEMPORAL_HEALTH_BELOW_CONTRACT")
                }
                if (
                    config.predictionHorizonMillis >
                        config.maximumWindowMillis
                ) {
                    add("PREDICTION_HORIZON_EXCEEDS_WINDOW")
                }
            }.distinct()

        val publishable =
            bounded.size >=
                min(
                    config.targetCameraCountPerWindow,
                    history.size.coerceAtLeast(2),
                ) &&
                coverageRatio >= config.minimumCoverageRatio &&
                cameraDiversity >= config.minimumCameraDiversity &&
                maximumSkew <= config.maximumSampleSkewMillis &&
                temporalHealth >= config.minimumTemporalHealth

        return WindowPlan(
            targetTimestampNanos = targetTimestampNanos,
            observations = observations,
            contributingCameraIds = contributingIds,
            cameraCount = bounded.size,
            coverageRatio = coverageRatio,
            cameraDiversity = cameraDiversity,
            temporalHealth = temporalHealth,
            maximumAbsoluteSkewMillis = maximumSkew,
            meanAbsoluteSkewMillis = meanSkew,
            outlierRatio = outlierRatio,
            effectiveSamplingMultiplier = effectiveMultiplier,
            publishable = publishable,
            warnings = warnings,
        )
    }

    private data class Selection(
        val sample: CameraSample,
        val temporalDistanceMillis: Double,
    )

    private fun selectClosest(
        queue: ArrayDeque<CameraSample>,
        targetTimestampNanos: Long,
        maximumWindowNanos: Long,
    ): Selection? {
        var best: CameraSample? = null
        var bestDistance = Long.MAX_VALUE

        queue.forEach { sample ->
            val distance =
                abs(
                    sample.timestampNanos -
                        targetTimestampNanos,
                )

            if (distance < bestDistance) {
                best = sample
                bestDistance = distance
            }
        }

        val selected = best ?: return null
        if (bestDistance > maximumWindowNanos) {
            return null
        }

        return Selection(
            sample = selected,
            temporalDistanceMillis =
                bestDistance.toDouble() / 1_000_000.0,
        )
    }

    private fun coverageRatio(
        cameraCount: Int,
        targetTimestampNanos: Long,
    ): Double {
        if (history.isEmpty()) return 0.0

        val expectedIntervalNanos =
            (
                config.nominalCameraFrameIntervalMillis *
                    1_000_000.0
            ).toLong().coerceAtLeast(1L)

        val active =
            history.values.count { queue ->
                val latest = queue.lastOrNull() ?: return@count false
                val age =
                    targetTimestampNanos -
                        latest.timestampNanos
                age <=
                    expectedIntervalNanos *
                        config.maximumTimestampGapMultiplier
            }

        return (
            active.toDouble() /
                min(
                    history.size,
                    config.targetCameraCountPerWindow,
                ).coerceAtLeast(1)
        ).coerceIn(0.0, 1.0)
    }

    private fun cameraDiversity(
        bounded: List<Selection>,
    ): Double {
        if (bounded.size <= 1) return 0.0

        val distinctSectors =
            bounded
                .map { it.sample.sector }
                .distinct()
                .size
                .toDouble()

        val sectorCoverage =
            (
                distinctSectors /
                    min(
                        8.0,
                        bounded.size.toDouble(),
                    )
            ).coerceIn(0.0, 1.0)

        val pairwiseSectorSpread =
            bounded
                .flatMapIndexed { index, first ->
                    bounded.drop(index + 1).map { second ->
                        abs(
                            first.sample.sector -
                                second.sample.sector,
                        )
                    }
                }
                .averageOrZero()
                .coerceIn(0.0, 8.0) /
                8.0

        return (
            0.60 * sectorCoverage +
                0.40 * pairwiseSectorSpread
        ).coerceIn(0.0, 1.0)
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}

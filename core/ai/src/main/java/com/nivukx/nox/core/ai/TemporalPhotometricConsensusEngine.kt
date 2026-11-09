package com.nivukx.nox.core.ai

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Bounded multi-frame consensus over SparsePhotometricMotionFieldEngine evidence.
 *
 * The underlying flow estimator remains the measurement source. This wrapper adds a short temporal
 * memory so one-frame correspondence failures, brief occlusions, or motion-boundary ambiguities do
 * not immediately become landmark-driving evidence.
 *
 * The consensus is intentionally local and lightweight:
 * - at most four frame summaries are retained;
 * - vector correspondence uses normalized image position;
 * - robust median/MAD filtering rejects temporal spikes;
 * - confidence combines temporal agreement with source quality;
 * - warm-up is explicit;
 * - stale and regressed timestamps reset the temporal window.
 */
class TemporalPhotometricConsensusEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        val maximumHistoryFrames: Int = 4,
        val minimumFramesForConsensus: Int = 3,
        val maximumTimestampGapNanos: Long = 120_000_000L,
        val positionMatchRadiusNormalized: Double = 0.045,
        val madScale: Double = 3.25,
        val maximumDisplacementResidualPixels: Double = 4.0,
        val temporalAgreementWeight: Double = 0.34,
        val sourceConfidenceWeight: Double = 0.26,
        val sourceConsistencyWeight: Double = 0.20,
        val textureWeight: Double = 0.10,
        val subpixelWeight: Double = 0.10,
    ) {
        init {
            require(maximumHistoryFrames in 2..8)
            require(
                minimumFramesForConsensus in
                    2..maximumHistoryFrames,
            )
            require(maximumTimestampGapNanos > 0L)
            require(positionMatchRadiusNormalized > 0.0)
            require(madScale > 0.0)
            require(maximumDisplacementResidualPixels > 0.0)
            listOf(
                temporalAgreementWeight,
                sourceConfidenceWeight,
                sourceConsistencyWeight,
                textureWeight,
                subpixelWeight,
            ).forEach {
                require(it.isFinite() && it >= 0.0)
            }
        }
    }

    data class Vector(
        val normalizedX: Double,
        val normalizedY: Double,
        val dxPixels: Double,
        val dyPixels: Double,
        val confidence: Double,
        val temporalAgreement: Double,
        val sourceConfidence: Double,
        val sourceConsistency: Double,
        val gradientAgreement: Double,
        val forwardBackwardConsistency: Double,
        val photometricError: Double,
        val textureStrength: Double,
        val subpixelRefinedRatio: Double,
        val temporalResidualPixels: Double,
    )

    data class Result(
        val timestampNanos: Long,
        val valid: Boolean,
        val sourceFrameCount: Int,
        val temporalConsensusScore: Double,
        val rejectedTemporalSpikes: Int,
        val vectors: List<Vector>,
        val warnings: List<String>,
    ) {
        companion object {
            fun empty(
                timestampNanos: Long,
            ): Result =
                Result(
                    timestampNanos = timestampNanos,
                    valid = false,
                    sourceFrameCount = 0,
                    temporalConsensusScore = 0.0,
                    rejectedTemporalSpikes = 0,
                    vectors = emptyList(),
                    warnings = emptyList(),
                )
        }
    }

    private data class Frame(
        val timestampNanos: Long,
        val result: SparsePhotometricMotionFieldEngine.Result,
    )

    private var history = ArrayDeque<Frame>()
    private var lastTimestampNanos = Long.MIN_VALUE
    private var closed = false

    @Synchronized
    fun fuse(
        source: SparsePhotometricMotionFieldEngine.Result,
    ): Result {
        check(!closed)

        if (
            lastTimestampNanos != Long.MIN_VALUE &&
            (
                source.timestampNanos <=
                    lastTimestampNanos ||
                    source.timestampNanos -
                        lastTimestampNanos >
                        config.maximumTimestampGapNanos
                )
        ) {
            history.clear()
        }

        lastTimestampNanos =
            source.timestampNanos

        if (source.valid) {
            history.addLast(
                Frame(
                    timestampNanos = source.timestampNanos,
                    result = source,
                ),
            )
        }

        while (
            history.size >
                config.maximumHistoryFrames
        ) {
            history.removeFirst()
        }

        if (
            history.size <
                config.minimumFramesForConsensus
        ) {
            return Result(
                timestampNanos = source.timestampNanos,
                valid = false,
                sourceFrameCount = history.size,
                temporalConsensusScore = 0.0,
                rejectedTemporalSpikes = 0,
                vectors = source.vectors.map {
                    Vector(
                        normalizedX = it.normalizedX,
                        normalizedY = it.normalizedY,
                        dxPixels = it.dxPixels,
                        dyPixels = it.dyPixels,
                        confidence = it.confidence * 0.72,
                        temporalAgreement = 0.0,
                        sourceConfidence = it.confidence,
                        sourceConsistency =
                            it.forwardBackwardConsistency,
                        gradientAgreement =
                            it.gradientAgreement,
                        forwardBackwardConsistency =
                            it.forwardBackwardConsistency,
                        photometricError =
                            it.photometricError,
                        textureStrength =
                            it.textureStrength,
                        subpixelRefinedRatio =
                            if (it.subpixelRefined) 1.0 else 0.0,
                        temporalResidualPixels = 0.0,
                    )
                },
                warnings =
                    listOf(
                        "PHOTOMETRIC_TEMPORAL_CONSENSUS_WARMUP",
                    ),
            )
        }

        val currentVectors =
            source.vectors

        if (currentVectors.isEmpty()) {
            return Result(
                timestampNanos = source.timestampNanos,
                valid = false,
                sourceFrameCount = history.size,
                temporalConsensusScore = 0.0,
                rejectedTemporalSpikes = 0,
                vectors = emptyList(),
                warnings =
                    listOf(
                        "PHOTOMETRIC_TEMPORAL_CONSENSUS_NO_CURRENT_VECTORS",
                    ),
            )
        }

        val retained =
            ArrayList<Vector>(
                currentVectors.size,
            )
        var rejected =
            0

        currentVectors.forEach { current ->
            val temporalSamples =
                history.mapNotNull { frame ->
                    val candidate =
                        frame.result.vectors
                            .mapNotNull { vector ->
                                val distance =
                                    hypotNormalized(
                                        current.normalizedX -
                                            vector.normalizedX,
                                        current.normalizedY -
                                            vector.normalizedY,
                                    )
                                if (
                                    distance <=
                                        config.positionMatchRadiusNormalized
                                ) {
                                    vector to distance
                                } else {
                                    null
                                }
                            }
                            .minByOrNull { it.second }
                            ?: return@mapNotNull null

                    candidate.first
                }

            if (
                temporalSamples.size <
                    config.minimumFramesForConsensus
            ) {
                retained +=
                    Vector(
                        normalizedX = current.normalizedX,
                        normalizedY = current.normalizedY,
                        dxPixels = current.dxPixels,
                        dyPixels = current.dyPixels,
                        confidence = current.confidence * 0.74,
                        temporalAgreement = 0.0,
                        sourceConfidence = current.confidence,
                        sourceConsistency =
                            current.forwardBackwardConsistency,
                        gradientAgreement =
                            current.gradientAgreement,
                        forwardBackwardConsistency =
                            current.forwardBackwardConsistency,
                        photometricError =
                            current.photometricError,
                        textureStrength =
                            current.textureStrength,
                        subpixelRefinedRatio =
                            if (current.subpixelRefined) 1.0 else 0.0,
                        temporalResidualPixels = 0.0,
                    )
                return@forEach
            }

            val dxValues =
                temporalSamples.map {
                    it.dxPixels
                }
            val dyValues =
                temporalSamples.map {
                    it.dyPixels
                }

            val medianDx =
                median(dxValues)
            val medianDy =
                median(dyValues)
            val residuals =
                temporalSamples.map { sample ->
                    hypot(
                        sample.dxPixels -
                            medianDx,
                        sample.dyPixels -
                            medianDy,
                    )
                }
            val residualMedian =
                median(residuals)
            val residualMad =
                median(
                    residuals.map {
                        abs(
                            it -
                                residualMedian,
                        )
                    },
                )

            val robustLimit =
                max(
                    config.maximumDisplacementResidualPixels,
                    residualMedian +
                        config.madScale *
                            max(
                                residualMad,
                                0.15,
                            ),
                )

            val inliers =
                temporalSamples.filter { sample ->
                    hypot(
                        sample.dxPixels -
                            medianDx,
                        sample.dyPixels -
                            medianDy,
                    ) <= robustLimit
                }

            val spikeCount =
                temporalSamples.size -
                    inliers.size

            if (
                inliers.size <
                    config.minimumFramesForConsensus
            ) {
                rejected +=
                    max(
                        1,
                        spikeCount,
                    )
                return@forEach
            }

            val fusedDx =
                median(
                    inliers.map {
                        it.dxPixels
                    },
                )
            val fusedDy =
                median(
                    inliers.map {
                        it.dyPixels
                    },
                )

            val temporalResidual =
                inliers.map {
                    hypot(
                        it.dxPixels -
                            fusedDx,
                        it.dyPixels -
                            fusedDy,
                    )
                }.let(::median)

            val temporalAgreement =
                (
                    1.0 -
                        temporalResidual /
                            config.maximumDisplacementResidualPixels
                ).coerceIn(
                    0.0,
                    1.0,
                )

            val sourceConfidence =
                inliers.map {
                    it.confidence
                }.average().coerceIn(
                    0.0,
                    1.0,
                )

            val sourceConsistency =
                inliers.map {
                    it.forwardBackwardConsistency
                }.average().coerceIn(
                    0.0,
                    1.0,
                )

            val gradientAgreement =
                inliers.map {
                    it.gradientAgreement
                }.average().coerceIn(
                    0.0,
                    1.0,
                )

            val forwardBackwardConsistency =
                inliers.map {
                    it.forwardBackwardConsistency
                }.average().coerceIn(
                    0.0,
                    1.0,
                )

            val photometricError =
                inliers.map {
                    it.photometricError
                }.average().coerceAtLeast(0.0)

            val textureStrength =
                inliers.map {
                    it.textureStrength
                }.average().coerceIn(
                    0.0,
                    1.0,
                )

            val subpixelRatio =
                inliers.count {
                    it.subpixelRefined
                }.toDouble() /
                    inliers.size.toDouble()

            val confidence =
                (
                    temporalAgreement *
                        config.temporalAgreementWeight +
                        sourceConfidence *
                        config.sourceConfidenceWeight +
                        sourceConsistency *
                        config.sourceConsistencyWeight +
                        textureStrength *
                        config.textureWeight +
                        subpixelRatio *
                        config.subpixelWeight
                ).coerceIn(
                    0.0,
                    1.0,
                )

            retained +=
                Vector(
                    normalizedX = current.normalizedX,
                    normalizedY = current.normalizedY,
                    dxPixels = fusedDx,
                    dyPixels = fusedDy,
                    confidence = confidence,
                    temporalAgreement =
                        temporalAgreement,
                    sourceConfidence =
                        sourceConfidence,
                    sourceConsistency =
                        sourceConsistency,
                    gradientAgreement =
                        gradientAgreement,
                    forwardBackwardConsistency =
                        forwardBackwardConsistency,
                    photometricError =
                        photometricError,
                    textureStrength =
                        textureStrength,
                    subpixelRefinedRatio =
                        subpixelRatio,
                    temporalResidualPixels =
                        temporalResidual,
                )

            rejected += spikeCount
        }

        val consensus =
            retained
                .map { it.temporalAgreement }
                .averageOrZero()

        val warnings =
            buildList {
                if (rejected > 0) {
                    add(
                        "PHOTOMETRIC_TEMPORAL_SPIKES_REJECTED",
                    )
                }
                if (consensus < 0.55) {
                    add(
                        "PHOTOMETRIC_TEMPORAL_CONSENSUS_LOW",
                    )
                }
            }

        return Result(
            timestampNanos =
                source.timestampNanos,
            valid =
                retained.isNotEmpty() &&
                    consensus >= 0.45,
            sourceFrameCount = history.size,
            temporalConsensusScore =
                consensus.coerceIn(
                    0.0,
                    1.0,
                ),
            rejectedTemporalSpikes =
                rejected,
            vectors = retained,
            warnings = warnings.distinct(),
        )
    }

    @Synchronized
    fun reset() {
        history.clear()
        lastTimestampNanos = Long.MIN_VALUE
    }

    @Synchronized
    override fun close() {
        if (closed) return
        reset()
        closed = true
    }

    private fun median(
        values: List<Double>,
    ): Double {
        if (values.isEmpty()) return 0.0
        val sorted =
            values.sorted()
        val middle =
            sorted.size / 2
        return if (
            sorted.size % 2 == 0
        ) {
            (
                sorted[middle - 1] +
                    sorted[middle]
                ) * 0.5
        } else {
            sorted[middle]
        }
    }

    private fun averageOrZero(
        values: List<Double>,
    ): Double =
        if (values.isEmpty()) 0.0
        else values.average()

    private fun hypotNormalized(
        x: Double,
        y: Double,
    ): Double =
        hypot(
            x,
            y,
        )

    private fun hypot(
        x: Double,
        y: Double,
    ): Double =
        sqrt(
            x * x +
                y * y,
        )
}

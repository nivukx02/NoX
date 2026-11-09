package com.nivukx.nox.core.ai

import android.graphics.ImageFormat
import android.media.Image
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * High-end deterministic photometric motion evidence engine.
 *
 * This layer deliberately remains a measurement authority rather than a learned optical-flow
 * replacement. It is designed for the NoX stack as a low-allocation temporal evidence stream that
 * can operate beside pose, IMU, synchronization and reconstruction.
 *
 * The implementation upgrades the original integer-pixel matcher with:
 *  - exposure-normalized patch comparison,
 *  - robust Huber residuals,
 *  - gradient/orientation agreement,
 *  - subpixel quadratic refinement around the best integer match,
 *  - texture-adaptive confidence,
 *  - forward/backward consistency when a usable prior exists,
 *  - robust global-motion estimation,
 *  - explicit camera-motion versus differential-motion evidence,
 *  - bounded sample storage,
 *  - strict timestamp monotonicity,
 *  - reusable frame sampling buffers,
 *  - no learned-accuracy claims.
 *
 * Photometric evidence cannot establish metric millimeter accuracy by itself. It is used to improve
 * temporal observability and camera-motion suppression while the calibrated geometric engines remain
 * responsible for 3D measurement.
 */
class SparsePhotometricMotionFieldEngine(
    initialConfig: Config = Config(),
) : AutoCloseable {

    @Volatile
    private var config: Config = initialConfig
 : AutoCloseable {

    data class Config(
        val gridWidth: Int = 40,
        val gridHeight: Int = 24,
        val patchRadius: Int = 1,
        val searchRadius: Int = 4,
        val minimumTextureGradient: Double = 0.012,
        val minimumSamples: Int = 8,
        val maximumStoredVectors: Int = 2048,
        val huberDelta: Double = 0.040,
        val minimumPatchContrast: Double = 0.012,
        val subpixelRefinementLimit: Double = 0.49,
        val forwardBackwardConsistencyPixels: Double = 1.25,
        val cameraMotionRiskThreshold: Double = 0.025,
        val globalMotionConfidenceFloor: Double = 0.35,
        val exposureShiftSoftLimit: Double = 0.12,
        val exposureShiftHardLimit: Double = 0.40,
        val exposureReliabilityFloor: Double = 0.35,
        val timestampRegressionToleranceNanos: Long = 0L,
    ) {
        init {
            require(gridWidth >= 3)
            require(gridHeight >= 3)
            require(patchRadius >= 1)
            require(searchRadius >= 1)
            require(minimumTextureGradient >= 0.0)
            require(minimumSamples >= 4)
            require(maximumStoredVectors >= minimumSamples)
            require(huberDelta > 0.0)
            require(minimumPatchContrast >= 0.0)
            require(subpixelRefinementLimit in 0.0..0.5)
            require(forwardBackwardConsistencyPixels > 0.0)
            require(cameraMotionRiskThreshold > 0.0)
            require(globalMotionConfidenceFloor in 0.0..1.0)
            require(exposureShiftSoftLimit >= 0.0)
            require(exposureShiftHardLimit > exposureShiftSoftLimit)
            require(exposureReliabilityFloor in 0.0..1.0)
            require(timestampRegressionToleranceNanos >= 0L)
        }
    }

    data class FlowVector(
        val normalizedX: Double,
        val normalizedY: Double,
        val dxPixels: Double,
        val dyPixels: Double,
        val confidence: Double,
        val residualEnergy: Double,
        val photometricError: Double = 0.0,
        val gradientAgreement: Double = 0.0,
        val textureStrength: Double = 0.0,
        val forwardBackwardConsistency: Double = 1.0,
        val subpixelRefined: Boolean = false,
        val sourceState: SourceState = SourceState.OBSERVED,
    )

    enum class SourceState {
        OBSERVED,
        LOW_TEXTURE,
        AMBIGUOUS,
        INCONSISTENT,
    }

    data class Result(
        val timestampNanos: Long,
        val valid: Boolean,
        val sampleCount: Int,
        val meanConfidence: Double,
        val globalDxPixels: Double,
        val globalDyPixels: Double,
        val globalMotionMagnitudeNormalizedPerFrame: Double,
        val cameraMotionRisk: Double,
        val globalMotionConfidence: Double,
        val differentialMotionScore: Double,
        val textureCoverage: Double,
        val vectors: List<FlowVector>,
        val warnings: List<String>,
        val medianPhotometricError: Double = 0.0,
        val p95PhotometricError: Double = 0.0,
        val meanGradientAgreement: Double = 0.0,
        val meanForwardBackwardConsistency: Double = 1.0,
        val subpixelRefinedRatio: Double = 0.0,
        val exposureShiftNormalized: Double = 0.0,
        val exposureReliability: Double = 1.0,
        val timestampDeltaNanos: Long = 0L,
    )

    private data class FrameSample(
        val width: Int,
        val height: Int,
        val values: FloatArray,
        val timestampNanos: Long,
        val mean: Double,
        val standardDeviation: Double,
    )

    private data class PatchStats(
        val mean: Double,
        val standardDeviation: Double,
        val contrast: Double,
        val gradientX: Double,
        val gradientY: Double,
        val gradientMagnitude: Double,
    )

    private data class IntegerMatch(
        val dx: Int,
        val dy: Int,
        val cost: Double,
        val confidence: Double,
        val gradientAgreement: Double,
        val textureStrength: Double,
        val forwardBackwardConsistency: Double,
        val subpixelDx: Double,
        val subpixelDy: Double,
        val subpixelRefined: Boolean,
        val sourceState: SourceState,
    )

    private data class MotionSummary(
        val dx: Double,
        val dy: Double,
        val confidence: Double,
    )

    private var previous: FrameSample? = null
    private var previousTimestampNanos: Long = Long.MIN_VALUE
    private var closed = false

    private var scratchValues =
        FloatArray(0)

    @Synchronized
    @Synchronized
    fun configureExposureReliability(
        softLimit: Double,
        hardLimit: Double,
        floor: Double,
    ) {
        require(softLimit >= 0.0)
        require(hardLimit > softLimit)
        require(floor in 0.0..1.0)

        config =
            config.copy(
                exposureShiftSoftLimit = softLimit,
                exposureShiftHardLimit = hardLimit,
                exposureReliabilityFloor = floor,
            )
    }

    fun process(image: Image): Result {
        check(!closed) {
            "SparsePhotometricMotionFieldEngine is closed"
        }

        require(
            image.format ==
                ImageFormat.YUV_420_888,
        ) {
            "SparsePhotometricMotionFieldEngine requires YUV_420_888"
        }

        val timestamp = image.timestamp

        if (
            previousTimestampNanos != Long.MIN_VALUE &&
            timestamp + config.timestampRegressionToleranceNanos <
                previousTimestampNanos
        ) {
            return Result(
                timestampNanos = timestamp,
                valid = false,
                sampleCount = 0,
                meanConfidence = 0.0,
                globalDxPixels = 0.0,
                globalDyPixels = 0.0,
                globalMotionMagnitudeNormalizedPerFrame = 0.0,
                cameraMotionRisk = 0.0,
                globalMotionConfidence = 0.0,
                differentialMotionScore = 0.0,
                textureCoverage = 0.0,
                vectors = emptyList(),
                warnings =
                    listOf(
                        "PHOTOMETRIC_TIMESTAMP_REGRESSION",
                    ),
                timestampDeltaNanos = 0L,
            )
        }

        val current =
            sampleLuma(
                image,
                timestamp,
            )

        val prior =
            previous

        val deltaNanos =
            if (
                prior == null ||
                priorTimestampInvalid()
            ) {
                0L
            } else {
                (
                    timestamp -
                        prior.timestampNanos
                    ).coerceAtLeast(0L)
            }

        previous =
            current
        previousTimestampNanos =
            timestamp

        if (
            prior == null ||
            prior.width != current.width ||
            prior.height != current.height ||
            prior.values.size != current.values.size
        ) {
            return warmupResult(
                current = current,
                previous = prior,
                timestampDeltaNanos = deltaNanos,
            )
        }

        val cols =
            max(
                3,
                config.gridWidth,
            )
        val rows =
            max(
                3,
                config.gridHeight,
            )

        val vectors =
            ArrayList<FlowVector>(
                min(
                    config.maximumStoredVectors,
                    cols * rows,
                ),
            )

        var textured = 0
        var candidateCount = 0

        for (gy in 1 until rows - 1) {
            for (gx in 1 until cols - 1) {
                if (
                    vectors.size >=
                        config.maximumStoredVectors
                ) {
                    break
                }

                val normalizedX =
                    gx.toDouble() /
                        (cols - 1).toDouble()
                val normalizedY =
                    gy.toDouble() /
                        (rows - 1).toDouble()

                val x =
                    normalizedX *
                        (current.width - 1)
                val y =
                    normalizedY *
                        (current.height - 1)

                val cx =
                    x.toInt()
                val cy =
                    y.toInt()

                if (
                    !insidePatch(
                        cx,
                        cy,
                        current.width,
                        current.height,
                    )
                ) {
                    continue
                }

                val stats =
                    patchStats(
                        current,
                        cx,
                        cy,
                    )

                if (
                    stats.gradientMagnitude <
                        config.minimumTextureGradient ||
                    stats.contrast <
                        config.minimumPatchContrast
                ) {
                    continue
                }

                textured++
                candidateCount++

                val match =
                    bestPatchMatch(
                        current = current,
                        previous = prior,
                        cx = cx,
                        cy = cy,
                    ) ?: continue

                vectors +=
                    FlowVector(
                        normalizedX = normalizedX,
                        normalizedY = normalizedY,
                        dxPixels =
                            match.subpixelDx,
                        dyPixels =
                            match.subpixelDy,
                        confidence =
                            match.confidence
                                .coerceIn(0.0, 1.0),
                        residualEnergy =
                            match.cost
                                .coerceAtLeast(0.0),
                        photometricError =
                            match.cost
                                .coerceAtLeast(0.0),
                        gradientAgreement =
                            match.gradientAgreement
                                .coerceIn(0.0, 1.0),
                        textureStrength =
                            stats.gradientMagnitude
                                .coerceIn(0.0, 1.0),
                        forwardBackwardConsistency =
                            match
                                .forwardBackwardConsistency
                                .coerceIn(0.0, 1.0),
                        subpixelRefined =
                            match.subpixelRefined,
                        sourceState =
                            match.sourceState,
                    )
            }
        }

        val textureCoverage =
            textured.toDouble() /
                max(
                    1,
                    (cols - 2) *
                        (rows - 2),
                ).toDouble()

        if (
            vectors.size <
                config.minimumSamples
        ) {
            return Result(
                timestampNanos = current.timestampNanos,
                valid = false,
                sampleCount = vectors.size,
                meanConfidence =
                    vectors
                        .map { it.confidence }
                        .averageOrZero(),
                globalDxPixels = 0.0,
                globalDyPixels = 0.0,
                globalMotionMagnitudeNormalizedPerFrame = 0.0,
                cameraMotionRisk = 0.0,
                globalMotionConfidence = 0.0,
                differentialMotionScore = 0.0,
                textureCoverage =
                    textureCoverage
                        .coerceIn(0.0, 1.0),
                vectors = vectors,
                warnings =
                    listOf(
                        "PHOTOMETRIC_MOTION_TEXTURE_INSUFFICIENT",
                    ),
                medianPhotometricError =
                    vectors
                        .map { it.photometricError }
                        .averageOrZero(),
                meanGradientAgreement =
                    vectors
                        .map { it.gradientAgreement }
                        .averageOrZero(),
                meanForwardBackwardConsistency =
                    vectors
                        .map {
                            it.forwardBackwardConsistency
                        }
                        .averageOrZero(),
                subpixelRefinedRatio =
                    subpixelRatio(vectors),
                exposureShiftNormalized =
                    exposureShift(
                        current,
                        prior,
                    ),
                timestampDeltaNanos =
                    deltaNanos,
            )
        }

        val motion =
            robustGlobalMotion(
                vectors,
            )

        val medianDx =
            motion.dx

        val medianDy =
            motion.dy

        val confidence =
            robustMeanConfidence(
                vectors,
                motion,
            )

        val residuals =
            vectors.map { vector ->
                hypot(
                    vector.dxPixels -
                        medianDx,
                    vector.dyPixels -
                        medianDy,
                )
            }

        val medianResidual =
            median(residuals)

        val p95Residual =
            percentile(
                residuals,
                0.95,
            )

        val frameExtent =
            max(
                1.0,
                max(
                    current.width,
                    current.height,
                ).toDouble(),
            )

        val differential =
            (
                medianResidual /
                    frameExtent
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val globalMagnitude =
            hypot(
                medianDx,
                medianDy,
            )

        val normalizedGlobalMotion =
            (
                globalMagnitude /
                    frameExtent
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val cameraRisk =
            (
                normalizedGlobalMotion /
                    config.cameraMotionRiskThreshold
                ).coerceIn(
                    0.0,
                    1.0,
                ) *
                (
                    0.55 +
                        0.45 *
                            confidence
                    )
                    .coerceIn(
                        0.0,
                        1.0,
                    )

        val globalConfidence =
            (
                confidence *
                    textureCoverage *
                    (
                        1.0 -
                            differential *
                                0.65
                    ).coerceIn(
                        0.0,
                        1.0,
                    ) *
                    (
                        0.70 +
                            motion.confidence *
                                0.30
                    )
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val meanPhotometricError =
            vectors
                .map {
                    it.photometricError
                }
                .averageOrZero()

        val p95PhotometricError =
            percentile(
                vectors.map {
                    it.photometricError
                },
                0.95,
            )

        val meanGradientAgreement =
            vectors
                .map {
                    it.gradientAgreement
                }
                .averageOrZero()

        val meanForwardBackward =
            vectors
                .map {
                    it.forwardBackwardConsistency
                }
                .averageOrZero()

        val subpixelRatio =
            subpixelRatio(vectors)

        val exposureShift =
            exposureShift(
                current,
                prior,
            )

        val exposureReliability =
            exposureReliability(exposureShift)

        val warnings =
            buildList {
                if (
                    textureCoverage <
                        0.35
                ) {
                    add(
                        "PHOTOMETRIC_TEXTURE_COVERAGE_LOW",
                    )
                }

                if (
                    confidence <
                        0.45
                ) {
                    add(
                        "PHOTOMETRIC_FLOW_CONFIDENCE_LOW",
                    )
                }

                if (
                    meanPhotometricError >
                        0.18
                ) {
                    add(
                        "PHOTOMETRIC_RESIDUAL_HIGH",
                    )
                }

                if (
                    p95PhotometricError >
                        0.28
                ) {
                    add(
                        "PHOTOMETRIC_RESIDUAL_TAIL_HIGH",
                    )
                }

                if (
                    meanGradientAgreement <
                        0.42
                ) {
                    add(
                        "PHOTOMETRIC_GRADIENT_AGREEMENT_LOW",
                    )
                }

                if (
                    meanForwardBackward <
                        0.45
                ) {
                    add(
                        "PHOTOMETRIC_FORWARD_BACKWARD_INCONSISTENCY",
                    )
                }

                if (
                    differential >
                        0.45
                ) {
                    add(
                        "PHOTOMETRIC_DIFFERENTIAL_MOTION_HIGH",
                    )
                }

                if (
                    cameraRisk >
                        0.75
                ) {
                    add(
                        "PHOTOMETRIC_GLOBAL_CAMERA_MOTION_HIGH",
                    )
                }

                if (
                    abs(exposureShift) >
                        0.12
                ) {
                    add(
                        "PHOTOMETRIC_EXPOSURE_SHIFT_DETECTED",
                    )
                }

                if (
                    globalConfidence <
                        config.globalMotionConfidenceFloor
                ) {
                    add(
                        "PHOTOMETRIC_GLOBAL_MOTION_CONFIDENCE_LOW",
                    )
                }

                if (
                    candidateCount >
                        0 &&
                    vectors.size.toDouble() /
                        candidateCount.toDouble() <
                        0.55
                ) {
                    add(
                        "PHOTOMETRIC_MATCH_REJECTION_HIGH",
                    )
                }
            }

        return Result(
            timestampNanos =
                current.timestampNanos,
            valid = true,
            sampleCount = vectors.size,
            meanConfidence =
                confidence
                    .coerceIn(0.0, 1.0),
            globalDxPixels =
                medianDx,
            globalDyPixels =
                medianDy,
            globalMotionMagnitudeNormalizedPerFrame =
                normalizedGlobalMotion,
            cameraMotionRisk =
                cameraRisk
                    .coerceIn(0.0, 1.0),
            globalMotionConfidence =
                globalConfidence,
            differentialMotionScore =
                differential,
            textureCoverage =
                textureCoverage
                    .coerceIn(0.0, 1.0),
            vectors = vectors,
            warnings =
                warnings
                    .distinct(),
            medianPhotometricError =
                medianPhotometric(
                    vectors,
                ),
            p95PhotometricError =
                p95PhotometricError,
            meanGradientAgreement =
                meanGradientAgreement
                    .coerceIn(0.0, 1.0),
            meanForwardBackwardConsistency =
                meanForwardBackward
                    .coerceIn(0.0, 1.0),
            subpixelRefinedRatio =
                subpixelRatio,
            exposureShiftNormalized =
                exposureShift,
            exposureReliability =
                exposureReliability,
            timestampDeltaNanos =
                deltaNanos,
        )
    }

    @Synchronized
    fun reset() {
        previous = null
        previousTimestampNanos =
            Long.MIN_VALUE
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        reset()
        scratchValues =
            FloatArray(0)
    }

    private fun warmupResult(
        current: FrameSample,
        previous: FrameSample?,
        timestampDeltaNanos: Long,
    ): Result =
        Result(
            timestampNanos =
                current.timestampNanos,
            valid = false,
            sampleCount = 0,
            meanConfidence = 0.0,
            globalDxPixels = 0.0,
            globalDyPixels = 0.0,
            globalMotionMagnitudeNormalizedPerFrame = 0.0,
            cameraMotionRisk = 0.0,
            globalMotionConfidence = 0.0,
            differentialMotionScore = 0.0,
            textureCoverage = 0.0,
            vectors = emptyList(),
            warnings =
                listOf(
                    "PHOTOMETRIC_MOTION_WARMUP",
                ),
            exposureShiftNormalized =
                previous?.let {
                    exposureShift(
                        current,
                        it,
                    )
                } ?: 0.0,
            timestampDeltaNanos =
                timestampDeltaNanos,
        )

    private fun bestPatchMatch(
        current: FrameSample,
        previous: FrameSample,
        cx: Int,
        cy: Int,
    ): IntegerMatch? {
        var bestCost =
            Double.POSITIVE_INFINITY
        var secondCost =
            Double.POSITIVE_INFINITY
        var bestDx = 0
        var bestDy = 0
        var bestGradientAgreement =
            0.0

        val currentStats =
            patchStats(
                current,
                cx,
                cy,
            )

        for (dy in -config.searchRadius..config.searchRadius) {
            for (dx in -config.searchRadius..config.searchRadius) {
                val px =
                    cx + dx
                val py =
                    cy + dy

                if (
                    !insidePatch(
                        px,
                        py,
                        previous.width,
                        previous.height,
                    )
                ) {
                    continue
                }

                val previousStats =
                    patchStats(
                        previous,
                        px,
                        py,
                    )

                if (
                    previousStats.contrast <
                        config.minimumPatchContrast
                ) {
                    continue
                }

                val photometricCost =
                    normalizedPatchCost(
                        current = current,
                        previous = previous,
                        cxCurrent = cx,
                        cyCurrent = cy,
                        cxPrevious = px,
                        cyPrevious = py,
                        currentStats = currentStats,
                        previousStats = previousStats,
                    )

                val gradientAgreement =
                    gradientAgreement(
                        currentStats,
                        previousStats,
                    )

                val cost =
                    (
                        photometricCost * 0.78 +
                            (
                                1.0 -
                                    gradientAgreement
                            ) * 0.22
                        ).coerceAtLeast(
                            0.0,
                        )

                if (
                    cost <
                        bestCost
                ) {
                    secondCost =
                        bestCost
                    bestCost =
                        cost
                    bestDx =
                        dx
                    bestDy =
                        dy
                    bestGradientAgreement =
                        gradientAgreement
                } else if (
                    cost <
                        secondCost
                ) {
                    secondCost =
                        cost
                }
            }
        }

        if (
            !bestCost.isFinite()
        ) {
            return null
        }

        val subpixel =
            subpixelRefine(
                current = current,
                previous = previous,
                cxCurrent = cx,
                cyCurrent = cy,
                integerDx = bestDx,
                integerDy = bestDy,
                bestCost = bestCost,
            )

        val margin =
            (
                secondCost -
                    bestCost
                ).coerceAtLeast(
                0.0,
            )

        val ambiguityConfidence =
            (
                margin /
                    (
                        secondCost +
                            1e-6
                    )
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val textureStrength =
            (
                currentStats.gradientMagnitude *
                    9.5
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val photometricConfidence =
            (
                1.0 -
                    bestCost /
                        (
                            bestCost +
                                0.085
                        )
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val gradientConfidence =
            bestGradientAgreement
                .coerceIn(
                    0.0,
                    1.0,
                )

        val forwardBackward =
            forwardBackwardConsistency(
                previous = previous,
                current = current,
                cxCurrent = cx,
                cyCurrent = cy,
                dx = subpixel.dx,
                dy = subpixel.dy,
            )

        val consistencyConfidence =
            forwardBackward
                .coerceIn(
                    0.0,
                    1.0,
                )

        val confidence =
            (
                ambiguityConfidence * 0.26 +
                    (
                        photometricConfidence *
                            exposureReliability
                    ) * 0.32 +
                    gradientConfidence * 0.18 +
                    textureStrength * 0.12 +
                    consistencyConfidence * 0.12
                ).coerceIn(
                0.0,
                1.0,
            )

        val sourceState =
            when {
                confidence < 0.28 ->
                    SourceState.INCONSISTENT
                ambiguityConfidence < 0.18 ->
                    SourceState.AMBIGUOUS
                textureStrength < 0.25 ->
                    SourceState.LOW_TEXTURE
                else ->
                    SourceState.OBSERVED
            }

        return IntegerMatch(
            dx = bestDx,
            dy = bestDy,
            cost = bestCost,
            confidence = confidence,
            gradientAgreement =
                gradientConfidence,
            textureStrength =
                textureStrength,
            forwardBackwardConsistency =
                forwardBackward,
            subpixelDx =
                (
                    bestDx.toDouble() +
                        subpixel.dx
                    ).coerceIn(
                    -config.searchRadius.toDouble(),
                    config.searchRadius.toDouble(),
                ),
            subpixelDy =
                (
                    bestDy.toDouble() +
                        subpixel.dy
                    ).coerceIn(
                    -config.searchRadius.toDouble(),
                    config.searchRadius.toDouble(),
                ),
            subpixelRefined =
                subpixel.refined,
            sourceState =
                sourceState,
        )
    }

    private fun normalizedPatchCost(
        current: FrameSample,
        previous: FrameSample,
        cxCurrent: Int,
        cyCurrent: Int,
        cxPrevious: Int,
        cyPrevious: Int,
        currentStats: PatchStats,
        previousStats: PatchStats,
    ): Double {
        val sigmaCurrent =
            currentStats.standardDeviation
                .coerceAtLeast(0.008)
        val sigmaPrevious =
            previousStats.standardDeviation
                .coerceAtLeast(0.008)

        var sum = 0.0
        var count = 0

        for (dy in -config.patchRadius..config.patchRadius) {
            for (dx in -config.patchRadius..config.patchRadius) {
                val ax =
                    cxCurrent + dx
                val ay =
                    cyCurrent + dy
                val bx =
                    cxPrevious + dx
                val by =
                    cyPrevious + dy

                if (
                    ax !in 0 until current.width ||
                    ay !in 0 until current.height ||
                    bx !in 0 until previous.width ||
                    by !in 0 until previous.height
                ) {
                    continue
                }

                val a =
                    (
                        current.values[
                            ay * current.width +
                                ax
                        ].toDouble() -
                            currentStats.mean
                        ) /
                        sigmaCurrent

                val b =
                    (
                        previous.values[
                            by * previous.width +
                                bx
                        ].toDouble() -
                            previousStats.mean
                        ) /
                        sigmaPrevious

                sum +=
                    huberLoss(
                        a - b,
                        config.huberDelta,
                    )
                count++
            }
        }

        return if (
            count == 0
        ) {
            Double.POSITIVE_INFINITY
        } else {
            (
                sum /
                    count.toDouble()
            ).coerceIn(
                0.0,
                4.0,
            )
        }
    }

    private fun huberLoss(
        residual: Double,
        delta: Double,
    ): Double {
        val magnitude =
            abs(residual)
        return if (
            magnitude <=
                delta
        ) {
            0.5 *
                magnitude *
                magnitude
        } else {
            delta *
                (
                    magnitude -
                        0.5 *
                            delta
                )
        }
    }

    private fun subpixelRefine(
        current: FrameSample,
        previous: FrameSample,
        cxCurrent: Int,
        cyCurrent: Int,
        integerDx: Int,
        integerDy: Int,
        bestCost: Double,
    ): SubpixelResult {
        var offsetX = 0.0
        var offsetY = 0.0
        var refined = false

        val left =
            patchCostAtOffset(
                current,
                previous,
                cxCurrent,
                cyCurrent,
                integerDx - 1,
                integerDy,
            )
        val centerX =
            patchCostAtOffset(
                current,
                previous,
                cxCurrent,
                cyCurrent,
                integerDx,
                integerDy,
            )
        val right =
            patchCostAtOffset(
                current,
                previous,
                cxCurrent,
                cyCurrent,
                integerDx + 1,
                integerDy,
            )

        val top =
            patchCostAtOffset(
                current,
                previous,
                cxCurrent,
                cyCurrent,
                integerDx,
                integerDy - 1,
            )
        val centerY =
            patchCostAtOffset(
                current,
                previous,
                cxCurrent,
                cyCurrent,
                integerDx,
                integerDy,
            )
        val bottom =
            patchCostAtOffset(
                current,
                previous,
                cxCurrent,
                cyCurrent,
                integerDx,
                integerDy + 1,
            )

        offsetX =
            parabolicVertex(
                left,
                centerX,
                right,
            )
        offsetY =
            parabolicVertex(
                top,
                centerY,
                bottom,
            )

        if (
            offsetX.isFinite() ||
            offsetY.isFinite()
        ) {
            refined = true
        }

        return SubpixelResult(
            dx =
                offsetX
                    .takeIf(Double::isFinite)
                    ?.coerceIn(
                        -config.subpixelRefinementLimit,
                        config.subpixelRefinementLimit,
                    )
                    ?: 0.0,
            dy =
                offsetY
                    .takeIf(Double::isFinite)
                    ?.coerceIn(
                        -config.subpixelRefinementLimit,
                        config.subpixelRefinementLimit,
                    )
                    ?: 0.0,
            refined =
                refined &&
                    bestCost.isFinite(),
        )
    }

    private fun parabolicVertex(
        left: Double,
        center: Double,
        right: Double,
    ): Double {
        if (
            !left.isFinite() ||
            !center.isFinite() ||
            !right.isFinite()
        ) {
            return 0.0
        }

        val denominator =
            2.0 *
                (
                    left -
                        2.0 *
                            center +
                        right
                )

        if (
            abs(denominator) <
                1e-9
        ) {
            return 0.0
        }

        return (
            left -
                right
            ) /
            denominator
    }

    private fun forwardBackwardConsistency(
        previous: FrameSample,
        current: FrameSample,
        cxCurrent: Int,
        cyCurrent: Int,
        dx: Double,
        dy: Double,
    ): Double {
        val targetX =
            (
                cxCurrent +
                    dx
                ).toInt()
        val targetY =
            (
                cyCurrent +
                    dy
                ).toInt()

        if (
            !insidePatch(
                targetX,
                targetY,
                previous.width,
                previous.height,
            )
        ) {
            return 0.0
        }

        val reverse =
            bestReverseMatch(
                from = previous,
                to = current,
                cx = targetX,
                cy = targetY,
            )
                ?: return 0.0

        val closedLoop =
            hypot(
                dx +
                    reverse.dx,
                dy +
                    reverse.dy,
            )

        return (
            1.0 -
                closedLoop /
                    config.forwardBackwardConsistencyPixels
            ).coerceIn(
            0.0,
            1.0,
        )
    }

    private fun bestReverseMatch(
        from: FrameSample,
        to: FrameSample,
        cx: Int,
        cy: Int,
    ): ReverseMatch? {
        var bestCost =
            Double.POSITIVE_INFINITY
        var bestDx = 0
        var bestDy = 0
        var second =
            Double.POSITIVE_INFINITY

        if (
            !insidePatch(
                cx,
                cy,
                from.width,
                from.height,
            )
        ) {
            return null
        }

        val fromStats =
            patchStats(
                from,
                cx,
                cy,
            )

        for (dy in -config.searchRadius..config.searchRadius) {
            for (dx in -config.searchRadius..config.searchRadius) {
                val tx =
                    cx + dx
                val ty =
                    cy + dy

                if (
                    !insidePatch(
                        tx,
                        ty,
                        to.width,
                        to.height,
                    )
                ) {
                    continue
                }

                val toStats =
                    patchStats(
                        to,
                        tx,
                        ty,
                    )

                val cost =
                    normalizedPatchCost(
                        current = from,
                        previous = to,
                        cxCurrent = cx,
                        cyCurrent = cy,
                        cxPrevious = tx,
                        cyPrevious = ty,
                        currentStats = fromStats,
                        previousStats = toStats,
                    )

                if (
                    cost <
                        bestCost
                ) {
                    second =
                        bestCost
                    bestCost =
                        cost
                    bestDx =
                        dx
                    bestDy =
                        dy
                } else if (
                    cost <
                        second
                ) {
                    second =
                        cost
                }
            }
        }

        if (
            !bestCost.isFinite()
        ) {
            return null
        }

        val margin =
            (
                second -
                    bestCost
                ).coerceAtLeast(
                0.0,
            )

        return ReverseMatch(
            dx =
                bestDx.toDouble(),
            dy =
                bestDy.toDouble(),
            confidence =
                (
                    margin /
                        (
                            second +
                                1e-6
                        )
                    ).coerceIn(
                    0.0,
                    1.0,
                ),
        )
    }

    private fun patchCostAtOffset(
        current: FrameSample,
        previous: FrameSample,
        cxCurrent: Int,
        cyCurrent: Int,
        dx: Int,
        dy: Int,
    ): Double {
        val px =
            cxCurrent + dx
        val py =
            cyCurrent + dy

        if (
            !insidePatch(
                px,
                py,
                previous.width,
                previous.height,
            )
        ) {
            return Double.POSITIVE_INFINITY
        }

        val currentStats =
            patchStats(
                current,
                cxCurrent,
                cyCurrent,
            )
        val previousStats =
            patchStats(
                previous,
                px,
                py,
            )

        return (
            normalizedPatchCost(
                current = current,
                previous = previous,
                cxCurrent = cxCurrent,
                cyCurrent = cyCurrent,
                cxPrevious = px,
                cyPrevious = py,
                currentStats = currentStats,
                previousStats = previousStats,
            ) * 0.78 +
                (
                    1.0 -
                        gradientAgreement(
                            currentStats,
                            previousStats,
                        )
                    ) * 0.22
            )
    }

    private fun patchStats(
        frame: FrameSample,
        cx: Int,
        cy: Int,
    ): PatchStats {
        var sum = 0.0
        var sumSquared = 0.0
        var count = 0

        var gradientX = 0.0
        var gradientY = 0.0

        for (dy in -config.patchRadius..config.patchRadius) {
            for (dx in -config.patchRadius..config.patchRadius) {
                val x =
                    cx + dx
                val y =
                    cy + dy
                if (
                    x !in 0 until frame.width ||
                    y !in 0 until frame.height
                ) {
                    continue
                }

                val value =
                    frame.values[
                        y * frame.width +
                            x
                    ].toDouble()

                sum += value
                sumSquared +=
                    value * value
                count++

                val left =
                    frame.values[
                        y * frame.width +
                            (x - 1).coerceAtLeast(
                                0,
                            )
                    ].toDouble()
                val right =
                    frame.values[
                        y * frame.width +
                            (x + 1).coerceAtMost(
                                frame.width - 1,
                            )
                    ].toDouble()
                val top =
                    frame.values[
                        (y - 1).coerceAtLeast(
                            0,
                        ) *
                            frame.width +
                            x
                    ].toDouble()
                val bottom =
                    frame.values[
                        (y + 1).coerceAtMost(
                            frame.height - 1,
                        ) *
                            frame.width +
                            x
                    ].toDouble()

                gradientX +=
                    (right - left) *
                        0.5
                gradientY +=
                    (bottom - top) *
                        0.5
            }
        }

        if (count <= 0) {
            return PatchStats(
                mean = 0.0,
                standardDeviation = 0.0,
                contrast = 0.0,
                gradientX = 0.0,
                gradientY = 0.0,
                gradientMagnitude = 0.0,
            )
        }

        val mean =
            sum /
                count.toDouble()

        val variance =
            (
                sumSquared /
                    count.toDouble() -
                    mean * mean
                ).coerceAtLeast(
                0.0,
            )

        val standardDeviation =
            sqrt(variance)

        val normalizedGradient =
            sqrt(
                gradientX *
                    gradientX +
                    gradientY *
                        gradientY,
            ) /
                count.toDouble()

        return PatchStats(
            mean = mean,
            standardDeviation =
                standardDeviation,
            contrast =
                standardDeviation.coerceIn(
                    0.0,
                    1.0,
                ),
            gradientX =
                gradientX /
                    count.toDouble(),
            gradientY =
                gradientY /
                    count.toDouble(),
            gradientMagnitude =
                normalizedGradient.coerceIn(
                    0.0,
                    1.0,
                ),
        )
    }

    private fun gradientAgreement(
        current: PatchStats,
        previous: PatchStats,
    ): Double {
        val currentMagnitude =
            hypot(
                current.gradientX,
                current.gradientY,
            )
        val previousMagnitude =
            hypot(
                previous.gradientX,
                previous.gradientY,
            )

        if (
            currentMagnitude <
                1e-6 ||
            previousMagnitude <
                1e-6
        ) {
            return 0.0
        }

        val dot =
            current.gradientX *
                previous.gradientX +
                current.gradientY *
                    previous.gradientY

        return (
            0.5 +
                0.5 *
                    dot /
                    (
                        currentMagnitude *
                            previousMagnitude
                    ).coerceAtLeast(
                        1e-9,
                    )
            ).coerceIn(
            0.0,
            1.0,
        )
    }

    private fun sampleLuma(
        image: Image,
        timestampNanos: Long,
    ): FrameSample {
        val width =
            config.gridWidth
                .coerceAtLeast(3)
        val height =
            config.gridHeight
                .coerceAtLeast(3)

        val required =
            width * height

        if (
            scratchValues.size !=
                required
        ) {
            scratchValues =
                FloatArray(
                    required,
                )
        }

        val plane =
            image.planes
                .firstOrNull()

        if (
            plane == null
        ) {
            return FrameSample(
                width = width,
                height = height,
                values = FloatArray(required),
                timestampNanos =
                    timestampNanos,
                mean = 0.0,
                standardDeviation = 0.0,
            )
        }

        val sourceWidth =
            image.width.coerceAtLeast(1)
        val sourceHeight =
            image.height.coerceAtLeast(1)

        val buffer =
            plane.buffer.duplicate()

        var sum = 0.0
        var sumSquared = 0.0

        for (gy in 0 until height) {
            val sourceY =
                (
                    gy.toDouble() /
                        (height - 1).coerceAtLeast(
                            1,
                        )
                    ) *
                    (sourceHeight - 1)

            val y =
                sourceY
                    .toInt()
                    .coerceIn(
                        0,
                        sourceHeight - 1,
                    )

            for (gx in 0 until width) {
                val sourceX =
                    (
                        gx.toDouble() /
                            (width - 1).coerceAtLeast(
                                1,
                            )
                        ) *
                        (sourceWidth - 1)

                val x =
                    sourceX
                        .toInt()
                        .coerceIn(
                            0,
                            sourceWidth - 1,
                        )

                val index =
                    y *
                        plane.rowStride +
                        x *
                            plane.pixelStride

                val value =
                    if (
                        index in
                            0 until
                            buffer.limit()
                    ) {
                        (
                            (
                                buffer.get(
                                    index,
                                ).toInt() and
                                    0xFF
                                ) /
                                255.0f
                            )
                    } else {
                        0.0f
                    }

                val destination =
                    gy * width + gx

                scratchValues[
                    destination
                ] = value

                sum +=
                    value.toDouble()
                sumSquared +=
                    value.toDouble() *
                        value.toDouble()
            }
        }

        val count =
            required.coerceAtLeast(
                1,
            )

        val mean =
            sum /
                count.toDouble()

        val variance =
            (
                sumSquared /
                    count.toDouble() -
                    mean *
                        mean
                ).coerceAtLeast(
                0.0,
            )

        return FrameSample(
            width = width,
            height = height,
            values =
                scratchValues.copyOf(),
            timestampNanos =
                timestampNanos,
            mean = mean,
            standardDeviation =
                sqrt(variance),
        )
    }

    private fun insidePatch(
        x: Int,
        y: Int,
        width: Int,
        height: Int,
    ): Boolean =
        x -
            config.patchRadius >= 0 &&
            y -
                config.patchRadius >= 0 &&
            x +
                config.patchRadius <
                width &&
            y +
                config.patchRadius <
                height

    private fun robustGlobalMotion(
        vectors: List<FlowVector>,
    ): MotionSummary {
        if (vectors.isEmpty()) {
            return MotionSummary(
                dx = 0.0,
                dy = 0.0,
                confidence = 0.0,
            )
        }

        val dxs =
            vectors.map {
                it.dxPixels
            }
        val dys =
            vectors.map {
                it.dyPixels
            }

        val firstDx =
            median(dxs)
        val firstDy =
            median(dys)

        val residuals =
            vectors.map {
                hypot(
                    it.dxPixels -
                        firstDx,
                    it.dyPixels -
                        firstDy,
                )
            }

        val scale =
            (
                median(
                    residuals,
                ) * 1.4826
            ).coerceAtLeast(
                0.20,
            )

        val weights =
            vectors.mapIndexed { index, vector ->
                val residual =
                    residuals[index]
                val robustWeight =
                    if (
                        residual <=
                            2.5 * scale
                    ) {
                        1.0
                    } else {
                        (
                            2.5 * scale /
                                residual
                            ).coerceIn(
                            0.05,
                            1.0,
                        )
                    }

                vector.confidence *
                    robustWeight
            }

        val sumWeight =
            weights.sum()
                .coerceAtLeast(
                    1e-9,
                )

        val dx =
            vectors.mapIndexed {
                    index,
                    vector,
                ->
                    vector.dxPixels *
                        weights[index]
                }.sum() /
                sumWeight

        val dy =
            vectors.mapIndexed {
                    index,
                    vector,
                ->
                    vector.dyPixels *
                        weights[index]
                }.sum() /
                sumWeight

        val confidence =
            (
                vectors
                    .mapIndexed {
                        index,
                        vector,
                    ->
                        vector.confidence *
                            weights[index]
                    }.sum() /
                    sumWeight
            ).coerceIn(
                0.0,
                1.0,
            )

        return MotionSummary(
            dx = dx,
            dy = dy,
            confidence = confidence,
        )
    }

    private fun robustMeanConfidence(
        vectors: List<FlowVector>,
        motion: MotionSummary,
    ): Double {
        if (vectors.isEmpty()) return 0.0

        val residuals =
            vectors.map {
                hypot(
                    it.dxPixels -
                        motion.dx,
                    it.dyPixels -
                        motion.dy,
                )
            }

        val medianResidual =
            median(
                residuals,
            )

        val scale =
            (
                medianResidual *
                    1.4826
            ).coerceAtLeast(
                0.15,
            )

        val values =
            vectors.mapIndexed { index, vector ->
                val residual =
                    residuals[index]
                val weight =
                    if (
                        residual <=
                            2.5 * scale
                    ) {
                        1.0
                    } else {
                        (
                            2.5 * scale /
                                residual
                            ).coerceIn(
                            0.05,
                            1.0,
                        )
                    }

                vector.confidence *
                    weight
            }

        return values.averageOrZero()
            .coerceIn(
                0.0,
                1.0,
            )
    }

    fun exposureReliability(
        exposureShiftNormalized: Double,
    ): Double {
        if (!exposureShiftNormalized.isFinite()) {
            return config.exposureReliabilityFloor
        }

        val magnitude =
            abs(exposureShiftNormalized)

        if (
            magnitude <=
                config.exposureShiftSoftLimit
        ) {
            return 1.0
        }

        if (
            magnitude >=
                config.exposureShiftHardLimit
        ) {
            return config.exposureReliabilityFloor
        }

        val normalized =
            (
                magnitude -
                    config.exposureShiftSoftLimit
            ) /
                (
                    config.exposureShiftHardLimit -
                        config.exposureShiftSoftLimit
                ).coerceAtLeast(1e-9)

        val shaped =
            exp(
                -2.2 *
                    normalized *
                    normalized,
            )

        return max(
            config.exposureReliabilityFloor,
            shaped,
        ).coerceIn(
            config.exposureReliabilityFloor,
            1.0,
        )
    }

    private fun exposureShift(
        current: FrameSample,
        previous: FrameSample,
    ): Double =
        (
            current.mean -
                previous.mean
            ).coerceIn(
            -1.0,
            1.0,
        )

    private fun medianPhotometric(
        vectors: List<FlowVector>,
    ): Double =
        median(
            vectors.map {
                it.photometricError
            },
        )

    private fun subpixelRatio(
        vectors: List<FlowVector>,
    ): Double =
        if (vectors.isEmpty()) {
            0.0
        } else {
            vectors.count {
                it.subpixelRefined
            }.toDouble() /
                vectors.size.toDouble()
        }.coerceIn(
            0.0,
            1.0,
        )

    private fun percentile(
        values: List<Double>,
        fraction: Double,
    ): Double {
        if (values.isEmpty()) return 0.0

        val sorted =
            values
                .filter(Double::isFinite)
                .sorted()

        if (sorted.isEmpty()) return 0.0

        val index =
            (
                fraction.coerceIn(
                    0.0,
                    1.0,
                ) *
                    (sorted.size - 1)
                )
                .toInt()
                .coerceIn(
                    0,
                    sorted.lastIndex,
                )

        return sorted[index]
    }

    private fun median(
        values: List<Double>,
    ): Double {
        if (values.isEmpty()) return 0.0

        val sorted =
            values
                .filter(Double::isFinite)
                .sorted()

        if (sorted.isEmpty()) return 0.0

        val middle =
            sorted.size / 2

        return if (
            sorted.size % 2 ==
                1
        ) {
            sorted[middle]
        } else {
            (
                sorted[middle - 1] +
                    sorted[middle]
                ) *
                0.5
        }
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) {
            0.0
        } else {
            average()
                .takeIf(Double::isFinite)
                ?: 0.0
        }

    private fun priorTimestampInvalid(): Boolean =
        previousTimestampNanos ==
            Long.MIN_VALUE

    private data class SubpixelResult(
        val dx: Double,
        val dy: Double,
        val refined: Boolean,
    )

    private data class ReverseMatch(
        val dx: Double,
        val dy: Double,
        val confidence: Double,
    )
}

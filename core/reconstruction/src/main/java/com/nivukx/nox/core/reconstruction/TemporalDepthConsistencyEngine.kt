package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.nativeengine.DepthSampleConvention
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

private const val MIN_TEMPORAL_GRID_DIMENSION = 8
private const val MAX_TEMPORAL_GRID_DIMENSION = 1024
private const val MAX_TEMPORAL_GRID_SAMPLES = 262_144
private const val PIXEL_WEIGHT_EPSILON = 1e-9
private const val DEPTH_EDGE_CONFIDENCE_SCALE = 0.5

/**
 * Camera-motion-compensated temporal depth consistency.
 *
 * A depth frame should not be treated as temporally consistent merely because its pixels are
 * smooth. Camera translation and rotation can move every valid surface point between frames.
 * This engine therefore stores a bounded sparse depth representation, transforms the previous
 * samples through the previous and current camera poses, and compares expected current depth
 * against observed current depth.
 *
 * The result is intentionally a scalar evidence signal plus robust diagnostics. It is designed to
 * feed downstream TSDF weighting rather than replace metric depth.
 *
 * Properties:
 * - bounded memory;
 * - no neural model or fabricated depth;
 * - robust residual statistics;
 * - camera-motion compensation;
 * - temporal hysteresis;
 * - hard rejection for timestamp/pose discontinuities;
 * - configurable sampling density;
 * - deterministic replay from the same depth/pose sequence.
 */
class TemporalDepthConsistencyEngine(
    private var config: Config = Config(),
) {

    data class Config(
        val gridWidth: Int = 64,
        val gridHeight: Int = 36,
        val minimumDepthMeters: Double = 0.05,
        val maximumDepthMeters: Double = 20.0,
        val residualScaleMeters: Double = 0.035,
        val warningResidualMeters: Double = 0.055,
        val hardResidualMeters: Double = 0.16,
        val minimumValidRatio: Double = 0.34,
        val historyAlpha: Double = 0.24,
        val maxFrameGapSeconds: Double = 0.22,
        val maxCameraTranslationMeters: Double = 0.75,
        val maxCameraRotationRadians: Double = 0.65,
        val robustHuberDeltaMeters: Double = 0.045,
        val minimumConfidence: Double = 0.10,
        val warmupConfidence: Double = 0.62,
    ) {
        init {
            require(gridWidth in MIN_TEMPORAL_GRID_DIMENSION..MAX_TEMPORAL_GRID_DIMENSION)
            require(gridHeight in MIN_TEMPORAL_GRID_DIMENSION..MAX_TEMPORAL_GRID_DIMENSION)
            require(
                gridWidth.toLong() * gridHeight.toLong() <=
                    MAX_TEMPORAL_GRID_SAMPLES.toLong(),
            ) {
                "Temporal depth grid exceeds the bounded sample budget"
            }
            require(
                listOf(
                    minimumDepthMeters,
                    maximumDepthMeters,
                    residualScaleMeters,
                    warningResidualMeters,
                    hardResidualMeters,
                    historyAlpha,
                    maxFrameGapSeconds,
                    maxCameraTranslationMeters,
                    maxCameraRotationRadians,
                    robustHuberDeltaMeters,
                    minimumConfidence,
                    warmupConfidence,
                ).all { it.isFinite() },
            ) {
                "Temporal depth configuration values must be finite"
            }
            require(minimumDepthMeters > 0.0)
            require(maximumDepthMeters > minimumDepthMeters)
            require(residualScaleMeters > 0.0)
            require(warningResidualMeters >= residualScaleMeters)
            require(hardResidualMeters > warningResidualMeters)
            require(minimumValidRatio in 0.0..1.0)
            require(historyAlpha in 0.0..1.0)
            require(maxFrameGapSeconds > 0.0)
            require(maxCameraTranslationMeters > 0.0)
            require(maxCameraRotationRadians > 0.0)
            require(robustHuberDeltaMeters > 0.0)
            require(minimumConfidence in 0.0..1.0)
            require(warmupConfidence in 0.0..1.0)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val confidence: Double,
        val rawConfidence: Double,
        val validSampleRatio: Double,
        val meanResidualMeters: Double,
        val medianResidualMeters: Double,
        val p95ResidualMeters: Double,
        val maxResidualMeters: Double,
        val cameraTranslationMeters: Double,
        val cameraRotationRadians: Double,
        val accepted: Boolean,
        val warmedUp: Boolean,
        val hardReset: Boolean,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private data class Sample(
        val x: Int,
        val y: Int,
        val depthMeters: Double,
        val confidence: Double,
    )

    private data class DepthAtPixel(
        val depthMeters: Double,
        val confidence: Double,
    )

    private data class ProjectedDepthPoint(
        val x: Double,
        val y: Double,
        val depthMeters: Double,
    )

    private data class ResidualSample(
        val residualMeters: Double,
        val confidenceWeight: Double,
    )

    private data class History(
        val timestampNanos: Long,
        val pose: Pose,
        val samples: List<Sample>,
        val confidence: Double,
    )

    private var history: History? = null

    @Synchronized
    fun configure(newConfig: Config) {
        if (newConfig == config) return
        config = newConfig
        history = null
    }

    fun configuration(): Config = config

    @Synchronized
    fun reset() {
        history = null
    }

    @Synchronized
    fun evaluate(
        depthMeters: FloatArray,
        width: Int,
        height: Int,
        intrinsics: CameraIntrinsics,
        cameraPose: Pose,
        timestampNanos: Long,
        sourceConfidence: Double = 1.0,
        depthConfidence: FloatArray? = null,
        depthConvention: DepthSampleConvention = DepthSampleConvention.EUCLIDEAN_RAY,
    ): Result {
        require(width > 0 && height > 0)
        val sampleCount = width.toLong() * height.toLong()
        require(sampleCount <= Int.MAX_VALUE.toLong()) {
            "Temporal depth frame exceeds JVM array indexing capacity"
        }
        require(sampleCount == depthMeters.size.toLong()) {
            "Depth array size must match the non-overflowing width × height product"
        }
        require(depthConfidence == null || depthConfidence.size.toLong() == sampleCount) {
            "Per-pixel confidence size must match the depth frame"
        }
        require(timestampNanos > 0L)
        val dimensionsUnspecified =
            intrinsics.imageWidth == 0 && intrinsics.imageHeight == 0
        require(
            dimensionsUnspecified ||
                (intrinsics.imageWidth == width && intrinsics.imageHeight == height),
        ) {
            "Calibrated intrinsics dimensions must match the depth frame"
        }
        require(
            intrinsics.fx.isFinite() && intrinsics.fx > 0.0 &&
                intrinsics.fy.isFinite() && intrinsics.fy > 0.0 &&
                intrinsics.cx.isFinite() && intrinsics.cy.isFinite() &&
                intrinsics.k1.isFinite() && intrinsics.k2.isFinite() &&
                intrinsics.k3.isFinite() && intrinsics.p1.isFinite() &&
                intrinsics.p2.isFinite(),
        ) {
            "Depth intrinsics must be finite with positive focal lengths"
        }
        require(
            cameraPose.position.x.isFinite() &&
                cameraPose.position.y.isFinite() &&
                cameraPose.position.z.isFinite(),
        ) {
            "Camera position must be finite"
        }
        val rotation = cameraPose.rotation
        val rotationNormSquared =
            rotation.w * rotation.w +
                rotation.x * rotation.x +
                rotation.y * rotation.y +
                rotation.z * rotation.z
        require(
            rotation.w.isFinite() && rotation.x.isFinite() &&
                rotation.y.isFinite() && rotation.z.isFinite() &&
                rotationNormSquared.isFinite() && rotationNormSquared > 1e-12,
        ) {
            "Camera rotation must be a finite, non-degenerate quaternion"
        }

        val samples =
            sampleDepth(
                depthMeters = depthMeters,
                width = width,
                height = height,
                depthConfidence = depthConfidence,
            )
        val sampledCoverage = sampleCoverage(samples)

        val previous = history
        val boundedWarmupEvidenceConfidence =
            min(currentConfidence, sampledCoverage)
        // NaN survives coerceIn; invalid evidence must reduce confidence, never poison it.
        val currentConfidence =
            sourceConfidence
                .takeIf { it.isFinite() }
                ?.coerceIn(0.0, 1.0)
                ?: 0.0

        if (previous == null) {
            history =
                History(
                    timestampNanos = timestampNanos,
                    pose = cameraPose,
                    samples = samples,
                    confidence = boundedWarmupEvidenceConfidence,
                )
            return Result(
                timestampNanos = timestampNanos,
                confidence = config.warmupConfidence *
                    boundedWarmupEvidenceConfidence,
                rawConfidence = config.warmupConfidence,
                validSampleRatio = sampledCoverage,
                meanResidualMeters = 0.0,
                medianResidualMeters = 0.0,
                p95ResidualMeters = 0.0,
                maxResidualMeters = 0.0,
                cameraTranslationMeters = 0.0,
                cameraRotationRadians = 0.0,
                accepted =
                    sampledCoverage >= config.minimumValidRatio &&
                        boundedWarmupEvidenceConfidence >= config.minimumConfidence,
                warmedUp = false,
                hardReset = false,
                warnings = buildList {
                    add("temporal-depth-warmup")
                    if (sampledCoverage < config.minimumValidRatio) {
                        add("temporal-depth-valid-coverage-low")
                    }
                    if (boundedWarmupEvidenceConfidence < config.minimumConfidence) {
                        add("temporal-depth-confidence-low")
                    }
                },
                recommendations = buildList {
                    add("Preserve the next synchronized depth frame to establish temporal evidence.")
                    if (sampledCoverage < config.minimumValidRatio) {
                        add("Increase synchronized depth coverage before granting the frame TSDF influence.")
                    }
                },
            )
        }

        // Out-of-order frames are telemetry, not a new temporal segment. Preserve the
        // latest accepted history so a delayed packet cannot move this camera's time backwards.
        if (timestampNanos <= previous.timestampNanos) {
            return Result(
                timestampNanos = timestampNanos,
                confidence = 0.0,
                rawConfidence = 0.0,
                validSampleRatio = sampledCoverage,
                meanResidualMeters = config.hardResidualMeters,
                medianResidualMeters = config.hardResidualMeters,
                p95ResidualMeters = config.hardResidualMeters,
                maxResidualMeters = config.hardResidualMeters,
                cameraTranslationMeters = 0.0,
                cameraRotationRadians = 0.0,
                accepted = false,
                warmedUp = false,
                hardReset = false,
                warnings = listOf("temporal-depth-stale-timestamp"),
                recommendations = listOf(
                    "Drop the out-of-order depth frame and retain the most recent temporal history.",
                ),
            )
        }

        val dtSeconds =
            (
                timestampNanos -
                    previous.timestampNanos
                ).toDouble() / 1e9

        val translation =
            cameraPose.position.distanceTo(
                previous.pose.position,
            )

        val rotation =
            quaternionAngularDistance(
                previous.pose.rotation,
                cameraPose.rotation,
            )

        val discontinuity =
            dtSeconds <= 0.0 ||
                dtSeconds > config.maxFrameGapSeconds ||
                translation > config.maxCameraTranslationMeters ||
                rotation > config.maxCameraRotationRadians

        if (discontinuity) {
            history =
                History(
                    timestampNanos = timestampNanos,
                    pose = cameraPose,
                    samples = samples,
                    confidence = boundedWarmupEvidenceConfidence,
                )

            return Result(
                timestampNanos = timestampNanos,
                confidence = config.warmupConfidence *
                    boundedWarmupEvidenceConfidence,
                rawConfidence = config.warmupConfidence,
                validSampleRatio = sampledCoverage,
                meanResidualMeters = 0.0,
                medianResidualMeters = 0.0,
                p95ResidualMeters = 0.0,
                maxResidualMeters = 0.0,
                cameraTranslationMeters = translation,
                cameraRotationRadians = rotation,
                accepted = false,
                warmedUp = false,
                hardReset = true,
                warnings = listOf(
                    "temporal-depth-history-reset",
                ) +
                    if (dtSeconds > config.maxFrameGapSeconds) {
                        listOf("temporal-depth-frame-gap")
                    } else {
                        emptyList()
                    } +
                    if (translation > config.maxCameraTranslationMeters) {
                        listOf("temporal-depth-camera-translation-jump")
                    } else {
                        emptyList()
                    } +
                    if (rotation > config.maxCameraRotationRadians) {
                        listOf("temporal-depth-camera-rotation-jump")
                    } else {
                        emptyList()
                    } +
                    if (sampledCoverage < config.minimumValidRatio) {
                        listOf("temporal-depth-valid-coverage-low")
                    } else {
                        emptyList()
                    } +
                    if (boundedWarmupEvidenceConfidence < config.minimumConfidence) {
                        listOf("temporal-depth-confidence-low")
                    } else {
                        emptyList()
                    },
                recommendations = buildList {
                    add("Treat the current depth frame as a new temporal segment after the camera-motion discontinuity.")
                    if (sampledCoverage < config.minimumValidRatio) {
                        add("Increase synchronized depth coverage before granting the frame TSDF influence.")
                    }
                },
            )
        }

        val residuals = ArrayList<ResidualSample>(
            min(previous.samples.size, config.gridWidth * config.gridHeight),
        )

        previous.samples.forEach { sample ->
            val projectedPoint =
                transformPreviousSampleToCurrentDepth(
                    sample = sample,
                    previousPose = previous.pose,
                    currentPose = cameraPose,
                    intrinsics = intrinsics,
                    width = width,
                    height = height,
                    depthConvention = depthConvention,
                )

            if (projectedPoint == null) return@forEach

            val observed =
                depthAtNormalizedPixel(
                    depthMeters = depthMeters,
                    width = width,
                    height = height,
                    x = projectedPoint.x,
                    y = projectedPoint.y,
                    depthConfidence = depthConfidence,
                )

            if (observed == null) return@forEach

            val residual =
                abs(
                    observed.depthMeters -
                        projectedPoint.depthMeters,
                )
            val confidenceWeight =
                min(sample.confidence, observed.confidence)

            if (residual.isFinite() && confidenceWeight > 0.0) {
                residuals +=
                    ResidualSample(
                        residualMeters = residual,
                        confidenceWeight = confidenceWeight,
                    )
            }
        }

        val totalConfidenceWeight =
            residuals.sumOf { it.confidenceWeight }
        val validRatio =
            (
                totalConfidenceWeight /
                    previous.samples.size.coerceAtLeast(1)
                ).coerceIn(0.0, 1.0)

        val sorted =
            residuals.sortedBy { it.residualMeters }

        val mean =
            if (totalConfidenceWeight <= 0.0) {
                config.hardResidualMeters
            } else {
                residuals.sumOf {
                    it.residualMeters * it.confidenceWeight
                } / totalConfidenceWeight
            }

        val median =
            percentile(
                sorted,
                0.50,
            )

        val p95 =
            percentile(
                sorted,
                0.95,
            )

        val maxResidual =
            sorted.lastOrNull()?.residualMeters
                ?: config.hardResidualMeters

        val robustResidual =
            robustMean(
                residuals,
            )

        val residualQuality =
            exp(
                -robustResidual /
                    config.residualScaleMeters,
            ).coerceIn(0.0, 1.0)

        val coverageQuality =
            (
                validRatio /
                    config.minimumValidRatio.coerceAtLeast(1e-6)
                ).coerceIn(0.0, 1.0)

        val motionQuality =
            cameraMotionQuality(
                translation = translation,
                rotation = rotation,
            )

        val raw =
            (
                residualQuality * 0.58 +
                    coverageQuality * 0.20 +
                    motionQuality * 0.14 +
                    currentConfidence * 0.08
                ).coerceIn(0.0, 1.0)

        val prior =
            previous.confidence.coerceIn(0.0, 1.0)

        val confidence =
            (
                prior * (1.0 - config.historyAlpha) +
                    raw * config.historyAlpha
                ).coerceIn(0.0, 1.0)

        val effectiveConfidence =
            if (p95 >= config.warningResidualMeters) {
                min(confidence, raw)
            } else {
                confidence
            }

        val accepted =
            validRatio >= config.minimumValidRatio &&
                effectiveConfidence >= config.minimumConfidence &&
                p95 < config.hardResidualMeters

        val warnings =
            buildList {
                if (validRatio < config.minimumValidRatio) {
                    add("temporal-depth-valid-coverage-low")
                }
                if (mean >= config.warningResidualMeters) {
                    add("temporal-depth-residual-elevated")
                }
                if (p95 >= config.warningResidualMeters) {
                    add("temporal-depth-p95-residual-elevated")
                }
                if (p95 >= config.hardResidualMeters) {
                    add("temporal-depth-hard-residual")
                }
                if (translation > config.maxCameraTranslationMeters * 0.55) {
                    add("temporal-depth-camera-motion-high")
                }
                if (rotation > config.maxCameraRotationRadians * 0.55) {
                    add("temporal-depth-camera-rotation-high")
                }
                if (effectiveConfidence < 0.40) {
                    add("temporal-depth-confidence-low")
                }
            }.distinct()

        val recommendations =
            buildList {
                if (validRatio < config.minimumValidRatio) {
                    add(
                        "Increase synchronized depth coverage before giving this frame full TSDF influence.",
                    )
                }
                if (p95 >= config.warningResidualMeters) {
                    add(
                        "Reduce temporal fusion weight for inconsistent depth regions.",
                    )
                }
                if (translation > config.maxCameraTranslationMeters * 0.55) {
                    add(
                        "Use camera-motion-compensated evidence rather than raw pixel-to-pixel depth agreement.",
                    )
                }
                if (rotation > config.maxCameraRotationRadians * 0.55) {
                    add(
                        "Preserve geometric reprojection and reduce temporal smoothing during rapid camera rotation.",
                    )
                }
            }.distinct().take(6)

        history =
            History(
                timestampNanos = timestampNanos,
                pose = cameraPose,
                samples = samples,
                confidence = effectiveConfidence,
            )

        return Result(
            timestampNanos = timestampNanos,
            confidence = effectiveConfidence,
            rawConfidence = raw,
            validSampleRatio = validRatio,
            meanResidualMeters = mean,
            medianResidualMeters = median,
            p95ResidualMeters = p95,
            maxResidualMeters = maxResidual,
            cameraTranslationMeters = translation,
            cameraRotationRadians = rotation,
            accepted = accepted,
            warmedUp = true,
            hardReset = false,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    /**
     * Coverage is the confidence-weighted share of configured grid probes with usable depth.
     * Weak per-pixel depth evidence therefore cannot masquerade as a fully covered warmup frame.
     */
    private fun sampleCoverage(samples: List<Sample>): Double =
        (
            samples.sumOf { it.confidence } /
                (config.gridWidth * config.gridHeight).toDouble()
            ).coerceIn(0.0, 1.0)

    private fun sampleDepth(
        depthMeters: FloatArray,
        width: Int,
        height: Int,
        depthConfidence: FloatArray?,
    ): List<Sample> {
        val result =
            ArrayList<Sample>(
                config.gridWidth * config.gridHeight,
            )

        for (gy in 0 until config.gridHeight) {
            val normalizedY =
                if (config.gridHeight == 1) {
                    0.5
                } else {
                    gy.toDouble() /
                        (config.gridHeight - 1)
                }

            val y =
                (
                    normalizedY *
                        (height - 1)
                    ).toInt().coerceIn(0, height - 1)

            for (gx in 0 until config.gridWidth) {
                val normalizedX =
                    if (config.gridWidth == 1) {
                        0.5
                    } else {
                        gx.toDouble() /
                            (config.gridWidth - 1)
                    }

                val x =
                    (
                        normalizedX *
                            (width - 1)
                        ).toInt().coerceIn(0, width - 1)

                val index = y * width + x
                val depth = depthMeters[index].toDouble()
                val pixelConfidence =
                    if (depthConfidence == null) {
                        1.0
                    } else {
                        depthConfidence[index]
                            .takeIf { it.isFinite() }
                            ?.toDouble()
                            ?.coerceIn(0.0, 1.0)
                            ?: 0.0
                    }

                if (
                    depth.isFinite() &&
                    depth >= config.minimumDepthMeters &&
                    depth <= config.maximumDepthMeters &&
                    pixelConfidence > 0.0
                ) {
                    result +=
                        Sample(
                            x = x,
                            y = y,
                            depthMeters = depth,
                            confidence = pixelConfidence,
                        )
                }
            }
        }

        return result
    }

    private fun transformPreviousSampleToCurrentDepth(
        sample: Sample,
        previousPose: Pose,
        currentPose: Pose,
        intrinsics: CameraIntrinsics,
        width: Int,
        height: Int,
        depthConvention: DepthSampleConvention,
    ): ProjectedDepthPoint? {
        val x =
            sample.x.toDouble()
        val y =
            sample.y.toDouble()

        // DEPTH16 carries sensor-defined range measurements. Respect the selected NoX depth
        // convention instead of mixing optical-axis depth with Euclidean ray distance.
        val cameraRay =
            Vec3(
                (x - intrinsics.cx) /
                    intrinsics.fx.coerceAtLeast(1e-8),
                (y - intrinsics.cy) /
                    intrinsics.fy.coerceAtLeast(1e-8),
                1.0,
            )

        val previousCameraPoint =
            when (depthConvention) {
                DepthSampleConvention.EUCLIDEAN_RAY ->
                    cameraRay.normalized() * sample.depthMeters
                DepthSampleConvention.OPTICAL_AXIS_Z ->
                    Vec3(
                        cameraRay.x * sample.depthMeters,
                        cameraRay.y * sample.depthMeters,
                        sample.depthMeters,
                    )
            }

        val worldPoint =
            previousPose.position +
                previousPose.rotation.rotate(
                    previousCameraPoint,
                )

        val currentCameraPoint =
            currentPose.rotation
                .conjugate()
                .rotate(
                    worldPoint -
                        currentPose.position,
                )

        if (
            !currentCameraPoint.x.isFinite() ||
                !currentCameraPoint.y.isFinite() ||
                !currentCameraPoint.z.isFinite() ||
                currentCameraPoint.z <= 0.0
        ) {
            return null
        }
        val projectedDepthMeters =
            when (depthConvention) {
                DepthSampleConvention.EUCLIDEAN_RAY ->
                    currentCameraPoint.norm()
                DepthSampleConvention.OPTICAL_AXIS_Z ->
                    currentCameraPoint.z
            }
        if (
            !projectedDepthMeters.isFinite() ||
                projectedDepthMeters < config.minimumDepthMeters ||
                projectedDepthMeters > config.maximumDepthMeters
        ) {
            return null
        }

        val projectedX =
            intrinsics.fx *
                (
                    currentCameraPoint.x /
                        currentCameraPoint.z
                    ) +
                intrinsics.cx

        val projectedY =
            intrinsics.fy *
                (
                    currentCameraPoint.y /
                        currentCameraPoint.z
                    ) +
                intrinsics.cy

        if (
            projectedX < 0.0 ||
            projectedX > width - 1.0 ||
            projectedY < 0.0 ||
            projectedY > height - 1.0
        ) {
            return null
        }

        return ProjectedDepthPoint(
            x = projectedX,
            y = projectedY,
            depthMeters = projectedDepthMeters,
        )
    }

    /**
     * Samples the current depth image at a motion-compensated sub-pixel projection.
     *
     * Bilinear sampling is used only across a locally continuous depth surface. If a neighbor is
     * invalid or the footprint crosses a depth discontinuity, use the nearest valid sample with
     * reduced confidence instead of blending foreground and background geometry.
     */
    private fun depthAtNormalizedPixel(
        depthMeters: FloatArray,
        width: Int,
        height: Int,
        x: Double,
        y: Double,
        depthConfidence: FloatArray?,
    ): DepthAtPixel? {
        if (!x.isFinite() || !y.isFinite()) return null
        if (x < 0.0 || x > width - 1.0 || y < 0.0 || y > height - 1.0) {
            return null
        }

        val x0 = floor(x).toInt()
        val y0 = floor(y).toInt()
        val x1 = (x0 + 1).coerceAtMost(width - 1)
        val y1 = (y0 + 1).coerceAtMost(height - 1)
        val dx = (x - x0).coerceIn(0.0, 1.0)
        val dy = (y - y0).coerceIn(0.0, 1.0)
        val w00 = (1.0 - dx) * (1.0 - dy)
        val w10 = dx * (1.0 - dy)
        val w01 = (1.0 - dx) * dy
        val w11 = dx * dy

        val index00 = y0 * width + x0
        val index10 = y0 * width + x1
        val index01 = y1 * width + x0
        val index11 = y1 * width + x1
        val d00 = depthMeters[index00].toDouble()
        val d10 = depthMeters[index10].toDouble()
        val d01 = depthMeters[index01].toDouble()
        val d11 = depthMeters[index11].toDouble()
        val c00 = confidenceAt(index00, depthConfidence)
        val c10 = confidenceAt(index10, depthConfidence)
        val c01 = confidenceAt(index01, depthConfidence)
        val c11 = confidenceAt(index11, depthConfidence)
        val v00 = isValidDepth(d00) && c00 > 0.0
        val v10 = isValidDepth(d10) && c10 > 0.0
        val v01 = isValidDepth(d01) && c01 > 0.0
        val v11 = isValidDepth(d11) && c11 > 0.0

        val footprintInvalid =
            (w00 > PIXEL_WEIGHT_EPSILON && !v00) ||
                (w10 > PIXEL_WEIGHT_EPSILON && !v10) ||
                (w01 > PIXEL_WEIGHT_EPSILON && !v01) ||
                (w11 > PIXEL_WEIGHT_EPSILON && !v11)

        if (footprintInvalid) {
            return nearestDepthAt(
                x = x,
                y = y,
                width = width,
                depthMeters = depthMeters,
                depthConfidence = depthConfidence,
                confidenceScale = DEPTH_EDGE_CONFIDENCE_SCALE,
            )
        }

        var minimumDepth = Double.POSITIVE_INFINITY
        var maximumDepth = Double.NEGATIVE_INFINITY
        if (w00 > PIXEL_WEIGHT_EPSILON) {
            minimumDepth = min(minimumDepth, d00)
            maximumDepth = max(maximumDepth, d00)
        }
        if (w10 > PIXEL_WEIGHT_EPSILON) {
            minimumDepth = min(minimumDepth, d10)
            maximumDepth = max(maximumDepth, d10)
        }
        if (w01 > PIXEL_WEIGHT_EPSILON) {
            minimumDepth = min(minimumDepth, d01)
            maximumDepth = max(maximumDepth, d01)
        }
        if (w11 > PIXEL_WEIGHT_EPSILON) {
            minimumDepth = min(minimumDepth, d11)
            maximumDepth = max(maximumDepth, d11)
        }

        if (
            maximumDepth - minimumDepth >
                max(config.hardResidualMeters, config.warningResidualMeters * 2.0)
        ) {
            return nearestDepthAt(
                x = x,
                y = y,
                width = width,
                depthMeters = depthMeters,
                depthConfidence = depthConfidence,
                confidenceScale = DEPTH_EDGE_CONFIDENCE_SCALE,
            )
        }

        val weight00 = if (w00 > PIXEL_WEIGHT_EPSILON) w00 * c00 else 0.0
        val weight10 = if (w10 > PIXEL_WEIGHT_EPSILON) w10 * c10 else 0.0
        val weight01 = if (w01 > PIXEL_WEIGHT_EPSILON) w01 * c01 else 0.0
        val weight11 = if (w11 > PIXEL_WEIGHT_EPSILON) w11 * c11 else 0.0
        val totalWeight = weight00 + weight10 + weight01 + weight11
        var weightedDepth = 0.0
        if (weight00 > 0.0) weightedDepth += d00 * weight00
        if (weight10 > 0.0) weightedDepth += d10 * weight10
        if (weight01 > 0.0) weightedDepth += d01 * weight01
        if (weight11 > 0.0) weightedDepth += d11 * weight11
        if (!totalWeight.isFinite() || totalWeight <= 0.0) return null
        return DepthAtPixel(
            depthMeters = weightedDepth / totalWeight,
            confidence = totalWeight.coerceIn(0.0, 1.0),
        )
    }

    private fun isValidDepth(value: Double): Boolean =
        value.isFinite() &&
            value >= config.minimumDepthMeters &&
            value <= config.maximumDepthMeters

    private fun confidenceAt(
        index: Int,
        depthConfidence: FloatArray?,
    ): Double =
        if (depthConfidence == null) {
            1.0
        } else {
            depthConfidence[index]
                .takeIf { it.isFinite() }
                ?.toDouble()
                ?.coerceIn(0.0, 1.0)
                ?: 0.0
        }

    private fun depthAtIndex(
        index: Int,
        depthMeters: FloatArray,
        depthConfidence: FloatArray?,
    ): DepthAtPixel? {
        val value = depthMeters[index].toDouble()
        if (!isValidDepth(value)) return null
        val confidence = confidenceAt(index, depthConfidence)
        if (confidence <= 0.0) return null
        return DepthAtPixel(value, confidence)
    }

    private fun nearestDepthAt(
        x: Double,
        y: Double,
        width: Int,
        depthMeters: FloatArray,
        depthConfidence: FloatArray?,
        confidenceScale: Double,
    ): DepthAtPixel? {
        val nearestX = x.roundToInt().coerceIn(0, width - 1)
        val nearestY = y.roundToInt()
        val nearest = depthAtIndex(
            index = nearestY * width + nearestX,
            depthMeters = depthMeters,
            depthConfidence = depthConfidence,
        ) ?: return null
        return nearest.copy(
            confidence = (nearest.confidence * confidenceScale).coerceIn(0.0, 1.0),
        )
    }

    private fun robustMean(
        residuals: List<ResidualSample>,
    ): Double {
        if (residuals.isEmpty()) {
            return config.hardResidualMeters
        }

        var weightedSum = 0.0
        var weightSum = 0.0

        residuals.forEach { sample ->
            val absolute =
                abs(sample.residualMeters)

            val huberWeight =
                if (
                    absolute <=
                        config.robustHuberDeltaMeters
                ) {
                    1.0
                } else {
                    config.robustHuberDeltaMeters /
                        absolute.coerceAtLeast(1e-9)
                }
            val weight = sample.confidenceWeight * huberWeight

            weightedSum +=
                sample.residualMeters * weight
            weightSum += weight
        }

        return (
            weightedSum /
                weightSum.coerceAtLeast(1e-9)
            ).coerceAtLeast(0.0)
    }

    private fun percentile(
        sorted: List<ResidualSample>,
        probability: Double,
    ): Double {
        if (sorted.isEmpty()) {
            return config.hardResidualMeters
        }

        val totalWeight =
            sorted.sumOf { it.confidenceWeight }
        if (totalWeight <= 0.0) {
            return config.hardResidualMeters
        }
        val targetWeight =
            probability.coerceIn(0.0, 1.0) * totalWeight
        var cumulativeWeight = 0.0
        sorted.forEach { sample ->
            cumulativeWeight += sample.confidenceWeight
            if (cumulativeWeight >= targetWeight) {
                return sample.residualMeters
            }
        }

        return sorted.last().residualMeters
    }

    private fun cameraMotionQuality(
        translation: Double,
        rotation: Double,
    ): Double {
        val translationQuality =
            (
                1.0 -
                    translation /
                        config.maxCameraTranslationMeters
                ).coerceIn(0.0, 1.0)

        val rotationQuality =
            (
                1.0 -
                    rotation /
                        config.maxCameraRotationRadians
                ).coerceIn(0.0, 1.0)

        return (
            translationQuality * 0.58 +
                rotationQuality * 0.42
            ).coerceIn(0.0, 1.0)
    }

    private fun quaternionAngularDistance(
        a: com.nivukx.nox.core.model.Quat,
        b: com.nivukx.nox.core.model.Quat,
    ): Double {
        val qa = a.normalized()
        val qb = b.normalized()

        val dot =
            abs(
                qa.w * qb.w +
                    qa.x * qb.x +
                    qa.y * qb.y +
                    qa.z * qb.z,
            ).coerceIn(0.0, 1.0)

        return (
            2.0 *
                kotlin.math.acos(dot)
            ).coerceIn(0.0, Math.PI)
    }
}

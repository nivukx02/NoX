package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DenseMeshQuality
import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Vec3
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Sparse TSDF fusion for calibrated multi-view depth.
 *
 * This is a bounded voxel-hash implementation rather than a dense cubic volume. It can therefore
 * operate on mobile memory budgets while keeping metric geometry. Surface extraction uses local
 * sign-crossing neighborhoods and emits a watertight-oriented manifold where the occupied voxel
 * topology supports it.
 */
class SparseTsdfFusionEngine(
    private val voxelSizeMeters: Double = 0.0125,
    private val truncationDistanceMeters: Double = 0.05,
    private val maxVoxels: Int = 1_000_000,
    private val temporalHalfLifeSeconds: Double = 0.10,
    private val minimumRetainedWeight: Double = 0.045,
    private val decayBudgetVoxels: Int = 24_000,
    private val extractionCellBudget: Int = 360_000,
    private val maxSurfaceSurfels: Int = 250_000,
    private val surfaceSurfelHalfLifeSeconds: Double = 0.18,
    private val surfaceSurfelMinWeight: Double = 0.035,
) {

    init {
        require(voxelSizeMeters.isFinite() && voxelSizeMeters > 0.0)
        require(
            truncationDistanceMeters.isFinite() &&
                truncationDistanceMeters > voxelSizeMeters
        )
        require(maxVoxels > 0)
        require(
            temporalHalfLifeSeconds.isFinite() &&
                temporalHalfLifeSeconds > 0.0
        )
        require(
            minimumRetainedWeight.isFinite() &&
                minimumRetainedWeight > 0.0
        )
        require(decayBudgetVoxels > 0)
        require(extractionCellBudget > 0)
        require(maxSurfaceSurfels > 0)
        require(
            surfaceSurfelHalfLifeSeconds.isFinite() &&
                surfaceSurfelHalfLifeSeconds > 0.0
        )
        require(
            surfaceSurfelMinWeight.isFinite() &&
                surfaceSurfelMinWeight > 0.0
        )
    }

    /**
     * Fine surface evidence retained beside the voxel field.
     *
     * Surfels are not a second reconstruction truth. They are a high-frequency evidence cache
     * that preserves local surface position, normal, uncertainty and motion for downstream
     * detail-aware meshing and temporal consistency checks.
     */
    data class SurfaceSurfel(
        val position: Vec3,
        val normal: Vec3,
        val confidence: Double,
        val uncertaintyMeters: Double,
        val weight: Double,
        val ageSeconds: Double,
        val observationCount: Long,
        val normalStability: Double,
        val speedMetersPerSecond: Double,
        val varianceX: Double = 0.0004,
        val varianceY: Double = 0.0004,
        val varianceZ: Double = 0.0004,
        val dynamicProbability: Double = 0.0,
        val temporalStability: Double = 1.0,
    ) {
        val finite: Boolean
            get() =
                position.x.isFinite() &&
                    position.y.isFinite() &&
                    position.z.isFinite() &&
                    normal.x.isFinite() &&
                    normal.y.isFinite() &&
                    normal.z.isFinite() &&
                    confidence.isFinite() &&
                    uncertaintyMeters.isFinite() &&
                    weight.isFinite() &&
                    ageSeconds.isFinite() &&
                    normalStability.isFinite() &&
                    speedMetersPerSecond.isFinite() &&
                    varianceX.isFinite() &&
                    varianceY.isFinite() &&
                    varianceZ.isFinite() &&
                    dynamicProbability.isFinite() &&
                    temporalStability.isFinite()
    }

    data class SurfaceDetailMetrics(
        val surfelCount: Int,
        val meanConfidence: Double,
        val meanUncertaintyMeters: Double,
        val meanNormalStability: Double,
        val meanSpeedMetersPerSecond: Double,
        val meanDynamicProbability: Double = 0.0,
        val meanTemporalStability: Double = 0.0,
        val trustedGaussianRatio: Double = 0.0,
        val temporallyStableRatio: Double,
        val memoryUtilization: Double,
    )

    data class AdaptiveEvidence(
        val motionEnergy: Double = 0.0,
        val uncertainty: Double = 0.0,
        val cameraCoverage: Double = 1.0,
        val cameraInformationGain: Double = 1.0,
        val reconstructionConfidence: Double = 1.0,
        val temporalDepthConsistency: Double = 1.0,
        val thermalHeadroom: Double = 1.0,
        val gpuHeadroom: Double = 1.0,
        val requestedDetail: Double = 1.0,
    ) {
        init {
            listOf(
                motionEnergy,
                uncertainty,
                cameraCoverage,
                cameraInformationGain,
                reconstructionConfidence,
                thermalHeadroom,
                gpuHeadroom,
                requestedDetail,
            ).forEach { require(it.isFinite() && it in 0.0..1.0) }
        }
    }

    data class AdaptiveDecision(
        val stride: Int,
        val confidenceScale: Double,
        val detailLevel: Double,
        val computePressure: Double,
        val reason: String,
        val acceptedSamples: Int = 0,
        val rejectedSamples: Int = 0,
        val meanEvidenceWeight: Double = 0.0,
        val meanSurfaceNormalWeight: Double = 1.0,
    )

    data class PerSampleEvidence(
        val depthUncertaintyMeters: FloatArray? = null,
        val staticConfidence: FloatArray? = null,
        val temporalConfidence: FloatArray? = null,
        val photometricConfidence: FloatArray? = null,
    )

    data class EvidenceFusionDecision(
        val acceptedSamples: Int,
        val rejectedSamples: Int,
        val meanEvidenceWeight: Double,
        val meanRobustWeight: Double,
        val meanAngularWeight: Double,
        val meanSurfaceNormalWeight: Double = 1.0,
        val reason: String,
    ) {
        init {
            require(acceptedSamples >= 0)
            require(rejectedSamples >= 0)
            require(meanEvidenceWeight.isFinite())
            require(meanRobustWeight.isFinite())
            require(meanAngularWeight.isFinite())
            require(meanSurfaceNormalWeight.isFinite())
        }
    }

    data class UncertaintyMetrics(
        val meanSignedDistanceStdDev: Double,
        val highUncertaintyVoxelRatio: Double,
        val confidenceFactor: Double,
    ) {
        init {
            require(meanSignedDistanceStdDev.isFinite())
            require(highUncertaintyVoxelRatio.isFinite())
            require(confidenceFactor.isFinite())
        }
    }

    private data class Voxel(
        var sdf: Double,
        var weight: Double,
        var lastTimestampNanos: Long,
        var observationCount: Long,
        var sdfVariance: Double,
    )

    private data class MutableSurfel(
        var position: Vec3,
        var normal: Vec3,
        var confidence: Double,
        var uncertaintyMeters: Double,
        var weight: Double,
        var lastTimestampNanos: Long,
        var observationCount: Long,
        var normalStability: Double,
        var velocity: Vec3,
        var varianceX: Double,
        var varianceY: Double,
        var varianceZ: Double,
        var dynamicProbability: Double,
        var temporalStability: Double,
    )

    private val voxels = LinkedHashMap<Long, Voxel>(4096, 0.75f, true)
    private val surfaceSurfels =
        LinkedHashMap<Long, MutableSurfel>(16_384, 0.75f, true)
    private var lastDecayTimestampNanos: Long = Long.MIN_VALUE
    private var lastSurfaceSurfelDecayTimestampNanos: Long = Long.MIN_VALUE
    private var outOfOrderFrameCount: Long = 0L
    private var voxelEvictionCount: Long = 0L
    private var surfaceSurfelEvictionCount: Long = 0L
    private var surfaceSurfelRejectedCount: Long = 0L

    @Synchronized
    fun reset() {
        voxels.clear()
        surfaceSurfels.clear()
        lastDecayTimestampNanos = Long.MIN_VALUE
        lastSurfaceSurfelDecayTimestampNanos = Long.MIN_VALUE
        outOfOrderFrameCount = 0L
        voxelEvictionCount = 0L
        surfaceSurfelEvictionCount = 0L
        surfaceSurfelRejectedCount = 0L
    }

    @Synchronized
    fun integrateDepth(
        depthMeters: FloatArray,
        width: Int,
        height: Int,
        intrinsics: CameraIntrinsics,
        cameraPose: Pose,
        timestampNanos: Long,
        stride: Int = 3,
        confidence: Double = 1.0,
    ) {
        require(depthMeters.size == width * height)
        require(width > 0 && height > 0)
        require(timestampNanos >= 0L)
        require(confidence.isFinite() && confidence in 0.0..1.0)
        require(intrinsics.fx.isFinite() && intrinsics.fx > 0.0)
        require(intrinsics.fy.isFinite() && intrinsics.fy > 0.0)

        if (!acceptTimestamp(timestampNanos)) return
        decayTemporalState(timestampNanos)

        val invFx = 1.0 / intrinsics.fx.coerceAtLeast(1e-8)
        val invFy = 1.0 / intrinsics.fy.coerceAtLeast(1e-8)

        for (y in 0 until height step stride.coerceAtLeast(1)) {
            for (x in 0 until width step stride.coerceAtLeast(1)) {
                val depth = depthMeters[y * width + x].toDouble()
                if (!depth.isFinite() || depth <= 0.05 || depth > 20.0) continue

                val rayCamera = Vec3(
                    (x - intrinsics.cx) * invFx,
                    (y - intrinsics.cy) * invFy,
                    1.0,
                ).normalized()

                val rayWorld = cameraPose.rotation
                    .rotate(rayCamera)
                    .normalized()
                val origin = cameraPose.position

                val normalCamera =
                    estimateSurfaceNormal(
                        depthMeters = depthMeters,
                        width = width,
                        height = height,
                        x = x,
                        y = y,
                        stride = stride.coerceAtLeast(1),
                        centerDepthMeters = depth,
                        intrinsics = intrinsics,
                    )
                val normalWorld =
                    cameraPose.rotation
                        .rotate(normalCamera)
                        .normalized()

                val surfacePoint =
                    origin + rayWorld * depth

                val normalObservability =
                    kotlin.math.abs(
                        normalCamera.dot(rayCamera * -1.0),
                    ).coerceIn(0.10, 1.0)
                val geometricUncertainty =
                    (
                        voxelSizeMeters *
                            (
                                0.22 +
                                    0.78 * normalObservability
                            ) /
                            kotlin.math.sqrt(
                                confidence.coerceAtLeast(0.05),
                            )
                    )
                    .coerceIn(0.0015, 0.08)
                val rangeScaledUncertainty =
                    (
                        geometricUncertainty *
                            (
                                0.55 +
                                    (depth / 2.0)
                                        .coerceIn(0.0, 3.0) * 0.45
                            )
                    ).coerceIn(0.0015, 0.12)

                updateSurfaceSurfel(
                    position = surfacePoint,
                    normal = normalWorld,
                    confidence = confidence,
                    uncertaintyMeters = rangeScaledUncertainty,
                    dynamicProbability = 0.0,
                    temporalConsistency = 1.0,
                    timestampNanos = timestampNanos,
                )

                val start = max(
                    0.05,
                    depth - truncationDistanceMeters,
                )
                val end = depth + truncationDistanceMeters

                var sample = start
                while (sample <= end) {
                    val sdf = (
                        depth - sample
                    ) / truncationDistanceMeters

                    if (sdf >= -1.0) {
                        val world = origin + rayWorld * sample
                        integrateVoxel(
                            world = world,
                            sdf = sdf,
                            confidence = confidence,
                            timestampNanos = timestampNanos,
                        )
                    }

                    sample += voxelSizeMeters * 0.75
                }
            }
        }
    }

    private fun validateDepthInput(
        depthMeters: FloatArray,
        width: Int,
        height: Int,
        depthConfidence: FloatArray?,
    ) {
        require(width > 0 && height > 0)
        val sampleCount = width.toLong() * height.toLong()
        require(sampleCount <= Int.MAX_VALUE.toLong()) {
            "Depth dimensions exceed JVM array sample capacity"
        }
        require(depthMeters.size.toLong() == sampleCount) {
            "Depth samples must match depth dimensions"
        }
        require(depthConfidence == null || depthConfidence.size.toLong() == sampleCount) {
            "Per-pixel confidence must match depth dimensions"
        }
    }

    @Synchronized
    fun integrateDepthAdaptive(
        depthMeters: FloatArray,
        width: Int,
        height: Int,
        intrinsics: CameraIntrinsics,
        cameraPose: Pose,
        timestampNanos: Long,
        baseStride: Int = 2,
        confidence: Double = 1.0,
        evidence: AdaptiveEvidence = AdaptiveEvidence(),
        uncertaintyModel: DepthUncertaintyModelEngine? = null,
        depthConfidence: FloatArray? = null,
    ): AdaptiveDecision {
        validateDepthInput(depthMeters, width, height, depthConfidence)
        require(timestampNanos >= 0L)
        require(confidence.isFinite() && confidence in 0.0..1.0)
        require(intrinsics.fx.isFinite() && intrinsics.fx > 0.0)
        require(intrinsics.fy.isFinite() && intrinsics.fy > 0.0)
        if (!acceptTimestamp(timestampNanos)) {
            return AdaptiveDecision(
                stride = baseStride.coerceAtLeast(1),
                confidenceScale = 0.0,
                detailLevel = 0.0,
                computePressure = 1.0,
                reason = "stale_frame_rejected",
                acceptedSamples = 0,
                rejectedSamples = depthMeters.size,
                meanEvidenceWeight = 0.0,
            )
        }

        val pressure =
            evidence.motionEnergy * 0.23 +
                evidence.uncertainty * 0.22 +
                (1.0 - evidence.cameraInformationGain) * 0.12 +
                (1.0 - evidence.thermalHeadroom) * 0.18 +
                (1.0 - evidence.gpuHeadroom) * 0.16 +
                (1.0 - evidence.reconstructionConfidence) * 0.10 +
                (1.0 - evidence.temporalDepthConsistency) * 0.12 +
                (1.0 - evidence.cameraCoverage) * 0.07

        val demand =
            evidence.requestedDetail * 0.30 +
                evidence.cameraCoverage * 0.20 +
                evidence.cameraInformationGain * 0.12 +
                evidence.reconstructionConfidence * 0.20 +
                evidence.temporalDepthConsistency * 0.12 +
                evidence.gpuHeadroom * 0.12 +
                evidence.thermalHeadroom * 0.12

        val detailLevel =
            (demand * (1.0 - pressure * 0.74))
                .coerceIn(0.0, 1.0)

        val strideMultiplier = when {
            detailLevel >= 0.82 -> 1
            detailLevel >= 0.62 -> 2
            detailLevel >= 0.44 -> 3
            else -> 4
        }

        val effectiveStride =
            (baseStride.coerceAtLeast(1) * strideMultiplier)
                .coerceAtMost(12)

        val confidenceScale =
            (0.56 + detailLevel * 0.44)
                .coerceIn(0.50, 1.0)

        val reason = when {
            detailLevel >= 0.82 -> "fine_detail"
            evidence.motionEnergy >= 0.78 -> "fast_motion"
            evidence.uncertainty >= 0.65 -> "uncertainty_limited"
            evidence.temporalDepthConsistency < 0.42 ->
                "temporal_depth_limited"
            evidence.cameraInformationGain < 0.42 ->
                "view_information_limited"
            evidence.cameraCoverage < 0.45 -> "coverage_limited"
            pressure >= 0.68 -> "compute_pressure"
            else -> "balanced"
        }

        val evidenceDecision =
            integrateDepthEvidenceWeighted(
                depthMeters = depthMeters,
                width = width,
                height = height,
                intrinsics = intrinsics,
                cameraPose = cameraPose,
                timestampNanos = timestampNanos,
                stride = effectiveStride,
                confidence = confidence * confidenceScale,
                uncertaintyModel = uncertaintyModel,
                depthConfidence = depthConfidence,
            )

        return AdaptiveDecision(
            stride = effectiveStride,
            confidenceScale = confidenceScale,
            detailLevel = detailLevel,
            computePressure = pressure.coerceIn(0.0, 1.0),
            reason = reason,
            acceptedSamples =
                evidenceDecision.acceptedSamples,
            rejectedSamples =
                evidenceDecision.rejectedSamples,
            meanEvidenceWeight =
                evidenceDecision.meanEvidenceWeight,
            meanSurfaceNormalWeight =
                evidenceDecision.meanSurfaceNormalWeight,
        )
    }

    @Synchronized
    fun integrateDepthEvidenceWeighted(
        depthMeters: FloatArray,
        width: Int,
        height: Int,
        intrinsics: CameraIntrinsics,
        cameraPose: Pose,
        timestampNanos: Long,
        stride: Int = 2,
        confidence: Double = 1.0,
        evidence: PerSampleEvidence = PerSampleEvidence(),
        uncertaintyModel: DepthUncertaintyModelEngine? = null,
        depthConfidence: FloatArray? = null,
    ): EvidenceFusionDecision {
        validateDepthInput(depthMeters, width, height, depthConfidence)
        require(timestampNanos >= 0L)
        require(confidence.isFinite() && confidence in 0.0..1.0)
        require(intrinsics.fx.isFinite() && intrinsics.fx > 0.0)
        require(intrinsics.fy.isFinite() && intrinsics.fy > 0.0)
        if (!acceptTimestamp(timestampNanos)) {
            return EvidenceFusionDecision(
                acceptedSamples = 0,
                rejectedSamples = depthMeters.size,
                meanEvidenceWeight = 0.0,
                meanRobustWeight = 0.0,
                meanAngularWeight = 0.0,
                reason = "stale_frame_rejected",
            )
        }
        require(width > 0 && height > 0)

        val sampleCount = width * height
        fun validateEvidenceArray(
            name: String,
            values: FloatArray?,
        ) {
            require(
                values == null || values.size >= sampleCount,
            ) {
                "$name must match depth dimensions"
            }
        }

        validateEvidenceArray(
            "depthUncertaintyMeters",
            evidence.depthUncertaintyMeters,
        )
        validateEvidenceArray(
            "staticConfidence",
            evidence.staticConfidence,
        )
        validateEvidenceArray(
            "temporalConfidence",
            evidence.temporalConfidence,
        )
        validateEvidenceArray(
            "photometricConfidence",
            evidence.photometricConfidence,
        )

        decayTemporalState(timestampNanos)

        val invFx =
            1.0 / intrinsics.fx.coerceAtLeast(1e-8)
        val invFy =
            1.0 / intrinsics.fy.coerceAtLeast(1e-8)
        val safeStride =
            stride.coerceAtLeast(1)
        val globalConfidence =
            confidence.coerceIn(0.01, 32.0)

        var accepted = 0
        var rejected = 0
        var evidenceWeightSum = 0.0
        var robustWeightSum = 0.0
        var angularWeightSum = 0.0
        var surfaceNormalWeightSum = 0.0

        for (
            y in 0 until height step safeStride
        ) {
            for (
                x in 0 until width step safeStride
            ) {
                val index = y * width + x
                val depth =
                    depthMeters[index].toDouble()

                if (
                    !depth.isFinite() ||
                    depth <= 0.05 ||
                    depth > 20.0
                ) {
                    rejected += 1
                    continue
                }

                // DEPTH16 confidence is per sample. Non-finite values are rejected and zero
                // confidence must not sneak into the scalar-weighted fallback path.
                val pixelConfidence =
                    depthConfidence?.get(index)?.toDouble()
                        ?.takeIf { it.isFinite() }
                        ?.coerceIn(0.0, 1.0)
                        ?: if (depthConfidence == null) 1.0 else 0.0
                if (pixelConfidence <= 0.0) {
                    rejected += 1
                    continue
                }

                val rayCamera =
                    Vec3(
                        (x - intrinsics.cx) * invFx,
                        (y - intrinsics.cy) * invFy,
                        1.0,
                    ).normalized()

                val angularWeight =
                    kotlin.math.abs(
                        rayCamera.z.coerceIn(0.0, 1.0),
                    ).let {
                        kotlin.math.sqrt(it)
                    }.coerceIn(0.20, 1.0)

                val surfaceNormalWeight =
                    estimateSurfaceNormalObservability(
                        depthMeters = depthMeters,
                        width = width,
                        height = height,
                        x = x,
                        y = y,
                        stride = safeStride,
                        centerDepthMeters = depth,
                        intrinsics = intrinsics,
                        rayCamera = rayCamera,
                    )

                val modeledUncertainty =
                    uncertaintyModel?.estimate(
                        DepthUncertaintyModelEngine.Input(
                            depthMeters = depth,
                            neighborDepthX =
                                depthMeters[
                                    y * width +
                                        (x + safeStride).coerceAtMost(
                                            width - 1,
                                        )
                                ].toDouble().takeIf {
                                    it.isFinite() && it > 0.05
                                },
                            neighborDepthY =
                                depthMeters[
                                    (y + safeStride).coerceAtMost(
                                        height - 1,
                                    ) * width + x
                                ].toDouble().takeIf {
                                    it.isFinite() && it > 0.05
                                },
                            rayZ = rayCamera.z,
                            surfaceNormalObservability =
                                surfaceNormalWeight,
                            temporalConsistency =
                                evidence.temporalConfidence
                                    ?.get(index)
                                    ?.toDouble()
                                    ?.coerceIn(0.0, 1.0)
                                    ?: 1.0,
                            sourceConfidence =
                                evidence.photometricConfidence
                                    ?.get(index)
                                    ?.toDouble()
                                    ?.coerceIn(0.0, 1.0)
                                    ?: globalConfidence
                                        .coerceIn(0.0, 1.0),
                            sourceClass =
                                DepthUncertaintyModelEngine.SourceClass.DEPTH_SENSOR,
                        ),
                    )?.sigmaMeters

                val uncertainty =
                    (
                        evidence.depthUncertaintyMeters
                            ?.get(index)
                            ?.toDouble()
                            ?.takeIf { it.isFinite() && it >= 0.0 }
                            ?: modeledUncertainty
                            ?: 0.035
                        ).coerceIn(0.0, 0.75)

                val staticConfidence =
                    (
                        evidence.staticConfidence
                            ?.get(index)
                            ?.toDouble()
                            ?: 1.0
                        ).coerceIn(0.0, 1.0)

                val temporalConfidence =
                    (
                        evidence.temporalConfidence
                            ?.get(index)
                            ?.toDouble()
                            ?: 1.0
                        ).coerceIn(0.0, 1.0)

                val photometricConfidence =
                    (
                        evidence.photometricConfidence
                            ?.get(index)
                            ?.toDouble()
                            ?: 1.0
                        ).coerceIn(0.0, 1.0)

                // Do not admit near-untrusted samples into the static TSDF path. A very
                // low static score combined with weak temporal support is explicitly treated
                // as dynamic/unreliable evidence rather than allowing it to accumulate.
                if (
                    staticConfidence < 0.05 &&
                    temporalConfidence < 0.25
                ) {
                    rejected += 1
                    continue
                }

                val uncertaintyWeight =
                    kotlin.math.exp(
                        -uncertainty /
                            0.055,
                    ).coerceIn(0.0, 1.0)

                val evidenceWeight =
                    (
                        uncertaintyWeight * 0.34 +
                            staticConfidence * 0.22 +
                            temporalConfidence * 0.18 +
                            photometricConfidence * 0.16 +
                            surfaceNormalWeight * 0.10
                        ) *
                        angularWeight *
                        pixelConfidence

                if (evidenceWeight < 0.035) {
                    rejected += 1
                    continue
                }

                val normalCamera =
                    estimateSurfaceNormal(
                        depthMeters = depthMeters,
                        width = width,
                        height = height,
                        x = x,
                        y = y,
                        stride = safeStride,
                        centerDepthMeters = depth,
                        intrinsics = intrinsics,
                    )
                val normalWorld =
                    cameraPose.rotation
                        .rotate(normalCamera)
                        .normalized()

                val surfacePoint =
                    cameraPose.position +
                        cameraPose.rotation
                            .rotate(rayCamera)
                            .normalized() *
                        depth

                updateSurfaceSurfel(
                    position = surfacePoint,
                    normal = normalWorld,
                    confidence =
                        globalConfidence *
                            evidenceWeight,
                    uncertaintyMeters = uncertainty,
                    dynamicProbability =
                        (1.0 - staticConfidence)
                            .coerceIn(0.0, 1.0),
                    temporalConsistency =
                        temporalConfidence,
                    timestampNanos = timestampNanos,
                )

                val start =
                    max(
                        0.05,
                        depth -
                            truncationDistanceMeters,
                    )
                val end =
                    depth +
                        truncationDistanceMeters
                var sample =
                    start
                var sampleAccepted = false
                var sampleRobustWeight = 0.0

                while (sample <= end) {
                    val sdf =
                        (
                            depth - sample
                        ) /
                            truncationDistanceMeters

                    if (sdf >= -1.0) {
                        val world =
                            cameraPose.position +
                                cameraPose.rotation
                                    .rotate(
                                        rayCamera * 1.0,
                                    )
                                    .normalized() *
                                    sample

                        val robust =
                            integrateVoxelRobust(
                                world = world,
                                sdf = sdf,
                                confidence =
                                    globalConfidence *
                                        evidenceWeight,
                                timestampNanos =
                                    timestampNanos,
                            )

                        if (robust > 0.0) {
                            sampleAccepted = true
                            sampleRobustWeight =
                                max(
                                    sampleRobustWeight,
                                    robust,
                                )
                        }
                    }

                    sample +=
                        voxelSizeMeters * 0.75
                }

                if (sampleAccepted) {
                    accepted += 1
                    evidenceWeightSum +=
                        evidenceWeight
                    robustWeightSum +=
                        sampleRobustWeight
                    angularWeightSum +=
                        angularWeight
                    surfaceNormalWeightSum +=
                        surfaceNormalWeight
                } else {
                    rejected += 1
                }
            }
        }

        val reason =
            when {
                accepted == 0 -> "all_samples_rejected"
                surfaceNormalWeightSum / accepted >= 0.78 &&
                    evidenceWeightSum / accepted >= 0.78 ->
                    "high_confidence_surface_evidence"
                surfaceNormalWeightSum / accepted < 0.40 ->
                    "surface_observability_limited"
                robustWeightSum / accepted < 0.35 ->
                    "outlier_suppressed"
                else ->
                    "uncertainty_weighted"
            }

        return EvidenceFusionDecision(
            acceptedSamples = accepted,
            rejectedSamples = rejected,
            meanEvidenceWeight =
                (
                    evidenceWeightSum /
                        accepted.coerceAtLeast(1)
                    ).coerceIn(0.0, 1.0),
            meanRobustWeight =
                (
                    robustWeightSum /
                        accepted.coerceAtLeast(1)
                    ).coerceIn(0.0, 1.0),
            meanAngularWeight =
                (
                    angularWeightSum /
                        accepted.coerceAtLeast(1)
                    ).coerceIn(0.0, 1.0),
            meanSurfaceNormalWeight =
                (
                    surfaceNormalWeightSum /
                        accepted.coerceAtLeast(1)
                    ).coerceIn(0.0, 1.0),
            reason = reason,
        )
    }

    private fun estimateSurfaceNormalObservability(
        depthMeters: FloatArray,
        width: Int,
        height: Int,
        x: Int,
        y: Int,
        stride: Int,
        centerDepthMeters: Double,
        intrinsics: CameraIntrinsics,
        rayCamera: Vec3,
    ): Double {
        val neighborX = x + stride
        val neighborY = y + stride

        if (
            neighborX >= width ||
                neighborY >= height
        ) {
            return 1.0
        }

        val depthX =
            depthMeters[y * width + neighborX].toDouble()
        val depthY =
            depthMeters[neighborY * width + x].toDouble()

        if (
            !depthX.isFinite() ||
                !depthY.isFinite() ||
                depthX <= 0.05 ||
                depthY <= 0.05
        ) {
            return 0.72
        }

        val center =
            rayCamera * centerDepthMeters

        val rayX =
            Vec3(
                (neighborX - intrinsics.cx) /
                    intrinsics.fx.coerceAtLeast(1e-8),
                (y - intrinsics.cy) /
                    intrinsics.fy.coerceAtLeast(1e-8),
                1.0,
            ).normalized()

        val rayY =
            Vec3(
                (x - intrinsics.cx) /
                    intrinsics.fx.coerceAtLeast(1e-8),
                (neighborY - intrinsics.cy) /
                    intrinsics.fy.coerceAtLeast(1e-8),
                1.0,
            ).normalized()

        val tangentX =
            rayX * depthX -
                center
        val tangentY =
            rayY * depthY -
                center

        val rawNormal =
            tangentX.cross(tangentY)
        val normalMagnitude =
            rawNormal.norm()

        if (!normalMagnitude.isFinite() ||
            normalMagnitude < 1e-6
        ) {
            return 0.72
        }

        val normal =
            rawNormal /
                normalMagnitude

        val observability =
            kotlin.math.abs(
                normal.dot(
                    rayCamera * -1.0,
                ),
            )

        return (
            0.35 +
                observability * 0.65
            ).coerceIn(0.35, 1.0)
    }

    @Synchronized
    fun extractMesh(
        timestampNanos: Long,
        isoBand: Double = 0.13,
        cellBudget: Int = extractionCellBudget,
        maxVertices: Int = 350_000,
    ): Pair<ReconstructionMesh, DenseMeshQuality>? {
        if (voxels.size < 16) return null

        decayTemporalState(timestampNanos)

        val vertices = ArrayList<MeshVertex>()
        val triangles = ArrayList<MeshTriangle>()
        val vertexConfidence = ArrayList<Double>()

        data class EdgeKey(
            val a: Long,
            val b: Long,
        )

        val edgeCache = HashMap<EdgeKey, Int>(voxels.size / 2)
        val offsets = arrayOf(
            Triple(0, 0, 0),
            Triple(1, 0, 0),
            Triple(1, 1, 0),
            Triple(0, 1, 0),
            Triple(0, 0, 1),
            Triple(1, 0, 1),
            Triple(1, 1, 1),
            Triple(0, 1, 1),
        )

        val tetrahedra = arrayOf(
            intArrayOf(0, 5, 1, 6),
            intArrayOf(0, 1, 2, 6),
            intArrayOf(0, 2, 3, 6),
            intArrayOf(0, 3, 7, 6),
            intArrayOf(0, 7, 4, 6),
            intArrayOf(0, 4, 5, 6),
        )

        fun edgeVertex(
            globalA: Long,
            globalB: Long,
            positionA: Vec3,
            positionB: Vec3,
            sdfA: Double,
            sdfB: Double,
            confidence: Double,
        ): Int {
            val key =
                if (globalA <= globalB) {
                    EdgeKey(globalA, globalB)
                } else {
                    EdgeKey(globalB, globalA)
                }

            edgeCache[key]?.let { return it }

            val denominator = sdfA - sdfB
            val interpolation =
                if (kotlin.math.abs(denominator) < 1e-9) {
                    0.5
                } else {
                    (sdfA / denominator).coerceIn(0.0, 1.0)
                }

            val position = positionA + (
                positionB - positionA
                ) * interpolation

            val index = vertices.size
            vertices += MeshVertex(position)
            vertexConfidence += confidence.coerceIn(0.0, 1.0)
            edgeCache[key] = index
            return index
        }

        fun emit(
            a: Int,
            b: Int,
            c: Int,
        ) {
            if (a == b || b == c || a == c) return
            if (
                a !in vertices.indices ||
                b !in vertices.indices ||
                c !in vertices.indices
            ) return

            triangles += MeshTriangle(a, b, c)
        }

        fun processTetra(
            cubeKeys: LongArray,
            points: Array<Vec3>,
            sdf: DoubleArray,
            weights: DoubleArray,
            tetra: IntArray,
        ) {
            val inside = BooleanArray(4) {
                sdf[tetra[it]] <= 0.0
            }

            val insideIndices =
                (0 until 4).filter { inside[it] }
            val outsideIndices =
                (0 until 4).filterNot { inside[it] }

            if (
                insideIndices.isEmpty() ||
                outsideIndices.isEmpty()
            ) {
                return
            }

            fun intersection(
                localA: Int,
                localB: Int,
            ): Int {
                val a = tetra[localA]
                val b = tetra[localB]
                return edgeVertex(
                    globalA = cubeKeys[a],
                    globalB = cubeKeys[b],
                    positionA = points[a],
                    positionB = points[b],
                    sdfA = sdf[a],
                    sdfB = sdf[b],
                    confidence = (
                        weights[a] * 0.5 +
                            weights[b] * 0.5
                        ).coerceIn(0.0, 1.0),
                )
            }

            when (insideIndices.size) {
                1 -> {
                    val i = insideIndices[0]
                    val o0 = outsideIndices[0]
                    val o1 = outsideIndices[1]
                    val o2 = outsideIndices[2]

                    emit(
                        intersection(i, o0),
                        intersection(i, o1),
                        intersection(i, o2),
                    )
                }

                3 -> {
                    val o = outsideIndices[0]
                    val i0 = insideIndices[0]
                    val i1 = insideIndices[1]
                    val i2 = insideIndices[2]

                    emit(
                        intersection(o, i2),
                        intersection(o, i1),
                        intersection(o, i0),
                    )
                }

                2 -> {
                    val i0 = insideIndices[0]
                    val i1 = insideIndices[1]
                    val o0 = outsideIndices[0]
                    val o1 = outsideIndices[1]

                    val a = intersection(i0, o0)
                    val b = intersection(i0, o1)
                    val c = intersection(i1, o1)
                    val d = intersection(i1, o0)

                    emit(a, b, c)
                    emit(a, c, d)
                }
            }
        }

        val extractionBudget =
            cellBudget.coerceAtLeast(256)
        val candidateKeys =
            ArrayList<Long>(
                min(
                    voxels.size,
                    extractionBudget,
                ),
            )
        val keyIterator = voxels.keys.iterator()
        while (
            keyIterator.hasNext() &&
            candidateKeys.size < extractionBudget
        ) {
            candidateKeys += keyIterator.next()
        }

        var inspectedCells = 0
        for (key in candidateKeys) {
            if (vertices.size >= maxVertices.coerceAtLeast(4)) break
            if (inspectedCells >= extractionBudget) break
            inspectedCells += 1

            val (gx, gy, gz) = unpackKey(key)
            val cubeKeys = LongArray(8)
            val samples = arrayOfNulls<Voxel>(8)

            var validCube = true

            for (index in offsets.indices) {
                val o = offsets[index]
                val sampleKey =
                    packKey(
                        gx + o.first,
                        gy + o.second,
                        gz + o.third,
                    )
                cubeKeys[index] = sampleKey
                val sample = voxels[sampleKey]
                samples[index] = sample

                if (
                    sample == null ||
                    sample.weight < 0.5 ||
                    !sample.sdf.isFinite()
                ) {
                    validCube = false
                    break
                }
            }

            if (!validCube) continue

            val values = DoubleArray(8) {
                samples[it]!!.sdf
            }

            var minSdf = Double.POSITIVE_INFINITY
            var maxSdf = Double.NEGATIVE_INFINITY
            for (value in values) {
                minSdf = min(minSdf, value)
                maxSdf = max(maxSdf, value)
            }

            if (minSdf > 0.0 || maxSdf < 0.0) continue
            if (
                minSdf > isoBand &&
                maxSdf < -isoBand
            ) continue

            val points = Array(8) { index ->
                val o = offsets[index]
                Vec3(
                    (gx + o.first) * voxelSizeMeters,
                    (gy + o.second) * voxelSizeMeters,
                    (gz + o.third) * voxelSizeMeters,
                )
            }

            val weights = DoubleArray(8) { index ->
                val sample =
                    samples[index]!!
                val uncertaintyPenalty =
                    kotlin.math.exp(
                        -kotlin.math.sqrt(
                            sample.sdfVariance
                                .coerceAtLeast(0.0)
                        ) * 2.5
                    )
                (
                    sample.weight /
                        8.0 *
                        uncertaintyPenalty
                    ).coerceIn(0.0, 1.0)
            }

            tetrahedra.forEach {
                processTetra(
                    cubeKeys = cubeKeys,
                    points = points,
                    sdf = values,
                    weights = weights,
                    tetra = it,
                )
            }
        }

        if (vertices.size < 4 || triangles.isEmpty()) {
            return null
        }

        val normals =
            MeshNormalEstimator.estimate(
                vertices = vertices,
                triangles = triangles,
            )
        val validNormalCount =
            normals.count {
                it.norm() > 0.5 &&
                    it.x.isFinite() &&
                    it.y.isFinite() &&
                    it.z.isFinite()
            }
        val normalCoverage =
            (
                validNormalCount.toDouble() /
                    normals.size.coerceAtLeast(1)
                ).coerceIn(0.0, 1.0)

        val meanConfidence =
            vertexConfidence.average().coerceIn(0.0, 1.0)

        val meanSdfVariance =
            candidateKeys
                .asSequence()
                .mapNotNull { key ->
                    voxels[key]?.sdfVariance
                }
                .average()
                .takeIf { it.isFinite() }
                ?: 0.0

        val uncertaintyConfidence =
            kotlin.math.exp(
                -kotlin.math.sqrt(
                    meanSdfVariance
                        .coerceAtLeast(0.0)
                ) * 2.5
            ).coerceIn(0.0, 1.0)

        val fusedGeometricConfidence =
            (
                meanConfidence *
                    (0.72 + 0.28 * normalCoverage) *
                    (0.70 + 0.30 * uncertaintyConfidence)
                ).coerceIn(0.0, 1.0)

        val edgeUse = HashMap<Pair<Int, Int>, Int>()
        triangles.forEach { triangle ->
            val edges = listOf(
                Pair(
                    min(triangle.a, triangle.b),
                    max(triangle.a, triangle.b),
                ),
                Pair(
                    min(triangle.b, triangle.c),
                    max(triangle.b, triangle.c),
                ),
                Pair(
                    min(triangle.c, triangle.a),
                    max(triangle.c, triangle.a),
                ),
            )
            edges.forEach { edge ->
                edgeUse[edge] = (edgeUse[edge] ?: 0) + 1
            }
        }

        val manifoldEdges =
            edgeUse.count { it.value == 2 }
        val watertightness =
            if (edgeUse.isEmpty()) 0.0
            else manifoldEdges.toDouble() / edgeUse.size.toDouble()

        val quality = DenseMeshQuality(
            timestampNanos = timestampNanos,
            vertexCount = vertices.size,
            triangleCount = triangles.size,
            meanReprojectionErrorPx = 0.0,
            watertightnessScore = watertightness.coerceIn(0.0, 1.0),
            temporalStability = estimateTemporalStability(),
            geometricConfidence =
                fusedGeometricConfidence,
            normalCoverage = normalCoverage,
            reprojectionEvidenceAvailable = false,
            watertightnessEvidenceAvailable = true,
            temporalEvidenceAvailable = true,
        )

        return ReconstructionMesh(
            vertices = vertices,
            triangles = triangles,
            timestampNanos = timestampNanos,
            confidence = fusedGeometricConfidence,
            normals = normals,
        ) to quality
    }

    @Synchronized
    fun voxelCount(): Int = voxels.size

    fun uncertaintyMetrics(): UncertaintyMetrics {
        if (voxels.isEmpty()) {
            return UncertaintyMetrics(
                meanSignedDistanceStdDev = 0.0,
                highUncertaintyVoxelRatio = 0.0,
                confidenceFactor = 1.0,
            )
        }

        var varianceSum = 0.0
        var highUncertaintyCount = 0

        voxels.values.forEach { voxel ->
            val standardDeviation =
                kotlin.math.sqrt(
                    voxel.sdfVariance.coerceAtLeast(0.0),
                )
            varianceSum += standardDeviation

            if (standardDeviation >= 0.30) {
                highUncertaintyCount += 1
            }
        }

        val meanStdDev =
            (varianceSum / voxels.size.coerceAtLeast(1))
                .coerceAtLeast(0.0)

        val highUncertaintyRatio =
            highUncertaintyCount.toDouble() /
                voxels.size.coerceAtLeast(1).toDouble()

        val confidenceFactor =
            kotlin.math.exp(
                -meanStdDev * 1.8,
            ).coerceIn(0.0, 1.0)

        return UncertaintyMetrics(
            meanSignedDistanceStdDev = meanStdDev.coerceIn(0.0, 1.0),
            highUncertaintyVoxelRatio =
                highUncertaintyRatio.coerceIn(0.0, 1.0),
            confidenceFactor = confidenceFactor,
        )
    }

    data class RuntimeStats(
        val voxelCount: Int,
        val maxVoxels: Int,
        val outOfOrderFrameCount: Long,
        val voxelEvictionCount: Long,
        val lastTimestampNanos: Long?,
        val capacityUtilization: Double,
        val memoryPressure: Double,
        val surfaceSurfelCount: Int = 0,
        val maxSurfaceSurfels: Int = 0,
        val surfaceSurfelEvictionCount: Long = 0L,
        val surfaceSurfelRejectedCount: Long = 0L,
        val surfaceDetailConfidence: Double = 0.0,
        val surfaceDetailNormalStability: Double = 0.0,
    )

    @Synchronized
    fun runtimeStats(): RuntimeStats =
        RuntimeStats(
            voxelCount = voxels.size,
            maxVoxels = maxVoxels,
            outOfOrderFrameCount = outOfOrderFrameCount,
            voxelEvictionCount = voxelEvictionCount,
            lastTimestampNanos =
                lastDecayTimestampNanos.takeIf {
                    it != Long.MIN_VALUE
                },
            capacityUtilization =
                (voxels.size.toDouble() /
                    maxVoxels.coerceAtLeast(1).toDouble())
                    .coerceIn(0.0, 1.0),
            memoryPressure =
                (
                    max(
                        voxels.size.toDouble() /
                            maxVoxels.coerceAtLeast(1).toDouble(),
                        surfaceSurfels.size.toDouble() /
                            maxSurfaceSurfels.coerceAtLeast(1).toDouble(),
                    )
                ).coerceIn(0.0, 1.0),
            surfaceSurfelCount = surfaceSurfels.size,
            maxSurfaceSurfels = maxSurfaceSurfels,
            surfaceSurfelEvictionCount = surfaceSurfelEvictionCount,
            surfaceSurfelRejectedCount = surfaceSurfelRejectedCount,
            surfaceDetailConfidence =
                surfaceDetailMetrics(lastDecayTimestampNanos)
                    .meanConfidence,
            surfaceDetailNormalStability =
                surfaceDetailMetrics(lastDecayTimestampNanos)
                    .meanNormalStability,
        )

    @Synchronized
    fun surfaceDetailMetrics(): SurfaceDetailMetrics =
        if (lastDecayTimestampNanos == Long.MIN_VALUE) {
            surfaceDetailMetrics(
                timestampNanos = 0L,
                decay = false,
            )
        } else {
            surfaceDetailMetrics(
                timestampNanos = lastDecayTimestampNanos,
                decay = false,
            )
        }

    fun surfaceDetailMetrics(
        timestampNanos: Long,
    ): SurfaceDetailMetrics =
        surfaceDetailMetrics(timestampNanos, decay = true)

    @Synchronized
    fun extractSurfaceDetail(
        timestampNanos: Long,
        maxPoints: Int = 100_000,
    ): List<SurfaceSurfel> {
        if (surfaceSurfels.isEmpty()) return emptyList()
        surfaceDetailMetrics(timestampNanos, decay = true)
        return surfaceSurfels.values
            .asSequence()
            .map { surfel ->
                val ageSeconds =
                    (
                        timestampNanos -
                            surfel.lastTimestampNanos
                        ).coerceAtLeast(0L).toDouble() / 1e9
                SurfaceSurfel(
                    position = surfel.position,
                    normal = surfel.normal.normalized(),
                    confidence = surfel.confidence.coerceIn(0.0, 1.0),
                    uncertaintyMeters =
                        surfel.uncertaintyMeters.coerceAtLeast(0.0),
                    weight = surfel.weight.coerceAtLeast(0.0),
                    ageSeconds = ageSeconds,
                    observationCount = surfel.observationCount,
                    normalStability =
                        surfel.normalStability.coerceIn(0.0, 1.0),
                    speedMetersPerSecond =
                        surfel.velocity.norm().coerceAtLeast(0.0),
                    varianceX = surfel.varianceX.coerceAtLeast(0.0),
                    varianceY = surfel.varianceY.coerceAtLeast(0.0),
                    varianceZ = surfel.varianceZ.coerceAtLeast(0.0),
                    dynamicProbability =
                        surfel.dynamicProbability.coerceIn(0.0, 1.0),
                    temporalStability =
                        surfel.temporalStability.coerceIn(0.0, 1.0),
                )
            }
            .filter { it.finite }
            .sortedWith(
                compareByDescending<SurfaceSurfel> { it.confidence }
                    .thenByDescending { it.normalStability }
                    .thenByDescending { it.weight }
                    .thenBy { it.ageSeconds },
            )
            .take(maxPoints.coerceAtLeast(1))
            .toList()
    }

    @Synchronized
    fun surfaceSurfelCount(): Int =
        surfaceSurfels.size

    @Synchronized
    fun surfaceSurfelMemoryPressure(): Double =
        (
            surfaceSurfels.size.toDouble() /
                maxSurfaceSurfels.coerceAtLeast(1).toDouble()
            ).coerceIn(0.0, 1.0)

    private fun acceptTimestamp(timestampNanos: Long): Boolean {
        if (timestampNanos < 0L) return false
        if (
            lastDecayTimestampNanos != Long.MIN_VALUE &&
            timestampNanos < lastDecayTimestampNanos
        ) {
            outOfOrderFrameCount += 1L
            return false
        }
        return true
    }

    private fun surfaceDetailMetrics(
        timestampNanos: Long,
        decay: Boolean,
    ): SurfaceDetailMetrics {
        if (decay) {
            decaySurfaceSurfels(timestampNanos)
        }
        if (surfaceSurfels.isEmpty()) {
            return SurfaceDetailMetrics(
                surfelCount = 0,
                meanConfidence = 0.0,
                meanUncertaintyMeters = 0.0,
                meanNormalStability = 0.0,
                meanSpeedMetersPerSecond = 0.0,
                temporallyStableRatio = 0.0,
                memoryUtilization = 0.0,
            )
        }

        var confidence = 0.0
        var uncertainty = 0.0
        var normalStability = 0.0
        var speed = 0.0
        var dynamicProbability = 0.0
        var temporalStability = 0.0
        var trustedGaussian = 0
        var stable = 0

        surfaceSurfels.values.forEach { surfel ->
            confidence += surfel.confidence.coerceIn(0.0, 1.0)
            uncertainty += surfel.uncertaintyMeters.coerceAtLeast(0.0)
            normalStability += surfel.normalStability.coerceIn(0.0, 1.0)
            speed += surfel.velocity.norm().coerceAtLeast(0.0)
            dynamicProbability +=
                surfel.dynamicProbability.coerceIn(0.0, 1.0)
            temporalStability +=
                surfel.temporalStability.coerceIn(0.0, 1.0)

            val covarianceTrace =
                (
                    surfel.varianceX +
                        surfel.varianceY +
                        surfel.varianceZ
                    ) / 3.0

            if (
                surfel.weight >= 1.5 &&
                surfel.normalStability >= 0.72
            ) {
                stable++
            }

            if (
                surfel.confidence >= 0.70 &&
                kotlin.math.sqrt(covarianceTrace) <= 0.010 &&
                surfel.dynamicProbability <= 0.35 &&
                surfel.temporalStability >= 0.75
            ) {
                trustedGaussian++
            }
        }

        val count = surfaceSurfels.size.coerceAtLeast(1)
        return SurfaceDetailMetrics(
            surfelCount = surfaceSurfels.size,
            meanConfidence = (confidence / count).coerceIn(0.0, 1.0),
            meanUncertaintyMeters =
                (uncertainty / count).coerceAtLeast(0.0),
            meanNormalStability =
                (normalStability / count).coerceIn(0.0, 1.0),
            meanSpeedMetersPerSecond =
                (speed / count).coerceAtLeast(0.0),
            temporallyStableRatio =
                (stable.toDouble() / count).coerceIn(0.0, 1.0),
            memoryUtilization =
                (
                    surfaceSurfels.size.toDouble() /
                        maxSurfaceSurfels.coerceAtLeast(1).toDouble()
                    ).coerceIn(0.0, 1.0),
        )
    }

    private fun updateSurfaceSurfel(
        position: Vec3,
        normal: Vec3,
        confidence: Double,
        uncertaintyMeters: Double,
        dynamicProbability: Double,
        temporalConsistency: Double,
        timestampNanos: Long,
    ) {
        if (
            !position.isFinite() ||
            !normal.isFinite() ||
            !confidence.isFinite() ||
            !uncertaintyMeters.isFinite() ||
            timestampNanos < 0L
        ) {
            surfaceSurfelRejectedCount += 1L
            return
        }

        val normalizedNormal = normal.normalized()
        if (!normalizedNormal.isFinite() || normalizedNormal.norm() < 0.5) {
            surfaceSurfelRejectedCount += 1L
            return
        }

        val key = surfacePointKey(position)
        val existing = surfaceSurfels[key]

        if (existing == null) {
            if (surfaceSurfels.size >= maxSurfaceSurfels) {
                evictLowestSurfaceSurfel(timestampNanos)
            }
            if (surfaceSurfels.size >= maxSurfaceSurfels) {
                surfaceSurfelRejectedCount += 1L
                return
            }

            val initialVariance =
                uncertaintyMeters
                    .coerceIn(0.0005, 0.12)
                    .let { it * it }

            surfaceSurfels[key] = MutableSurfel(
                position = position,
                normal = normalizedNormal,
                confidence = confidence.coerceIn(0.0, 1.0),
                uncertaintyMeters = uncertaintyMeters.coerceAtLeast(0.0),
                weight = confidence.coerceIn(0.01, 32.0),
                lastTimestampNanos = timestampNanos,
                observationCount = 1L,
                normalStability = 0.55,
                velocity = Vec3(0.0, 0.0, 0.0),
                varianceX = initialVariance,
                varianceY = initialVariance,
                varianceZ = initialVariance,
                dynamicProbability =
                    dynamicProbability.coerceIn(0.0, 1.0),
                temporalStability =
                    temporalConsistency.coerceIn(0.0, 1.0),
            )
            return
        }

        val dtSeconds =
            (
                timestampNanos -
                    existing.lastTimestampNanos
                ).coerceAtLeast(0L).toDouble() / 1e9

        val velocity =
            if (dtSeconds > 1e-6 && dtSeconds < 1.0) {
                (
                    position -
                        existing.position
                    ) / dtSeconds
            } else {
                Vec3(0.0, 0.0, 0.0)
            }

        val normalAgreement =
            (
                existing.normal.dot(normalizedNormal) +
                    1.0
                ) * 0.5

        val motionResidual =
            (
                position -
                    existing.position
                ).norm().coerceAtMost(2.0)

        val residualWeight =
            exp(
                -motionResidual /
                    max(
                        voxelSizeMeters * 3.0,
                        1e-6,
                    ),
            ).coerceIn(0.04, 1.0)

        val effectiveWeight =
            (
                confidence.coerceIn(0.01, 32.0) *
                    residualWeight
                ).coerceAtLeast(0.01)

        val combinedWeight =
            (
                existing.weight +
                    effectiveWeight
                ).coerceAtLeast(1e-9)

        val positionBlend =
            (
                existing.position * existing.weight +
                    position * effectiveWeight
                ) / combinedWeight

        val normalBlend =
            (
                existing.normal * existing.weight +
                    normalizedNormal * effectiveWeight
                ).normalized()

        existing.position = positionBlend
        existing.normal = normalBlend
        existing.confidence =
            (
                existing.confidence * existing.weight +
                    confidence.coerceIn(0.0, 1.0) *
                        effectiveWeight
                ) / combinedWeight
        val priorUncertainty =
            existing.uncertaintyMeters.coerceIn(
                0.001,
                1.0,
            )
        val measurementUncertainty =
            uncertaintyMeters.coerceIn(
                0.001,
                1.0,
            )
        val priorPrecision =
            1.0 / (priorUncertainty * priorUncertainty)
        val measurementPrecision =
            effectiveWeight /
                (measurementUncertainty *
                    measurementUncertainty)
        val posteriorPrecision =
            priorPrecision +
                measurementPrecision
        val posteriorUncertainty =
            kotlin.math.sqrt(
                1.0 /
                    posteriorPrecision.coerceAtLeast(1.0),
            ).coerceIn(0.0005, 0.12)

        existing.uncertaintyMeters =
            posteriorUncertainty

        val safeMeasurementVariance =
            measurementUncertainty *
                measurementUncertainty
        val priorVariance =
            priorUncertainty *
                priorUncertainty

        existing.varianceX =
            (
                priorVariance *
                    measurementPrecision +
                    safeMeasurementVariance *
                    priorPrecision
                ) /
                posteriorPrecision.coerceAtLeast(1e-9)

        existing.varianceY =
            existing.varianceX

        existing.varianceZ =
            (
                existing.varianceY * 0.50 +
                    (
                        (
                            (
                                priorVariance *
                                    measurementPrecision +
                                    safeMeasurementVariance *
                                    priorPrecision
                            ) /
                            posteriorPrecision.coerceAtLeast(1e-9)
                        ) * 0.50
                    )
                ).coerceIn(
                    1e-7,
                    0.0144,
                )

        existing.dynamicProbability =
            (
                existing.dynamicProbability * 0.64 +
                    dynamicProbability.coerceIn(0.0, 1.0) * 0.24 +
                    (
                        1.0 -
                            temporalConsistency.coerceIn(0.0, 1.0)
                    ) * 0.12
                ).coerceIn(0.0, 1.0)

        existing.temporalStability =
            (
                existing.temporalStability * 0.62 +
                    temporalConsistency.coerceIn(0.0, 1.0) * 0.38
                ).coerceIn(0.0, 1.0)

        existing.weight =
            min(64.0, combinedWeight)
        existing.normalStability =
            (
                existing.normalStability * 0.76 +
                    normalAgreement * 0.24
                ).coerceIn(0.0, 1.0)
        existing.velocity =
            (
                existing.velocity * 0.68 +
                    limitVectorMagnitude(
                        velocity,
                        6.0,
                    ) * 0.32
                )
        existing.lastTimestampNanos = timestampNanos
        existing.observationCount =
            (existing.observationCount + 1L)
                .coerceAtMost(1_000_000L)
    }

    private fun decaySurfaceSurfels(
        timestampNanos: Long,
    ) {
        if (timestampNanos < 0L) return
        if (lastSurfaceSurfelDecayTimestampNanos == timestampNanos) return

        val halfLifeNanos =
            (
                surfaceSurfelHalfLifeSeconds *
                    1_000_000_000.0
                ).toLong().coerceAtLeast(1L)

        val iterator = surfaceSurfels.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            val surfel = entry.value
            val ageNanos =
                (
                    timestampNanos -
                        surfel.lastTimestampNanos
                    ).coerceAtLeast(0L)

            if (ageNanos <= 0L) continue

            val decay =
                exp(
                    -0.6931471805599453 *
                        ageNanos.toDouble() /
                        halfLifeNanos.toDouble(),
                )

            surfel.weight *= decay
            surfel.confidence *= 0.995 + decay * 0.005
            surfel.temporalStability =
                (
                    surfel.temporalStability *
                        (0.985 + 0.015 * decay)
                ).coerceIn(0.0, 1.0)
            surfel.varianceX =
                (
                    surfel.varianceX +
                        (1.0 - decay) * 0.0002
                ).coerceAtMost(0.0144)
            surfel.varianceY =
                (
                    surfel.varianceY +
                        (1.0 - decay) * 0.0002
                ).coerceAtMost(0.0144)
            surfel.varianceZ =
                (
                    surfel.varianceZ +
                        (1.0 - decay) * 0.0002
                ).coerceAtMost(0.0144)

            if (
                surfel.weight < surfaceSurfelMinWeight &&
                ageNanos > halfLifeNanos * 8L
            ) {
                iterator.remove()
            }
        }

        while (surfaceSurfels.size > maxSurfaceSurfels) {
            evictLowestSurfaceSurfel(timestampNanos)
        }

        lastSurfaceSurfelDecayTimestampNanos = timestampNanos
    }

    private fun evictLowestSurfaceSurfel(
        timestampNanos: Long,
    ) {
        val victim = surfaceSurfels.entries.minByOrNull { entry ->
            val surfel = entry.value
            val ageSeconds =
                (
                    timestampNanos -
                        surfel.lastTimestampNanos
                    ).coerceAtLeast(0L).toDouble() / 1e9
            val retention =
                0.45 * (
                    surfel.weight / 16.0
                ).coerceIn(0.0, 1.0) +
                    0.30 * surfel.normalStability.coerceIn(0.0, 1.0) +
                    0.15 * surfel.confidence.coerceIn(0.0, 1.0) +
                    0.10 * exp(-2.0 * ageSeconds)
            retention
        } ?: return

        surfaceSurfels.remove(victim.key)
        surfaceSurfelEvictionCount += 1L
    }

    private fun surfacePointKey(position: Vec3): Long =
        packKey(
            floor(position.x / voxelSizeMeters).toInt(),
            floor(position.y / voxelSizeMeters).toInt(),
            floor(position.z / voxelSizeMeters).toInt(),
        )

    private fun estimateSurfaceNormal(
        depthMeters: FloatArray,
        width: Int,
        height: Int,
        x: Int,
        y: Int,
        stride: Int,
        centerDepthMeters: Double,
        intrinsics: CameraIntrinsics,
    ): Vec3 {
        val rayCenter = Vec3(
            (x - intrinsics.cx) /
                intrinsics.fx.coerceAtLeast(1e-8),
            (y - intrinsics.cy) /
                intrinsics.fy.coerceAtLeast(1e-8),
            1.0,
        ).normalized()

        val nx = (x + stride).coerceAtMost(width - 1)
        val ny = (y + stride).coerceAtMost(height - 1)

        if (nx == x && ny == y) {
            return Vec3(0.0, 0.0, 1.0)
        }

        val depthX =
            depthMeters[y * width + nx].toDouble()
        val depthY =
            depthMeters[ny * width + x].toDouble()

        if (
            !depthX.isFinite() ||
            !depthY.isFinite() ||
            depthX <= 0.05 ||
            depthY <= 0.05
        ) {
            return rayCenter * -1.0
        }

        val center = rayCenter * centerDepthMeters
        val rayX = Vec3(
            (nx - intrinsics.cx) /
                intrinsics.fx.coerceAtLeast(1e-8),
            (y - intrinsics.cy) /
                intrinsics.fy.coerceAtLeast(1e-8),
            1.0,
        ).normalized()
        val rayY = Vec3(
            (x - intrinsics.cx) /
                intrinsics.fx.coerceAtLeast(1e-8),
            (ny - intrinsics.cy) /
                intrinsics.fy.coerceAtLeast(1e-8),
            1.0,
        ).normalized()

        val normal =
            (
                (rayX * depthX - center)
                    .cross(
                        rayY * depthY - center,
                    )
                ).normalized()

        return if (normal.isFinite() && normal.norm() >= 0.5) {
            normal
        } else {
            rayCenter * -1.0
        }
    }

    private fun decayTemporalState(
        timestampNanos: Long,
    ) {
        if (lastDecayTimestampNanos == timestampNanos) return
        if (lastDecayTimestampNanos != Long.MIN_VALUE &&
            timestampNanos < lastDecayTimestampNanos
        ) {
            outOfOrderFrameCount += 1L
            return
        }

        val referenceTimestamp =
            if (lastDecayTimestampNanos == Long.MIN_VALUE) {
                timestampNanos
            } else {
                timestampNanos
            }

        val halfLifeNanos = (
            temporalHalfLifeSeconds.coerceAtLeast(0.01) *
                1_000_000_000.0
            ).toLong().coerceAtLeast(1L)

        val iterator = voxels.entries.iterator()
        var processed = 0

        while (
            iterator.hasNext() &&
            processed < decayBudgetVoxels.coerceAtLeast(256)
        ) {
            val entry = iterator.next()
            val voxel = entry.value
            processed += 1

            val ageNanos = (
                referenceTimestamp - voxel.lastTimestampNanos
                ).coerceAtLeast(0L)

            if (ageNanos <= 0L) continue

            val decay = kotlin.math.exp(
                -0.6931471805599453 *
                    ageNanos.toDouble() /
                    halfLifeNanos.toDouble(),
            )

            voxel.weight *= decay

            if (
                voxel.weight < minimumRetainedWeight &&
                voxel.observationCount <= 2L
            ) {
                iterator.remove()
            }
        }

        lastDecayTimestampNanos = timestampNanos
    }

    private fun limitVectorMagnitude(
        vector: Vec3,
        maxLength: Double,
    ): Vec3 {
        val magnitude = vector.norm()
        if (!magnitude.isFinite() || magnitude <= maxLength) {
            return vector
        }
        return vector * (maxLength / magnitude.coerceAtLeast(1e-9))
    }

    private fun integrateVoxelRobust(
        world: Vec3,
        sdf: Double,
        confidence: Double,
        timestampNanos: Long,
    ): Double {
        val x =
            floor(world.x / voxelSizeMeters)
                .toInt()
        val y =
            floor(world.y / voxelSizeMeters)
                .toInt()
        val z =
            floor(world.z / voxelSizeMeters)
                .toInt()

        val key = packKey(x, y, z)
        val existing = voxels[key]

        if (existing == null) {
            if (voxels.size >= maxVoxels) {
                evictLowestWeight()
            }

            val weight =
                confidence.coerceIn(0.01, 32.0)

            voxels[key] =
                Voxel(
                    sdf = sdf,
                    weight = weight,
                    lastTimestampNanos =
                        timestampNanos,
                    observationCount = 1L,
                    sdfVariance = 0.0,
                )

            return 1.0
        }

        val residual =
            kotlin.math.abs(
                existing.sdf - sdf,
            )

        val huberScale = 0.18
        val robustWeight =
            if (residual <= huberScale) {
                1.0
            } else {
                (
                    huberScale /
                        residual.coerceAtLeast(
                            1e-9,
                        )
                    ).coerceIn(0.04, 1.0)
            }

        val effectiveWeight =
            (
                confidence *
                    robustWeight
                ).coerceIn(0.0, 32.0)

        if (effectiveWeight < 0.015) {
            return 0.0
        }

        val combinedWeight =
            (
                existing.weight +
                    effectiveWeight
                ).coerceAtLeast(1e-9)

        val previousMean =
            existing.sdf

        val blendedMean =
            (
                previousMean *
                    existing.weight +
                    sdf *
                        effectiveWeight
                ) /
                combinedWeight

        val previousVariance =
            existing.sdfVariance.coerceAtLeast(
                0.0,
            )

        val meanShift =
            previousMean -
                blendedMean
        val incomingResidual =
            sdf - blendedMean

        val blendedVariance =
            (
                existing.weight *
                    (
                        previousVariance +
                            meanShift *
                            meanShift
                        ) +
                    effectiveWeight *
                        incomingResidual *
                        incomingResidual
                ) /
                combinedWeight

        existing.sdf = blendedMean
        existing.sdfVariance =
            blendedVariance.coerceIn(0.0, 1.0)
        existing.weight =
            min(
                64.0,
                combinedWeight,
            )
        existing.lastTimestampNanos =
            timestampNanos
        existing.observationCount =
            (
                existing.observationCount + 1L
                ).coerceAtMost(1_000_000L)

        return robustWeight
    }

    private fun integrateVoxel(
        world: Vec3,
        sdf: Double,
        confidence: Double,
        timestampNanos: Long,
    ) {
        val x = floor(world.x / voxelSizeMeters).toInt()
        val y = floor(world.y / voxelSizeMeters).toInt()
        val z = floor(world.z / voxelSizeMeters).toInt()
        val key = packKey(x, y, z)

        val weight = confidence.coerceIn(0.01, 32.0)
        val existing = voxels[key]

        if (existing == null) {
            if (voxels.size >= maxVoxels) {
                evictLowestWeight()
            }

            voxels[key] = Voxel(
                sdf = sdf,
                weight = weight,
                lastTimestampNanos = timestampNanos,
                observationCount = 1L,
                sdfVariance = 0.0,
            )
            return
        }

        // Blend with the true combined evidence before capping stored confidence.
        // Dividing by the capped weight would over-weight late observations once the voxel
        // reached the storage ceiling.
        val combinedWeight =
            (
                existing.weight +
                    weight
            ).coerceAtLeast(1e-9)

        val previousMean = existing.sdf
        val blendedMean = (
            previousMean * existing.weight +
                sdf * weight
            ) / combinedWeight

        val previousVariance =
            existing.sdfVariance.coerceAtLeast(0.0)
        val meanShift = previousMean - blendedMean
        val incomingResidual = sdf - blendedMean
        val blendedVariance = (
            existing.weight * (
                previousVariance +
                    meanShift * meanShift
                ) +
                weight * incomingResidual * incomingResidual
            ) / combinedWeight

        existing.sdf = blendedMean
        existing.sdfVariance =
            blendedVariance
                .coerceIn(0.0, 1.0)

        existing.weight =
            min(
                64.0,
                combinedWeight,
            )
        existing.lastTimestampNanos = timestampNanos
        existing.observationCount =
            (existing.observationCount + 1L)
                .coerceAtMost(1_000_000L)
    }

    private fun evictLowestWeight() {
        if (voxels.isEmpty()) return

        val now = voxels.values.maxOfOrNull {
            it.lastTimestampNanos
        } ?: 0L

        val victim = voxels.entries.minByOrNull { entry ->
            val voxel = entry.value
            val ageSeconds =
                (
                    now - voxel.lastTimestampNanos
                ).coerceAtLeast(0L) / 1e9

            val recency =
                kotlin.math.exp(
                    -1.8 * ageSeconds,
                )

            val observationStability = (
                voxel.observationCount /
                    12.0
                ).coerceIn(0.0, 1.0)

            val retentionScore =
                0.68 * (
                    voxel.weight / 16.0
                ).coerceIn(0.0, 1.0) +
                    0.22 * observationStability +
                    0.10 * recency

            retentionScore
        } ?: return

        voxels.remove(victim.key)
        voxelEvictionCount += 1L
    }

    private fun estimateTemporalStability(): Double {
        if (voxels.isEmpty()) return 0.0
        val stable = voxels.values.count {
            it.weight >= 2.0
        }
        return stable.toDouble() / voxels.size
    }

    private fun packKey(x: Int, y: Int, z: Int): Long {
        val mask = (1L shl 21) - 1L
        val bias = 1_048_576L
        return (
            ((x.toLong() + bias) and mask) shl 42
        ) or (
            ((y.toLong() + bias) and mask) shl 21
        ) or (
            (z.toLong() + bias) and mask
        )
    }

    private fun unpackKey(key: Long): Triple<Int, Int, Int> {
        val mask = (1L shl 21) - 1L
        val bias = 1_048_576L

        val x = ((key ushr 42) and mask) - bias
        val y = ((key ushr 21) and mask) - bias
        val z = (key and mask) - bias

        return Triple(
            x.toInt(),
            y.toInt(),
            z.toInt(),
        )
    }
}

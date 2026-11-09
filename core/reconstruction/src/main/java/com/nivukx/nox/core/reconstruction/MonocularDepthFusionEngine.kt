package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.ai.PoseDetectionEngine
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DepthRepresentation
import com.nivukx.nox.core.model.DynamicTopology
import com.nivukx.nox.core.model.MonocularDepthFrame
import com.nivukx.nox.core.model.MonocularDepthSample
import com.nivukx.nox.core.model.MonocularReconstructionFrame
import com.nivukx.nox.core.model.MonocularReconstructionJoint
import com.nivukx.nox.core.model.MotionQualityGate
import com.nivukx.nox.core.model.ReconstructionConfidenceBand
import com.nivukx.nox.core.model.ScaleSource
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Production monocular RGB reconstruction path.
 *
 * A single RGB camera cannot recover metric 3D perfectly without scale information. This engine
 * does not hide that ambiguity. It combines calibrated intrinsics, optional dense-depth samples,
 * temporal state and articulation constraints to produce a stable 3D estimate together with
 * uncertainty and confidence. When the device has no metric depth source the output is explicitly
 * tagged as anthropometric/relative rather than pretending it is equivalent to a calibrated rig.
 *
 * The solver operates on one scalar depth value per observed landmark. Each iteration minimizes:
 *   - articulated edge-length residual;
 *   - temporal depth residual;
 *   - external depth residual;
 *   - root-depth prior;
 *   - confidence-weighted regularization.
 *
 * This is intentionally independent of a neural depth model. A future NPU/TFLite/ONNX depth
 * provider can feed [MonocularDepthFrame] without changing the optimization layer.
 */
class MonocularDepthFusionEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val maxIterations: Int = 18,
        val learningRate: Double = 0.055,
        val minimumDepthMeters: Double = 0.18,
        val maximumDepthMeters: Double = 18.0,
        val humanHeightPriorMeters: Double = 1.72,
        val animalHeightPriorMeters: Double = 0.85,
        val objectDepthPriorMeters: Double = 2.50,
        val temporalWeight: Double = 0.70,
        val depthSensorWeight: Double = 1.35,
        val relativeDepthWeight: Double = 0.65,
        val rootPriorWeight: Double = 0.52,
        val topologyWeight: Double = 1.0,
        val visibilityWeight: Double = 0.20,
        val maxDtSeconds: Double = 0.20,
        val minimumEdgeConfidence: Double = 0.08,
        val maxDepthJumpMeters: Double = 1.20,
        val uncertaintyBaseMeters: Double = 0.035,
        val anthropometricScaleConfidence: Double = 0.62,
        val huberDeltaMeters: Double = 0.045,
        val minimumLearningRate: Double = 0.006,
        val gradientClip: Double = 6.0,
    ) {
        init {
            require(maxIterations in 2..128)
            require(learningRate in 0.001..0.5)
            require(minimumDepthMeters > 0.0)
            require(maximumDepthMeters > minimumDepthMeters)
            require(temporalWeight >= 0.0)
            require(depthSensorWeight >= 0.0)
            require(rootPriorWeight >= 0.0)
            require(topologyWeight >= 0.0)
            require(maxDtSeconds > 0.0)
            require(huberDeltaMeters > 0.0)
            require(minimumLearningRate in 0.0001..learningRate)
            require(gradientClip > 0.0)
        }
    }

    data class Observation(
        val id: Int,
        val positionNormalized: Vec2,
        val confidence: Double,
    ) {
        init {
            require(positionNormalized.x.isFinite())
            require(positionNormalized.y.isFinite())
            require(confidence in 0.0..1.0)
        }
    }

    data class Result(
        val frame: MonocularReconstructionFrame,
        val depthFrame: MonocularDepthFrame,
        val gate: MotionQualityGate,
        val iterations: Int,
    )

    fun reconstruct(
        timestampNanos: Long,
        subjectType: SubjectType,
        trackId: Long,
        imageWidth: Int,
        imageHeight: Int,
        intrinsics: CameraIntrinsics,
        observations: List<Observation>,
        topology: DynamicTopology,
        denseDepth: MonocularDepthFrame? = null,
        previous: MonocularReconstructionFrame? = null,
        calibratedScaleMeters: Double? = null,
    ): Result {
        require(imageWidth > 0)
        require(imageHeight > 0)

        val usable = observations
            .filter {
                it.confidence >= 0.01 &&
                    it.positionNormalized.x.isFinite() &&
                    it.positionNormalized.y.isFinite()
            }
            .associateBy { it.id }

        if (usable.isEmpty()) {
            val empty = emptyFrame(
                timestampNanos,
                subjectType,
                trackId,
                calibratedScaleMeters,
            )
            return Result(
                frame = empty,
                depthFrame = emptyDepthFrame(timestampNanos, imageWidth, imageHeight),
                gate = MotionQualityGate.evaluate(
                    confidence = 0.0,
                    uncertaintyMeters = Double.POSITIVE_INFINITY,
                    reprojectionErrorPx = Double.POSITIVE_INFINITY,
                    temporalStability = 0.0,
                    hardFailureReasons = listOf("no usable 2D observations"),
                ),
                iterations = 0,
            )
        }

        val rays = usable.mapValues { (_, observation) ->
            pixelRay(
                observation.positionNormalized,
                imageWidth,
                imageHeight,
                intrinsics,
            )
        }

        val prior = resolveScale(
            subjectType = subjectType,
            observations = usable.values.toList(),
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            intrinsics = intrinsics,
            calibratedScaleMeters = calibratedScaleMeters,
            denseDepth = denseDepth,
        )

        val depths = initializeDepths(
            observations = usable,
            rays = rays,
            topology = topology,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            intrinsics = intrinsics,
            prior = prior,
            denseDepth = denseDepth,
            previous = previous,
        ).toMutableMap()

        var iterationsCompleted = 0
        var previousEnergy = Double.POSITIVE_INFINITY

        for (iteration in 0 until config.maxIterations) {
            val gradients = depths.keys.associateWith { 0.0 }.toMutableMap()
            var energy = 0.0

            topology.edges.forEach { edge ->
                val a = usable[edge.a] ?: return@forEach
                val b = usable[edge.b] ?: return@forEach
                val za = depths[edge.a] ?: return@forEach
                val zb = depths[edge.b] ?: return@forEach

                val pa = rays[edge.a]!! * za
                val pb = rays[edge.b]!! * zb
                val delta = pa - pb
                val distance = delta.norm().coerceAtLeast(1e-6)
                val error = distance - edge.preferredLengthMeters
                val confidence =
                    min(
                        a.confidence,
                        b.confidence,
                    ).coerceIn(config.minimumEdgeConfidence, 1.0)
                val baseWeight =
                    config.topologyWeight * edge.stiffness * confidence
                val robustWeight =
                    huberWeight(
                        residual = error,
                        delta = config.huberDeltaMeters,
                    )
                val weight = baseWeight * robustWeight

                energy += baseWeight *
                    huberLoss(
                        residual = error,
                        delta = config.huberDeltaMeters,
                    )

                val derivativeA = delta.dot(rays[edge.a]!!) / distance
                val derivativeB = -delta.dot(rays[edge.b]!!) / distance

                gradients[edge.a] =
                    gradients.getValue(edge.a) +
                        2.0 * weight * error * derivativeA

                gradients[edge.b] =
                    gradients.getValue(edge.b) +
                        2.0 * weight * error * derivativeB
            }

            previous?.joints?.associateBy { it.landmarkId }?.forEach { (id, old) ->
                val z = depths[id] ?: return@forEach
                val dtSeconds = (
                    timestampNanos - previous.timestampNanos
                    ).coerceAtLeast(1L) / 1e9
                val predictedZ = (
                    old.depthMeters +
                        old.velocityCameraMetersPerSecond.z *
                        dtSeconds.coerceAtMost(config.maxDtSeconds)
                    ).coerceIn(
                        config.minimumDepthMeters,
                        config.maximumDepthMeters,
                    )
                val residual = z - predictedZ
                val baseWeight =
                    config.temporalWeight *
                        old.confidence.coerceIn(0.0, 1.0)
                val robustWeight =
                    huberWeight(
                        residual = residual,
                        delta = config.huberDeltaMeters,
                    )
                val weight = baseWeight * robustWeight
                energy += baseWeight *
                    huberLoss(
                        residual = residual,
                        delta = config.huberDeltaMeters,
                    )
                gradients[id] =
                    gradients.getValue(id) +
                        2.0 * weight * residual
            }

            denseDepth?.let { external ->
                usable.forEach { (id, observation) ->
                    val depth = nearestDenseDepth(
                        external,
                        observation.positionNormalized,
                    ) ?: return@forEach
                    val z = depths[id] ?: return@forEach
                    val targetDepth =
                        externalDepthMeters(
                            external = external,
                            sample = depth,
                            rootDepthMeters = prior.rootDepthMeters,
                        )
                    val residual = z - targetDepth
                    val representationWeight =
                        if (
                            external.representation ==
                                DepthRepresentation.METRIC_METERS
                        ) {
                            config.depthSensorWeight
                        } else {
                            config.relativeDepthWeight
                        }
                    val baseWeight =
                        representationWeight *
                            depth.confidence.coerceIn(0.0, 1.0) *
                            observation.confidence
                    val robustWeight =
                        huberWeight(
                            residual = residual,
                            delta = config.huberDeltaMeters,
                        )
                    val weight = baseWeight * robustWeight
                    energy += baseWeight *
                        huberLoss(
                            residual = residual,
                            delta = config.huberDeltaMeters,
                        )
                    gradients[id] =
                        gradients.getValue(id) +
                            2.0 * weight * residual
                }
            }

            val rootId = topology.rootNodeId
            if (rootId != null && rootId in depths) {
                val rootZ = depths.getValue(rootId)
                val residual = rootZ - prior.rootDepthMeters
                val baseWeight =
                    config.rootPriorWeight * prior.scaleConfidence
                val robustWeight =
                    huberWeight(
                        residual = residual,
                        delta = config.huberDeltaMeters,
                    )
                val weight = baseWeight * robustWeight
                energy += baseWeight *
                    huberLoss(
                        residual = residual,
                        delta = config.huberDeltaMeters,
                    )
                gradients[rootId] =
                    gradients.getValue(rootId) +
                        2.0 * weight * residual
            }

            val adaptiveLearningRate =
                (
                    config.learningRate /
                        (1.0 + iteration.toDouble() * 0.08)
                    ).coerceAtLeast(config.minimumLearningRate)

            var maxStep = 0.0
            for (id in depths.keys) {
                val z = depths.getValue(id)
                val gradient = gradients.getValue(id)
                val clippedGradient =
                    gradient.coerceIn(
                        -config.gradientClip,
                        config.gradientClip,
                    )
                val step = (
                    adaptiveLearningRate * clippedGradient
                ).coerceIn(
                    -config.maxDepthJumpMeters * 0.18,
                    config.maxDepthJumpMeters * 0.18,
                )

                val updated = (
                    z - step
                ).coerceIn(
                    config.minimumDepthMeters,
                    config.maximumDepthMeters,
                )
                maxStep = max(maxStep, abs(updated - z))
                depths[id] = updated
            }

            iterationsCompleted = iteration + 1

            if (
                energy >= previousEnergy &&
                maxStep < 0.0008
            ) {
                break
            }

            previousEnergy = energy
            if (maxStep < 0.00025) {
                break
            }
        }

        val oldJoints = previous?.joints?.associateBy { it.landmarkId }.orEmpty()
        val joints = usable.values.map { observation ->
            val ray = rays.getValue(observation.id)
            val depth = depths.getValue(observation.id)
            val position = ray * depth
            val old = oldJoints[observation.id]
            val dtSeconds = if (previous == null) {
                1.0 / 30.0
            } else {
                (
                    timestampNanos - previous.timestampNanos
                    ).coerceAtLeast(1L) / 1e9
            }.coerceAtMost(config.maxDtSeconds)

            val velocity = if (old == null) {
                Vec3(0.0, 0.0, 0.0)
            } else {
                (
                    position - old.positionCameraMeters
                ) / dtSeconds.coerceAtLeast(1e-4)
            }

            val acceleration = if (old == null) {
                Vec3(0.0, 0.0, 0.0)
            } else {
                (
                    velocity - old.velocityCameraMetersPerSecond
                ) / dtSeconds.coerceAtLeast(1e-4)
            }

            val temporalConfidence =
                if (old == null) 0.40
                else old.confidence * 0.78 + 0.22

            val depthEvidence =
                denseDepth?.let {
                    nearestDenseDepth(it, observation.positionNormalized)
                        ?.confidence
                        ?.coerceIn(0.0, 1.0)
                } ?: prior.scaleConfidence

            val topologyResidual = localTopologyResidual(
                landmarkId = observation.id,
                depths = depths,
                rays = rays,
                observations = usable,
                topology = topology,
            )

            val geometryConfidence =
                exp(
                    -topologyResidual /
                        max(0.02, prior.expectedScaleMeters * 0.07),
                ).coerceIn(0.0, 1.0)

            val depthUncertainty = (
                config.uncertaintyBaseMeters +
                    (1.0 - observation.confidence) * 0.18 +
                    topologyResidual * 0.65 +
                    (1.0 - depthEvidence) * 0.22
                ).coerceAtLeast(0.005)

            val confidence =
                (
                    0.34 * observation.confidence +
                        0.18 * geometryConfidence +
                        0.17 * temporalConfidence +
                        0.15 * topology.globalConfidence +
                        0.16 * depthEvidence
                    ).coerceIn(0.0, 1.0)

            MonocularReconstructionJoint(
                landmarkId = observation.id,
                semanticName =
                    topology.nodes
                        .firstOrNull { it.id == observation.id }
                        ?.semanticName
                        ?: "landmark_" + observation.id,
                positionCameraMeters = position,
                velocityCameraMetersPerSecond = velocity,
                accelerationCameraMetersPerSecond2 = acceleration,
                confidence = confidence,
                uncertaintyMeters = depthUncertainty,
                depthMeters = depth,
                depthConfidence = depthEvidence,
                predicted = observation.confidence < 0.20,
            )
        }

        val root = topology.rootNodeId
            ?.let { rootId ->
                joints.firstOrNull { it.landmarkId == rootId }?.positionCameraMeters
            }
            ?: joints.map { it.positionCameraMeters }
                .averageVectorOrZero()

        val edgeResiduals = topology.edges.mapNotNull { edge ->
            val a = joints.firstOrNull { it.landmarkId == edge.a }
            val b = joints.firstOrNull { it.landmarkId == edge.b }
            if (a == null || b == null) null
            else abs(
                a.positionCameraMeters.distanceTo(b.positionCameraMeters) -
                    edge.preferredLengthMeters,
            )
        }

        val meanResidualMeters =
            if (edgeResiduals.isEmpty()) 0.0
            else edgeResiduals.average()

        val reprojectionPx = (
            meanResidualMeters /
                prior.rootDepthMeters.coerceAtLeast(0.05)
        ) * (
            (intrinsics.fx + intrinsics.fy) * 0.5
        )

        val meanUncertainty =
            if (joints.isEmpty()) Double.POSITIVE_INFINITY
            else joints.map { it.uncertaintyMeters }.average()

        val temporalStability =
            temporalStability(
                previous = previous,
                current = joints,
                timestampNanos = timestampNanos,
            )

        val scaleSource = prior.scaleSource
        val scaleConfidence = prior.scaleConfidence
        val globalConfidence =
            (
                joints.map { it.confidence }.averageOrNull() * 0.48 +
                    topology.globalConfidence * 0.18 +
                    temporalStability * 0.18 +
                    scaleConfidence * 0.16
                ).coerceIn(0.0, 1.0)

        val band = MotionQualityGate.evaluate(
            confidence = globalConfidence,
            uncertaintyMeters = meanUncertainty,
            reprojectionErrorPx = reprojectionPx,
            temporalStability = temporalStability,
        ).confidenceBand

        val frame = MonocularReconstructionFrame(
            timestampNanos = timestampNanos,
            subjectType = subjectType,
            trackId = trackId,
            joints = joints,
            rootPositionMeters = root,
            globalScaleMeters = prior.expectedScaleMeters,
            scaleSource = scaleSource,
            scaleConfidence = scaleConfidence,
            globalConfidence = globalConfidence,
            reprojectionErrorPx = reprojectionPx.coerceAtLeast(0.0),
            meanUncertaintyMeters = meanUncertainty,
            temporalStability = temporalStability,
            confidenceBand = band,
        )

        val depthSamples = joints.map {
            MonocularDepthSample(
                x = usable.getValue(it.landmarkId).positionNormalized.x,
                y = usable.getValue(it.landmarkId).positionNormalized.y,
                depthMeters = it.depthMeters,
                confidence = it.depthConfidence,
                source = scaleSource,
            )
        }

        val depthFrame = MonocularDepthFrame(
            timestampNanos = timestampNanos,
            width = imageWidth,
            height = imageHeight,
            samples = depthSamples,
            meanDepthMeters = joints.map { it.depthMeters }.averageOrZero(),
            medianDepthMeters = median(joints.map { it.depthMeters }),
            scaleSource = scaleSource,
            confidence =
                joints.map { it.depthConfidence }.averageOrZero(),
            temporalStability = temporalStability,
            metric = scaleSource != ScaleSource.ANTHROPOMETRIC_PRIOR &&
                scaleSource != ScaleSource.UNKNOWN,
        )

        val gate = MotionQualityGate.evaluate(
            confidence = globalConfidence,
            uncertaintyMeters = meanUncertainty,
            reprojectionErrorPx = reprojectionPx,
            temporalStability = temporalStability,
        )

        return Result(
            frame = frame,
            depthFrame = depthFrame,
            gate = gate,
            iterations = iterationsCompleted,
        )
    }

    fun reconstruct(
        timestampNanos: Long,
        pose: PoseDetectionEngine.Result,
        intrinsics: CameraIntrinsics,
        topology: DynamicTopology,
        subjectType: SubjectType = SubjectType.HUMAN,
        trackId: Long = -1L,
        denseDepth: MonocularDepthFrame? = null,
        previous: MonocularReconstructionFrame? = null,
        calibratedScaleMeters: Double? = null,
    ): Result =
        reconstruct(
            timestampNanos = timestampNanos,
            subjectType = subjectType,
            trackId = trackId,
            imageWidth = pose.imageWidth,
            imageHeight = pose.imageHeight,
            intrinsics = intrinsics,
            observations = pose.landmarks.map {
                Observation(
                    id = it.id,
                    positionNormalized = Vec2(
                        it.x.toDouble() / pose.imageWidth.coerceAtLeast(1),
                        it.y.toDouble() / pose.imageHeight.coerceAtLeast(1),
                    ),
                    confidence = it.confidence.toDouble(),
                )
            },
            topology = topology,
            denseDepth = denseDepth,
            previous = previous,
            calibratedScaleMeters = calibratedScaleMeters,
        )

    private data class ScaleResolution(
        val expectedScaleMeters: Double,
        val rootDepthMeters: Double,
        val scaleSource: ScaleSource,
        val scaleConfidence: Double,
    )

    private fun resolveScale(
        subjectType: SubjectType,
        observations: List<Observation>,
        imageWidth: Int,
        imageHeight: Int,
        intrinsics: CameraIntrinsics,
        calibratedScaleMeters: Double?,
        denseDepth: MonocularDepthFrame?,
    ): ScaleResolution {
        if (calibratedScaleMeters != null && calibratedScaleMeters > 0.01) {
            val root =
                estimateRootDepth(
                    observations,
                    imageWidth,
                    imageHeight,
                    intrinsics,
                    calibratedScaleMeters,
                )
            return ScaleResolution(
                expectedScaleMeters = calibratedScaleMeters,
                rootDepthMeters = root,
                scaleSource = ScaleSource.CALIBRATED_METRIC,
                scaleConfidence = 0.98,
            )
        }

        denseDepth?.let {
            if (it.metric && it.confidence >= 0.45) {
                return ScaleResolution(
                    expectedScaleMeters = when (subjectType) {
                        SubjectType.HUMAN -> config.humanHeightPriorMeters
                        SubjectType.ANIMAL -> config.animalHeightPriorMeters
                        else -> config.objectDepthPriorMeters
                    },
                    rootDepthMeters =
                        it.medianDepthMeters.coerceIn(
                            config.minimumDepthMeters,
                            config.maximumDepthMeters,
                        ),
                    scaleSource = ScaleSource.DEPTH_SENSOR,
                    scaleConfidence = it.confidence.coerceIn(0.0, 1.0),
                )
            }
        }

        val subjectScale = when (subjectType) {
            SubjectType.HUMAN -> config.humanHeightPriorMeters
            SubjectType.ANIMAL -> config.animalHeightPriorMeters
            SubjectType.OBJECT -> config.objectDepthPriorMeters
            SubjectType.UNKNOWN -> config.objectDepthPriorMeters
        }

        val root =
            estimateRootDepth(
                observations,
                imageWidth,
                imageHeight,
                intrinsics,
                subjectScale,
            )

        return ScaleResolution(
            expectedScaleMeters = subjectScale,
            rootDepthMeters = root,
            scaleSource = if (subjectType == SubjectType.OBJECT || subjectType == SubjectType.UNKNOWN) {
                ScaleSource.UNKNOWN
            } else {
                ScaleSource.ANTHROPOMETRIC_PRIOR
            },
            scaleConfidence =
                if (subjectType == SubjectType.HUMAN || subjectType == SubjectType.ANIMAL) {
                    config.anthropometricScaleConfidence
                } else {
                    0.22
                },
        )
    }

    private fun estimateRootDepth(
        observations: List<Observation>,
        imageWidth: Int,
        imageHeight: Int,
        intrinsics: CameraIntrinsics,
        subjectScaleMeters: Double,
    ): Double {
        if (observations.size < 2) {
            return (
                (
                    (intrinsics.fx + intrinsics.fy) * 0.5
                ) * subjectScaleMeters / imageHeight.coerceAtLeast(1)
            ).coerceIn(
                config.minimumDepthMeters,
                config.maximumDepthMeters,
            )
        }

        val minY = observations.minOf { it.positionNormalized.y }
        val maxY = observations.maxOf { it.positionNormalized.y }
        val heightPx = (
            (maxY - minY) *
                imageHeight.toDouble()
        ).coerceAtLeast(imageHeight * 0.08)

        val focal = (intrinsics.fx + intrinsics.fy) * 0.5
        return (
            focal * subjectScaleMeters / heightPx
        ).coerceIn(
            config.minimumDepthMeters,
            config.maximumDepthMeters,
        )
    }

    private fun initializeDepths(
        observations: Map<Int, Observation>,
        rays: Map<Int, Vec3>,
        topology: DynamicTopology,
        imageWidth: Int,
        imageHeight: Int,
        intrinsics: CameraIntrinsics,
        prior: ScaleResolution,
        denseDepth: MonocularDepthFrame?,
        previous: MonocularReconstructionFrame?,
    ): Map<Int, Double> {
        val old = previous?.joints?.associateBy { it.landmarkId }.orEmpty()
        val output = LinkedHashMap<Int, Double>()

        observations.forEach { (id, observation) ->
            val external = denseDepth?.let {
                nearestDenseDepth(it, observation.positionNormalized)
            }

            val previousDepth = old[id]?.depthMeters
            val priorDepth = (
                prior.rootDepthMeters *
                    depthShapeCorrection(
                        observation,
                        observations.values.toList(),
                    )
                ).coerceIn(
                    config.minimumDepthMeters,
                    config.maximumDepthMeters,
                )

            val candidates = buildList {
                external?.let {
                    val targetDepth = externalDepthMeters(
                        external = denseDepth,
                        sample = it,
                        rootDepthMeters = prior.rootDepthMeters,
                    )
                    val representationWeight =
                        if (
                            denseDepth?.representation ==
                                DepthRepresentation.METRIC_METERS
                        ) {
                            it.confidence
                        } else {
                            it.confidence * config.relativeDepthWeight
                        }
                    add(targetDepth to representationWeight)
                }
                previousDepth?.let {
                    add(it to old[id]?.confidence?.coerceIn(0.0, 1.0).orZero())
                }
                add(priorDepth to prior.scaleConfidence)
            }

            val weightedDepth = (
                candidates.sumOf { it.first * it.second } /
                    candidates.sumOf { it.second }
                        .coerceAtLeast(1e-6)
                ).coerceIn(
                    config.minimumDepthMeters,
                    config.maximumDepthMeters,
                )

            output[id] = weightedDepth
        }

        return output
    }

    private fun depthShapeCorrection(
        observation: Observation,
        all: List<Observation>,
    ): Double {
        val centerY = all.map { it.positionNormalized.y }.averageOrZero()
        val vertical = (
            abs(observation.positionNormalized.y - centerY)
        ).coerceIn(0.0, 0.5)
        return (
            1.0 +
                (vertical - 0.15) * 0.12
            ).coerceIn(0.86, 1.14)
    }

    private fun pixelRay(
        normalized: Vec2,
        width: Int,
        height: Int,
        intrinsics: CameraIntrinsics,
    ): Vec3 {
        val px = normalized.x * width.toDouble()
        val py = normalized.y * height.toDouble()
        val x = (
            px - intrinsics.cx
        ) / intrinsics.fx.coerceAtLeast(1e-6)
        val y = (
            py - intrinsics.cy
        ) / intrinsics.fy.coerceAtLeast(1e-6)
        return Vec3(x, y, 1.0).normalized()
    }

    private fun localTopologyResidual(
        landmarkId: Int,
        depths: Map<Int, Double>,
        rays: Map<Int, Vec3>,
        observations: Map<Int, Observation>,
        topology: DynamicTopology,
    ): Double {
        val residuals = topology.edges
            .asSequence()
            .filter { it.a == landmarkId || it.b == landmarkId }
            .mapNotNull { edge ->
                val za = depths[edge.a] ?: return@mapNotNull null
                val zb = depths[edge.b] ?: return@mapNotNull null
                val a = observations[edge.a] ?: return@mapNotNull null
                val b = observations[edge.b] ?: return@mapNotNull null
                val wa = a.confidence
                val wb = b.confidence
                val actual =
                    (
                        rays[edge.a]!! * za -
                            rays[edge.b]!! * zb
                        ).norm()
                abs(actual - edge.preferredLengthMeters) *
                    min(wa, wb)
            }
            .toList()
        return residuals.averageOrZero()
    }

    private fun temporalStability(
        previous: MonocularReconstructionFrame?,
        current: List<MonocularReconstructionJoint>,
        timestampNanos: Long,
    ): Double {
        if (previous == null || current.isEmpty()) return 0.48
        val old = previous.joints.associateBy { it.landmarkId }
        val dt = (
            timestampNanos - previous.timestampNanos
        ).coerceAtLeast(1L) / 1e9

        val residuals = current.mapNotNull { joint ->
            val prior = old[joint.landmarkId] ?: return@mapNotNull null
            val predicted =
                prior.positionCameraMeters +
                    prior.velocityCameraMetersPerSecond *
                    dt.coerceAtMost(config.maxDtSeconds)
            val error = joint.positionCameraMeters.distanceTo(predicted)
            exp(
                -error /
                    (
                        0.05 +
                            prior.uncertaintyMeters.coerceAtLeast(0.01)
                    ),
            ).coerceIn(0.0, 1.0)
        }

        return residuals.averageOrZero()
    }

    private fun externalDepthMeters(
        external: MonocularDepthFrame,
        sample: MonocularDepthSample,
        rootDepthMeters: Double,
    ): Double =
        when (external.representation) {
            DepthRepresentation.METRIC_METERS ->
                sample.depthMeters.coerceIn(
                    config.minimumDepthMeters,
                    config.maximumDepthMeters,
                )

            DepthRepresentation.RELATIVE_ORDERED_DEPTH -> {
                val normalized =
                    sample.depthMeters.coerceIn(0.0, 1.0)
                (
                    rootDepthMeters *
                        (0.70 + normalized * 0.60)
                ).coerceIn(
                    config.minimumDepthMeters,
                    config.maximumDepthMeters,
                )
            }
        }

    private fun nearestDenseDepth(
        frame: MonocularDepthFrame,
        normalized: Vec2,
    ): MonocularDepthSample? {
        if (frame.samples.isEmpty()) return null
        return frame.samples.minByOrNull { sample ->
            val dx = sample.x - normalized.x
            val dy = sample.y - normalized.y
            dx * dx + dy * dy
        }?.takeIf {
            it.depthMeters.isFinite() &&
                it.depthMeters in config.minimumDepthMeters..config.maximumDepthMeters
        }
    }

    private fun emptyFrame(
        timestampNanos: Long,
        subjectType: SubjectType,
        trackId: Long,
        calibratedScaleMeters: Double?,
    ): MonocularReconstructionFrame =
        MonocularReconstructionFrame(
            timestampNanos = timestampNanos,
            subjectType = subjectType,
            trackId = trackId,
            joints = emptyList(),
            rootPositionMeters = Vec3(0.0, 0.0, 0.0),
            globalScaleMeters = calibratedScaleMeters ?: 0.0,
            scaleSource =
                if (calibratedScaleMeters != null) {
                    ScaleSource.CALIBRATED_METRIC
                } else {
                    ScaleSource.UNKNOWN
                },
            scaleConfidence = 0.0,
            globalConfidence = 0.0,
            reprojectionErrorPx = Double.POSITIVE_INFINITY,
            meanUncertaintyMeters = Double.POSITIVE_INFINITY,
            temporalStability = 0.0,
            confidenceBand = ReconstructionConfidenceBand.UNTRUSTED,
        )

    private fun emptyDepthFrame(
        timestampNanos: Long,
        width: Int,
        height: Int,
    ): MonocularDepthFrame =
        MonocularDepthFrame(
            timestampNanos = timestampNanos,
            width = width,
            height = height,
            samples = emptyList(),
            meanDepthMeters = 0.0,
            medianDepthMeters = 0.0,
            scaleSource = ScaleSource.UNKNOWN,
            confidence = 0.0,
            temporalStability = 0.0,
            metric = false,
        )

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[mid - 1] + sorted[mid]) * 0.5
        } else {
            sorted[mid]
        }
    }

    private fun Vec3.isFinite(): Boolean =
        x.isFinite() && y.isFinite() && z.isFinite()

    private fun List<Vec3>.averageVectorOrZero(): Vec3 {
        if (isEmpty()) return Vec3(0.0, 0.0, 0.0)
        return Vec3(
            map { it.x }.average(),
            map { it.y }.average(),
            map { it.z }.average(),
        )
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private fun Double?.orZero(): Double = this ?: 0.0

    private fun List<Double>.averageOrNull(): Double =
        if (isEmpty()) 0.0 else average()

    private fun huberWeight(
        residual: Double,
        delta: Double,
    ): Double {
        val magnitude = abs(residual)
        if (!magnitude.isFinite()) return 0.0
        if (magnitude <= delta) return 1.0
        return (
            delta /
                magnitude.coerceAtLeast(1e-9)
            ).coerceIn(0.02, 1.0)
    }

    private fun huberLoss(
        residual: Double,
        delta: Double,
    ): Double {
        val magnitude = abs(residual)
        if (!magnitude.isFinite()) return delta * delta * 8.0
        return if (magnitude <= delta) {
            0.5 * residual * residual
        } else {
            delta * (
                magnitude - 0.5 * delta
            )
        }
    }


}

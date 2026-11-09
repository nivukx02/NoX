package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.ai.MotionQualityEngine
import com.nivukx.nox.core.ai.PoseDetectionEngine
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DynamicTopology
import com.nivukx.nox.core.model.MonocularReconstructionFrame
import com.nivukx.nox.core.model.MotionQualityGate
import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3

/**
 * Single-camera RGB reconstruction façade.
 *
 * This class deliberately composes specialized engines instead of becoming a second monolith:
 * perception produces 2D landmarks, topology defines the subject graph, depth fusion estimates
 * metric/relative depth, and temporal stabilization preserves continuity. The result is translated
 * into NoX's existing [SkeletonFrame] contract so multi-view and monocular capture can share
 * downstream avatar/session tooling.
 */
class MonocularMotionReconstructionPipeline(
    private val topologyEngine: DynamicTopologyEngine = DynamicTopologyEngine(),
    private val depthEngine: MonocularDepthFusionEngine = MonocularDepthFusionEngine(),
    private val qualityEngine: com.nivukx.nox.core.ai.MotionQualityEngine =
        com.nivukx.nox.core.ai.MotionQualityEngine(),
    private val occlusionRecovery: OcclusionRecoveryEngine =
        OcclusionRecoveryEngine(),
    private val kinematicSolver: HighEndKinematicConstraintSolver =
        HighEndKinematicConstraintSolver(
            HighEndKinematicConstraintSolver.Config(
                iterations = 18,
                temporalBlend = 0.34,
                edgeStiffness = 0.84,
                anchorStiffness = 0.90,
                predictionWeight = 0.72,
                microMotionPreservation = 0.94,
            ),
        ),
) {

    data class Input(
        val timestampNanos: Long,
        val subjectType: SubjectType,
        val trackId: Long,
        val pose: PoseDetectionEngine.Result,
        val intrinsics: CameraIntrinsics,
        val denseDepth:
            com.nivukx.nox.core.model.MonocularDepthFrame? = null,
        val previous: MonocularReconstructionFrame? = null,
        val calibratedScaleMeters: Double? = null,
        val captureTimingConfidence: Double = 1.0,
        val lightingConfidence: Double = 1.0,
        val cameraMotionRisk: Double = 0.0,
        val cameraMotionSuppression: Double = 0.72,
        val microMotionPreservation: Double = 0.88,
        val preserveMicroMotion: Boolean = true,
        val temporalEvidenceScale: Double = 1.0,
        val topologyConstraintScale: Double = 1.0,
        val maximumReconstructionUncertaintyMeters: Double = 0.35,
    )

    data class Result(
        val skeleton: SkeletonFrame,
        val monocular: MonocularReconstructionFrame,
        val topology: DynamicTopology,
        val depth:
            com.nivukx.nox.core.model.MonocularDepthFrame,
        val gate: com.nivukx.nox.core.model.MotionQualityGate,
        val quality: MotionQualityEngine.Result,
        val iterations: Int,
        val worldConsistency: WorldSpaceMotionConsistencyEngine.Result,
    )

    private val groundContactEngine: GroundContactAndRootMotionEngine =
        GroundContactAndRootMotionEngine()
    private val scaleRefinementEngine: AnthropometricScaleRefinementEngine =
        AnthropometricScaleRefinementEngine()
    private val worldConsistencyEngine =
        WorldSpaceMotionConsistencyEngine()

    private var previousTopology: DynamicTopology? = null

    @Synchronized
    fun configureCapturePolicy(
        maximumOcclusionPredictionFrames: Int,
        captureFps: Int,
    ) {
        occlusionRecovery.configureMaximumPredictionAgeFrames(
            frames = maximumOcclusionPredictionFrames,
            captureFps = captureFps,
        )
    }

    @Synchronized
    fun reconstruct(input: Input): Result {
        val topology = topologyEngine.buildFromPose(
            timestampNanos = input.timestampNanos,
            subjectType = input.subjectType,
            trackId = input.trackId,
            observations = input.pose.landmarks.map {
                com.nivukx.nox.core.model.LandmarkObservation(
                    landmarkId = it.id,
                    x = it.x.toDouble(),
                    y = it.y.toDouble(),
                    confidence = it.confidence.toDouble(),
                    timestampNanos = input.timestampNanos,
                    space = com.nivukx.nox.core.model.LandmarkSpace.PIXELS,
                    trackingId = input.trackId.toInt().takeIf { id -> id >= 0 },
                )
            },
            previous = previousTopology,
        )

        val solved = depthEngine.reconstruct(
            timestampNanos = input.timestampNanos,
            subjectType = input.subjectType,
            trackId = input.trackId,
            imageWidth = input.pose.imageWidth,
            imageHeight = input.pose.imageHeight,
            intrinsics = input.intrinsics,
            observations = input.pose.landmarks.map {
                MonocularDepthFusionEngine.Observation(
                    id = it.id,
                    positionNormalized = com.nivukx.nox.core.model.Vec2(
                        it.x.toDouble() / input.pose.imageWidth.coerceAtLeast(1),
                        it.y.toDouble() / input.pose.imageHeight.coerceAtLeast(1),
                    ),
                    confidence = it.confidence.toDouble(),
                )
            },
            topology = topology,
            denseDepth = input.denseDepth,
            previous = input.previous,
            calibratedScaleMeters = input.calibratedScaleMeters,
        )

        val recovered = occlusionRecovery.recover(
            current = solved.frame,
            topology = topology,
            previous = input.previous,
        )

        val recoveredFrame = recovered.frame

        val kinematicResult = kinematicSolver.solve(
            HighEndKinematicConstraintSolver.Frame(
                timestampNanos = recoveredFrame.timestampNanos,
                subjectType = recoveredFrame.subjectType,
                joints = recoveredFrame.joints.map { joint ->
                    HighEndKinematicConstraintSolver.Joint(
                        id = joint.landmarkId,
                        semanticName = joint.semanticName,
                        position = joint.positionCameraMeters,
                        confidence = joint.confidence,
                        predicted = joint.predicted,
                        visible = !joint.predicted,
                    )
                },
                edges = topology.edges.map { edge ->
                    HighEndKinematicConstraintSolver.Edge(
                        a = edge.a,
                        b = edge.b,
                        preferredLengthMeters = edge.preferredLengthMeters,
                        minimumLengthMeters = edge.minimumLengthMeters,
                        maximumLengthMeters = edge.maximumLengthMeters,
                        stiffness = edge.stiffness,
                    )
                },
                rootId = topology.rootNodeId,
            ),
        )

        val solvedById = kinematicResult.joints.associateBy {
            it.source.id
        }

        val reconstructed = recoveredFrame.copy(
            joints = recoveredFrame.joints.map { joint ->
                val refined = solvedById[joint.landmarkId]
                    ?: return@map joint

                joint.copy(
                    positionCameraMeters = refined.position,
                    velocityCameraMetersPerSecond =
                        refined.velocityMetersPerSecond,
                    accelerationCameraMetersPerSecond2 =
                        refined.accelerationMetersPerSecond2,
                    confidence = minOf(
                        joint.confidence,
                        refined.confidence,
                    ),
                    uncertaintyMeters = (
                        joint.uncertaintyMeters +
                            refined.residualMeters * 0.35
                        ).coerceAtLeast(joint.uncertaintyMeters),
                    depthMeters = refined.position.z,
                    predicted = joint.predicted || refined.predicted,
                )
            },
            globalConfidence = minOf(
                recoveredFrame.globalConfidence,
                kinematicResult.globalConfidence,
            ),
            meanUncertaintyMeters = maxOf(
                recoveredFrame.meanUncertaintyMeters,
                kinematicResult.meanResidualMeters * 0.40,
            ),
            temporalStability = minOf(
                recoveredFrame.temporalStability,
                kinematicResult.temporalStability,
            ),
        )

        val scaleRefinement =
            scaleRefinementEngine.refine(
                frame = reconstructed,
                topology = topology,
            )

        val scaledReconstruction = scaleRefinement.frame

        previousTopology = topology.copy(
            nodes = topology.nodes.map { node ->
                val joint = reconstructed.joints.firstOrNull {
                    it.landmarkId == node.id
                }
                node.copy(
                    position3D = joint?.positionCameraMeters ?: node.position3D,
                    uncertaintyMeters =
                        joint?.uncertaintyMeters ?: node.uncertaintyMeters,
                    confidence =
                        joint?.confidence ?: node.confidence,
                    predicted = joint?.predicted ?: node.predicted,
                )
            },
            scaleSource = scaledReconstruction.scaleSource,
            globalConfidence = scaledReconstruction.globalConfidence,
            temporalStability = scaledReconstruction.temporalStability,
            timestampNanos = scaledReconstruction.timestampNanos,
        )

        val groundResult =
            if (input.subjectType == SubjectType.HUMAN) {
                groundContactEngine.process(scaledReconstruction.toSkeletonFrame(accepted = true))
            } else {
                null
            }

        val contactFrame =
            groundResult?.skeleton
                ?: scaledReconstruction.toSkeletonFrame(accepted = true)

        val worldConsistency =
            worldConsistencyEngine.optimize(
                WorldSpaceMotionConsistencyEngine.Input(
                    frame = contactFrame,
                    topology = topology,
                    captureTimingConfidence =
                        input.captureTimingConfidence,
                    lightingConfidence =
                        input.lightingConfidence,
                    cameraMotionRisk =
                        (
                            input.cameraMotionRisk.coerceIn(0.0, 1.0) *
                                (
                                    1.0 -
                                        0.75 *
                                            input.cameraMotionSuppression
                                                .coerceIn(0.0, 1.0)
                                    )
                            ).coerceIn(0.0, 1.0),
                    groundHeightMeters =
                        groundResult?.plane?.offsetMeters?.let { -it },
                    preserveMicroMotion =
                        input.preserveMicroMotion &&
                            input.microMotionPreservation >= 0.35,
                    temporalEvidenceScale =
                        (
                            input.temporalEvidenceScale *
                                (
                                    0.55 +
                                        0.65 *
                                            input.microMotionPreservation
                                                .coerceIn(0.0, 1.0)
                                    )
                            ).coerceIn(0.25, 1.50),
                    topologyConstraintScale =
                        input.topologyConstraintScale,
                ),
            )

        val presentationFrame =
            worldConsistency.skeleton

        val quality = qualityEngine.evaluate(
            MotionQualityEngine.Input(
                timestampNanos = input.timestampNanos,
                subjectType = input.subjectType,
                detectorConfidence = input.pose.confidence.toDouble(),
                geometryConfidence = (
                    topology.globalConfidence *
                        kinematicResult.globalConfidence *
                        (
                            groundResult?.confidence?.coerceAtLeast(0.55)
                                ?: 1.0
                            )
                    ).coerceIn(0.0, 1.0),
                temporalStability = reconstructed.temporalStability,
                topologyStability = topology.topologyStability,
                occlusionRisk =
                    if (presentationFrame.joints.isEmpty()) {
                        1.0
                    } else {
                        presentationFrame.joints.count { it.observationCount <= 0 }
                            .toDouble() /
                            presentationFrame.joints.size.toDouble()
                    },
                depthConfidence = solved.depthFrame.confidence,
                scaleConfidence = scaledReconstruction.scaleConfidence,
                meanUncertaintyMeters =
                    presentationFrame.joints
                        .map { it.positionUncertaintyMeters }
                        .average()
                        .takeIf { it.isFinite() }
                        ?: 0.35,
                reprojectionErrorPx =
                    presentationFrame.joints
                        .map { it.reprojectionErrorPx }
                        .filter { it.isFinite() }
                        .maxOrNull()
                        ?.coerceAtLeast(0.0)
                        ?: 0.0,
                predictedFraction =
                    if (presentationFrame.joints.isEmpty()) {
                        1.0
                    } else {
                        presentationFrame.joints.count { it.observationCount <= 0 }
                            .toDouble() /
                            presentationFrame.joints.size.toDouble()
                    },
                visibleNodes =
                    presentationFrame.joints.count { it.observationCount > 0 },
                totalNodes = presentationFrame.joints.size,
                modelFresh = true,
            ),
        )

        val skeleton = presentationFrame

        return Result(
            skeleton = skeleton,
            monocular = reconstructed,
            topology = previousTopology ?: topology,
            depth = solved.depthFrame,
            gate = MotionQualityGate.evaluate(
                confidence = presentationFrame.globalConfidence,
                uncertaintyMeters =
                    presentationFrame.joints
                        .map { it.positionUncertaintyMeters }
                        .average()
                        .takeIf { it.isFinite() }
                        ?: 0.35,
                reprojectionErrorPx =
                    presentationFrame.joints
                        .map { it.reprojectionErrorPx }
                        .filter { it.isFinite() }
                        .maxOrNull()
                        ?.coerceAtLeast(0.0)
                        ?: 0.0,
                temporalStability = reconstructed.temporalStability,
            ),
            quality = quality,
            iterations =
                solved.iterations +
                    kinematicResult.iterations,
            worldConsistency = worldConsistency,
        )
    }

    @Synchronized
    fun reset() {
        previousTopology = null
        occlusionRecovery.clear()
        kinematicSolver.reset()
        groundContactEngine.reset()
        worldConsistencyEngine.reset()
    }

    private fun MonocularReconstructionFrame.toSkeletonFrame(
        accepted: Boolean,
    ): SkeletonFrame =
        SkeletonFrame(
            timestampNanos = timestampNanos,
            subjectType = subjectType,
            trackId = trackId,
            joints = joints.map { joint ->
                ReconstructionJoint(
                    landmarkId = joint.landmarkId,
                    position = joint.positionCameraMeters,
                    velocity = joint.velocityCameraMetersPerSecond,
                    confidence =
                        if (accepted) joint.confidence
                        else joint.confidence * 0.72,
                    reprojectionErrorPx = reprojectionErrorPx,
                    observationCount = if (joint.predicted) 0 else 1,
                    positionUncertaintyMeters = joint.uncertaintyMeters,
                )
            },
            globalConfidence =
                if (accepted) globalConfidence else globalConfidence * 0.72,
        )

    private fun Vec3.isFiniteSafe(): Boolean =
        x.isFinite() && y.isFinite() && z.isFinite()
}

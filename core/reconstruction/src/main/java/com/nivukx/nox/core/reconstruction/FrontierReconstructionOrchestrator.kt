package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.MeasurementAuthority
import com.nivukx.nox.core.model.ReconstructionMesh
import java.util.concurrent.atomic.AtomicBoolean

/** Bounded reconstruction pipeline composing sparse geometry, dynamic surface refinement, global 4D, and provenance. */
class FrontierReconstructionOrchestrator(
    private val sparse: SparseViewHumanReconstructionEngine = SparseViewHumanReconstructionEngine(),
    private val gaussian: DynamicGaussianSurfaceRefinementEngine = DynamicGaussianSurfaceRefinementEngine(),
    private val global: GlobalHumanScene4DOptimizationEngine = GlobalHumanScene4DOptimizationEngine(),
    private val provenance: ReconstructionProvenanceEngine = ReconstructionProvenanceEngine(),
    private val meshGaussianBridge: FrontierMeshGaussianRefinementBridge = FrontierMeshGaussianRefinementBridge(),
    private val frontier: FrontierReconstructionCoordinator = FrontierReconstructionCoordinator(),
) : AutoCloseable {

    private val closed = AtomicBoolean(false)

    data class FrontierInput(
        val cameras: List<com.nivukx.nox.core.model.CameraFederationEvidence>,
        val landmarks: List<com.nivukx.nox.core.model.FrontierLandmarkMeasurement>,
        val actors: List<com.nivukx.nox.core.model.FrontierActorRuntimeContract> = emptyList(),
        val surfaceObservations:
            List<ProbabilisticDynamicSurfaceFusionEngine.SurfaceObservation> =
            emptyList(),
        val precisionEvidence:
            com.nivukx.nox.core.model.PrecisionAuthoritySnapshot? = null,
        val headroom:
            com.nivukx.nox.core.ai.MultiPersonRuntimeCapacityGovernor.RuntimeHeadroom =
            com.nivukx.nox.core.ai.MultiPersonRuntimeCapacityGovernor.RuntimeHeadroom(
                cpuHeadroom = 1.0,
                gpuHeadroom = 1.0,
                npuHeadroom = 1.0,
                thermalHeadroom = 1.0,
                memoryHeadroom = 1.0,
            ),
        val captureFps: Int = 60,
    )

    data class Input(
        val timestampNanos: Long,
        val regions: List<SparseViewHumanReconstructionEngine.RegionEvidence>,
        val surfacePoints: List<DynamicGaussianSurfaceRefinementEngine.SurfacePoint>,
        val globalCameras: List<GlobalHumanScene4DOptimizationEngine.CameraState>,
        val globalPoints: List<GlobalHumanScene4DOptimizationEngine.ActorPoint>,
        val surfaceMesh: ReconstructionMesh? = null,
        val surfaceMeshConfidence: Double = 1.0,
        val surfaceMeshOrigin: EvidenceOrigin = EvidenceOrigin.OBSERVED,
        val bundleCameras: List<BundleAdjustmentEngine.Camera> = emptyList(),
        val bundlePoints: List<BundleAdjustmentEngine.Point> = emptyList(),
        val bundleObservations: List<BundleAdjustmentEngine.Observation> = emptyList(),
        val humanBundleEvidence: List<HumanAwareBundleAdjustmentEvidenceEngine.Evidence> = emptyList(),
        val globalSurfaceConstraints: List<GlobalHumanScene4DOptimizationEngine.SurfaceConstraint> = emptyList(),
        val interactionConstraints: List<GlobalHumanScene4DOptimizationEngine.InteractionConstraint> = emptyList(),
        val actorInteractionConstraints: List<GlobalHumanScene4DOptimizationEngine.ActorInteractionConstraint> = emptyList(),
        val frontier: FrontierInput? = null,
    )

    data class Result(
        val sparseRegions: List<SparseViewHumanReconstructionEngine.RegionResult>,
        val gaussians: List<DynamicGaussianSurfaceRefinementEngine.Gaussian>,
        val globalRefinement: GlobalHumanScene4DOptimizationEngine.Result,
        val provenance: ReconstructionProvenanceEngine.Result,
        val authoritativeRegionCount: Int,
        val humanAwareBundle: BundleAdjustmentEngine.HumanAwareResult? = null,
        val frontier: FrontierReconstructionCoordinator.Result? = null,
    )

    @Synchronized
    fun process(input: Input): Result {
        check(!closed.get()) {
            "FrontierReconstructionOrchestrator is closed"
        }
        val frontierResult =
            input.frontier?.let { request ->
                val derivedSurfaceObservations =
                    input.surfacePoints
                        .asSequence()
                        .filter {
                            it.positionMeters.x.isFinite() &&
                                it.positionMeters.y.isFinite() &&
                                it.positionMeters.z.isFinite() &&
                                it.normal.x.isFinite() &&
                                it.normal.y.isFinite() &&
                                it.normal.z.isFinite() &&
                                it.confidence.isFinite() &&
                                it.confidence > 0.0
                        }
                        .map { point ->
                            ProbabilisticDynamicSurfaceFusionEngine.SurfaceObservation(
                                patchId = point.id,
                                timestampNanos =
                                    point.timestampNanos
                                        .takeIf { it > 0L }
                                        ?: input.timestampNanos,
                                positionMeters = point.positionMeters,
                                normal = point.normal,
                                confidence =
                                    point.confidence.coerceIn(0.0, 1.0),
                                visibility =
                                    point.visibility.coerceIn(0.0, 1.0),
                                temporalConsistency =
                                    point.temporalStability.coerceIn(0.0, 1.0),
                                photometricConsistency =
                                    point.photometricConsistency.coerceIn(
                                        0.0,
                                        1.0,
                                    ),
                                motionBoundaryConfidence =
                                    point.motionBoundaryConfidence.coerceIn(
                                        0.0,
                                        1.0,
                                    ),
                                cameraCount =
                                    point.independentCameraCount.coerceAtLeast(0),
                                uncertaintyMeters =
                                    kotlin.math.sqrt(
                                        point.covariance
                                            .principalVarianceUpperBound()
                                            .coerceAtLeast(1e-12),
                                    ),
                                authority =
                                    when (point.origin) {
                                        EvidenceOrigin.OBSERVED ->
                                            MeasurementAuthority.OBSERVED
                                        EvidenceOrigin.GEOMETRICALLY_CONSTRAINED ->
                                            MeasurementAuthority.GEOMETRIC
                                        EvidenceOrigin.TEMPORALLY_PROPAGATED ->
                                            MeasurementAuthority.TEMPORAL
                                        EvidenceOrigin.LEARNED_PRIOR ->
                                            MeasurementAuthority.LEARNED
                                        EvidenceOrigin.GENERATIVE_COMPLETION ->
                                            MeasurementAuthority.COMPLETED
                                    },
                            )
                        }
                        .toList()

                frontier.processFrame(
                    timestampNanos = input.timestampNanos,
                    cameras = request.cameras,
                    landmarks = request.landmarks,
                    actors = request.actors,
                    surfaceObservations =
                        request.surfaceObservations +
                            derivedSurfaceObservations,
                    precisionEvidence = request.precisionEvidence,
                    headroom = request.headroom,
                    captureFps = request.captureFps,
                )
            }

        val sparseRegions = sparse.reconstruct(input.timestampNanos, input.regions)
        val priorGaussians = gaussian.refine(input.surfacePoints)
        val meshGaussians = input.surfaceMesh?.let { mesh ->
            meshGaussianBridge.refine(
                mesh = mesh,
                surfaceConfidence = input.surfaceMeshConfidence,
                origin = input.surfaceMeshOrigin,
            ).gaussians
        }.orEmpty()
        val meshIds = meshGaussians.map { it.id }.toSet()
        val gaussians = meshGaussians + priorGaussians.filter { it.id !in meshIds }
        val globalPointKeys = input.globalPoints
            .map { globalKey(it.actorId, it.pointId) }
            .toSet()
        val autoSurfaceConstraints = sparseRegions
            .asSequence()
            .filter {
                it.state !=
                    SparseViewHumanReconstructionEngine.State.INSUFFICIENT_EVIDENCE
            }
            .filter {
                globalKey(it.actorId, it.regionId) in globalPointKeys
            }
            .map { region ->
                val stateWeight =
                    when (region.state) {
                        SparseViewHumanReconstructionEngine.State.OBSERVED -> 0.88
                        SparseViewHumanReconstructionEngine.State.CONSTRAINED -> 0.64
                        SparseViewHumanReconstructionEngine.State.PROPAGATED -> 0.30
                        SparseViewHumanReconstructionEngine.State.LEARNED -> 0.18
                        SparseViewHumanReconstructionEngine.State.INSUFFICIENT_EVIDENCE -> 0.0
                    }
                GlobalHumanScene4DOptimizationEngine.SurfaceConstraint(
                    actorId = region.actorId,
                    pointId = region.regionId,
                    targetPositionMeters = region.positionMeters,
                    weight = stateWeight,
                    confidence = region.confidence,
                )
            }
            .toList()
        val mergedSurfaceConstraints =
            input.globalSurfaceConstraints + autoSurfaceConstraints
        val globalResult = global.optimize(
            cameras = input.globalCameras,
            points = input.globalPoints,
            surfaceConstraints = mergedSurfaceConstraints,
            interactionConstraints = input.interactionConstraints,
            actorInteractionConstraints = input.actorInteractionConstraints,
        )
        val humanAwareBundle =
            input.bundleCameras
                .takeIf { it.size >= 2 && it.any { camera -> camera.fixed } }
                ?.let {
                    BundleAdjustmentEngine().optimizeHumanAware(
                        inputCameras = input.bundleCameras,
                        inputPoints = input.bundlePoints,
                        observations = input.bundleObservations,
                        motionEvidence = input.humanBundleEvidence,
                    )
                }
        val regionInputs = sparseRegions.map { region ->
            val observed = (region.measuredRatio * 1000.0).toInt().coerceAtLeast(0)
            val inferred = (region.inferredRatio * 1000.0).toInt().coerceAtLeast(0)
            ReconstructionProvenanceEngine.RegionInput(
                actorId=region.actorId, regionId=region.regionId,
                counts=mapOf(EvidenceOrigin.OBSERVED to observed, EvidenceOrigin.GEOMETRICALLY_CONSTRAINED to 0, EvidenceOrigin.TEMPORALLY_PROPAGATED to inferred, EvidenceOrigin.LEARNED_PRIOR to 0, EvidenceOrigin.GENERATIVE_COMPLETION to 0),
                confidence=region.confidence,
            )
        }
        val prov = provenance.aggregate(regionInputs)
        return Result(
            sparseRegions = sparseRegions,
            gaussians = gaussians,
            globalRefinement = globalResult,
            provenance = prov,
            authoritativeRegionCount = sparseRegions.count {
                it.state != SparseViewHumanReconstructionEngine.State.INSUFFICIENT_EVIDENCE
            },
            humanAwareBundle = humanAwareBundle,
            frontier = frontierResult,
        )
    }

    @Synchronized
    override fun close() {
        if (!closed.compareAndSet(false, true)) return

        var failure: Throwable? = null
        runCatching { sparse.close() }
            .onFailure { error -> failure = error }
        runCatching { frontier.close() }
            .onFailure { error ->
                val prior = failure
                if (prior == null) {
                    failure = error
                } else {
                    prior.addSuppressed(error)
                }
            }

        failure?.let { throw it }
    }

    private fun globalKey(
        actorId: Int,
        pointId: Int,
    ): Long =
        (actorId.toLong() shl 32) xor
            (pointId.toLong() and 0xffffffffL)

}
package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.ai.MultiPersonRuntimeCapacityGovernor
import com.nivukx.nox.core.ai.UltraFederatedMotionFusionEngine
import com.nivukx.nox.core.model.CameraFederationEvidence
import com.nivukx.nox.core.model.FrontierActorRuntimeContract
import com.nivukx.nox.core.model.FrontierLandmark3dResult
import com.nivukx.nox.core.model.FrontierLandmarkMeasurement
import com.nivukx.nox.core.model.FrontierReconstructionProvenance
import com.nivukx.nox.core.model.MeasurementAuthority
import com.nivukx.nox.core.model.MeasurementFailure
import com.nivukx.nox.core.model.Vec3
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

/**
 * Production composition root for the new world-space fusion and dynamic surface layers.
 *
 * The coordinator is deliberately thin. Heavy mathematics stays in bounded specialist engines,
 * while this layer owns lifecycle, frame admission, evidence lineage, memory limits and a single
 * coherent result for the operator/runtime layer.
 */
class FrontierReconstructionCoordinator(
    private val fusion: UltraFederatedMotionFusionEngine =
        UltraFederatedMotionFusionEngine(),
    private val surface: ProbabilisticDynamicSurfaceFusionEngine =
        ProbabilisticDynamicSurfaceFusionEngine(),
    private val precisionQualifier: FrontierPrecisionQualificationEngine =
        FrontierPrecisionQualificationEngine(),
    private val detailAllocator: AdaptiveActorReconstructionDetailAllocator =
        AdaptiveActorReconstructionDetailAllocator(),
    private val representationAuthority:
        ReconstructionRepresentationAuthorityEngine =
        ReconstructionRepresentationAuthorityEngine(),
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        val maximumFrameHistory: Int = 8,
        val maximumSurfaceObservationsPerFrame: Int = 24_000,
        val maximumLandmarksPerFrame: Int = 65_536,
        val minimumAuthoritativeLandmarkConfidence: Double = 0.60,
        val maximumSurfacePredictionFrames: Int = 12,
    ) {
        init {
            require(maximumFrameHistory in 2..4096) {
                "Frame history must remain within the bounded retention envelope"
            }
            require(maximumSurfaceObservationsPerFrame in 1024..250_000) {
                "Per-frame surface observation capacity is outside the supported memory envelope"
            }
            require(maximumLandmarksPerFrame in 1024..250_000) {
                "Per-frame landmark capacity is outside the supported memory envelope"
            }
            require(
                minimumAuthoritativeLandmarkConfidence.isFinite() &&
                    minimumAuthoritativeLandmarkConfidence in 0.0..1.0,
            ) {
                "Authoritative landmark confidence must be finite and in [0, 1]"
            }
            require(maximumSurfacePredictionFrames in 1..120) {
                "Surface prediction retention must be bounded to 120 frames"
            }
        }
    }

    data class SurfaceFrameSummary(
        val integrated: Int,
        val rejected: Int,
        val predicted: Int,
        val patchCount: Int,
        val directMeasurementRatio: Double,
        val topologyStability: Double,
        val p95UncertaintyMeters: Double,
    )

    data class Result(
        val timestampNanos: Long,
        val worldLandmarks: List<FrontierLandmark3dResult>,
        val surfaceSummary: SurfaceFrameSummary,
        val provenance: FrontierReconstructionProvenance,
        val cameraCount: Int,
        val activeActorCount: Int,
        val heldActorCount: Int,
        val authoritativeLandmarkCount: Int,
        val failedLandmarkCount: Int,
        val warnings: List<String>,
        val failure: MeasurementFailure,
        val precisionQualification:
            FrontierPrecisionQualificationEngine.Result,
        val detailAllocation:
            AdaptiveActorReconstructionDetailAllocator.Result,
        val representationAuthority:
            ReconstructionRepresentationAuthorityEngine.Result,
    )

    private data class FrameRecord(
        val timestampNanos: Long,
        val worldLandmarkCount: Int,
        val authoritativeLandmarkCount: Int,
        val p95UncertaintyMeters: Double,
    )

    private val history = ArrayDeque<FrameRecord>()
    private val closed = AtomicBoolean(false)
    private var lastTimestampNanos = 0L

    @Synchronized
    fun processFrame(
        timestampNanos: Long,
        cameras: Collection<CameraFederationEvidence>,
        landmarks: Collection<FrontierLandmarkMeasurement>,
        actors: Collection<FrontierActorRuntimeContract> = emptyList(),
        surfaceObservations: Collection<ProbabilisticDynamicSurfaceFusionEngine.SurfaceObservation> =
            emptyList(),
        precisionEvidence:
            com.nivukx.nox.core.model.PrecisionAuthoritySnapshot? = null,
        headroom: MultiPersonRuntimeCapacityGovernor.RuntimeHeadroom =
            MultiPersonRuntimeCapacityGovernor.RuntimeHeadroom(
                cpuHeadroom = 1.0,
                gpuHeadroom = 1.0,
                npuHeadroom = 1.0,
                thermalHeadroom = 1.0,
                memoryHeadroom = 1.0,
            ),
        captureFps: Int = 60,
    ): Result {
        check(!closed.get()) { "FrontierReconstructionCoordinator is closed" }
        require(timestampNanos > 0L)
        require(lastTimestampNanos == 0L || timestampNanos >= lastTimestampNanos)
        require(captureFps in 15..240)

        if (lastTimestampNanos != 0L && timestampNanos == lastTimestampNanos) {
            return repeatResult(timestampNanos)
        }

        val prioritizedCameraIds =
            cameras.asSequence()
                .filter { it.usable }
                .distinctBy { it.cameraId }
                .sortedWith(
                    compareByDescending<CameraFederationEvidence> { it.authorityScore }
                        .thenBy { it.cameraId },
                )
                .take(ReconstructionFrameCapacityGuard.MAX_CAMERA_INPUTS)
                .map { it.cameraId }
                .toList()

        val selectedLandmarks =
            BoundedMultiCameraLandmarkSelector.select(
                items = landmarks,
                timestampNanos = timestampNanos,
                maximumItems = config.maximumLandmarksPerFrame,
                prioritizedCameraIds = prioritizedCameraIds,
            )
        val boundedLandmarks = selectedLandmarks.items

        val selectedSurfaceObservations =
            BoundedSurfaceObservationSelector.select(
                observations = surfaceObservations,
                timestampNanos = timestampNanos,
                maximumObservations = config.maximumSurfaceObservationsPerFrame,
            )

        // Capacity diagnostics measure unique identities, not raw collection entries.
        // Each selector retains at most cap entries and one overflow witness, even when a peer
        // delivers a very large list with repeated camera or actor identifiers.
        val cameraCapacitySelection =
            BoundedDistinctEligibleSelector.select(
                items = cameras,
                isEligible = { true },
                keyOf = { it.cameraId },
                maximumItems = ReconstructionFrameCapacityGuard.MAX_CAMERA_INPUTS,
            )
        val actorCapacitySelection =
            BoundedDistinctEligibleSelector.select(
                items = actors,
                isEligible = { true },
                keyOf = { it.actorId },
                maximumItems = ReconstructionFrameCapacityGuard.MAX_ACTOR_INPUTS,
            )

        val capacityDiagnostics =
            ReconstructionFrameCapacityGuard.inspect(
                cameraInputCount = cameraCapacitySelection.eligibleCountCappedAtOverflow,
                actorInputCount = actorCapacitySelection.eligibleCountCappedAtOverflow,
                uniqueEligibleLandmarkCount = selectedLandmarks.eligibleCountCappedAtOverflow,
                eligibleSurfaceObservationCount = selectedSurfaceObservations.eligibleCountCappedAtOverflow,
                maximumLandmarks = config.maximumLandmarksPerFrame,
                maximumSurfaceObservations = config.maximumSurfaceObservationsPerFrame,
            )

        val fusionResult =
            fusion.fuseFrame(
                timestampNanos = timestampNanos,
                cameras = cameras,
                observations = boundedLandmarks,
                actors = actors,
                headroom = headroom,
                captureFps = captureFps,
            )

        var integrated = 0
        var rejected = 0

        selectedSurfaceObservations.observations
            .forEach { observation ->
                if (surface.integrate(observation)) {
                    integrated++
                } else {
                    rejected++
                }
            }

        val predicted =
            if (selectedSurfaceObservations.observations.isEmpty()) {
                surface.predict(timestampNanos)
                    .coerceAtMost(config.maximumSurfacePredictionFrames *
                        surface.snapshot().size.coerceAtLeast(1))
            } else {
                0
            }

        surface.prune(timestampNanos)
        val surfaceStats = surface.stats()

        /*
         * Precision feasibility is a qualification signal, not measurement authority. An observed
         * or predicted point must never become authoritative merely because modeled feasibility is
         * high. Authoritative geometry requires the fusion engine to explicitly classify the point
         * as GEOMETRIC and to pass the minimum confidence contract.
         */
        val authoritative =
            fusionResult.landmarks.count {
                isAuthoritativeLandmark(
                    landmark = it,
                    minimumConfidence =
                        config.minimumAuthoritativeLandmarkConfidence,
                )
            }

        val failed =
            fusionResult.landmarks.count {
                it.failure != MeasurementFailure.NONE
            }

        val landmarkUncertainty =
            fusionResult.landmarks
                .map { it.p95UncertaintyMeters }
                .filter { it.isFinite() }

        val p95LandmarkUncertainty =
            percentile(
                landmarkUncertainty,
                0.95,
            )

        val combinedP95 =
            max(
                p95LandmarkUncertainty
                    .takeIf { it.isFinite() }
                    ?: 0.0,
                surfaceStats.p95UncertaintyMeters,
            )

        val p95ReprojectionPixels =
            percentile(
                fusionResult.landmarks
                    .map { it.reprojectionErrorPixels }
                    .filter { it.isFinite() },
                0.95,
            )

        val measuredRatio =
            if (fusionResult.landmarks.isEmpty()) {
                0.0
            } else {
                fusionResult.landmarks
                    .count {
                        it.authority == MeasurementAuthority.GEOMETRIC &&
                            it.failure == MeasurementFailure.NONE
                    }
                    .toDouble() /
                    fusionResult.landmarks.size.toDouble()
            }

        val provenance =
            FrontierReconstructionProvenance(
                observedRatio =
                    (
                        measuredRatio * 0.42 +
                            surfaceStats.directlyMeasuredRatio * 0.58
                        ).coerceIn(0.0, 1.0),
                geometricRatio =
                    (
                        measuredRatio * 0.72 +
                            surfaceStats.directlyMeasuredRatio * 0.28
                        ).coerceIn(0.0, 1.0),
                temporalRatio =
                    (
                        surfaceStats.predictedPatchCount.toDouble() /
                            surfaceStats.patchCount.coerceAtLeast(1)
                        ).coerceIn(0.0, 1.0),
                learnedRatio =
                    if (surfaceStats.patchCount == 0) {
                        0.0
                    } else {
                        surface.snapshot()
                            .count {
                                it.authority == MeasurementAuthority.LEARNED
                            }
                            .toDouble() /
                            surfaceStats.patchCount.toDouble()
                    },
                completedRatio =
                    if (surfaceStats.patchCount == 0) {
                        0.0
                    } else {
                        surface.snapshot()
                            .count {
                                it.authority == MeasurementAuthority.COMPLETED
                            }
                            .toDouble() /
                            surfaceStats.patchCount.toDouble()
                    },
                meanUncertaintyMeters =
                    surfaceStats.meanUncertaintyMeters
                        .coerceAtLeast(
                            if (landmarkUncertainty.isEmpty()) 0.0
                            else landmarkUncertainty.average(),
                        ),
                p95UncertaintyMeters = combinedP95,
                topologyContinuity = surfaceStats.topologyStability,
                temporalStability =
                    (
                        0.56 * surfaceStats.topologyStability +
                            0.44 *
                                (
                                    1.0 -
                                        surfaceStats.dynamicMotionMagnitudeMetersPerSecond /
                                            12.0
                                ).coerceIn(0.0, 1.0)
                        ).coerceIn(0.0, 1.0),
                viewDiversity = fusionResult.cameraPlan.diversityScore,
            )

        val detailAllocation =
            detailAllocator.allocate(
                actors =
                    actors
                        .asSequence()
                        .distinctBy { it.actorId }
                        .take(64)
                        .map { actor ->
                            AdaptiveActorReconstructionDetailAllocator.ActorRequest(
                                actorId = actor.actorId,
                                operatorPriority = actor.priority,
                                identityConfidence = actor.identityConfidence,
                                visibilityRatio = actor.visibleRatio,
                                occlusionRatio = actor.occlusionRatio,
                                motionComplexity = actor.priority,
                                interactionRisk = actor.interactionRisk,
                                cameraSupportCount =
                                    fusionResult.cameraPlan.authority.size,
                                requestedSurfaceDetail =
                                    actor.requestedSurfaceDetail,
                                requestedFaceDetail =
                                    actor.requestedFaceDetail,
                                requestedEyeDetail =
                                    actor.requestedEyeDetail,
                                requestedHandDetail =
                                    actor.requestedHandDetail,
                                faceGeometryConfidence =
                                    actor.faceGeometryConfidence,
                                eyeGeometryConfidence =
                                    actor.eyeGeometryConfidence,
                                handGeometryConfidence =
                                    actor.handGeometryConfidence,
                                surfaceGeometryConfidence =
                                    actor.surfaceGeometryConfidence,
                                occlusionRecoveryConfidence =
                                    actor.occlusionRecoveryConfidence,
                                identityStability =
                                    actor.identityStability,
                            )
                        }
                        .toList(),
                memoryHeadroom = headroom.memoryHeadroom,
                gpuHeadroom = headroom.gpuHeadroom,
                npuHeadroom = headroom.npuHeadroom,
            )

        val warnings =
            buildList {
                addAll(capacityDiagnostics.warnings)
                addAll(fusionResult.health.warnings)
                if (surfaceStats.predictedPatchCount > surfaceStats.patchCount / 3) {
                    add("Surface state is prediction-heavy; direct measurement authority is reduced.")
                }
                if (rejected > 0) {
                    add("Some surface observations were rejected by the bounded probabilistic state.")
                }
                if (failed > 0) {
                    add(
                        failed.toString() +
                            " world landmarks did not satisfy the geometric reconstruction gates.",
                    )
                }
                if (headroom.memoryHeadroom < 0.30) {
                    add("Memory headroom is low; reconstruction detail is being bounded.")
                }
                detailAllocation.warnings.forEach { add("detail-$it") }
            }
                .distinct()
                .take(32)

        val failure =
            when {
                fusionResult.health.failure != MeasurementFailure.NONE ->
                    fusionResult.health.failure
                failed > 0 &&
                    authoritative == 0 &&
                    boundedLandmarks.isNotEmpty() ->
                    MeasurementFailure.HIGH_UNCERTAINTY
                headroom.memoryHeadroom < 0.10 ->
                    MeasurementFailure.MEMORY_PRESSURE
                headroom.thermalHeadroom < 0.10 ->
                    MeasurementFailure.THERMAL_PRESSURE
                capacityDiagnostics.exceededStreams.isNotEmpty() ->
                    MeasurementFailure.HIGH_UNCERTAINTY
                else ->
                    MeasurementFailure.NONE
            }


        val maximumBaselineMeters =
            fusionResult.cameraPlan.authority
                .flatMapIndexed { index, first ->
                    fusionResult.cameraPlan.authority
                        .drop(index + 1)
                        .map { second ->
                            first.positionMeters.distanceTo(
                                second.positionMeters,
                            )
                        }
                }
                .filter { it.isFinite() }
                .maxOrNull()
                ?: 0.0

        val precisionQualification =
            precisionQualifier.evaluate(
                input =
                    FrontierPrecisionQualificationEngine.Input(
                        evidenceSamples =
                            fusionResult.landmarks.sumOf {
                                it.viewCount.coerceAtLeast(0)
                            },
                        measuredP95Millimeters =
                            precisionEvidence
                                ?.p95ErrorMeters
                                ?.times(1000.0),
                        measuredP99Millimeters =
                            precisionEvidence
                                ?.p99ErrorMeters
                                ?.times(1000.0),
                        externalReferenceP95Millimeters = null,
                        externalReferenceUncertaintyMillimeters = null,
                        systemMeasurementUncertaintyMillimeters =
                            precisionEvidence
                                ?.bootstrapHalfWidthMeters
                                ?.times(1000.0)
                                ?: combinedP95
                                    .takeIf { it.isFinite() }
                                    ?.times(1000.0),
                        matchedReference =
                            precisionEvidence
                                ?.matchedReferenceQualified
                                ?: false,
                        independentViews =
                            maxOf(
                                precisionEvidence
                                    ?.independentViewCount
                                    ?: 0,
                                fusionResult.landmarks
                                    .maxOfOrNull { it.viewCount }
                                    ?: 0,
                            ),
                        baselineMeters = maximumBaselineMeters,
                        parallaxDegrees =
                            maxOf(
                                precisionEvidence
                                    ?.viewDiversity
                                    ?.times(45.0)
                                    ?: 0.0,
                                fusionResult.landmarks
                                    .maxOfOrNull {
                                        it.angularDiversityDegrees
                                    }
                                    ?: 0.0,
                            ),
                        calibrationConfidence =
                            maxOf(
                                fusionResult.health.calibrationConfidence,
                                precisionEvidence
                                    ?.calibrationResidualPixelsP95
                                    ?.let {
                                        (1.0 - it / 2.0)
                                            .coerceIn(0.0, 1.0)
                                    }
                                    ?: 0.0,
                            ),
                        synchronizationConfidence =
                            maxOf(
                                fusionResult.health.synchronizationConfidence,
                                precisionEvidence
                                    ?.timingResidualNanosP95
                                    ?.let {
                                        (1.0 -
                                            it / 12_000_000.0
                                            )
                                            .coerceIn(0.0, 1.0)
                                    }
                                    ?: 0.0,
                            ),
                        viewDiversity =
                            maxOf(
                                fusionResult.health.viewDiversity,
                                precisionEvidence?.viewDiversity ?: 0.0,
                            ).coerceIn(0.0, 1.0),
                        crossViewAgreement =
                            if (p95ReprojectionPixels.isFinite()) {
                                (
                                    1.0 -
                                        p95ReprojectionPixels / 3.0
                                ).coerceIn(0.0, 1.0)
                            } else {
                                0.0
                            },
                        occlusionRatio =
                            actors
                                .map { it.occlusionRatio }
                                .average()
                                .takeIf { actors.isNotEmpty() }
                                ?.coerceIn(0.0, 1.0)
                                ?: 0.0,
                        identityContinuity =
                            actors
                                .map { it.identityConfidence }
                                .average()
                                .takeIf { actors.isNotEmpty() }
                                ?.coerceIn(0.0, 1.0)
                                ?: 1.0,
                        surfaceTemporalStability =
                            surfaceStats.topologyStability,
                        thermalRisk =
                            (1.0 - headroom.thermalHeadroom)
                                .coerceIn(0.0, 1.0),
                        memoryPressure =
                            (1.0 - headroom.memoryHeadroom)
                                .coerceIn(0.0, 1.0),
                        timingUncertaintyMicros =
                            fusionResult.cameraPlan.authority
                                .map {
                                    it.timingResidualMillis * 1000.0
                                }
                                .average()
                                .takeIf {
                                    fusionResult.cameraPlan.authority.isNotEmpty()
                                }
                                ?: 0.0,
                        reprojectionP95Pixels =
                            p95ReprojectionPixels
                                .takeIf { it.isFinite() }
                                ?: Double.POSITIVE_INFINITY,
                        hardFailure =
                            failure != MeasurementFailure.NONE,
                    ),
                enabled = true,
            )


        val meanLandmarkViewCount =
            fusionResult.landmarks
                .map { it.viewCount.toDouble() }
                .average()
                .takeIf { fusionResult.landmarks.isNotEmpty() }
                ?: 0.0

        val reconstructionRepresentations =
            representationAuthority.decide(
                evidence =
                    listOf(
                        ReconstructionRepresentationAuthorityEngine.Evidence(
                            representation =
                                ReconstructionRepresentationAuthorityEngine.Representation.PROBABILISTIC_SURFACE,
                            coverage =
                                (
                                    surfaceStats.patchCount.toDouble() /
                                        750_000.0
                                ).coerceIn(0.0, 1.0),
                            metricConfidence =
                                surfaceStats.directlyMeasuredRatio,
                            uncertaintyMeters =
                                surfaceStats.p95UncertaintyMeters
                                    .coerceAtLeast(1e-6),
                            temporalStability =
                                surfaceStats.topologyStability,
                            topologyStability =
                                surfaceStats.topologyStability,
                            appearanceConfidence =
                                surfaceStats.directlyMeasuredRatio,
                            dynamicCompatibility =
                                (
                                    1.0 -
                                        surfaceStats.dynamicMotionMagnitudeMetersPerSecond /
                                            12.0
                                ).coerceIn(0.0, 1.0),
                            memoryCostMegabytes =
                                detailAllocation.totalMemoryBudgetBytes.toDouble() /
                                    (1024.0 * 1024.0),
                            gpuCostNormalized =
                                detailAllocation.budgetUtilization,
                            directlyMeasuredRatio =
                                surfaceStats.directlyMeasuredRatio,
                        ),
                        ReconstructionRepresentationAuthorityEngine.Evidence(
                            representation =
                                ReconstructionRepresentationAuthorityEngine.Representation.GAUSSIAN,
                            coverage =
                                if (surfaceStats.patchCount == 0) {
                                    0.0
                                } else {
                                    (
                                        detailAllocation.totalGaussianBudget.toDouble() /
                                            1_500_000.0
                                    ).coerceIn(0.0, 1.0)
                                },
                            metricConfidence =
                                (
                                    precisionEvidence?.modeledSubMillimeterFeasible == true ||
                                        precisionQualification.measuredWithinTarget
                                    )
                                    .let { if (it) 0.88 else 0.64 },
                            uncertaintyMeters =
                                (precisionQualification.measuredP95Millimeters?.div(1000.0) ?: combinedP95)
                                    .coerceAtLeast(0.001),
                            temporalStability =
                                (precisionQualification.readinessScore * 6.0)
                                    .div(6.0)
                                    .coerceIn(0.0, 1.0),
                            topologyStability = 0.80,
                            appearanceConfidence = 0.92,
                            dynamicCompatibility = 0.96,
                            memoryCostMegabytes =
                                detailAllocation.totalGaussianBudget.toDouble() *
                                    96.0 /
                                    (1024.0 * 1024.0),
                            gpuCostNormalized =
                                (
                                    detailAllocation.totalGaussianBudget.toDouble() /
                                        1_500_000.0
                                ).coerceIn(0.0, 1.0),
                            directlyMeasuredRatio =
                                provenance.directlyMeasuredRatio,
                        ),
                        ReconstructionRepresentationAuthorityEngine.Evidence(
                            representation =
                                ReconstructionRepresentationAuthorityEngine.Representation.TSDF,
                            coverage =
                                fusionResult.cameraPlan.authority.size.toDouble() /
                                    32.0,
                            metricConfidence =
                                fusionResult.health.calibrationConfidence *
                                    fusionResult.health.synchronizationConfidence,
                            uncertaintyMeters =
                                combinedP95.coerceAtLeast(0.001),
                            temporalStability = 0.92,
                            topologyStability = 0.97,
                            appearanceConfidence = 0.64,
                            dynamicCompatibility = 0.82,
                            memoryCostMegabytes =
                                detailAllocation.totalSurfaceSamplesPerFrame.toDouble() *
                                    64.0 /
                                    (1024.0 * 1024.0),
                            gpuCostNormalized = 0.72,
                            directlyMeasuredRatio =
                                provenance.directlyMeasuredRatio,
                        ),
                        ReconstructionRepresentationAuthorityEngine.Evidence(
                            representation =
                                ReconstructionRepresentationAuthorityEngine.Representation.LEARNED_COMPLETION,
                            coverage =
                                (
                                    1.0 -
                                        provenance.directlyMeasuredRatio
                                ).coerceIn(0.0, 1.0),
                            metricConfidence = 0.40,
                            uncertaintyMeters =
                                combinedP95
                                    .coerceAtLeast(0.010),
                            temporalStability = 0.74,
                            topologyStability = 0.70,
                            appearanceConfidence = 0.84,
                            dynamicCompatibility = 0.70,
                            memoryCostMegabytes = 128.0,
                            gpuCostNormalized = 0.34,
                            directlyMeasuredRatio = 0.0,
                        ),
                    ),
                memoryHeadroom = headroom.memoryHeadroom,
                gpuHeadroom = headroom.gpuHeadroom,
                dynamicScene =
                    surfaceStats.dynamicMotionMagnitudeMetersPerSecond >
                        0.25,
            )

        history.addLast(
            FrameRecord(
                timestampNanos = timestampNanos,
                worldLandmarkCount = fusionResult.landmarks.size,
                authoritativeLandmarkCount = authoritative,
                p95UncertaintyMeters = combinedP95,
            ),
        )
        while (history.size > config.maximumFrameHistory) {
            history.removeFirst()
        }

        lastTimestampNanos = timestampNanos

        return Result(
            timestampNanos = timestampNanos,
            worldLandmarks = fusionResult.landmarks,
            surfaceSummary =
                SurfaceFrameSummary(
                    integrated = integrated,
                    rejected = rejected,
                    predicted = predicted,
                    patchCount = surfaceStats.patchCount,
                    directMeasurementRatio =
                        surfaceStats.directlyMeasuredRatio,
                    topologyStability =
                        surfaceStats.topologyStability,
                    p95UncertaintyMeters =
                        surfaceStats.p95UncertaintyMeters,
                ),
            provenance = provenance,
            cameraCount =
                cameras
                    .asSequence()
                    .distinctBy { it.cameraId }
                    .take(128)
                    .count(),
            activeActorCount =
                fusionResult.actorPlan.activeActorIds.size,
            heldActorCount =
                fusionResult.actorPlan.heldActorIds.size,
            authoritativeLandmarkCount = authoritative,
            failedLandmarkCount = failed,
            warnings =
                (
                    warnings +
                        precisionQualification.warnings
                            .map { "precision-$it" }
                )
                    .distinct()
                    .take(32),
            failure = failure,
            precisionQualification = precisionQualification,
            detailAllocation = detailAllocation,
            representationAuthority = reconstructionRepresentations,
        )
    }

    @Synchronized
    fun frameHistory(): List<Pair<Long, Int>> =
        history.map {
            it.timestampNanos to
                it.authoritativeLandmarkCount
        }

    @Synchronized
    fun reset() {
        if (closed.get()) return
        history.clear()
        lastTimestampNanos = 0L
        fusion.reset()
        surface.reset()
        detailAllocator.reset()
        representationAuthority
            .let {
                // Stateless arbitration engine: no persistent representation history is retained.
            }
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            history.clear()
            fusion.closeIfSupported()
            surface.close()
        }
    }

    internal fun isAuthoritativeLandmark(
        landmark: FrontierLandmark3dResult,
        minimumConfidence: Double,
    ): Boolean =
        landmark.failure == MeasurementFailure.NONE &&
            landmark.authority == MeasurementAuthority.GEOMETRIC &&
            landmark.precisionFeasibility >= minimumConfidence

    private fun repeatResult(timestampNanos: Long): Result =
        Result(
            timestampNanos = timestampNanos,
            worldLandmarks = emptyList(),
            surfaceSummary =
                SurfaceFrameSummary(
                    integrated = 0,
                    rejected = 0,
                    predicted = 0,
                    patchCount = surface.stats().patchCount,
                    directMeasurementRatio = surface.stats().directlyMeasuredRatio,
                    topologyStability = surface.stats().topologyStability,
                    p95UncertaintyMeters = surface.stats().p95UncertaintyMeters,
                ),
            provenance =
                FrontierReconstructionProvenance(
                    observedRatio = 0.0,
                    geometricRatio = 0.0,
                    temporalRatio = 1.0,
                    learnedRatio = 0.0,
                    completedRatio = 0.0,
                    meanUncertaintyMeters = surface.stats().meanUncertaintyMeters,
                    p95UncertaintyMeters = surface.stats().p95UncertaintyMeters,
                    topologyContinuity = surface.stats().topologyStability,
                    temporalStability = surface.stats().topologyStability,
                    viewDiversity = 0.0,
                ),
            cameraCount = 0,
            activeActorCount = 0,
            heldActorCount = 0,
            authoritativeLandmarkCount = 0,
            failedLandmarkCount = 0,
            warnings = listOf("Duplicate frame timestamp was ignored."),
            failure = MeasurementFailure.INVALID_TIMESTAMP,
            detailAllocation =
                detailAllocator.allocate(
                    actors = emptyList(),
                    memoryHeadroom = 1.0,
                    gpuHeadroom = 1.0,
                    npuHeadroom = 1.0,
                ),
            precisionQualification =
                precisionQualifier.evaluate(
                    input =
                        FrontierPrecisionQualificationEngine.Input(
                            evidenceSamples = 0,
                            matchedReference = false,
                            independentViews = 0,
                            baselineMeters = 0.0,
                            parallaxDegrees = 0.0,
                            calibrationConfidence = 0.0,
                            synchronizationConfidence = 0.0,
                            viewDiversity = 0.0,
                            crossViewAgreement = 0.0,
                            occlusionRatio = 1.0,
                            identityContinuity = 0.0,
                            surfaceTemporalStability = 0.0,
                            thermalRisk = 1.0,
                            memoryPressure = 1.0,
                            timingUncertaintyMicros = 0.0,
                            reprojectionP95Pixels = Double.POSITIVE_INFINITY,
                            hardFailure = true,
                        ),
                    enabled = true,
                ),
        )

    private fun percentile(
        values: List<Double>,
        probability: Double,
    ): Double {
        if (values.isEmpty()) return Double.POSITIVE_INFINITY
        val sorted = values.sorted()
        if (sorted.size == 1) return sorted.first()
        val index =
            probability.coerceIn(0.0, 1.0) *
                (sorted.size - 1)
        val lower = index.toInt()
        val upper = (lower + 1).coerceAtMost(sorted.lastIndex)
        val alpha = index - lower
        return sorted[lower] * (1.0 - alpha) +
            sorted[upper] * alpha
    }

    private fun UltraFederatedMotionFusionEngine.closeIfSupported() {
        reset()
    }
}

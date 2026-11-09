package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraFederationEvidence
import com.nivukx.nox.core.model.FrontierActorRuntimeContract
import com.nivukx.nox.core.model.FrontierCaptureHealth
import com.nivukx.nox.core.model.FrontierLandmark3dResult
import com.nivukx.nox.core.model.FrontierLandmarkMeasurement
import com.nivukx.nox.core.model.MeasurementAuthority
import com.nivukx.nox.core.model.MeasurementFailure
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * High-end, bounded multi-camera world-space fusion fabric.
 *
 * The engine is intentionally geometry-first:
 * - learned/semantic layers supply observations;
 * - calibrated camera models lift them to rays;
 * - robust weighted line intersection produces world-space estimates;
 * - uncertainty and timing quality control contribution authority;
 * - selection uses quality plus angular diversity rather than "best camera wins";
 * - all state is bounded for 128 cameras and 64 actor slots.
 *
 * This is an implementation layer, not a claim of sub-millimetre physical accuracy.
 */
class UltraFederatedMotionFusionEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumCameras: Int = 128,
        val maximumAuthorityCameras: Int = 32,
        val maximumWitnessCameras: Int = 96,
        val maximumActors: Int = 64,
        val targetActors: Int = 50,
        val minimumViews: Int = 2,
        val preferredViews: Int = 6,
        val minimumParallaxDegrees: Double = 6.0,
        val preferredParallaxDegrees: Double = 14.0,
        val maximumP95UncertaintyMeters: Double = 0.008,
        val maximumReprojectionPixels: Double = 3.0,
        val robustCutoffPixels: Double = 2.5,
        val maximumTemporalUncertaintyNanos: Long = 12_000_000L,
        val maximumSceneSpeedMetersPerSecond: Double = 12.0,
        val cameraDiversityWeight: Double = 0.42,
        val geometricContributionWeight: Double = 0.24,
        val minimumGeometricContribution: Double = 0.12,
        val triangulationPeakParallaxDegrees: Double = 90.0,
        val triangulationPeakWidthDegrees: Double = 45.0,
        val minimumTriangulationConditionScore: Double = 0.30,
    ) {
        init {
            require(maximumCameras in 1..128)
            require(maximumAuthorityCameras in 1..32)
            require(maximumWitnessCameras in 0..96)
            require(maximumActors in 1..64)
            require(targetActors in 1..maximumActors)
            require(minimumViews >= 2)
            require(preferredViews >= minimumViews)
            require(minimumParallaxDegrees > 0.0)
            require(preferredParallaxDegrees >= minimumParallaxDegrees)
            require(maximumP95UncertaintyMeters > 0.0)
            require(maximumReprojectionPixels > 0.0)
            require(robustCutoffPixels > 0.0)
            require(maximumTemporalUncertaintyNanos >= 0L)
            require(maximumSceneSpeedMetersPerSecond > 0.0)
            require(cameraDiversityWeight in 0.0..1.0)
            require(geometricContributionWeight in 0.0..1.0)
            require(minimumGeometricContribution in 0.0..1.0)
            require(triangulationPeakParallaxDegrees in 45.0..135.0)
            require(triangulationPeakWidthDegrees in 5.0..90.0)
            require(minimumTriangulationConditionScore in 0.0..1.0)
        }
    }

    data class ActorPlan(
        val activeActorIds: List<Long>,
        val heldActorIds: List<Long>,
        val fairnessBoostedActorIds: List<Long>,
        val budgets: Map<Long, MultiPersonRuntimeCapacityGovernor.ActorBudgetView>,
        val starvationRisk: Double,
        val warning: String?,
    )

    data class CameraPlan(
        val authority: List<CameraFederationEvidence>,
        val witness: List<CameraFederationEvidence>,
        val rejected: List<CameraFederationEvidence>,
        val diversityScore: Double,
    )

    data class FrameResult(
        val timestampNanos: Long,
        val cameraPlan: CameraPlan,
        val actorPlan: ActorPlan,
        val landmarks: List<FrontierLandmark3dResult>,
        val health: FrontierCaptureHealth,
    )

    private val runtimeGovernor =
        MultiPersonRuntimeCapacityGovernor(
            policy =
                com.nivukx.nox.core.model.MultiPersonCapacityPolicy
                    .fiftyPersonProduction(),
        )

    private var lastTimestampNanos: Long = 0L

    fun planActors(
        actors: Collection<FrontierActorRuntimeContract>,
        headroom: MultiPersonRuntimeCapacityGovernor.RuntimeHeadroom,
        captureFps: Int,
    ): ActorPlan {
        val bounded =
            actors
                .filter { it.actorId >= 0L }
                .distinctBy { it.actorId }
                .sortedByDescending { it.urgency }
                .take(config.maximumActors)

        val demands =
            bounded.map {
                MultiPersonRuntimeCapacityGovernor.SubjectDemand(
                    trackId = it.actorId,
                    confidence = it.identityConfidence,
                    visibleRatio = it.visibleRatio,
                    motionEnergy = it.priority,
                    faceRequested = it.requestedFaceDetail,
                    eyeRequested = it.requestedEyeDetail,
                    handsRequested = it.requestedHandDetail,
                    surfaceRequested = it.requestedSurfaceDetail,
                    interactionPriority = it.interactionRisk,
                    identityPriority = it.identityConfidence,
                )
            }

        val allocation =
            runtimeGovernor.allocate(
                demands = demands,
                headroom = headroom,
                captureFps = captureFps,
            )

        val views =
            allocation.budgets.associate {
                it.trackId to
                    MultiPersonRuntimeCapacityGovernor.ActorBudgetView(
                        bodyHz = it.bodyInferenceHz,
                        faceHz = it.faceInferenceHz,
                        eyeHz = it.eyeInferenceHz,
                        handHz = it.handInferenceHz,
                        surfaceHz = it.denseSurfaceHz,
                        reconstructionPriority = it.reconstructionPriority,
                        reserveStateBytes = it.reserveStateBytes,
                    )
            }

        return ActorPlan(
            activeActorIds = allocation.admittedSubjectIds,
            heldActorIds =
                allocation.heldSubjectIds
                    .take(config.maximumActors),
            fairnessBoostedActorIds =
                allocation.fairnessBoostedSubjectIds
                    .take(config.maximumActors),
            budgets = views,
            starvationRisk = allocation.starvationRisk,
            warning = allocation.warnings.firstOrNull(),
        )
    }

    fun selectCameras(
        candidates: Collection<CameraFederationEvidence>,
    ): CameraPlan {
        val bounded =
            candidates
                .distinctBy { it.cameraId }
                .filter { it.usable }
                .sortedWith(
                    compareByDescending<CameraFederationEvidence> {
                        it.authorityScore
                    }.thenBy { it.cameraId },
                )
                .take(config.maximumCameras)

        val authority = ArrayList<CameraFederationEvidence>(
            config.maximumAuthorityCameras,
        )
        val witness = ArrayList<CameraFederationEvidence>(
            config.maximumWitnessCameras,
        )
        val remaining = bounded.toMutableList()

        while (
            authority.size < config.maximumAuthorityCameras &&
            remaining.isNotEmpty()
        ) {
            var bestIndex = 0
            var bestScore = Double.NEGATIVE_INFINITY
            for (index in remaining.indices) {
                val candidate = remaining[index]
                val diversity =
                    if (authority.isEmpty()) {
                        1.0
                    } else {
                        authority.minOf { prior ->
                            angularSeparationDegrees(
                                prior.forward.normalized(),
                                candidate.forward.normalized(),
                            ) / 90.0
                        }.coerceIn(0.0, 1.0)
                    }

                val sectorBonus =
                    if (
                        authority.none {
                            it.sector == candidate.sector
                        }
                    ) {
                        0.12
                    } else {
                        0.0
                    }

                val score =
                    (
                        candidate.authorityScore *
                            (
                                1.0 +
                                    config.cameraDiversityWeight *
                                        diversity
                            ) +
                            sectorBonus
                        )

                if (
                    score > bestScore ||
                    (
                        score == bestScore &&
                            candidate.cameraId <
                                remaining[bestIndex].cameraId
                    )
                ) {
                    bestScore = score
                    bestIndex = index
                }
            }

            authority += remaining.removeAt(bestIndex)
        }

        remaining
            .sortedWith(
                compareByDescending<CameraFederationEvidence> {
                    it.authorityScore
                }.thenBy { it.cameraId },
            )
            .take(config.maximumWitnessCameras)
            .forEach { witness += it }

        val selected = authority.map { it } + witness.map { it }
        val rejected =
            bounded.filter { candidate ->
                selected.none { it.cameraId == candidate.cameraId }
            }

        val diversity =
            selected
                .take(config.maximumAuthorityCameras)
                .let(::cameraDiversityScore)

        return CameraPlan(
            authority = authority,
            witness = witness,
            rejected = rejected,
            diversityScore = diversity,
        )
    }

    fun fuseFrame(
        timestampNanos: Long,
        cameras: Collection<CameraFederationEvidence>,
        observations: Collection<FrontierLandmarkMeasurement>,
        actors: Collection<FrontierActorRuntimeContract> = emptyList(),
        headroom: MultiPersonRuntimeCapacityGovernor.RuntimeHeadroom =
            MultiPersonRuntimeCapacityGovernor.RuntimeHeadroom(
                cpuHeadroom = 1.0,
                gpuHeadroom = 1.0,
                npuHeadroom = 1.0,
                thermalHeadroom = 1.0,
                memoryHeadroom = 1.0,
            ),
        captureFps: Int = 60,
    ): FrameResult {
        require(timestampNanos > 0L)
        require(lastTimestampNanos == 0L || timestampNanos >= lastTimestampNanos)
        require(captureFps in 15..240)

        if (
            lastTimestampNanos != 0L &&
            timestampNanos < lastTimestampNanos
        ) {
            throw IllegalArgumentException("Timestamp regression")
        }

        lastTimestampNanos = timestampNanos

        val cameraPlan = selectCameras(cameras)
        val actorPlan = planActors(actors, headroom, captureFps)

        val authorityIds =
            cameraPlan.authority
                .map { it.cameraId }
                .toSet()

        val usableObservations =
            observations
                .filter { it.observedAtNanos > 0L }
                .distinctBy {
                    Triple(
                        it.landmarkId,
                        it.cameraId,
                        it.observedAtNanos,
                    )
                }
                .filter {
                    it.cameraId in authorityIds ||
                        cameraPlan.witness.any { camera ->
                            camera.cameraId == it.cameraId
                        }
                }
                .take(config.maximumCameras * 512)

        val grouped =
            usableObservations
                .groupBy { it.landmarkId }
                .entries
                .sortedBy { it.key }

        val fused =
            ArrayList<FrontierLandmark3dResult>(grouped.size)

        for ((landmarkId, group) in grouped) {
            fused +=
                fuseLandmark(
                    timestampNanos = timestampNanos,
                    landmarkId = landmarkId,
                    observations = group.value,
                    authorityCameras = cameraPlan.authority,
                )
        }

        val valid =
            fused.filter {
                it.failure == MeasurementFailure.NONE
            }

        val confidence =
            valid
                .map {
                    (
                        0.45 * (1.0 - it.reprojectionErrorPixels / config.maximumReprojectionPixels) +
                            0.35 * it.precisionFeasibility +
                            0.20 * (it.viewCount / 8.0).coerceIn(0.0, 1.0)
                        ).coerceIn(0.0, 1.0)
                }
                .average()
                .takeIf { valid.isNotEmpty() }
                ?: 0.0

        val p95 =
            percentile(
                valid.map { it.p95UncertaintyMeters },
                0.95,
            )

        val reprojection =
            percentile(
                valid.map { it.reprojectionErrorPixels },
                0.95,
            )

        val sync =
            cameraPlan.authority
                .map { it.synchronizationConfidence }
                .average()
                .takeIf { cameraPlan.authority.isNotEmpty() }
                ?: 0.0

        val calibration =
            cameraPlan.authority
                .map { it.calibrationConfidence }
                .average()
                .takeIf { cameraPlan.authority.isNotEmpty() }
                ?: 0.0

        val authoritative =
            valid.isNotEmpty() &&
                sync >= 0.85 &&
                calibration >= 0.85 &&
                cameraPlan.diversityScore >= 0.40 &&
                p95 <= config.maximumP95UncertaintyMeters

        val warnings =
            buildList {
                actorPlan.warning?.let(::add)
                if (cameraPlan.authority.size < 3) {
                    add("Fewer than three authority cameras are available.")
                }
                if (sync < 0.85) {
                    add("Timing authority is below geometric qualification.")
                }
                if (calibration < 0.85) {
                    add("Calibration authority is below geometric qualification.")
                }
                if (cameraPlan.diversityScore < 0.40) {
                    add("View diversity is insufficient for robust depth conditioning.")
                }
                if (p95.isFinite() && p95 > config.maximumP95UncertaintyMeters) {
                    add("P95 world-space uncertainty exceeds the configured gate.")
                }
                if (valid.any { it.failure == MeasurementFailure.INSUFFICIENT_VIEWS }) {
                    add("Some landmarks lack sufficient independent views.")
                }
            }.take(24)

        val failure =
            when {
                observations.isEmpty() -> MeasurementFailure.INSUFFICIENT_VIEWS
                cameraPlan.authority.size < 2 -> MeasurementFailure.INSUFFICIENT_VIEWS
                sync < 0.45 -> MeasurementFailure.POOR_TIMING
                calibration < 0.45 -> MeasurementFailure.POOR_CALIBRATION
                cameraPlan.diversityScore < 0.20 -> MeasurementFailure.POOR_VIEW_DIVERSITY
                else -> MeasurementFailure.NONE
            }

        val health =
            FrontierCaptureHealth(
                timestampNanos = timestampNanos,
                cameraCount = cameras.distinctBy { it.cameraId }.size.coerceAtMost(128),
                selectedAuthorityCameraCount = cameraPlan.authority.size,
                witnessCameraCount = cameraPlan.witness.size,
                activeActorCount = actorPlan.activeActorIds.size,
                heldActorCount = actorPlan.heldActorIds.size,
                meanGeometryConfidence = confidence,
                p95UncertaintyMeters =
                    if (p95.isFinite()) p95 else Double.POSITIVE_INFINITY,
                meanReprojectionErrorPixels =
                    if (reprojection.isFinite()) reprojection else Double.POSITIVE_INFINITY,
                synchronizationConfidence = sync,
                calibrationConfidence = calibration,
                viewDiversity = cameraPlan.diversityScore,
                memoryHeadroom = headroom.memoryHeadroom,
                thermalHeadroom = headroom.thermalHeadroom,
                networkHeadroom =
                    cameraPlan.authority
                        .map { it.networkHeadroom }
                        .average()
                        .takeIf { cameraPlan.authority.isNotEmpty() }
                        ?: 0.0,
                precisionFeasibility =
                    valid
                        .map { it.precisionFeasibility }
                        .average()
                        .takeIf { valid.isNotEmpty() }
                        ?: 0.0,
                authoritative = authoritative,
                failure = failure,
                warnings = warnings,
            )

        return FrameResult(
            timestampNanos = timestampNanos,
            cameraPlan = cameraPlan,
            actorPlan = actorPlan,
            landmarks = fused,
            health = health,
        )
    }

    fun reset() {
        lastTimestampNanos = 0L
        runtimeGovernor.reset()
    }

    private fun fuseLandmark(
        timestampNanos: Long,
        landmarkId: Int,
        observations: List<FrontierLandmarkMeasurement>,
        authorityCameras: List<CameraFederationEvidence>,
    ): FrontierLandmark3dResult {
        val ranked =
            observations
                .distinctBy { it.cameraId }
                .mapNotNull { observation ->
                    val camera =
                        authorityCameras.firstOrNull {
                            it.cameraId == observation.cameraId
                        } ?: return@mapNotNull null
                    val ray = rayFromMeasurement(observation)
                        ?: return@mapNotNull null
                    val cameraWeight =
                        camera.authorityScore
                    val rowTimingConfidence =
                        (
                            1.0 -
                                abs(
                                    observation.rollingShutterFraction,
                                ) * 0.18
                        ).coerceIn(0.0, 1.0)

                    val observationWeight =
                        (
                            observation.confidence *
                                observation.timingQuality *
                                rowTimingConfidence *
                                observation.calibrationQuality *
                                cameraWeight
                            ).coerceIn(0.001, 1.0)
                    WeightedRay(
                        observation = observation,
                        origin = observation.cameraPose.positionMetersSafe(),
                        direction = ray,
                        weight = observationWeight,
                        camera = camera,
                    )
                }
                .let { rays ->
                    val candidates = rays.toList()
                    candidates.map { candidate ->
                        candidate.copy(
                            weight =
                                (
                                    candidate.weight *
                                        geometricContribution(
                                            candidate,
                                            candidates,
                                        )
                                ).coerceIn(0.001, 1.0),
                        )
                    }
                }
                .sortedWith(
                    compareByDescending<WeightedRay> { it.weight }
                        .thenBy { it.observation.cameraId },
                )

        if (ranked.size < config.minimumViews) {
            return FrontierLandmark3dResult(
                landmarkId = landmarkId,
                timestampNanos = timestampNanos,
                positionMeters = Vec3(0.0, 0.0, 0.0),
                covarianceMeters2 = Double.POSITIVE_INFINITY,
                p95UncertaintyMeters = Double.POSITIVE_INFINITY,
                reprojectionErrorPixels = Double.POSITIVE_INFINITY,
                viewCount = ranked.size,
                angularDiversityDegrees =
                    if (ranked.size <= 1) 0.0
                    else angularDiversity(ranked),
                cameraIds = ranked.map { it.observation.cameraId }.distinct(),
                authority = MeasurementAuthority.REJECTED,
                failure = MeasurementFailure.INSUFFICIENT_VIEWS,
            )
        }

        val diversity = angularDiversity(ranked)

        val triangulationConditionScore =
            triangulationConditionScore(ranked)

        if (
            triangulationConditionScore <
                config.minimumTriangulationConditionScore
        ) {
            return FrontierLandmark3dResult(
                landmarkId = landmarkId,
                timestampNanos = timestampNanos,
                positionMeters = Vec3(0.0, 0.0, 0.0),
                covarianceMeters2 = Double.POSITIVE_INFINITY,
                p95UncertaintyMeters = Double.POSITIVE_INFINITY,
                reprojectionErrorPixels = Double.POSITIVE_INFINITY,
                viewCount = ranked.size,
                angularDiversityDegrees = diversity,
                cameraIds = ranked.map { it.observation.cameraId }.distinct(),
                authority = MeasurementAuthority.REJECTED,
                failure = MeasurementFailure.POOR_VIEW_DIVERSITY,
            )
        }

        if (diversity < config.minimumParallaxDegrees) {
            return FrontierLandmark3dResult(
                landmarkId = landmarkId,
                timestampNanos = timestampNanos,
                positionMeters = Vec3(0.0, 0.0, 0.0),
                covarianceMeters2 = Double.POSITIVE_INFINITY,
                p95UncertaintyMeters = Double.POSITIVE_INFINITY,
                reprojectionErrorPixels = Double.POSITIVE_INFINITY,
                viewCount = ranked.size,
                angularDiversityDegrees = diversity,
                cameraIds = ranked.map { it.observation.cameraId }.distinct(),
                authority = MeasurementAuthority.REJECTED,
                failure = MeasurementFailure.POOR_VIEW_DIVERSITY,
            )
        }

        var active = ranked.take(config.preferredViews)
        var solution = solveRayIntersection(active)
        if (solution == null) {
            return FrontierLandmark3dResult(
                landmarkId = landmarkId,
                timestampNanos = timestampNanos,
                positionMeters = Vec3(0.0, 0.0, 0.0),
                covarianceMeters2 = Double.POSITIVE_INFINITY,
                p95UncertaintyMeters = Double.POSITIVE_INFINITY,
                reprojectionErrorPixels = Double.POSITIVE_INFINITY,
                viewCount = active.size,
                angularDiversityDegrees = diversity,
                cameraIds = active.map { it.observation.cameraId },
                authority = MeasurementAuthority.REJECTED,
                failure = MeasurementFailure.HIGH_UNCERTAINTY,
            )
        }

        repeat(3) {
            val residuals =
                active.map { ray ->
                    residualToRay(
                        solution.first,
                        ray.origin,
                        ray.direction,
                    )
                }

            val median = percentile(residuals, 0.50)
            val scale =
                max(
                    config.robustCutoffPixels * 0.0005,
                    median,
                )

            active =
                active.mapIndexedNotNull { index, ray ->
                    val distance = residuals[index]
                    val normalized = distance / max(scale, 1e-8)
                    val robust =
                        if (normalized <= 1.0) {
                            1.0 - 0.5 * normalized * normalized
                        } else {
                            0.5 / max(normalized, 1e-6)
                        }
                    val updated =
                        ray.copy(
                            weight =
                                (
                                    ray.weight * robust.coerceIn(0.05, 1.0)
                                ).coerceIn(0.001, 1.0),
                        )
                    if (updated.weight >= 0.02) updated else null
                }

            solveRayIntersection(active)?.let {
                solution = it
            }
        }

        val residuals =
            active.map {
                residualToRay(
                    solution.first,
                    it.origin,
                    it.direction,
                )
            }

        val rms =
            sqrt(
                residuals
                    .map { it * it }
                    .average()
                    .coerceAtLeast(0.0),
            )

        val covariance =
            solution.second

        val sigma =
            sqrt(
                covariance.trace / 3.0
            )

        val timingSigmaSeconds =
            active
                .map {
                    it.observation.temporalUncertaintyNanos
                        .coerceAtMost(
                            config.maximumTemporalUncertaintyNanos,
                        )
                        .toDouble() /
                        1_000_000_000.0
                }
                .average()
                .coerceAtLeast(0.0)

        // Temporal uncertainty is converted into a conservative spatial uncertainty using an
        // explicit maximum scene speed. This keeps units physically meaningful instead of
        // multiplying seconds by an arbitrary camera distance.
        val temporalSpatialSigma =
            timingSigmaSeconds *
                config.maximumSceneSpeedMetersPerSecond

        /*
         * Calibration uncertainty is metric geometry uncertainty, not merely a contribution
         * weight. Translation uncertainty directly moves a ray origin; rotation uncertainty
         * becomes a range-dependent lateral error. Propagate both into world-space sigma so
         * calibration error cannot disappear after the robust solve.
         */
        val calibrationTranslationSigma =
            sqrt(
                active
                    .map {
                        val value =
                            it.observation.calibrationTranslationSigmaMeters
                                .coerceAtLeast(0.0)
                        value * value
                    }
                    .average()
                    .coerceAtLeast(0.0),
            )

        val calibrationRotationSpatialSigma =
            sqrt(
                active
                    .map {
                        val dx =
                            solution.first.x - it.origin.x
                        val dy =
                            solution.first.y - it.origin.y
                        val dz =
                            solution.first.z - it.origin.z
                        val rangeMeters =
                            sqrt(
                                dx * dx +
                                    dy * dy +
                                    dz * dz,
                            ).coerceAtLeast(0.0)
                        val angularSigma =
                            it.observation.calibrationRotationSigmaRadians
                                .coerceAtLeast(0.0)
                                .coerceAtMost(
                                    Math.toRadians(15.0),
                                )
                        val spatial =
                            rangeMeters *
                                tan(angularSigma)
                        spatial * spatial
                    }
                    .average()
                    .coerceAtLeast(0.0),
            )

        val calibrationSpatialSigma =
            sqrt(
                calibrationTranslationSigma *
                    calibrationTranslationSigma +
                    calibrationRotationSpatialSigma *
                        calibrationRotationSpatialSigma,
            )

        val propagatedSigma =
            sqrt(
                sigma * sigma +
                    temporalSpatialSigma *
                        temporalSpatialSigma +
                    calibrationSpatialSigma *
                        calibrationSpatialSigma
            )

        val p95Uncertainty =
            propagatedSigma * 1.96

        val confidence =
            (
                active.map { it.weight }.average() *
                    (
                        0.55 +
                            0.45 *
                                (
                                    1.0 -
                                        rms /
                                            config.maximumReprojectionPixels
                                ).coerceIn(0.0, 1.0)
                    )
                ).coerceIn(0.0, 1.0)

        val authority =
            if (
                confidence >= 0.72 &&
                p95Uncertainty <= config.maximumP95UncertaintyMeters &&
                rms <= config.maximumReprojectionPixels
            ) {
                MeasurementAuthority.GEOMETRIC
            } else {
                MeasurementAuthority.OBSERVED
            }

        return FrontierLandmark3dResult(
            landmarkId = landmarkId,
            timestampNanos = timestampNanos,
            positionMeters = solution.first,
            covarianceMeters2 = propagatedSigma * propagatedSigma,
            p95UncertaintyMeters = p95Uncertainty,
            reprojectionErrorPixels = rms,
            viewCount = active.size,
            angularDiversityDegrees = angularDiversity(active),
            cameraIds = active.map { it.observation.cameraId }.distinct(),
            authority = authority,
            failure =
                if (p95Uncertainty.isFinite()) {
                    MeasurementFailure.NONE
                } else {
                    MeasurementFailure.HIGH_UNCERTAINTY
                },
        )
    }

    private data class WeightedRay(
        val observation: FrontierLandmarkMeasurement,
        val origin: Vec3,
        val direction: Vec3,
        val weight: Double,
        val camera: CameraFederationEvidence,
    )

    private fun geometricContribution(
        candidate: WeightedRay,
        candidates: List<WeightedRay>,
    ): Double {
        if (candidates.size <= 1) return 1.0

        val separations =
            candidates
                .asSequence()
                .filter {
                    it.camera.cameraId !=
                        candidate.camera.cameraId
                }
                .map {
                    angleBetweenDirections(
                        candidate.direction,
                        it.direction,
                    )
                }
                .filter { it.isFinite() }
                .toList()

        val bestAngularInformation =
            separations
                .map { separation ->
                    val radians =
                        Math.toRadians(
                            separation.coerceIn(0.0, 180.0),
                        )
                    val sineConditioning =
                        abs(
                            kotlin.math.sin(radians),
                        )
                    val peakProximity =
                        kotlin.math.exp(
                            -(
                                abs(
                                    separation -
                                        config.triangulationPeakParallaxDegrees,
                                ) /
                                    config.triangulationPeakWidthDegrees
                            ).let { it * it },
                        ).coerceIn(0.0, 1.0)
                    val minimumParallaxRamp =
                        (
                            separation -
                                config.minimumParallaxDegrees
                        ) /
                            (
                                config.preferredParallaxDegrees -
                                    config.minimumParallaxDegrees
                            ).coerceAtLeast(1e-6)

                    (
                        0.86 *
                            sineConditioning *
                            peakProximity +
                            0.14 *
                                minimumParallaxRamp.coerceIn(
                                    0.0,
                                    1.0,
                                )
                        ).coerceIn(0.0, 1.0)
                }
                .maxOrNull()
                ?: return config.minimumGeometricContribution

        val angularInformation =
            bestAngularInformation

        val calibrationInformation =
            (
                0.60 *
                    candidate.camera.calibrationConfidence +
                    0.40 *
                    candidate.camera.calibrationUncertaintyQuality
            ).coerceIn(0.0, 1.0)

        val synchronizationInformation =
            (
                0.70 *
                    candidate.camera.synchronizationConfidence +
                    0.30 *
                    candidate.camera.timingAuthority
            ).coerceIn(0.0, 1.0)

        return (
            (1.0 - config.geometricContributionWeight) +
                config.geometricContributionWeight *
                    (
                        0.72 * angularInformation +
                            0.16 * calibrationInformation +
                            0.12 * synchronizationInformation
                    )
            )
            .coerceIn(
                config.minimumGeometricContribution,
                1.20,
            )
    }

    private fun angleBetweenDirections(
        a: Vec3,
        b: Vec3,
    ): Double {
        val aNorm = a.norm()
        val bNorm = b.norm()

        if (
            !aNorm.isFinite() ||
            !bNorm.isFinite() ||
            aNorm < 1e-9 ||
            bNorm < 1e-9
        ) {
            return Double.NaN
        }

        val cosine =
            (
                a.dot(b) /
                    (aNorm * bNorm)
            ).coerceIn(-1.0, 1.0)

        return Math.toDegrees(
            acos(cosine),
        ).coerceIn(0.0, 180.0)
    }

    private data class Symmetric3(
        val xx: Double,
        val xy: Double,
        val xz: Double,
        val yy: Double,
        val yz: Double,
        val zz: Double,
    ) {
        val trace: Double
            get() = xx + yy + zz
    }

    private fun rayFromMeasurement(
        observation: FrontierLandmarkMeasurement,
    ): Vec3? {
        val normalized =
            undistort(
                observation.point,
                observation.intrinsics,
            )

        val x =
            (
                normalized.x -
                    observation.intrinsics.cx
            ) /
                observation.intrinsics.fx
        val y =
            (
                normalized.y -
                    observation.intrinsics.cy
            ) /
                observation.intrinsics.fy

        val cameraDirection =
            Vec3(x, y, 1.0).normalized()

        if (
            cameraDirection.norm() < 1e-9 ||
            !cameraDirection.x.isFinite() ||
            !cameraDirection.y.isFinite() ||
            !cameraDirection.z.isFinite()
        ) {
            return null
        }

        val world =
            observation.cameraPose.rotation
                .normalized()
                .rotate(cameraDirection)
                .normalized()

        return world.takeIf {
            it.norm() > 1e-9
        }
    }

    private fun undistort(
        point: Vec2,
        intrinsics: CameraIntrinsics,
    ): Vec2 {
        if (
            !listOf(
                intrinsics.k1,
                intrinsics.k2,
                intrinsics.k3,
                intrinsics.p1,
                intrinsics.p2,
            ).all { it.isFinite() }
        ) {
            return point
        }

        val fx = max(abs(intrinsics.fx), 1e-6)
        val fy = max(abs(intrinsics.fy), 1e-6)

        val xd = (point.x - intrinsics.cx) / fx
        val yd = (point.y - intrinsics.cy) / fy

        var x = xd
        var y = yd

        repeat(6) {
            val r2 = x * x + y * y
            val radial =
                1.0 +
                    intrinsics.k1 * r2 +
                    intrinsics.k2 * r2 * r2 +
                    intrinsics.k3 * r2 * r2 * r2

            val tx =
                2.0 * intrinsics.p1 * x * y +
                    intrinsics.p2 * (r2 + 2.0 * x * x)
            val ty =
                intrinsics.p1 * (r2 + 2.0 * y * y) +
                    2.0 * intrinsics.p2 * x * y

            if (abs(radial) < 1e-8) {
                return@repeat
            }

            x = (xd - tx) / radial
            y = (yd - ty) / radial
        }

        return Vec2(
            x * fx + intrinsics.cx,
            y * fy + intrinsics.cy,
        )
    }

    private fun solveRayIntersection(
        rays: List<WeightedRay>,
    ): Pair<Vec3, Symmetric3>? {
        if (rays.size < 2) return null

        var a00 = 0.0
        var a01 = 0.0
        var a02 = 0.0
        var a11 = 0.0
        var a12 = 0.0
        var a22 = 0.0
        var b0 = 0.0
        var b1 = 0.0
        var b2 = 0.0

        for (ray in rays) {
            val d = ray.direction.normalized()
            val w = ray.weight.coerceAtLeast(1e-4)
            val p00 = 1.0 - d.x * d.x
            val p01 = -d.x * d.y
            val p02 = -d.x * d.z
            val p11 = 1.0 - d.y * d.y
            val p12 = -d.y * d.z
            val p22 = 1.0 - d.z * d.z

            a00 += w * p00
            a01 += w * p01
            a02 += w * p02
            a11 += w * p11
            a12 += w * p12
            a22 += w * p22

            b0 += w * (p00 * ray.origin.x + p01 * ray.origin.y + p02 * ray.origin.z)
            b1 += w * (p01 * ray.origin.x + p11 * ray.origin.y + p12 * ray.origin.z)
            b2 += w * (p02 * ray.origin.x + p12 * ray.origin.y + p22 * ray.origin.z)
        }

        val matrix =
            doubleArrayOf(
                a00, a01, a02,
                a01, a11, a12,
                a02, a12, a22,
            )

        val rhs = doubleArrayOf(b0, b1, b2)

        val solution = solve3(matrix.copyOf(), rhs.copyOf()) ?: return null
        val inverse = invert3(matrix) ?: return null

        val position =
            Vec3(
                solution[0],
                solution[1],
                solution[2],
            )

        if (
            !position.x.isFinite() ||
            !position.y.isFinite() ||
            !position.z.isFinite()
        ) {
            return null
        }

        return Pair(
            position,
            Symmetric3(
                xx = max(inverse[0], 1e-12),
                xy = inverse[1],
                xz = inverse[2],
                yy = max(inverse[4], 1e-12),
                yz = inverse[5],
                zz = max(inverse[8], 1e-12),
            ),
        )
    }

    private fun solve3(
        m: DoubleArray,
        b: DoubleArray,
    ): DoubleArray? {
        for (pivot in 0 until 3) {
            var best = pivot
            var bestValue = abs(m[pivot * 3 + pivot])
            for (row in pivot + 1 until 3) {
                val value = abs(m[row * 3 + pivot])
                if (value > bestValue) {
                    best = row
                    bestValue = value
                }
            }

            if (!bestValue.isFinite() || bestValue < 1e-12) {
                return null
            }

            if (best != pivot) {
                for (column in pivot until 3) {
                    val tmp = m[pivot * 3 + column]
                    m[pivot * 3 + column] = m[best * 3 + column]
                    m[best * 3 + column] = tmp
                }
                val rhs = b[pivot]
                b[pivot] = b[best]
                b[best] = rhs
            }

            val divisor = m[pivot * 3 + pivot]
            for (column in pivot until 3) {
                m[pivot * 3 + column] /= divisor
            }
            b[pivot] /= divisor

            for (row in 0 until 3) {
                if (row == pivot) continue
                val factor = m[row * 3 + pivot]
                for (column in pivot until 3) {
                    m[row * 3 + column] -= factor * m[pivot * 3 + column]
                }
                b[row] -= factor * b[pivot]
            }
        }

        return b
    }

    private fun invert3(
        matrix: DoubleArray,
    ): DoubleArray? {
        val augmented =
            Array(3) { row ->
                DoubleArray(6).apply {
                    this[0] = matrix[row * 3]
                    this[1] = matrix[row * 3 + 1]
                    this[2] = matrix[row * 3 + 2]
                    this[3 + row] = 1.0
                }
            }

        for (pivot in 0 until 3) {
            var best = pivot
            var bestValue = abs(augmented[pivot][pivot])
            for (row in pivot + 1 until 3) {
                val value = abs(augmented[row][pivot])
                if (value > bestValue) {
                    best = row
                    bestValue = value
                }
            }

            if (!bestValue.isFinite() || bestValue < 1e-12) return null

            if (best != pivot) {
                val swap = augmented[pivot]
                augmented[pivot] = augmented[best]
                augmented[best] = swap
            }

            val divisor = augmented[pivot][pivot]
            for (column in 0 until 6) {
                augmented[pivot][column] /= divisor
            }

            for (row in 0 until 3) {
                if (row == pivot) continue
                val factor = augmented[row][pivot]
                for (column in 0 until 6) {
                    augmented[row][column] -=
                        factor * augmented[pivot][column]
                }
            }
        }

        return DoubleArray(9) { index ->
            augmented[index / 3][3 + index % 3]
        }
    }

    private fun residualToRay(
        point: Vec3,
        origin: Vec3,
        direction: Vec3,
    ): Double {
        val delta = point - origin
        return delta
            .cross(direction.normalized())
            .norm()
    }

    private fun angularDiversity(
        rays: List<WeightedRay>,
    ): Double =
        if (rays.size < 2) {
            0.0
        } else {
            rays
                .flatMapIndexed { firstIndex, first ->
                    rays.drop(firstIndex + 1).map { second ->
                        angularSeparationDegrees(
                            first.direction,
                            second.direction,
                        )
                    }
                }
                .maxOrNull()
                ?.coerceAtMost(180.0)
                ?: 0.0
        }

    private fun triangulationConditionScore(
        rays: List<WeightedRay>,
    ): Double {
        if (rays.size < 2) return 0.0

        return rays
            .flatMapIndexed { index, first ->
                rays.drop(index + 1).mapNotNull { second ->
                    val angle =
                        angularSeparationDegrees(
                            first.direction,
                            second.direction,
                        ).takeIf(Double::isFinite)
                            ?: return@mapNotNull null

                    val radians =
                        Math.toRadians(
                            angle.coerceIn(0.0, 180.0),
                        )
                    val sine =
                        abs(
                            kotlin.math.sin(radians),
                        )
                    val peak =
                        kotlin.math.exp(
                            -(
                                abs(
                                    angle -
                                        config.triangulationPeakParallaxDegrees,
                                ) /
                                    config.triangulationPeakWidthDegrees
                            ).let { value ->
                                value * value
                            },
                        ).coerceIn(0.0, 1.0)

                    (sine * peak).coerceIn(0.0, 1.0)
                }
            }
            .maxOrNull()
            ?: 0.0
    }

    private fun cameraDiversityScore(
        cameras: List<CameraFederationEvidence>,
    ): Double {
        if (cameras.size < 2) return 0.0
        val uniqueSectors =
            cameras
                .map { it.sector }
                .distinct()
                .size
        val sectorScore =
            (uniqueSectors.toDouble() /
                min(8, cameras.size).toDouble())
                .coerceIn(0.0, 1.0)

        val angularPairs =
            cameras
                .flatMapIndexed { index, first ->
                    cameras.drop(index + 1).map { second ->
                        angularSeparationDegrees(
                            first.forward.normalized(),
                            second.forward.normalized(),
                        )
                    }
                }

        val angularScore =
            (
                angularPairs
                    .filter { it.isFinite() }
                    .average()
                    .takeIf { angularPairs.isNotEmpty() }
                    ?: 0.0
            )
                .let { (it / 45.0).coerceIn(0.0, 1.0) }

        return (
            0.55 * sectorScore +
                0.45 * angularScore
            ).coerceIn(0.0, 1.0)
    }

    private fun angularSeparationDegrees(
        first: Vec3,
        second: Vec3,
    ): Double {
        val cosine =
            (
                first.normalized().dot(second.normalized())
            )
                .coerceIn(-1.0, 1.0)
        return Math.toDegrees(
            acos(cosine),
        )
    }

    private fun percentile(
        values: List<Double>,
        probability: Double,
    ): Double {
        val finite =
            values
                .filter { it.isFinite() }
                .sorted()

        if (finite.isEmpty()) return Double.POSITIVE_INFINITY
        if (finite.size == 1) return finite.first()

        val index =
            probability.coerceIn(0.0, 1.0) *
                (finite.size - 1)

        val lower = index.toInt()
        val upper = min(lower + 1, finite.lastIndex)
        val alpha = index - lower

        return (
            finite[lower] *
                (1.0 - alpha) +
                finite[upper] * alpha
            )
    }

    private fun Pose.positionMetersSafe(): Vec3 =
        position
}

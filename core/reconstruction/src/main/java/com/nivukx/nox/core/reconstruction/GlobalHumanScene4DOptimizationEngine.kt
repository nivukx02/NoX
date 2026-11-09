package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Bounded global human-scene 4D refinement over metric point observations.
 *
 * The optimizer is intentionally geometry-first:
 * - calibrated camera observations are the highest-authority measurement family;
 * - temporal observations provide continuity without becoming a false measurement source;
 * - surface constraints stabilize dense reconstruction where explicit support exists;
 * - robust IRLS weights suppress gross correspondence failures;
 * - bounded camera translation corrections are regularized and never alter camera rotation authority;
 * - deterministic ordering and hard budgets keep the solver safe for large scenes.
 *
 * This is a refinement engine, not a metrology certificate. Physical sub-millimetre accuracy still
 * requires matched hardware/reference measurements outside the source-code solver.
 */
class GlobalHumanScene4DOptimizationEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumActors: Int = 64,
        val maximumCameras: Int = 128,
        val maximumPoints: Int = 16_384,
        val maximumTemporalSamplesPerPoint: Int = 32,
        val maximumIterations: Int = 8,
        val maximumStepMeters: Double = 0.05,
        val maximumCameraCorrectionMeters: Double = 0.02,
        val temporalSmoothnessWeight: Double = 0.18,
        val temporalVelocityWeight: Double = 0.10,
        val temporalAccelerationWeight: Double = 0.08,
        val surfaceConstraintWeight: Double = 0.24,
        val cameraCorrectionWeight: Double = 0.12,
        val measurementWeight: Double = 1.0,
        val robustScaleMeters: Double = 0.03,
        val cauchyScaleMeters: Double = 0.055,
        val minimumImprovementMeters: Double = 1e-7,
        val maximumResidualMeters: Double = 5.0,
        val maximumVelocityMetersPerSecond: Double = 15.0,
        val maximumAccelerationMetersPerSecond2: Double = 120.0,
        val maximumJerkMetersPerSecond3: Double = 800.0,
        val interactionSeparationWeight: Double = 0.30,
        val contactAttractionWeight: Double = 0.10,
        val minimumActorSeparationMeters: Double = 0.06,
        val maximumInteractionPairs: Int = 4_096,
    ) {
        init {
            require(maximumActors in 1..64)
            require(maximumCameras in 1..128)
            require(maximumPoints > 0)
            require(maximumTemporalSamplesPerPoint in 1..128)
            require(maximumIterations in 1..32)
            require(maximumStepMeters > 0.0)
            require(maximumCameraCorrectionMeters > 0.0)
            require(temporalSmoothnessWeight >= 0.0)
            require(temporalVelocityWeight >= 0.0)
            require(temporalAccelerationWeight >= 0.0)
            require(surfaceConstraintWeight >= 0.0)
            require(cameraCorrectionWeight >= 0.0)
            require(measurementWeight > 0.0)
            require(robustScaleMeters > 0.0)
            require(cauchyScaleMeters >= robustScaleMeters)
            require(minimumImprovementMeters >= 0.0)
            require(maximumResidualMeters > 0.0)
            require(maximumVelocityMetersPerSecond > 0.0)
            require(maximumAccelerationMetersPerSecond2 > 0.0)
            require(maximumJerkMetersPerSecond3 > 0.0)
            require(interactionSeparationWeight >= 0.0)
            require(contactAttractionWeight >= 0.0)
            require(minimumActorSeparationMeters > 0.0)
            require(maximumInteractionPairs > 0)
        }
    }

    data class TemporalObservation(
        val timestampNanos: Long,
        val positionMeters: Vec3,
        val confidence: Double = 1.0,
        val velocityMetersPerSecond: Vec3? = null,
        val accelerationMetersPerSecond2: Vec3? = null,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(finite(positionMeters))
            require(confidence.isFinite() && confidence in 0.0..1.0)
            velocityMetersPerSecond?.let { require(finite(it)) }
            accelerationMetersPerSecond2?.let { require(finite(it)) }
        }
    }

    data class CameraState(
        val cameraId: String,
        val positionMeters: Vec3,
        val correctionMeters: Vec3 = Vec3(0.0, 0.0, 0.0),
        val uncertaintyMeters: Double = 0.01,
        val confidence: Double = 1.0,
        val fixed: Boolean = false,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(finite(positionMeters))
            require(finite(correctionMeters))
            require(uncertaintyMeters.isFinite() && uncertaintyMeters >= 0.0)
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }

        val correctedPositionMeters: Vec3
            get() = positionMeters + correctionMeters
    }

    data class ActorPoint(
        val actorId: Int,
        val pointId: Int,
        val positionMeters: Vec3,
        val cameraObservations: List<Vec3>,
        val confidence: Double,
        val temporalObservations: List<TemporalObservation> = emptyList(),
        val cameraObservationById: Map<String, Vec3> = emptyMap(),
    ) {
        init {
            require(actorId in 0..64)
            require(pointId >= 0)
            require(finite(positionMeters))
            require(cameraObservations.all(::finite))
            require(confidence.isFinite() && confidence in 0.0..1.0)
            require(temporalObservations.size <= 128)
            require(cameraObservationById.values.all(::finite))
        }

        fun effectiveCameraObservations(): List<Vec3> =
            if (cameraObservationById.isEmpty()) {
                cameraObservations
            } else {
                cameraObservationById.values.toList()
            }
    }

    data class ActorInteractionConstraint(
        val actorA: Int,
        val actorB: Int,
        val minimumDistanceMeters: Double,
        val contactProbability: Double = 0.0,
        val confidence: Double = 1.0,
    ) {
        init {
            require(actorA >= 0 && actorB >= 0)
            require(actorA != actorB)
            require(minimumDistanceMeters > 0.0)
            require(contactProbability in 0.0..1.0)
            require(confidence in 0.0..1.0)
        }
    }

    data class InteractionConstraint(
        val actorA: Int,
        val pointA: Int,
        val actorB: Int,
        val pointB: Int,
        val desiredDistanceMeters: Double,
        val contactProbability: Double = 0.0,
        val confidence: Double = 1.0,
    ) {
        init {
            require(actorA >= 0 && actorB >= 0)
            require(pointA >= 0 && pointB >= 0)
            require(actorA != actorB || pointA != pointB)
            require(desiredDistanceMeters > 0.0)
            require(contactProbability in 0.0..1.0)
            require(confidence in 0.0..1.0)
        }
    }

    data class SurfaceConstraint(
        val actorId: Int,
        val pointId: Int,
        val targetPositionMeters: Vec3,
        val weight: Double = 1.0,
        val confidence: Double = 1.0,
    ) {
        init {
            require(actorId >= 0)
            require(pointId >= 0)
            require(finite(targetPositionMeters))
            require(weight.isFinite() && weight >= 0.0)
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }
    }

    data class PointDiagnostic(
        val actorId: Int,
        val pointId: Int,
        val cameraObservationCount: Int,
        val temporalObservationCount: Int,
        val residualBeforeMeters: Double,
        val residualAfterMeters: Double,
        val displacementMeters: Double,
        val precisionWeight: Double,
    )

    data class TrajectoryDiagnostic(
        val actorId: Int,
        val pointId: Int,
        val sampleCount: Int,
        val maximumVelocityMetersPerSecond: Double,
        val maximumAccelerationMetersPerSecond2: Double,
        val maximumJerkMetersPerSecond3: Double,
        val kinematicEnvelopeScore: Double,
    )

    data class Result(
        val points: List<ActorPoint>,
        val cameras: List<CameraState>,
        val initialResidualMeters: Double,
        val finalResidualMeters: Double,
        val iterations: Int,
        val converged: Boolean,
        val rolledBack: Boolean,
        val optimizedCameraCorrectionCount: Int,
        val pointDiagnostics: List<PointDiagnostic> = emptyList(),
        val trajectoryDiagnostics: List<TrajectoryDiagnostic> = emptyList(),
        val rejectedObservationCount: Int = 0,
        val warnings: List<String> = emptyList(),
    )

    private data class MutablePoint(
        val actorId: Int,
        val pointId: Int,
        var position: Vec3,
        val cameraObservations: MutableList<Vec3>,
        val cameraObservationById: MutableMap<String, Vec3>,
        val temporalObservations: MutableList<TemporalObservation>,
        val confidence: Double,
        val originalPosition: Vec3,
    ) {
        fun toPublic(): ActorPoint =
            ActorPoint(
                actorId = actorId,
                pointId = pointId,
                positionMeters = position,
                cameraObservations = cameraObservations.toList(),
                confidence = confidence,
                temporalObservations = temporalObservations.toList(),
                cameraObservationById = cameraObservationById.toMap(),
            )
    }

    private val closedWarning = "global-4d-input-coherence-warning"
    private var lastRunSignature: Long = 0L

    fun optimize(
        cameras: List<CameraState>,
        points: List<ActorPoint>,
        surfaceConstraints: List<SurfaceConstraint> = emptyList(),
        interactionConstraints: List<InteractionConstraint> = emptyList(),
        actorInteractionConstraints: List<ActorInteractionConstraint> = emptyList(),
    ): Result {
        val boundedCameras = cameras
            .asSequence()
            .filter { it.cameraId.isNotBlank() && finite(it.positionMeters) }
            .distinctBy { it.cameraId }
            .sortedBy { it.cameraId }
            .take(config.maximumCameras)
            .toList()

        val overflowCameraCount = cameras.size - boundedCameras.size
        val pointCandidates = points
            .asSequence()
            .filter { finite(it.positionMeters) && it.confidence.isFinite() }
            .sortedWith(compareBy<ActorPoint> { it.actorId }.thenBy { it.pointId })
            .take(config.maximumPoints)
            .toList()
        val overflowPointCount = points.size - pointCandidates.size

        val state = pointCandidates.map { point ->
            MutablePoint(
                actorId = point.actorId,
                pointId = point.pointId,
                position = point.positionMeters,
                cameraObservations = point.cameraObservations
                    .filter(::finite)
                    .take(config.maximumCameras)
                    .toMutableList(),
                cameraObservationById = point.cameraObservationById
                    .filterKeys { key -> boundedCameras.any { it.cameraId == key } }
                    .filterValues(::finite)
                    .toMutableMap(),
                temporalObservations = point.temporalObservations
                    .filter { finite(it.positionMeters) }
                    .sortedBy { it.timestampNanos }
                    .take(config.maximumTemporalSamplesPerPoint)
                    .toMutableList(),
                confidence = point.confidence.coerceIn(0.0, 1.0),
                originalPosition = point.positionMeters,
            )
        }.toMutableList()

        val constraints = surfaceConstraints
            .asSequence()
            .filter { finite(it.targetPositionMeters) }
            .sortedWith(compareBy<SurfaceConstraint> { it.actorId }.thenBy { it.pointId })
            .take(min(config.maximumPoints * 2, 32_768))
            .toList()

        val pointMap = state.associateBy { key(it.actorId, it.pointId) }
        val constraintMap = constraints.groupBy { key(it.actorId, it.pointId) }
        val boundedInteractions = interactionConstraints
            .asSequence()
            .filter { it.confidence > 0.0 }
            .sortedWith(
                compareBy<InteractionConstraint> { min(it.actorA, it.actorB) }
                    .thenBy { max(it.actorA, it.actorB) }
                    .thenBy { min(it.pointA, it.pointB) }
                    .thenBy { max(it.pointA, it.pointB) },
            )
            .take(config.maximumInteractionPairs)
            .toList()
        val boundedActorInteractions = actorInteractionConstraints
            .asSequence()
            .filter {
                it.actorA in 0..64 &&
                    it.actorB in 0..64 &&
                    it.confidence > 0.0
            }
            .sortedWith(
                compareBy<ActorInteractionConstraint> {
                    min(it.actorA, it.actorB)
                }.thenBy {
                    max(it.actorA, it.actorB)
                },
            )
            .take(config.maximumInteractionPairs)
            .toList()

        val interactionMap = boundedInteractions
            .flatMap { constraint ->
                listOf(
                    key(constraint.actorA, constraint.pointA) to constraint,
                    key(constraint.actorB, constraint.pointB) to constraint,
                )
            }
            .groupBy({ it.first }, { it.second })

        val initialResidual = residual(
            points = state,
            cameras = boundedCameras,
            constraints = constraintMap,
            includeKinematics = true,
            interactions = boundedInteractions,
            actorInteractions = boundedActorInteractions,
        )

        var currentCameras = boundedCameras.toMutableList()
        var currentResidual = initialResidual
        var bestPoints = state.map { it.position }.toList()
        var bestCameras = currentCameras.toList()
        var acceptedIterations = 0
        var converged = false
        var rejectedObservationCount = 0

        val baselineResidual = if (initialResidual.isFinite()) initialResidual else config.maximumResidualMeters
        lastRunSignature = signature(state, currentCameras)

        repeat(config.maximumIterations) {
            if (state.isEmpty()) {
                converged = true
                return@repeat
            }

            val pointUpdates = Array(state.size) { Vec3(0.0, 0.0, 0.0) }
            val cameraById = currentCameras.associateBy { it.cameraId }
            val actorInteractionAdjustments =
                actorInteractionAdjustments(
                    points = state,
                    constraints = boundedActorInteractions,
                )

            state.forEachIndexed { index, point ->
                val update = solvePointUpdate(
                    point = point,
                    cameraById = cameraById,
                    constraints = constraintMap[key(point.actorId, point.pointId)].orEmpty(),
                    interactions = interactionMap[key(point.actorId, point.pointId)].orEmpty(),
                    actorInteractionAdjustment =
                        actorInteractionAdjustments[point.actorId]
                            ?: Vec3(0.0, 0.0, 0.0),
                    allPoints = state,
                )
                pointUpdates[index] = update.delta
                rejectedObservationCount += update.rejectedObservations
            }

            state.forEachIndexed { index, point ->
                val delta = clampNorm(
                    pointUpdates[index],
                    config.maximumStepMeters * point.confidence.coerceIn(0.1, 1.0),
                )
                point.position += delta
            }

            val trajectoryUpdates = temporalRegularizationStep(
                points = state,
                constraintMap = constraintMap,
            )
            trajectoryUpdates.forEachIndexed { index, delta ->
                state[index].position += clampNorm(
                    delta,
                    config.maximumStepMeters * config.temporalSmoothnessWeight.coerceIn(0.0, 1.0),
                )
            }

            val cameraResult = refineCameraCorrections(
                cameras = currentCameras,
                points = state,
            )
            currentCameras = cameraResult.cameras.toMutableList()

            val nextResidual = residual(
                points = state,
                cameras = currentCameras,
                constraints = constraintMap,
                includeKinematics = true,
                interactions = boundedInteractions,
                actorInteractions = boundedActorInteractions,
            )

            if (
                nextResidual.isFinite() &&
                nextResidual + config.minimumImprovementMeters < currentResidual
            ) {
                currentResidual = nextResidual
                bestPoints = state.map { it.position }.toList()
                bestCameras = currentCameras.toList()
                acceptedIterations += 1

                val improvement = (baselineResidual - currentResidual).coerceAtLeast(0.0)
                val fractional = improvement / baselineResidual.coerceAtLeast(1e-9)
                if (
                    fractional >= 0.999 ||
                    abs(initialResidual - currentResidual) <= config.minimumImprovementMeters
                ) {
                    converged = true
                    return@repeat
                }
            } else {
                state.forEachIndexed { index, point ->
                    point.position = bestPoints[index]
                }
                currentCameras = bestCameras.toMutableList()
                if (
                    nextResidual.isFinite() &&
                    abs(nextResidual - currentResidual) <= config.minimumImprovementMeters
                ) {
                    converged = true
                    return@repeat
                }
            }
        }

        val rolledBack = !currentResidual.isFinite() ||
            currentResidual > initialResidual + config.minimumImprovementMeters

        if (rolledBack) {
            state.forEachIndexed { index, point ->
                point.position = points.getOrNull(index)?.positionMeters ?: bestPoints[index]
            }
            currentCameras = boundedCameras.toMutableList()
            currentResidual = initialResidual
        }

        val publicPoints = if (rolledBack) {
            pointCandidates
        } else {
            state.map { it.toPublic() }
        }

        val cameraOutput = if (rolledBack) {
            boundedCameras
        } else {
            currentCameras
        }

        val pointDiagnostics = buildPointDiagnostics(
            before = pointCandidates,
            after = publicPoints,
            cameras = cameraOutput,
            constraints = constraintMap,
        )
        val trajectoryDiagnostics = buildTrajectoryDiagnostics(publicPoints)

        val warnings = buildList {
            if (overflowCameraCount > 0) add("global-camera-budget-truncated")
            if (overflowPointCount > 0) add("global-point-budget-truncated")
            if (constraints.size < surfaceConstraints.size) add("global-surface-constraint-budget-truncated")
            if (publicPoints.any { it.temporalObservations.size >= config.maximumTemporalSamplesPerPoint }) {
                add("global-temporal-sample-budget-active")
            }
            if (cameraOutput.none { it.fixed } && cameraOutput.isNotEmpty()) {
                add("global-camera-gauge-is-soft")
            }
            if (rejectedObservationCount > 0) add("global-nonfinite-observations-rejected")
            if (!currentResidual.isFinite()) add(closedWarning)
        }

        return Result(
            points = publicPoints,
            cameras = cameraOutput,
            initialResidualMeters = initialResidual,
            finalResidualMeters = currentResidual,
            iterations = acceptedIterations,
            converged = converged || acceptedIterations >= config.maximumIterations,
            rolledBack = rolledBack,
            optimizedCameraCorrectionCount = cameraOutput.count { it.correctionMeters.norm() > 1e-9 },
            pointDiagnostics = pointDiagnostics,
            trajectoryDiagnostics = trajectoryDiagnostics,
            rejectedObservationCount = rejectedObservationCount,
            warnings = warnings.distinct(),
        )
    }

    private data class PointUpdate(
        val delta: Vec3,
        val rejectedObservations: Int,
    )

    private data class Measurement(
        val position: Vec3,
        val weight: Double,
    )

    private fun solvePointUpdate(
        point: MutablePoint,
        cameraById: Map<String, CameraState>,
        constraints: List<SurfaceConstraint>,
        interactions: List<InteractionConstraint>,
        actorInteractionAdjustment: Vec3,
        allPoints: List<MutablePoint>,
    ): PointUpdate {
        val measurements = ArrayList<Measurement>()
        var rejected = 0

        if (point.cameraObservationById.isNotEmpty()) {
            point.cameraObservationById.entries
                .sortedBy { it.key }
                .forEach { (cameraId, observation) ->
                    val camera = cameraById[cameraId]
                    if (camera == null || !finite(observation)) {
                        rejected++
                        return@forEach
                    }
                    val corrected = observation - camera.correctionMeters
                    if (!finite(corrected)) {
                        rejected++
                        return@forEach
                    }
                    val geometricWeight = config.measurementWeight *
                        camera.confidence.coerceIn(0.02, 1.0) /
                        (1.0 + camera.uncertaintyMeters)
                    measurements += Measurement(corrected, geometricWeight)
                }
        } else {
            point.cameraObservations.forEach { observation ->
                if (!finite(observation)) {
                    rejected++
                } else {
                    measurements += Measurement(observation, config.measurementWeight)
                }
            }
        }

        point.temporalObservations
            .take(config.maximumTemporalSamplesPerPoint)
            .forEach { sample ->
                val temporalWeight = (
                    0.15 +
                        config.temporalSmoothnessWeight +
                        config.temporalVelocityWeight * velocityEvidence(sample) +
                        config.temporalAccelerationWeight * accelerationEvidence(sample)
                    ) * sample.confidence.coerceIn(0.02, 1.0)

                measurements += Measurement(
                    position = sample.positionMeters,
                    weight = temporalWeight,
                )
            }

        constraints.forEach { constraint ->
            val weight = (
                config.surfaceConstraintWeight *
                    constraint.weight *
                    constraint.confidence
                ).coerceAtLeast(0.0)
            if (weight > 0.0) {
                measurements += Measurement(
                    position = constraint.targetPositionMeters,
                    weight = weight,
                )
            }
        }

        val interactionDelta =
            interactionAdjustment(
                point = point,
                interactions = interactions,
                allPoints = allPoints,
            )

        if (measurements.isEmpty()) {
            return PointUpdate(interactionDelta, rejected)
        }

        var estimate = point.position
        var lastDelta = Vec3(0.0, 0.0, 0.0)

        repeat(4) {
            var weightedSum = Vec3(0.0, 0.0, 0.0)
            var totalWeight = 0.0

            measurements.forEach { measurement ->
                val error = measurement.position.distanceTo(estimate)
                if (!error.isFinite() || error > config.maximumResidualMeters) {
                    return@forEach
                }

                val robust = robustWeight(error)
                val weight = measurement.weight * robust
                if (weight <= 1e-9 || !weight.isFinite()) return@forEach

                weightedSum += measurement.position * weight
                totalWeight += weight
            }

            if (totalWeight <= 1e-9) return@repeat

            val desired = weightedSum / totalWeight
            lastDelta = desired - estimate
            estimate = desired
        }

        val temporalDelta = localTrajectoryGradient(point, allPoints)
        val combined =
            lastDelta +
                temporalDelta +
                interactionDelta +
                actorInteractionAdjustment

        return PointUpdate(
            delta = combined,
            rejectedObservations = rejected,
        )
    }

    private fun interactionAdjustment(
        point: MutablePoint,
        interactions: List<InteractionConstraint>,
        allPoints: List<MutablePoint>,
    ): Vec3 {
        if (interactions.isEmpty()) {
            return Vec3(0.0, 0.0, 0.0)
        }

        var adjustment = Vec3(0.0, 0.0, 0.0)

        interactions.forEach { constraint ->
            val pointIsA =
                point.actorId == constraint.actorA &&
                    point.pointId == constraint.pointA
            val otherActor =
                if (pointIsA) constraint.actorB else constraint.actorA
            val otherPoint =
                if (pointIsA) constraint.pointB else constraint.pointA

            val other =
                allPoints.firstOrNull {
                    it.actorId == otherActor &&
                        it.pointId == otherPoint
                } ?: return@forEach

            val offset = point.position - other.position
            val distance = offset.norm()
            if (!distance.isFinite()) return@forEach

            val direction =
                if (distance < 1e-9) {
                    Vec3(1.0, 0.0, 0.0)
                } else {
                    offset / distance
                }

            val desired =
                max(
                    config.minimumActorSeparationMeters,
                    constraint.desiredDistanceMeters,
                )

            val separationError =
                desired - distance

            val contactBlend =
                constraint.contactProbability.coerceIn(0.0, 1.0)

            val separationTerm =
                if (separationError > 0.0) {
                    direction *
                        (
                            separationError *
                                config.interactionSeparationWeight *
                                constraint.confidence
                        )
                } else {
                    Vec3(0.0, 0.0, 0.0)
                }

            val attractionTerm =
                if (
                    contactBlend > 0.0 &&
                    distance > desired
                ) {
                    direction *
                        (
                            -(distance - desired) *
                                config.contactAttractionWeight *
                                contactBlend *
                                constraint.confidence
                        )
                } else {
                    Vec3(0.0, 0.0, 0.0)
                }

            adjustment +=
                clampNorm(
                    separationTerm + attractionTerm,
                    config.maximumStepMeters * 0.50,
                )
        }

        return adjustment
    }

    private fun actorInteractionAdjustments(
        points: List<MutablePoint>,
        constraints: List<ActorInteractionConstraint>,
    ): Map<Int, Vec3> {
        if (points.isEmpty() || constraints.isEmpty()) return emptyMap()

        val groups =
            points.groupBy { it.actorId }
        val centroids =
            groups.mapValues { (_, values) ->
                val confidenceSum =
                    values.sumOf {
                        it.confidence.coerceIn(0.05, 1.0)
                    }.coerceAtLeast(1e-9)
                values.fold(Vec3(0.0, 0.0, 0.0)) { sum, point ->
                    sum +
                        point.position *
                        point.confidence.coerceIn(0.05, 1.0)
                } / confidenceSum
            }

        val output = HashMap<Int, Vec3>()
        constraints.forEach { constraint ->
            val first = centroids[constraint.actorA] ?: return@forEach
            val second = centroids[constraint.actorB] ?: return@forEach

            val delta = second - first
            val distance = delta.norm()
            if (!distance.isFinite()) return@forEach

            val direction =
                if (distance < 1e-9) {
                    Vec3(1.0, 0.0, 0.0)
                } else {
                    delta / distance
                }

            val error =
                constraint.minimumDistanceMeters - distance
            val contactBlend =
                constraint.contactProbability.coerceIn(0.0, 1.0)

            val separationMagnitude =
                if (error > 0.0) {
                    error * config.interactionSeparationWeight
                } else {
                    0.0
                }

            val attractionMagnitude =
                if (
                    contactBlend > 0.0 &&
                    error < 0.0
                ) {
                    error *
                        config.contactAttractionWeight *
                        contactBlend
                } else {
                    0.0
                }

            val signedMagnitude =
                (
                    separationMagnitude +
                        attractionMagnitude
                    ) *
                    constraint.confidence

            if (
                !signedMagnitude.isFinite() ||
                    abs(signedMagnitude) < 1e-9
            ) {
                return@forEach
            }

            val adjustment =
                clampNorm(
                    direction * (signedMagnitude * 0.5),
                    config.maximumStepMeters * 0.35,
                )

            output[constraint.actorA] =
                (output[constraint.actorA] ?: Vec3(0.0, 0.0, 0.0)) -
                    adjustment
            output[constraint.actorB] =
                (output[constraint.actorB] ?: Vec3(0.0, 0.0, 0.0)) +
                    adjustment
        }

        return output
    }

    private fun temporalRegularizationStep(
        points: List<MutablePoint>,
        constraintMap: Map<Long, List<SurfaceConstraint>>,
    ): List<Vec3> {
        if (
            config.temporalSmoothnessWeight <= 0.0 ||
            points.size < 2
        ) {
            return List(points.size) { Vec3(0.0, 0.0, 0.0) }
        }

        val byTrajectory = points
            .groupBy { it.actorId to it.pointId }
            .values
            .filter { it.size >= 2 }

        val deltaByIdentity = HashMap<Long, Vec3>()
        byTrajectory.forEach { trajectory ->
            val sorted = trajectory.sortedBy { temporalTimestamp(it) }
            sorted.forEachIndexed { index, point ->
                val previous = sorted.getOrNull(index - 1)
                val next = sorted.getOrNull(index + 1)
                val previousPosition = previous?.position
                val nextPosition = next?.position

                val candidate = when {
                    previousPosition != null && nextPosition != null ->
                        (
                            (previousPosition + nextPosition) / 2.0
                        ) - point.position

                    previousPosition != null ->
                        previousPosition - point.position

                    nextPosition != null ->
                        nextPosition - point.position

                    else -> Vec3(0.0, 0.0, 0.0)
                }

                val constraintBoost = constraintMap[key(point.actorId, point.pointId)]
                    .orEmpty()
                    .maxOfOrNull { it.confidence * it.weight }
                    ?.coerceIn(0.0, 1.0)
                    ?: 0.0

                deltaByIdentity[key(point.actorId, point.pointId)] =
                    candidate * (
                        config.temporalSmoothnessWeight *
                            (1.0 - 0.35 * constraintBoost)
                    )
            }
        }

        return points.map {
            deltaByIdentity[key(it.actorId, it.pointId)]
                ?: Vec3(0.0, 0.0, 0.0)
        }
    }

    private fun localTrajectoryGradient(
        point: MutablePoint,
        allPoints: List<MutablePoint>,
    ): Vec3 {
        if (point.temporalObservations.isEmpty()) {
            return Vec3(0.0, 0.0, 0.0)
        }

        val temporalTarget = point.temporalObservations
            .sortedBy { it.timestampNanos }
            .take(config.maximumTemporalSamplesPerPoint)
            .map { sample ->
                sample.positionMeters * (
                    sample.confidence.coerceIn(0.02, 1.0) *
                        (
                            0.5 +
                                0.5 * velocityEvidence(sample) +
                                0.25 * accelerationEvidence(sample)
                            )
                    )
            }
            .reduceOrNull(Vec3::plus)
            ?: return Vec3(0.0, 0.0, 0.0)

        val totalWeight = point.temporalObservations
            .sortedBy { it.timestampNanos }
            .take(config.maximumTemporalSamplesPerPoint)
            .sumOf {
                it.confidence.coerceIn(0.02, 1.0) *
                    (
                        0.5 +
                            0.5 * velocityEvidence(it) +
                            0.25 * accelerationEvidence(it)
                        )
            }
        if (totalWeight <= 1e-9) return Vec3(0.0, 0.0, 0.0)

        val target = temporalTarget / totalWeight
        val crossTrajectoryCount = allPoints.count {
            it.actorId == point.actorId &&
                it.pointId == point.pointId
        }

        val scale = (
            config.temporalVelocityWeight +
                config.temporalAccelerationWeight +
                min(crossTrajectoryCount, 4).toDouble() * 0.01
        ).coerceAtMost(0.5)

        return (target - point.position) * scale
    }

    private data class CameraRefinementResult(
        val cameras: List<CameraState>,
    )

    private fun refineCameraCorrections(
        cameras: List<CameraState>,
        points: List<MutablePoint>,
    ): CameraRefinementResult {
        if (cameras.isEmpty() || points.isEmpty() || config.cameraCorrectionWeight <= 0.0) {
            return CameraRefinementResult(cameras)
        }

        val referenceCameraId = cameras
            .filter { !it.fixed }
            .minByOrNull { it.cameraId }
            ?.cameraId

        val cameraById = cameras.associateBy { it.cameraId }
        val observationsByCamera = HashMap<String, MutableList<Pair<Vec3, Double>>>()
        points.forEach { point ->
            point.cameraObservationById.forEach { (cameraId, observation) ->
                if (!finite(observation)) return@forEach
                val camera = cameraById[cameraId] ?: return@forEach
                val residual = point.position - (observation - camera.correctionMeters)
                val error = residual.norm()
                val weight = point.confidence.coerceIn(0.02, 1.0) *
                    camera.confidence.coerceIn(0.02, 1.0) *
                    robustWeight(error)
                if (weight > 1e-8 && weight.isFinite()) {
                    observationsByCamera
                        .getOrPut(cameraId) { ArrayList() }
                        .add(residual to weight)
                }
            }
        }

        return CameraRefinementResult(
            cameras.map { camera ->
                if (camera.fixed || camera.cameraId == referenceCameraId) {
                    camera
                } else {
                    val evidence = observationsByCamera[camera.cameraId].orEmpty()
                    if (evidence.isEmpty()) {
                        camera
                    } else {
                        var weighted = Vec3(0.0, 0.0, 0.0)
                        var total = 0.0
                        evidence.forEach { (residual, weight) ->
                            weighted += residual * weight
                            total += weight
                        }
                        if (total <= 1e-9) {
                            camera
                        } else {
                            val desired = weighted / total
                            val bounded = clampNorm(
                                desired * config.cameraCorrectionWeight,
                                config.maximumCameraCorrectionMeters,
                            )
                            camera.copy(
                                correctionMeters = camera.correctionMeters + bounded,
                            )
                        }
                    }
                }
            },
        )
    }

    private fun residual(
        points: List<MutablePoint>,
        cameras: List<CameraState>,
        constraints: Map<Long, List<SurfaceConstraint>>,
        includeKinematics: Boolean,
        interactions: List<InteractionConstraint> = emptyList(),
        actorInteractions: List<ActorInteractionConstraint> = emptyList(),
    ): Double {
        val cameraById = cameras.associateBy { it.cameraId }
        val errors = ArrayList<Pair<Double, Double>>()
        points.forEach { point ->
            if (point.cameraObservationById.isNotEmpty()) {
                point.cameraObservationById.forEach { (cameraId, observation) ->
                    val camera = cameraById[cameraId] ?: return@forEach
                    if (!finite(observation)) return@forEach
                    val error = (observation - camera.correctionMeters).distanceTo(point.position)
                    if (error.isFinite() && error <= config.maximumResidualMeters) {
                        errors += error to (
                            camera.confidence.coerceIn(0.02, 1.0) /
                                (1.0 + camera.uncertaintyMeters)
                            )
                    }
                }
            } else {
                point.cameraObservations.forEach { observation ->
                    val error = observation.distanceTo(point.position)
                    if (error.isFinite() && error <= config.maximumResidualMeters) {
                        errors += error to config.measurementWeight
                    }
                }
            }

            point.temporalObservations
                .take(config.maximumTemporalSamplesPerPoint)
                .forEach { sample ->
                    val error = sample.positionMeters.distanceTo(point.position)
                    if (error.isFinite() && error <= config.maximumResidualMeters) {
                        errors += error to (
                            sample.confidence.coerceIn(0.02, 1.0) *
                                config.temporalSmoothnessWeight.coerceAtLeast(0.05)
                            )
                    }
                }

            constraints[key(point.actorId, point.pointId)]
                .orEmpty()
                .forEach { constraint ->
                    val error = constraint.targetPositionMeters.distanceTo(point.position)
                    if (error.isFinite() && error <= config.maximumResidualMeters) {
                        errors += error to (
                            constraint.weight *
                                constraint.confidence *
                                config.surfaceConstraintWeight
                            )
                    }
                }
        }

        val actorCentroids =
            points.groupBy { it.actorId }
                .mapValues { (_, values) ->
                    val total =
                        values.sumOf {
                            it.confidence.coerceIn(0.05, 1.0)
                        }.coerceAtLeast(1e-9)
                    values.fold(Vec3(0.0, 0.0, 0.0)) { sum, point ->
                        sum +
                            point.position *
                            point.confidence.coerceIn(0.05, 1.0)
                    } / total
                }

        actorInteractions.forEach { interaction ->
            val a = actorCentroids[interaction.actorA]
                ?: return@forEach
            val b = actorCentroids[interaction.actorB]
                ?: return@forEach
            val distance = a.distanceTo(b)
            if (!distance.isFinite()) return@forEach
            val desired = interaction.minimumDistanceMeters
            val error = abs(distance - desired)
            val weight =
                interaction.confidence *
                    (
                        config.interactionSeparationWeight +
                            config.contactAttractionWeight *
                            interaction.contactProbability
                    )
            if (
                weight > 1e-9 &&
                    error.isFinite() &&
                    error <= config.maximumResidualMeters
            ) {
                errors += error to weight
            }
        }

        interactions.forEach { interaction ->
            val a = points.firstOrNull {
                it.actorId == interaction.actorA &&
                    it.pointId == interaction.pointA
            } ?: return@forEach
            val b = points.firstOrNull {
                it.actorId == interaction.actorB &&
                    it.pointId == interaction.pointB
            } ?: return@forEach
            val distance =
                a.position.distanceTo(b.position)
            if (!distance.isFinite()) return@forEach
            val desired =
                max(
                    config.minimumActorSeparationMeters,
                    interaction.desiredDistanceMeters,
                )
            val error = abs(distance - desired)
            val weight =
                (
                    interaction.confidence *
                        (
                            config.interactionSeparationWeight +
                                config.contactAttractionWeight *
                                interaction.contactProbability
                        )
                    ).coerceAtLeast(0.0)
            if (
                weight > 1e-9 &&
                    error.isFinite() &&
                    error <= config.maximumResidualMeters
            ) {
                errors += error to weight
            }
        }

        if (errors.isEmpty()) return Double.POSITIVE_INFINITY

        var weightedSquared = 0.0
        var totalWeight = 0.0
        errors.forEach { (error, weight) ->
            val robust = robustWeight(error)
            val effective = weight * robust
            weightedSquared += effective * error * error
            totalWeight += effective
        }

        var result = if (totalWeight <= 1e-9) {
            Double.POSITIVE_INFINITY
        } else {
            sqrt(weightedSquared / totalWeight)
        }

        if (includeKinematics) {
            val kinematicPenalty = temporalKinematicPenalty(points)
            result *= 1.0 + 0.12 * kinematicPenalty
        }

        return result
            .takeIf { it.isFinite() }
            ?.coerceAtMost(config.maximumResidualMeters * 2.0)
            ?: Double.POSITIVE_INFINITY
    }

    private fun temporalKinematicPenalty(
        points: List<MutablePoint>,
    ): Double {
        var penaltySum = 0.0
        var trajectoryCount = 0

        points.forEach { point ->
            val samples = point.temporalObservations
                .sortedBy { it.timestampNanos }
                .take(config.maximumTemporalSamplesPerPoint)

            if (samples.size < 2) return@forEach

            trajectoryCount++
            var maximumVelocity = 0.0
            var maximumAcceleration = 0.0
            var maximumJerk = 0.0
            val velocities = ArrayList<Pair<Long, Vec3>>()

            for (index in 1 until samples.size) {
                val dt = secondsBetween(
                    samples[index - 1].timestampNanos,
                    samples[index].timestampNanos,
                )
                if (dt <= 1e-9) continue
                val velocity =
                    (
                        samples[index].positionMeters -
                            samples[index - 1].positionMeters
                        ) / dt
                maximumVelocity = max(
                    maximumVelocity,
                    velocity.norm(),
                )
                velocities += samples[index].timestampNanos to velocity
            }

            for (index in 1 until velocities.size) {
                val previous = velocities[index - 1]
                val current = velocities[index]
                val dt = secondsBetween(
                    previous.first,
                    current.first,
                )
                if (dt <= 1e-9) continue
                val acceleration =
                    (current.second - previous.second) / dt
                maximumAcceleration = max(
                    maximumAcceleration,
                    acceleration.norm(),
                )
            }

            for (index in 2 until velocities.size) {
                val first = velocities[index - 2]
                val second = velocities[index - 1]
                val third = velocities[index]
                val dt1 = secondsBetween(first.first, second.first)
                val dt2 = secondsBetween(second.first, third.first)
                if (min(dt1, dt2) <= 1e-9) continue
                val accelerationDelta =
                    (
                        (third.second - second.second) / dt2 -
                            (second.second - first.second) / dt1
                        )
                maximumJerk = max(
                    maximumJerk,
                    accelerationDelta.norm() /
                        ((dt1 + dt2) * 0.5),
                )
            }

            val velocityPenalty = (
                maximumVelocity /
                    config.maximumVelocityMetersPerSecond
            ).coerceIn(0.0, 1.0)
            val accelerationPenalty = (
                maximumAcceleration /
                    config.maximumAccelerationMetersPerSecond2
            ).coerceIn(0.0, 1.0)
            val jerkPenalty = (
                maximumJerk /
                    config.maximumJerkMetersPerSecond3
            ).coerceIn(0.0, 1.0)

            penaltySum += (
                0.45 * velocityPenalty +
                    0.35 * accelerationPenalty +
                    0.20 * jerkPenalty
                )
        }

        return if (trajectoryCount == 0) {
            0.0
        } else {
            (penaltySum / trajectoryCount).coerceIn(0.0, 1.0)
        }
    }

    private fun buildPointDiagnostics(
        before: List<ActorPoint>,
        after: List<ActorPoint>,
        cameras: List<CameraState>,
        constraints: Map<Long, List<SurfaceConstraint>>,
    ): List<PointDiagnostic> {
        val beforeMap = before.associateBy { key(it.actorId, it.pointId) }
        return after.map { point ->
            val beforePoint = beforeMap[key(point.actorId, point.pointId)]
            val beforeResidual = pointResidual(
                point = beforePoint ?: point,
                cameras = cameras,
                constraints = constraints[key(point.actorId, point.pointId)].orEmpty(),
            )
            val afterResidual = pointResidual(
                point = point,
                cameras = cameras,
                constraints = constraints[key(point.actorId, point.pointId)].orEmpty(),
            )
            PointDiagnostic(
                actorId = point.actorId,
                pointId = point.pointId,
                cameraObservationCount = point.cameraObservationById.size
                    .coerceAtLeast(point.cameraObservations.size),
                temporalObservationCount = point.temporalObservations.size,
                residualBeforeMeters = beforeResidual,
                residualAfterMeters = afterResidual,
                displacementMeters = point.positionMeters.distanceTo(pointOriginal(beforePoint, point)),
                precisionWeight = (
                    point.confidence *
                        (1.0 / (1.0 + afterResidual))
                ).coerceIn(0.0, 1.0),
            )
        }
    }

    private fun pointOriginal(before: ActorPoint?, fallback: ActorPoint): Vec3 =
        before?.positionMeters ?: fallback.positionMeters

    private fun pointResidual(
        point: ActorPoint,
        cameras: List<CameraState>,
        constraints: List<SurfaceConstraint>,
    ): Double {
        val observations = ArrayList<Pair<Vec3, Double>>()
        if (point.cameraObservationById.isNotEmpty()) {
            point.cameraObservationById.forEach { (cameraId, observation) ->
                val camera = cameras.firstOrNull { it.cameraId == cameraId } ?: return@forEach
                observations += (observation - camera.correctionMeters) to camera.confidence
            }
        } else {
            point.cameraObservations.forEach { observations += it to 1.0 }
        }
        point.temporalObservations.forEach {
            observations += it.positionMeters to (
                it.confidence * config.temporalSmoothnessWeight.coerceAtLeast(0.05)
            )
        }
        constraints.forEach {
            observations += it.targetPositionMeters to (
                it.weight * it.confidence * config.surfaceConstraintWeight
            )
        }
        if (observations.isEmpty()) return Double.POSITIVE_INFINITY

        var weighted = 0.0
        var total = 0.0
        observations.forEach { (position, weight) ->
            val error = position.distanceTo(point.positionMeters)
            if (!error.isFinite()) return@forEach
            val effective = weight * robustWeight(error)
            weighted += effective * error
            total += effective
        }
        return if (total <= 1e-9) Double.POSITIVE_INFINITY else weighted / total
    }

    private fun buildTrajectoryDiagnostics(
        points: List<ActorPoint>,
    ): List<TrajectoryDiagnostic> {
        return points
            .groupBy { it.actorId to it.pointId }
            .map { (identity, samples) ->
                val temporal = samples
                    .flatMap { point ->
                        point.temporalObservations.map { sample ->
                            sample.timestampNanos to sample.positionMeters
                        } + listOf(
                            temporalTimestamp(point) to point.positionMeters,
                        )
                    }
                    .distinctBy { it.first }
                    .sortedBy { it.first }
                    .take(config.maximumTemporalSamplesPerPoint + 1)

                if (temporal.size < 2) {
                    return@map TrajectoryDiagnostic(
                        actorId = identity.first,
                        pointId = identity.second,
                        sampleCount = temporal.size,
                        maximumVelocityMetersPerSecond = 0.0,
                        maximumAccelerationMetersPerSecond2 = 0.0,
                        maximumJerkMetersPerSecond3 = 0.0,
                        kinematicEnvelopeScore = 1.0,
                    )
                }

                val velocities = ArrayList<Double>()
                val accelerations = ArrayList<Double>()
                val jerks = ArrayList<Double>()

                for (index in 1 until temporal.size) {
                    val dt = secondsBetween(
                        temporal[index - 1].first,
                        temporal[index].first,
                    )
                    if (dt <= 1e-9) continue
                    velocities += temporal[index].second.distanceTo(
                        temporal[index - 1].second,
                    ) / dt
                }

                for (index in 2 until temporal.size) {
                    val dt1 = secondsBetween(
                        temporal[index - 2].first,
                        temporal[index - 1].first,
                    )
                    val dt2 = secondsBetween(
                        temporal[index - 1].first,
                        temporal[index].first,
                    )
                    if (dt1 <= 1e-9 || dt2 <= 1e-9) continue
                    val v1 = (
                        temporal[index - 1].second -
                            temporal[index - 2].second
                        ) / dt1
                    val v2 = (
                        temporal[index].second -
                            temporal[index - 1].second
                        ) / dt2
                    accelerations += (v2 - v1).norm() / ((dt1 + dt2) * 0.5)
                }

                for (index in 3 until temporal.size) {
                    val t0 = temporal[index - 3]
                    val t1 = temporal[index - 2]
                    val t2 = temporal[index - 1]
                    val t3 = temporal[index]
                    val dt1 = secondsBetween(t0.first, t1.first)
                    val dt2 = secondsBetween(t1.first, t2.first)
                    val dt3 = secondsBetween(t2.first, t3.first)
                    if (min(dt1, min(dt2, dt3)) <= 1e-9) continue

                    val a1 = (
                        (t2.second - t1.second) / dt2 -
                            (t1.second - t0.second) / dt1
                        ) / ((dt1 + dt2) * 0.5)
                    val a2 = (
                        (t3.second - t2.second) / dt3 -
                            (t2.second - t1.second) / dt2
                        ) / ((dt2 + dt3) * 0.5)
                    jerks += (a2 - a1).norm() / ((dt2 + dt3) * 0.5)
                }

                val maxVelocity = velocities.maxOrNull() ?: 0.0
                val maxAcceleration = accelerations.maxOrNull() ?: 0.0
                val maxJerk = jerks.maxOrNull() ?: 0.0
                val velocityScore = scoreEnvelope(
                    maxVelocity,
                    config.maximumVelocityMetersPerSecond,
                )
                val accelerationScore = scoreEnvelope(
                    maxAcceleration,
                    config.maximumAccelerationMetersPerSecond2,
                )
                val jerkScore = scoreEnvelope(
                    maxJerk,
                    config.maximumJerkMetersPerSecond3,
                )

                TrajectoryDiagnostic(
                    actorId = identity.first,
                    pointId = identity.second,
                    sampleCount = temporal.size,
                    maximumVelocityMetersPerSecond = maxVelocity,
                    maximumAccelerationMetersPerSecond2 = maxAcceleration,
                    maximumJerkMetersPerSecond3 = maxJerk,
                    kinematicEnvelopeScore = (
                        0.45 * velocityScore +
                            0.35 * accelerationScore +
                            0.20 * jerkScore
                        ).coerceIn(0.0, 1.0),
                )
            }
            .sortedWith(compareBy<TrajectoryDiagnostic> { it.actorId }.thenBy { it.pointId })
    }

    private fun scoreEnvelope(value: Double, maximum: Double): Double {
        if (!value.isFinite() || value < 0.0) return 0.0
        return (1.0 - (value / maximum)).coerceIn(0.0, 1.0)
    }

    private fun velocityEvidence(sample: TemporalObservation): Double =
        sample.velocityMetersPerSecond?.let {
            val normalized = it.norm() / config.maximumVelocityMetersPerSecond
            1.0 - normalized.coerceIn(0.0, 1.0)
        } ?: 0.5

    private fun accelerationEvidence(sample: TemporalObservation): Double =
        sample.accelerationMetersPerSecond2?.let {
            val normalized = it.norm() / config.maximumAccelerationMetersPerSecond2
            1.0 - normalized.coerceIn(0.0, 1.0)
        } ?: 0.5

    private fun temporalTimestamp(point: MutablePoint): Long =
        point.temporalObservations.maxOfOrNull { it.timestampNanos } ?: 0L

    private fun temporalTimestamp(point: ActorPoint): Long =
        point.temporalObservations.maxOfOrNull { it.timestampNanos } ?: 0L

    private fun robustWeight(error: Double): Double {
        if (!error.isFinite() || error < 0.0) return 0.0
        val huber = if (error <= config.robustScaleMeters) {
            1.0
        } else {
            config.robustScaleMeters / error
        }
        val cauchy =
            1.0 / (
                1.0 +
                    (error / config.cauchyScaleMeters) *
                    (error / config.cauchyScaleMeters)
                )
        return (
            0.58 * huber +
                0.42 * cauchy
            ).coerceIn(0.0, 1.0)
    }

    private fun clampNorm(value: Vec3, maximum: Double): Vec3 {
        val norm = value.norm()
        if (!norm.isFinite() || norm <= maximum || norm < 1e-12) return value
        return value * (maximum / norm)
    }

    private fun secondsBetween(from: Long, to: Long): Double =
        max(0.0, (to - from).toDouble() / 1e9)

    private fun key(actorId: Int, pointId: Int): Long =
        (actorId.toLong() shl 32) xor (pointId.toLong() and 0xffffffffL)

    private fun signature(
        points: List<MutablePoint>,
        cameras: List<CameraState>,
    ): Long {
        var hash = 17L
        points.forEach { point ->
            hash = hash * 31 + point.actorId
            hash = hash * 31 + point.pointId
            hash = hash * 31 + point.position.hashCode()
        }
        cameras.forEach { camera ->
            hash = hash * 31 + camera.cameraId.hashCode()
            hash = hash * 31 + camera.correctionMeters.hashCode()
        }
        return hash
    }

    private fun finite(value: Vec3): Boolean =
        value.x.isFinite() && value.y.isFinite() && value.z.isFinite()
}

private fun Vec3.distanceTo(other: Vec3): Double =
    sqrt(
        (x - other.x) * (x - other.x) +
            (y - other.y) * (y - other.y) +
            (z - other.z) * (z - other.z),
    )

private fun List<Double>.averageOrNull(): Double? =
    if (isEmpty()) null else average()

package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Global camera-load allocator for multi-actor capture.
 *
 * Actor-specific selection can otherwise converge on the same camera subset. This allocator
 * preserves per-actor view diversity while distributing expensive authority work across the
 * federation. Cameras remain reusable for witness evidence; only the expensive authority load is
 * bounded.
 */
class UltraMultiActorCameraLoadBalancer(
    initialConfig: Config = Config(),
) {
    @Volatile
    private var config: Config = initialConfig
    data class Config(
        val maximumActors: Int = 64,
        val authorityCamerasPerActor: Int = 12,
        val minimumAuthorityCamerasPerActor: Int = 4,
        val maximumAuthorityActorLoad: Int = 24,
        val maximumWitnessCamerasPerActor: Int = 24,
        val diversityWeight: Double = 0.32,
        val visibilityWeight: Double = 0.30,
        val authorityWeight: Double = 0.26,
        val loadBalanceWeight: Double = 0.12,
        val hysteresisWeight: Double = 0.06,
        val metricGeometryWeight: Double = 0.12,
        val metricNoveltyWeight: Double = 0.10,
        val adaptiveAuthorityDepthEnabled: Boolean = true,
        val maximumAdaptiveAuthorityBonus: Int = 4,
        val adaptiveDifficultyThreshold: Double = 0.45,
        val adaptivePriorityThreshold: Double = 0.85,
        val maximumAdaptiveAuthorityCamerasPerActor: Int = 16,
    ) {
        init {
            require(maximumActors in 1..64)
            require(authorityCamerasPerActor in 1..32)
            require(minimumAuthorityCamerasPerActor in 2..authorityCamerasPerActor)
            require(maximumAuthorityActorLoad >= 1)
            require(maximumWitnessCamerasPerActor >= 0)
            listOf(
                diversityWeight,
                visibilityWeight,
                authorityWeight,
                loadBalanceWeight,
                hysteresisWeight,
                metricGeometryWeight,
                metricNoveltyWeight,
            ).forEach { require(it >= 0.0) }
            require(maximumAdaptiveAuthorityBonus in 0..8)
            require(adaptiveDifficultyThreshold in 0.0..1.0)
            require(adaptivePriorityThreshold in 0.0..1.0)
            require(
                maximumAdaptiveAuthorityCamerasPerActor in
                    authorityCamerasPerActor..32,
            )
        }
    }

    data class Actor(
        val actorId: Int,
        val priority: Double,
        val visibilityByCamera: Map<String, Double>,
        val worldPositionMeters: Vec3? = null,
        val occlusionRisk: Double = 0.0,
    ) {
        init {
            require(actorId >= 0)
            require(priority in 0.0..1.0)
            require(visibilityByCamera.keys.all { it.isNotBlank() })
            require(
                visibilityByCamera.values.all {
                    it.isFinite() && it in 0.0..1.0
                },
            )
            require(occlusionRisk in 0.0..1.0)
            worldPositionMeters?.let {
                require(
                    it.x.isFinite() &&
                        it.y.isFinite() &&
                        it.z.isFinite(),
                )
            }
        }
    }

    data class Assignment(
        val actorId: Int,
        val authorityCameraIds: List<String>,
        val witnessCameraIds: List<String>,
        val viewDiversity: Double,
        val authorityCoverage: Double,
        val loadPenalty: Double,
        val score: Double,
    )

    data class Result(
        val assignments: List<Assignment>,
        val cameraAuthorityLoad: Map<String, Int>,
        val requestedAuthorityDepthByActor: Map<Int, Int>,
        val meanActorCoverage: Double,
        val minimumActorCoverage: Double,
        val meanViewDiversity: Double,
        val overloadedCameraCount: Int,
        val starvationRisk: Double,
        val warnings: List<String>,
    )

    private val previousAssignments =
        LinkedHashMap<Int, Set<String>>()

    private val placementInformationEngine =
        CameraPlacementInformationEngine()

    @Synchronized
    fun configure(newConfig: Config) {
        config = newConfig
    }

    @Synchronized
    fun allocate(
        actors: Collection<Actor>,
        cameras: Collection<FrontierCameraAuthorityFederationEngine.CameraEvidence>,
    ): Result {
        val cameraList =
            cameras
                .distinctBy { it.cameraId }
                .filter {
                    it.eligible(
                        FrontierCameraAuthorityFederationEngine.Config(),
                    )
                }
                .take(128)

        val boundedActors =
            actors
                .distinctBy { it.actorId }
                .take(config.maximumActors)
                .sortedWith(
                    compareByDescending<Actor> { it.priority }
                        .thenBy { it.actorId },
                )

        if (boundedActors.isEmpty() || cameraList.isEmpty()) {
            return Result(
                assignments = emptyList(),
                cameraAuthorityLoad = emptyMap(),
                requestedAuthorityDepthByActor = emptyMap(),
                meanActorCoverage = 0.0,
                minimumActorCoverage = 0.0,
                meanViewDiversity = 0.0,
                overloadedCameraCount = 0,
                starvationRisk =
                    if (boundedActors.isEmpty()) 0.0 else 1.0,
                warnings =
                    listOf(
                        "actor-camera-allocation-empty",
                    ),
            )
        }

        val byId =
            cameraList.associateBy { it.cameraId }

        val authorityLoad =
            LinkedHashMap<String, Int>()
        val selectedByActor =
            LinkedHashMap<Int, MutableList<String>>()
        val requestedAuthorityDepthByActor =
            LinkedHashMap<Int, Int>()

        // Seed the assignment from high-priority actors first. The remaining actors can still use
        // already-loaded cameras, but gain a score boost for under-utilized alternatives.
        for (actor in boundedActors) {
            val requestedAuthorityDepth =
                authorityDepthForActor(actor)
            requestedAuthorityDepthByActor[actor.actorId] =
                requestedAuthorityDepth

            val selected =
                ArrayList<String>(
                    requestedAuthorityDepth,
                )

            repeat(requestedAuthorityDepth) {
                var bestId: String? = null
                var bestScore = Double.NEGATIVE_INFINITY

                for (camera in cameraList) {
                    if (camera.cameraId in selected) continue

                    val baseVisibility =
                        actor.visibilityByCamera[camera.cameraId]
                            ?: camera.actorVisibility

                    val metricScore =
                        metricScore(
                            actor = actor,
                            camera = camera,
                            selectedCameraIds = selected,
                            cameraById = byId,
                        )

                    val visibility =
                        (
                            0.55 * baseVisibility +
                                0.45 * metricScore.actorVisibilityScore
                        ).coerceIn(0.0, 1.0)

                    val authority =
                        camera.score(actor.actorId)
                            .coerceIn(0.0, 1.0)

                    val currentLoad =
                        authorityLoad[camera.cameraId] ?: 0
                    val loadRatio =
                        currentLoad.toDouble() /
                            max(
                                1,
                                config.maximumAuthorityActorLoad,
                            ).toDouble()

                    val loadPenalty =
                        loadRatio.coerceIn(0.0, 1.0)

                    val diversity =
                        if (selected.isEmpty()) {
                            1.0
                        } else {
                            val selectedCameras =
                                selected.mapNotNull(byId::get)
                            val minimumSectorDifference =
                                selectedCameras
                                    .minOfOrNull { prior ->
                                        if (prior.sector == camera.sector) 0.0 else 1.0
                                    }
                                    ?: 1.0

                            val visibilityDifference =
                                selectedCameras
                                    .map { prior ->
                                        abs(
                                            (actor.visibilityByCamera[prior.cameraId]
                                                ?: prior.actorVisibility) -
                                                visibility,
                                        )
                                    }
                                    .averageOrZero()

                            (
                                0.65 * minimumSectorDifference +
                                    0.35 * visibilityDifference
                                )
                                .coerceIn(0.0, 1.0)
                        }

                    val hysteresis =
                        if (
                            camera.cameraId in
                                previousAssignments[actor.actorId].orEmpty()
                        ) {
                            config.hysteresisWeight
                        } else {
                            0.0
                        }

                    val priority =
                        actor.priority.coerceIn(0.0, 1.0)

                    val score =
                        (
                            config.visibilityWeight *
                                visibility +
                                config.authorityWeight *
                                    authority +
                                config.diversityWeight *
                                    diversity +
                                config.loadBalanceWeight *
                                    (1.0 - loadPenalty) +
                                config.metricGeometryWeight *
                                    metricScore.combinedGeometryScore +
                                config.metricNoveltyWeight *
                                    metricScore.noveltyScore +
                                hysteresis +
                                0.04 * priority
                            )

                    if (
                        score > bestScore ||
                        (
                            score == bestScore &&
                                (
                                    bestId == null ||
                                        camera.cameraId <
                                            bestId!!
                                    )
                            )
                    ) {
                        bestScore = score
                        bestId = camera.cameraId
                    }
                }

                val chosen = bestId ?: return@repeat

                val currentLoad =
                    authorityLoad[chosen] ?: 0

                if (
                    currentLoad >=
                        config.maximumAuthorityActorLoad
                ) {
                    // A saturated camera can still be selected only when there are not enough
                    // remaining cameras to satisfy the hard per-actor minimum.
                    if (
                        selected.size >=
                            config.minimumAuthorityCamerasPerActor
                    ) {
                        return@repeat
                    }
                }

                selected += chosen
                authorityLoad[chosen] = currentLoad + 1
            }

            selectedByActor[actor.actorId] = selected
        }

        // Repair starvation deterministically.
        for (actor in boundedActors) {
            val selected =
                selectedByActor[actor.actorId]
                    ?: continue

            if (
                selected.size >=
                    config.minimumAuthorityCamerasPerActor
            ) {
                continue
            }

            cameraList
                .asSequence()
                .filter { it.cameraId !in selected }
                .sortedWith(
                    compareByDescending<FrontierCameraAuthorityFederationEngine.CameraEvidence> {
                        actor.visibilityByCamera[it.cameraId]
                            ?: it.actorVisibility
                    }.thenBy { it.cameraId },
                )
                .take(
                    config.minimumAuthorityCamerasPerActor -
                        selected.size,
                )
                .forEach { camera ->
                    selected += camera.cameraId
                    authorityLoad[camera.cameraId] =
                        (authorityLoad[camera.cameraId] ?: 0) + 1
                }
        }

        val assignments =
            boundedActors.map { actor ->
                val selected =
                    selectedByActor[actor.actorId].orEmpty()

                val witnesses =
                    cameraList
                        .asSequence()
                        .filter {
                            it.cameraId !in selected
                        }
                        .sortedWith(
                            compareByDescending<FrontierCameraAuthorityFederationEngine.CameraEvidence> {
                                actor.visibilityByCamera[it.cameraId]
                                    ?: it.actorVisibility
                            }.thenBy { it.cameraId },
                        )
                        .take(config.maximumWitnessCamerasPerActor)
                        .map { it.cameraId }
                        .toList()

                val coverage =
                    selected
                        .map {
                            actor.visibilityByCamera[it]
                                ?: byId[it]?.actorVisibility
                                ?: 0.0
                        }
                        .averageOrZero()

                val diversity =
                    selected
                        .mapNotNull(byId::get)
                        .let(::diversityScore)

                val loadPenalty =
                    if (selected.isEmpty()) {
                        1.0
                    } else {
                        selected
                            .map {
                                (
                                    authorityLoad[it] ?: 0
                                ).toDouble() /
                                    config.maximumAuthorityActorLoad
                            }
                            .average()
                            .coerceIn(0.0, 1.0)
                    }

                val score =
                    (
                        config.visibilityWeight * coverage +
                            config.diversityWeight * diversity +
                            config.authorityWeight *
                                selected
                                    .mapNotNull(byId::get)
                                    .map {
                                        it.score(actor.actorId)
                                    }
                                    .averageOrZero() +
                            config.loadBalanceWeight *
                                (1.0 - loadPenalty)
                    )
                        .coerceIn(0.0, 1.0)

                Assignment(
                    actorId = actor.actorId,
                    authorityCameraIds = selected,
                    witnessCameraIds = witnesses,
                    viewDiversity = diversity,
                    authorityCoverage = coverage,
                    loadPenalty = loadPenalty,
                    score = score,
                )
            }

        assignments.forEach {
            previousAssignments[it.actorId] =
                it.authorityCameraIds.toSet()
        }
        while (previousAssignments.size > 64) {
            previousAssignments.entries
                .firstOrNull()
                ?.let { previousAssignments.remove(it.key) }
        }

        val coverages =
            assignments.map {
                it.authorityCoverage
            }
        val diversities =
            assignments.map {
                it.viewDiversity
            }

        val overloaded =
            authorityLoad.values.count {
                it > config.maximumAuthorityActorLoad
            }

        val starvation =
            assignments
                .map {
                    when {
                        it.authorityCameraIds.size <
                            config.minimumAuthorityCamerasPerActor ->
                            1.0
                        it.authorityCoverage < 0.25 ->
                            0.85
                        it.viewDiversity < 0.20 ->
                            0.75
                        else ->
                            0.0
                    }
                }
                .averageOrZero()

        val warnings =
            buildList {
                if (overloaded > 0) {
                    add("camera-authority-load-overflow")
                }
                if (assignments.any {
                    it.authorityCameraIds.size <
                        config.minimumAuthorityCamerasPerActor
                }) {
                    add("actor-camera-view-starvation")
                }
                if (assignments.any { it.viewDiversity < 0.20 }) {
                    add("actor-camera-diversity-low")
                }
                if (starvation >= 0.35) {
                    add("global-actor-camera-starvation-risk-high")
                }
            }

        return Result(
            assignments = assignments,
            cameraAuthorityLoad =
                authorityLoad.toMap(),
            requestedAuthorityDepthByActor =
                requestedAuthorityDepthByActor.toMap(),
            meanActorCoverage =
                coverages.averageOrZero(),
            minimumActorCoverage =
                coverages.minOrNull() ?: 0.0,
            meanViewDiversity =
                diversities.averageOrZero(),
            overloadedCameraCount = overloaded,
            starvationRisk = starvation,
            warnings = warnings.distinct(),
        )
    }

    fun reset() {
        synchronized(this) {
            previousAssignments.clear()
        }
    }

    private fun authorityDepthForActor(
        actor: Actor,
    ): Int {
        val base =
            config.authorityCamerasPerActor.coerceAtLeast(
                config.minimumAuthorityCamerasPerActor,
            )

        if (!config.adaptiveAuthorityDepthEnabled) {
            return base
        }

        val meanVisibility =
            actor.visibilityByCamera.values
                .filter { it.isFinite() }
                .averageOrZero()
                .coerceIn(0.0, 1.0)

        val difficulty =
            max(
                actor.occlusionRisk,
                1.0 - meanVisibility,
            )

        var bonus = 0
        if (difficulty >= config.adaptiveDifficultyThreshold) {
            bonus += 2
        }
        if (difficulty >= 0.80) {
            bonus += 1
        }
        if (actor.priority >= config.adaptivePriorityThreshold) {
            bonus += 1
        }

        return (
            base +
                bonus.coerceAtMost(
                    config.maximumAdaptiveAuthorityBonus,
                )
            ).coerceAtMost(
                config.maximumAdaptiveAuthorityCamerasPerActor,
            )
    }

    private fun diversityScore(
        cameras: List<FrontierCameraAuthorityFederationEngine.CameraEvidence>,
    ): Double {
        if (cameras.size < 2) return 0.0

        val sectorScore =
            (
                cameras
                    .map { it.sector }
                    .distinct()
                    .size.toDouble() /
                    min(
                        8,
                        cameras.size,
                    ).toDouble()
                )
                .coerceIn(0.0, 1.0)

        val pairwise =
            cameras
                .flatMapIndexed { index, first ->
                    cameras
                        .drop(index + 1)
                        .map { second ->
                            if (first.sector == second.sector) {
                                0.0
                            } else {
                                1.0
                            }
                        }
                }

        val angular =
            pairwise.averageOrZero()

        return (
            0.60 * sectorScore +
                0.40 * angular
            )
            .coerceIn(0.0, 1.0)
    }

    private fun metricScore(
        actor: Actor,
        camera: FrontierCameraAuthorityFederationEngine.CameraEvidence,
        selectedCameraIds: List<String>,
        cameraById: Map<String, FrontierCameraAuthorityFederationEngine.CameraEvidence>,
    ): CameraPlacementInformationEngine.Score {
        val baseVisibility =
            actor.visibilityByCamera[camera.cameraId]
                ?: camera.actorVisibility
        val selectedAnchors =
            selectedCameraIds.mapNotNull { id ->
                cameraById[id]?.metricAnchor()
            }
        val actorPosition =
            actor.worldPositionMeters
        val cameraAnchor =
            camera.metricAnchor()

        if (actorPosition == null || cameraAnchor == null) {
            return CameraPlacementInformationEngine.Score(
                cameraId = camera.cameraId,
                actorId = actor.actorId,
                geometryAvailable = false,
                distanceMeters = Double.POSITIVE_INFINITY,
                distanceScore = 0.42,
                facingScore = 0.42,
                baseVisibilityScore = baseVisibility,
                actorVisibilityScore = baseVisibility,
                occlusionPenalty = actor.occlusionRisk,
                authorityScore = camera.score(actor.actorId),
                calibrationScore = camera.calibration,
                synchronizationScore = camera.synchronization,
                resourceScore =
                    (
                        camera.networkHeadroom +
                            camera.thermalHeadroom
                    ) * 0.5,
                noveltyScore =
                    if (selectedAnchors.isEmpty()) 1.0 else 0.0,
                noveltyDegrees = 0.0,
                baselineScore = 0.0,
                combinedGeometryScore = 0.42,
                finalScore = camera.score(actor.actorId),
                diagnostics =
                    listOf(
                        "METRIC_CAMERA_OR_ACTOR_ANCHOR_UNAVAILABLE",
                    ),
            )
        }

        return placementInformationEngine.evaluate(
            actor =
                CameraPlacementInformationEngine.ActorAnchor(
                    actorId = actor.actorId,
                    positionMeters = actorPosition,
                    priority = actor.priority,
                    baseVisibility = baseVisibility,
                    occlusionRisk = actor.occlusionRisk,
                ),
            camera = cameraAnchor,
            selectedCameras = selectedAnchors,
        )
    }

    private fun FrontierCameraAuthorityFederationEngine.CameraEvidence.metricAnchor():
        CameraPlacementInformationEngine.CameraAnchor? {
        val position = positionMeters ?: return null
        val direction = forward ?: return null
        return CameraPlacementInformationEngine.CameraAnchor(
            cameraId = cameraId,
            positionMeters = position,
            forward = direction,
            authorityScore = score(),
            sector = sector,
            calibrationConfidence = calibration,
            synchronizationConfidence = synchronization,
            networkHeadroom = networkHeadroom,
            thermalHeadroom = thermalHeadroom,
            motionBlurRisk = motionBlurRisk,
        )
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}

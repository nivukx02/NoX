package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.Vec3
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * High-end multi-camera calibration authority engine.
 *
 * The engine treats a camera rig as a probabilistic graph rather than a flat list of cameras.
 * Every edge carries geometric, temporal and uncertainty evidence. Reliable paths are preferred,
 * path covariance is accumulated, and loop closure disagreement is used as a global consistency
 * signal. The output is an authority plan and an active-calibration recommendation set.
 *
 * This layer does not rewrite calibration parameters. It determines whether the current calibration
 * artifact is strong enough for metric reconstruction and which observations would reduce the most
 * uncertainty on the next calibration pass.
 */
class UltraMultiCameraCalibrationAuthorityEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumCameras: Int = 128,
        val maximumAuthoritativeCameras: Int = 32,
        val minimumEdgeConfidence: Double = 0.35,
        val minimumLoopConfidence: Double = 0.45,
        val maximumTranslationSigmaMeters: Double = 0.010,
        val maximumRotationSigmaRadians: Double = 0.0025,
        val maximumTimingResidualMillis: Double = 4.0,
        val maximumReprojectionErrorPixels: Double = 2.5,
        val minimumParallaxDegrees: Double = 6.0,
        val preferredParallaxDegrees: Double = 14.0,
        val maximumCycleResidualMeters: Double = 0.020,
    ) {
        init {
            require(maximumCameras in 2..128)
            require(maximumAuthoritativeCameras in 2..32)
            require(minimumEdgeConfidence in 0.0..1.0)
            require(minimumLoopConfidence in 0.0..1.0)
            require(maximumTranslationSigmaMeters > 0.0)
            require(maximumRotationSigmaRadians > 0.0)
            require(maximumTimingResidualMillis > 0.0)
            require(maximumReprojectionErrorPixels > 0.0)
            require(minimumParallaxDegrees > 0.0)
            require(preferredParallaxDegrees >= minimumParallaxDegrees)
            require(maximumCycleResidualMeters > 0.0)
        }
    }

    data class Camera(
        val id: String,
        val sector: Int,
        val positionMeters: Vec3,
        val forward: Vec3,
        val reprojectionErrorPixels: Double,
        val synchronizationConfidence: Double,
        val calibrationConfidence: Double,
        val coverage: Double,
        val uncertainty: CalibrationUncertaintyProfile =
            CalibrationUncertaintyProfile(),
        val timingResidualMillis: Double = 0.0,
        val available: Boolean = true,
    ) {
        init {
            require(id.isNotBlank())
            require(sector >= 0)
            require(positionMeters.finite())
            require(forward.finite() && forward.norm() > 1e-9)
            require(reprojectionErrorPixels.isFinite() && reprojectionErrorPixels >= 0.0)
            require(synchronizationConfidence in 0.0..1.0)
            require(calibrationConfidence in 0.0..1.0)
            require(coverage in 0.0..1.0)
            require(timingResidualMillis.isFinite() && timingResidualMillis >= 0.0)
        }
    }

    data class Edge(
        val from: String,
        val to: String,
        val baselineMeters: Double,
        val parallaxDegrees: Double,
        val reprojectionErrorPixels: Double,
        val timingResidualMillis: Double,
        val translationSigmaMeters: Double,
        val rotationSigmaRadians: Double,
        val confidence: Double,
        val cycleResidualMeters: Double = 0.0,
    )

    data class CameraAuthority(
        val cameraId: String,
        val score: Double,
        val pathConfidence: Double,
        val uncertaintyPenalty: Double,
        val loopConsistency: Double,
        val sectorDiversity: Double,
        val authoritative: Boolean,
        val reasons: List<String>,
    )

    data class CalibrationTarget(
        val cameraId: String,
        val priority: Double,
        val reason: String,
        val estimatedTranslationSigmaMeters: Double,
        val estimatedRotationSigmaRadians: Double,
    )

    data class Result(
        val authoritativeCameras: List<CameraAuthority>,
        val witnessCameras: List<CameraAuthority>,
        val rejectedCameras: List<CameraAuthority>,
        val edges: List<Edge>,
        val globalConsistency: Double,
        val graphConnectivity: Double,
        val loopConsistency: Double,
        val uncertaintyQuality: Double,
        val metricReady: Boolean,
        val activeCalibrationTargets: List<CalibrationTarget>,
        val warnings: List<String>,
        val diagnostics: List<String>,
    )

    fun evaluate(
        cameras: Collection<Camera>,
        relativeEdges: Collection<Edge> = emptyList(),
    ): Result {
        val unique =
            cameras
                .distinctBy { it.id }
                .take(config.maximumCameras)

        if (unique.size < 2) {
            return emptyResult(
                warning = "calibration-graph-insufficient-cameras",
            )
        }

        val cameraById = unique.associateBy { it.id }

        val derivedEdges =
            if (relativeEdges.isEmpty()) {
                deriveEdges(unique)
            } else {
                relativeEdges
                    .filter {
                        it.from in cameraById &&
                            it.to in cameraById
                    }
                    .map(::sanitizeEdge)
            }

        val adjacency =
            derivedEdges
                .filter { it.confidence >= config.minimumEdgeConfidence }
                .flatMap { edge ->
                    listOf(
                        edge.from to edge,
                        edge.to to edge,
                    )
                }
                .groupBy(
                    keySelector = { it.first },
                    valueTransform = { it.second },
                )

        val reachable =
            unique.associate { camera ->
                camera.id to reachableConfidence(
                    camera.id,
                    unique,
                    adjacency,
                )
            }

        val loopConsistency =
            if (derivedEdges.isEmpty()) {
                0.0
            } else {
                (
                    1.0 -
                        derivedEdges
                            .map { it.cycleResidualMeters }
                            .averageOrZero() /
                            config.maximumCycleResidualMeters
                    ).coerceIn(0.0, 1.0)
            }

        val uncertaintyQuality =
            unique
                .map {
                    val profileEvidenceFactor =
                        if (it.uncertainty.hasEvidence) {
                            1.0
                        } else {
                            0.82
                        }

                    (
                        (
                            0.62 *
                                translationQuality(
                                    it.uncertainty.translationSigmaMeters,
                                ) +
                                0.22 *
                                    rotationQuality(
                                        it.uncertainty.rotationSigmaRadians,
                                    ) +
                                0.16 *
                                    timingQuality(
                                        it.timingResidualMillis,
                                    )
                            ) *
                            profileEvidenceFactor
                    ).coerceIn(0.0, 1.0)
                }
                .averageOrZero()

        val ranked =
            unique
                .map { camera ->
                    val pathConfidence =
                        reachable[camera.id] ?: 0.0
                    val uncertaintyPenalty =
                        1.0 -
                            (
                                0.68 *
                                    (
                                        1.0 -
                                            translationQuality(
                                                camera.uncertainty.translationSigmaMeters,
                                            )
                                    ) +
                                    0.32 *
                                        (
                                            1.0 -
                                                rotationQuality(
                                                    camera.uncertainty.rotationSigmaRadians,
                                                )
                                        )
                            ).coerceIn(0.0, 1.0)
                    val timing =
                        timingQuality(camera.timingResidualMillis)
                    val loop =
                        loopCameraConsistency(
                            camera.id,
                            derivedEdges,
                        )
                    val profileEvidenceFactor =
                        if (camera.uncertainty.hasEvidence) {
                            1.0
                        } else {
                            0.88
                        }

                    val cameraBase =
                        (
                            0.22 * camera.calibrationConfidence +
                                0.16 * camera.synchronizationConfidence +
                                0.13 * (
                                    1.0 -
                                        camera.reprojectionErrorPixels /
                                            config.maximumReprojectionErrorPixels
                                ).coerceIn(0.0, 1.0) +
                                0.12 * camera.coverage +
                                0.12 * timing +
                                0.11 * pathConfidence +
                                0.08 * uncertaintyPenalty +
                                0.06 * loop +
                                0.08 * (
                                    camera.positionMeters.norm() /
                                        (camera.positionMeters.norm() + 1.0)
                                )
                            ).coerceIn(0.0, 1.0) *
                            profileEvidenceFactor

                    CameraAuthority(
                        cameraId = camera.id,
                        score = cameraBase,
                        pathConfidence = pathConfidence,
                        uncertaintyPenalty = uncertaintyPenalty,
                        loopConsistency = loop,
                        sectorDiversity = 0.0,
                        authoritative = false,
                        reasons =
                            reasonsFor(
                                camera = camera,
                                pathConfidence = pathConfidence,
                                loop = loop,
                                timing = timing,
                            ),
                    )
                }
                .sortedWith(
                    compareByDescending<CameraAuthority> { it.score }
                        .thenBy { it.cameraId },
                )

        val selected = ArrayList<CameraAuthority>(
            config.maximumAuthoritativeCameras,
        )

        val remaining = ranked.toMutableList()

        // First pass guarantees broad sector coverage before pure score ranking.
        unique
            .map { it.sector }
            .distinct()
            .sorted()
            .forEach { sector ->
                if (selected.size >= config.maximumAuthoritativeCameras) {
                    return@forEach
                }
                val candidate =
                    remaining
                        .map { authority ->
                            authority to cameraById[authority.cameraId]
                        }
                        .firstOrNull { (_, camera) ->
                            camera?.sector == sector
                        }
                        ?.first
                candidate?.let {
                    selected +=
                        it.copy(
                            authoritative = true,
                            sectorDiversity = 1.0,
                        )
                    remaining.remove(it)
                }
            }

        while (
            selected.size < config.maximumAuthoritativeCameras &&
            remaining.isNotEmpty()
        ) {
            val best =
                remaining
                    .maxWithOrNull(
                        compareBy<CameraAuthority> { authority ->
                            diversityBonus(
                                authority.cameraId,
                                selected.map { it.cameraId },
                                cameraById,
                            )
                        }
                            .thenBy { it.score }
                            .thenByDescending { it.pathConfidence },
                    )
                    ?: break

            val updated =
                best.copy(
                    authoritative = true,
                    sectorDiversity =
                        diversityBonus(
                            best.cameraId,
                            selected.map { it.cameraId },
                            cameraById,
                        ),
                )
            selected += updated
            remaining.remove(best)
        }

        val selectedIds = selected.map { it.cameraId }.toSet()
        val witnesses =
            remaining
                .take(128 - config.maximumAuthoritativeCameras)
                .map {
                    it.copy(authoritative = false)
                }

        val rejected =
            remaining
                .drop(witnesses.size)
                .map {
                    it.copy(authoritative = false)
                }

        val graphConnectivity =
            selected
                .map { reachable[it.cameraId] ?: 0.0 }
                .averageOrZero()

        val diversity =
            selected
                .map { it.sectorDiversity }
                .averageOrZero()

        val globalConsistency =
            (
                0.24 * graphConnectivity +
                    0.26 * loopConsistency +
                    0.26 * uncertaintyQuality +
                    0.14 * diversity +
                    0.10 *
                        selected
                            .map {
                                it.score
                            }
                            .averageOrZero()
                )
                .coerceIn(0.0, 1.0)

        val metricReady =
            selected.size >= 2 &&
                graphConnectivity >= 0.55 &&
                loopConsistency >= 0.55 &&
                uncertaintyQuality >= 0.55 &&
                globalConsistency >= 0.60 &&
                selected.all { authority ->
                    authority.score >= 0.55
                }

        val targets =
            unique
                .mapNotNull { camera ->
                    calibrationTarget(
                        camera = camera,
                        authority =
                            ranked.firstOrNull {
                                it.cameraId == camera.id
                            },
                        loopConsistency = loopConsistency,
                    )
                }
                .sortedByDescending { it.priority }
                .take(16)

        val warnings =
            buildList {
                if (graphConnectivity < 0.55) {
                    add("calibration-graph-connectivity-low")
                }
                if (loopConsistency < 0.55) {
                    add("calibration-loop-consistency-low")
                }
                if (uncertaintyQuality < 0.55) {
                    add("calibration-uncertainty-high")
                }
                if (selected.size < min(6, unique.size)) {
                    add("calibration-authority-camera-diversity-limited")
                }
                if (targets.any { it.priority >= 0.70 }) {
                    add("active-calibration-recommended")
                }
                if (!metricReady) {
                    add("metric-reconstruction-not-authoritative")
                }
            }

        val diagnostics =
            buildList {
                add("cameraCount=" + unique.size)
                add("edgeCount=" + derivedEdges.size)
                add("authoritativeCount=" + selected.size)
                add("witnessCount=" + witnesses.size)
                add("rejectedCount=" + rejected.size)
                add("graphConnectivity=" + format(graphConnectivity))
                add("loopConsistency=" + format(loopConsistency))
                add("uncertaintyQuality=" + format(uncertaintyQuality))
                add("globalConsistency=" + format(globalConsistency))
                add("metricReady=" + metricReady)
                add("activeCalibrationTargetCount=" + targets.size)
            }

        return Result(
            authoritativeCameras = selected,
            witnessCameras = witnesses,
            rejectedCameras = rejected,
            edges = derivedEdges,
            globalConsistency = globalConsistency,
            graphConnectivity = graphConnectivity,
            loopConsistency = loopConsistency,
            uncertaintyQuality = uncertaintyQuality,
            metricReady = metricReady,
            activeCalibrationTargets = targets,
            warnings = warnings.distinct(),
            diagnostics = diagnostics,
        )
    }

    private fun deriveEdges(
        cameras: List<Camera>,
    ): List<Edge> {
        val result = ArrayList<Edge>(
            cameras.size * (cameras.size - 1) / 2,
        )

        for (firstIndex in 0 until cameras.lastIndex) {
            for (secondIndex in firstIndex + 1 until cameras.size) {
                val first = cameras[firstIndex]
                val second = cameras[secondIndex]

                val baseline =
                    first.positionMeters
                        .distanceTo(second.positionMeters)

                // Optical axes already point from each camera into the scene. Their angular
                // separation is the triangulation parallax used by the authority gate.
                val parallax =
                    angularSeparation(
                        first.forward.normalized(),
                        second.forward.normalized(),
                    )

                val reprojection =
                    (
                        first.reprojectionErrorPixels +
                            second.reprojectionErrorPixels
                        ) * 0.5

                val timing =
                    max(
                        first.timingResidualMillis,
                        second.timingResidualMillis,
                    )

                val translationSigma =
                    sqrt(
                        first.uncertainty.translationSigmaMeters *
                            first.uncertainty.translationSigmaMeters +
                            second.uncertainty.translationSigmaMeters *
                                second.uncertainty.translationSigmaMeters
                    )

                val rotationSigma =
                    sqrt(
                        first.uncertainty.rotationSigmaRadians *
                            first.uncertainty.rotationSigmaRadians +
                            second.uncertainty.rotationSigmaRadians *
                                second.uncertainty.rotationSigmaRadians
                    )

                val parallaxQuality =
                    when {
                        parallax < config.minimumParallaxDegrees ->
                            0.15
                        parallax < config.preferredParallaxDegrees ->
                            0.55 +
                                0.45 *
                                    (
                                        (parallax - config.minimumParallaxDegrees) /
                                            (
                                                config.preferredParallaxDegrees -
                                                    config.minimumParallaxDegrees
                                            )
                                    ).coerceIn(0.0, 1.0)
                        else ->
                            1.0
                    }

                val confidence =
                    (
                        0.22 *
                            first.calibrationConfidence *
                            second.calibrationConfidence +
                            0.18 *
                                first.synchronizationConfidence *
                                second.synchronizationConfidence +
                            0.14 * (
                                1.0 -
                                    reprojection /
                                        config.maximumReprojectionErrorPixels
                            ).coerceIn(0.0, 1.0) +
                            0.14 *
                                timingQuality(timing) +
                            0.16 *
                                translationQuality(
                                    translationSigma,
                                ) +
                            0.08 *
                                rotationQuality(
                                    rotationSigma,
                                ) +
                            0.08 * parallaxQuality
                        )
                        .coerceIn(0.0, 1.0)

                result +=
                    Edge(
                        from = first.id,
                        to = second.id,
                        baselineMeters = baseline,
                        parallaxDegrees = parallax,
                        reprojectionErrorPixels = reprojection,
                        timingResidualMillis = timing,
                        translationSigmaMeters = translationSigma,
                        rotationSigmaRadians = rotationSigma,
                        confidence = confidence,
                    )
            }
        }

        return result
    }

    private fun sanitizeEdge(
        edge: Edge,
    ): Edge =
        edge.copy(
            baselineMeters =
                max(0.0, edge.baselineMeters),
            parallaxDegrees =
                edge.parallaxDegrees.coerceIn(0.0, 180.0),
            reprojectionErrorPixels =
                max(0.0, edge.reprojectionErrorPixels),
            timingResidualMillis =
                max(0.0, edge.timingResidualMillis),
            translationSigmaMeters =
                max(0.0, edge.translationSigmaMeters),
            rotationSigmaRadians =
                max(0.0, edge.rotationSigmaRadians),
            confidence =
                edge.confidence.coerceIn(0.0, 1.0),
            cycleResidualMeters =
                max(0.0, edge.cycleResidualMeters),
        )

    private fun reachableConfidence(
        source: String,
        cameras: List<Camera>,
        adjacency: Map<String, List<Edge>>,
    ): Double {
        val distances =
            mutableMapOf<String, Double>()
        val open =
            java.util.PriorityQueue<Pair<String, Double>>(
                compareBy { it.second },
            )
        distances[source] = 0.0
        open += source to 0.0

        while (open.isNotEmpty()) {
            val (node, distance) = open.poll()
            if (distance > (distances[node] ?: Double.POSITIVE_INFINITY)) {
                continue
            }
            adjacency[node].orEmpty().forEach { edge ->
                val next =
                    if (edge.from == node) edge.to else edge.from
                // Positive edge cost for Dijkstra: high-confidence edges are
                // information-cheap while weak edges are expensive.
                val weight =
                    -kotlin.math.ln(
                        edge.confidence.coerceAtLeast(1e-6),
                    )
                val nextDistance =
                    distance + weight

                if (
                    next !in distances ||
                    nextDistance <
                        distances[next]!!
                ) {
                    distances[next] = nextDistance
                    open += next to nextDistance
                }
            }
        }

        if (cameras.size <= 1) return 0.0

        var weightedReach = 0.0
        distances.forEach { (node, cost) ->
            if (node !in cameras.map { it.id }) return@forEach
            val pathConfidence =
                kotlin.math.exp(
                    -cost.coerceAtLeast(0.0),
                )
                    .coerceIn(0.0, 1.0)
            weightedReach += pathConfidence
        }

        return (
            weightedReach /
                cameras.size.toDouble()
            ).coerceIn(0.0, 1.0)
    }

    private fun loopCameraConsistency(
        cameraId: String,
        edges: List<Edge>,
    ): Double {
        val local =
            edges.filter {
                it.from == cameraId ||
                    it.to == cameraId
            }
        if (local.isEmpty()) return 0.0

        return (
            0.65 *
                local.map { it.confidence }.average() +
                0.35 *
                    (
                        1.0 -
                            local
                                .map { it.cycleResidualMeters }
                                .average() /
                                config.maximumCycleResidualMeters
                    ).coerceIn(0.0, 1.0)
            )
            .coerceIn(0.0, 1.0)
    }

    private fun diversityBonus(
        cameraId: String,
        selected: List<String>,
        cameras: Map<String, Camera>,
    ): Double {
        val candidate = cameras[cameraId] ?: return 0.0
        if (selected.isEmpty()) return 1.0

        val candidateSectors =
            selected
                .mapNotNull { cameras[it]?.sector }
                .toSet()

        val sectorBonus =
            if (candidate.sector !in candidateSectors) {
                0.20
            } else {
                0.0
            }

        val angular =
            selected
                .mapNotNull { selectedId ->
                    cameras[selectedId]?.let {
                        angularSeparation(
                            candidate.forward.normalized(),
                            it.forward.normalized(),
                        )
                    }
                }
                .averageOrZero()

        return (
            sectorBonus +
                (angular / 90.0).coerceIn(0.0, 1.0) * 0.80
        )
            .coerceIn(0.0, 1.0)
    }

    private fun calibrationTarget(
        camera: Camera,
        authority: CameraAuthority?,
        loopConsistency: Double,
    ): CalibrationTarget? {
        if (authority == null) return null

        val translation =
            camera.uncertainty.translationSigmaMeters
        val rotation =
            camera.uncertainty.rotationSigmaRadians
        val timing =
            camera.timingResidualMillis

        val translationRisk =
            (
                translation /
                    config.maximumTranslationSigmaMeters
            ).coerceIn(0.0, 1.0)
        val rotationRisk =
            (
                rotation /
                    config.maximumRotationSigmaRadians
            ).coerceIn(0.0, 1.0)
        val timingRisk =
            (
                timing /
                    config.maximumTimingResidualMillis
            ).coerceIn(0.0, 1.0)
        val reprojectionRisk =
            (
                camera.reprojectionErrorPixels /
                    config.maximumReprojectionErrorPixels
            ).coerceIn(0.0, 1.0)

        val priority =
            (
                0.32 * translationRisk +
                    0.20 * rotationRisk +
                    0.18 * timingRisk +
                    0.18 * reprojectionRisk +
                    0.12 * (
                        1.0 - loopConsistency
                    )
                )
                .coerceIn(0.0, 1.0)

        if (priority < 0.35) return null

        return CalibrationTarget(
            cameraId = camera.id,
            priority = priority,
            reason =
                buildString {
                    if (translationRisk >= 0.45) append("translation-uncertainty;")
                    if (rotationRisk >= 0.45) append("rotation-uncertainty;")
                    if (timingRisk >= 0.45) append("timing-residual;")
                    if (reprojectionRisk >= 0.45) append("reprojection;")
                    if (loopConsistency < 0.55) append("loop-consistency;")
                }.ifBlank {
                    "active-uncertainty-reduction"
                },
            estimatedTranslationSigmaMeters =
                translation,
            estimatedRotationSigmaRadians =
                rotation,
        )
    }

    private fun reasonsFor(
        camera: Camera,
        pathConfidence: Double,
        loop: Double,
        timing: Double,
    ): List<String> =
        buildList {
            if (!camera.available) add("camera-unavailable")
            if (camera.calibrationConfidence < 0.55) add("calibration-confidence-low")
            if (pathConfidence < 0.55) add("graph-path-weak")
            if (loop < 0.55) add("loop-consistency-weak")
            if (timing < 0.55) add("timing-residual-high")
            if (
                camera.reprojectionErrorPixels >
                    config.maximumReprojectionErrorPixels
            ) {
                add("reprojection-error-high")
            }
            if (
                camera.uncertainty.translationSigmaMeters >
                    config.maximumTranslationSigmaMeters
            ) {
                add("translation-uncertainty-high")
            }
            if (
                camera.uncertainty.rotationSigmaRadians >
                    config.maximumRotationSigmaRadians
            ) {
                add("rotation-uncertainty-high")
            }
        }

    private fun translationQuality(
        sigma: Double,
    ): Double =
        if (!sigma.isFinite()) {
            0.0
        } else {
            exp(
                -sigma /
                    config.maximumTranslationSigmaMeters
            ).coerceIn(0.0, 1.0)
        }

    private fun rotationQuality(
        sigma: Double,
    ): Double =
        if (!sigma.isFinite()) {
            0.0
        } else {
            exp(
                -sigma /
                    config.maximumRotationSigmaRadians
            ).coerceIn(0.0, 1.0)
        }

    private fun timingQuality(
        residualMillis: Double,
    ): Double =
        if (!residualMillis.isFinite()) {
            0.0
        } else {
            exp(
                -residualMillis /
                    config.maximumTimingResidualMillis
            ).coerceIn(0.0, 1.0)
        }

    private fun angularSeparation(
        first: Vec3,
        second: Vec3,
    ): Double {
        val dot =
            first.normalized()
                .dot(second.normalized())
                .coerceIn(-1.0, 1.0)

        return Math.toDegrees(
            acos(dot),
        )
    }

    private fun emptyResult(
        warning: String,
    ): Result =
        Result(
            authoritativeCameras = emptyList(),
            witnessCameras = emptyList(),
            rejectedCameras = emptyList(),
            edges = emptyList(),
            globalConsistency = 0.0,
            graphConnectivity = 0.0,
            loopConsistency = 0.0,
            uncertaintyQuality = 0.0,
            metricReady = false,
            activeCalibrationTargets = emptyList(),
            warnings = listOf(warning),
            diagnostics = listOf(warning),
        )

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private fun Vec3.finite(): Boolean =
        x.isFinite() && y.isFinite() && z.isFinite()
}


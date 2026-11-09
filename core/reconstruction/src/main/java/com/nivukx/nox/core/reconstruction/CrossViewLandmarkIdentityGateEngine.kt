package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Calibrated cross-view identity gate for dense markerless landmarks.
 *
 * The dense reconstruction stack normally assumes that landmark ID N from camera A and landmark
 * ID N from camera B represent the same physical point. That assumption is unsafe during close
 * interaction, occlusion, identity swaps and detector ambiguity.
 *
 * This gate performs a geometry-only leave-one-out consistency check before downstream triangulation:
 * - duplicate camera observations are collapsed;
 * - stale and non-finite evidence is rejected;
 * - each camera is scored by how well it agrees with a 3D point reconstructed from the other views;
 * - symmetric pairwise reprojection is used when only two independent cameras exist;
 * - low-consistency views are rejected or downweighted;
 * - support is bounded so a large camera count cannot turn bad correspondence into false certainty.
 *
 * This is a correspondence-quality gate, not an accuracy guarantee. It requires real calibration and
 * real benchmark evidence before any physical precision claim can be made.
 */
class CrossViewLandmarkIdentityGateEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        val maximumObservations: Int = 96,
        // Dense reconstruction now selects up to 32 authoritative views.
        val maximumIndependentCameras: Int = 32,
        val staleObservationMillis: Double = 120.0,
        val warningReprojectionErrorPx: Double = 3.0,
        val hardReprojectionErrorPx: Double = 12.0,
        val robustReprojectionScalePx: Double = 3.5,
        val minimumObservationConfidence: Double = 0.05,
        val minimumGeometryScore: Double = 0.18,
        val minimumAcceptedViews: Int = 2,
        val minimumParallaxDegrees: Double = 0.50,
        val pairOnlyConfidencePenalty: Double = 0.84,
        val leaveOneOutConfidenceFloor: Double = 0.20,
        val duplicatePenalty: Double = 0.18,
        val residualExponent: Double = 1.0,
        val diagnosticsLimit: Int = 128,
    ) {
        init {
            require(maximumObservations >= 2)
            require(maximumIndependentCameras >= 2)
            require(staleObservationMillis > 0.0)
            require(warningReprojectionErrorPx > 0.0)
            require(hardReprojectionErrorPx > warningReprojectionErrorPx)
            require(robustReprojectionScalePx > 0.0)
            require(minimumObservationConfidence in 0.0..1.0)
            require(minimumGeometryScore in 0.0..1.0)
            require(minimumAcceptedViews >= 2)
            require(minimumParallaxDegrees > 0.0)
            require(pairOnlyConfidencePenalty in 0.0..1.0)
            require(leaveOneOutConfidenceFloor in 0.0..1.0)
            require(duplicatePenalty in 0.0..1.0)
            require(residualExponent > 0.0)
            require(diagnosticsLimit >= 8)
        }
    }

    data class Observation(
        val cameraId: String,
        val timestampNanos: Long,
        val u: Double,
        val v: Double,
        val cameraPose: Pose,
        val intrinsics: CameraIntrinsics,
        val confidence: Double = 1.0,
        val uncertainty: Double = 0.0,
        val visibility: Double = 1.0,
        val temporalStability: Double = 1.0,
        val sourceQuality: Double = 1.0,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(timestampNanos >= 0L)
            require(cameraPose.position.x.isFinite())
            require(cameraPose.position.y.isFinite())
            require(cameraPose.position.z.isFinite())
            require(intrinsics.fx > 0.0)
            require(intrinsics.fy > 0.0)
            require(confidence.isFinite())
            require(uncertainty.isFinite())
            require(visibility.isFinite())
            require(temporalStability.isFinite())
            require(sourceQuality.isFinite())
        }

        fun finite(): Boolean =
            timestampNanos >= 0L &&
                u.isFinite() &&
                v.isFinite() &&
                confidence.isFinite() &&
                uncertainty.isFinite() &&
                visibility.isFinite() &&
                temporalStability.isFinite() &&
                sourceQuality.isFinite() &&
                cameraPose.position.x.isFinite() &&
                cameraPose.position.y.isFinite() &&
                cameraPose.position.z.isFinite()
    }

    data class Input(
        val timestampNanos: Long,
        val observations: List<Observation>,
    ) {
        init {
            require(timestampNanos >= 0L)
        }
    }

    data class ObservationScore(
        val cameraId: String,
        val geometryScore: Double,
        val reprojectionP50Px: Double,
        val reprojectionP95Px: Double,
        val leaveOneOutSupported: Boolean,
        val accepted: Boolean,
        val confidenceScale: Double,
        val warnings: List<String>,
    )

    data class Result(
        val timestampNanos: Long,
        val acceptedObservations: List<Observation>,
        val rejectedObservations: List<Observation>,
        val scores: List<ObservationScore>,
        val identityConsistency: Double,
        val medianReprojectionErrorPx: Double,
        val p95ReprojectionErrorPx: Double,
        val uniqueCameraCount: Int,
        val duplicateObservationCount: Int,
        val staleObservationCount: Int,
        val nonFiniteObservationCount: Int,
        val warningCount: Int,
        val hardFailure: Boolean,
        val pairOnlyGeometry: Boolean,
        val warnings: List<String>,
        val diagnostics: List<String>,
    ) {
        val authoritative: Boolean
            get() =
                !hardFailure &&
                    acceptedObservations.size >= 2 &&
                    identityConsistency >= 0.80 &&
                    p95ReprojectionErrorPx <= 8.0
    }

    private data class PreparedObservation(
        val observation: Observation,
        val confidenceWeight: Double,
    )

    private val triangulation = TriangulationEngine(
        minParallaxDegrees = config.minimumParallaxDegrees,
    )

    private val diagnostics = ArrayDeque<String>()

    private var closed = false
    private var lastTimestampNanos = Long.MIN_VALUE
    private var latestResult: Result? = null

    @Synchronized
    fun evaluate(input: Input): Result {
        check(!closed) { "CrossViewLandmarkIdentityGateEngine is closed" }

        if (
            lastTimestampNanos != Long.MIN_VALUE &&
            input.timestampNanos < lastTimestampNanos
        ) {
            appendDiagnostic(
                "timestamp reversal rejected before cross-view identity gating",
            )
            val failed = emptyResult(
                timestampNanos = input.timestampNanos,
                hardFailure = true,
                warning = "Timestamp contract failed.",
            )
            latestResult = failed
            return failed
        }

        lastTimestampNanos = input.timestampNanos

        var staleCount = 0
        var nonFiniteCount = 0
        var duplicateCount = 0

        val byCamera = LinkedHashMap<String, PreparedObservation>()

        for (raw in input.observations.take(config.maximumObservations)) {
            if (!raw.finite()) {
                nonFiniteCount++
                continue
            }

            val ageMillis =
                (
                    input.timestampNanos -
                        raw.timestampNanos
                    ).coerceAtLeast(0L).toDouble() / 1_000_000.0

            if (
                !ageMillis.isFinite() ||
                ageMillis > config.staleObservationMillis
            ) {
                staleCount++
                continue
            }

            val confidence = raw.confidence.coerceIn(0.0, 1.0)
            if (confidence < config.minimumObservationConfidence) {
                continue
            }

            val candidate = PreparedObservation(
                observation = sanitize(raw),
                confidenceWeight = observationWeight(raw),
            )

            val existing = byCamera[raw.cameraId]
            if (existing == null) {
                byCamera[raw.cameraId] = candidate
            } else {
                duplicateCount++
                byCamera[raw.cameraId] = chooseDuplicate(
                    first = existing,
                    second = candidate,
                )
            }
        }

        val unique = byCamera.values
            .take(config.maximumIndependentCameras)

        if (unique.size < config.minimumAcceptedViews) {
            return emptyResult(
                timestampNanos = input.timestampNanos,
                hardFailure =
                    staleCount > 0 ||
                        nonFiniteCount > 0 ||
                        input.observations.isNotEmpty(),
                warning =
                    "Insufficient independent calibrated views survived identity gating.",
            ).copy(
                uniqueCameraCount = unique.size,
                duplicateObservationCount = duplicateCount,
                staleObservationCount = staleCount,
                nonFiniteObservationCount = nonFiniteCount,
                diagnostics = diagnostics.toList(),
            )
        }

        val observations =
            unique.map { it.observation }

        val pairOnly = observations.size == 2
        val scores =
            scoreObservations(
                observations = observations,
                pairOnly = pairOnly,
            )

        val primaryAccepted =
            observations.indices.mapNotNull { index ->
                if (scores[index].accepted) {
                    observations[index]
                } else {
                    null
                }
            }

        val accepted =
            if (
                primaryAccepted.size >= config.minimumAcceptedViews ||
                    observations.size < 3
            ) {
                primaryAccepted
            } else {
                scores
                    .mapIndexed { index, score -> index to score }
                    .sortedByDescending { it.second.geometryScore }
                    .take(config.minimumAcceptedViews)
                    .filter {
                        it.second.geometryScore >=
                            config.minimumGeometryScore
                    }
                    .map { observations[it.first] }
            }

        val rejected =
            observations.indices.mapNotNull { index ->
                if (!scores[index].accepted) {
                    observations[index]
                } else {
                    null
                }
            }

        val residuals =
            scores.map { it.reprojectionP95Px }
                .filter(Double::isFinite)
                .sorted()

        val medianResidual =
            percentile(
                scores.map { it.reprojectionP50Px },
                0.50,
            )

        val p95Residual =
            percentile(
                scores.map { it.reprojectionP95Px },
                0.95,
            )

        val identityConsistency =
            (
                scores.map { it.geometryScore }
                    .average()
                    .coerceIn(0.0, 1.0) *
                    (
                        accepted.size.toDouble() /
                            observations.size.toDouble()
                                .coerceAtLeast(1.0)
                    )
                ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (pairOnly) {
                add("pair-only identity check; no leave-one-out redundancy is available")
            }
            if (duplicateCount > 0) {
                add("duplicate same-camera evidence was collapsed")
            }
            if (staleCount > 0) {
                add("stale cross-view observations were rejected")
            }
            if (nonFiniteCount > 0) {
                add("non-finite cross-view observations were rejected")
            }
            if (p95Residual > config.warningReprojectionErrorPx) {
                add("cross-view reprojection residual is elevated")
            }
            if (accepted.size < config.minimumAcceptedViews) {
                add("cross-view identity support is insufficient")
            }
            if (identityConsistency < 0.60) {
                add("cross-view identity consistency is degraded")
            }
        }

        val hardFailure =
            accepted.size < config.minimumAcceptedViews ||
                (
                    accepted.isNotEmpty() &&
                        identityConsistency < 0.35
                    ) ||
                p95Residual > config.hardReprojectionErrorPx ||
                (
                    nonFiniteCount > 0 &&
                        accepted.isEmpty()
                    )

        val result = Result(
            timestampNanos = input.timestampNanos,
            acceptedObservations = accepted,
            rejectedObservations = rejected,
            scores = scores,
            identityConsistency = identityConsistency,
            medianReprojectionErrorPx = medianResidual,
            p95ReprojectionErrorPx = p95Residual,
            uniqueCameraCount = observations.size,
            duplicateObservationCount = duplicateCount,
            staleObservationCount = staleCount,
            nonFiniteObservationCount = nonFiniteCount,
            warningCount = warnings.size,
            hardFailure = hardFailure,
            pairOnlyGeometry = pairOnly,
            warnings = warnings.distinct(),
            diagnostics = diagnostics.toList(),
        )

        latestResult = result
        appendDiagnostic(
            "cross-view gate cameras=" + observations.size +
                " accepted=" + accepted.size +
                " identity=" + metric(identityConsistency) +
                " p95=" + metric(p95Residual),
        )

        return result
    }

    @Synchronized
    fun latest(): Result? = latestResult

    @Synchronized
    fun diagnosticHistory(): List<String> =
        diagnostics.toList()

    @Synchronized
    fun reset() {
        check(!closed) { "CrossViewLandmarkIdentityGateEngine is closed" }
        lastTimestampNanos = Long.MIN_VALUE
        latestResult = null
        diagnostics.clear()
    }

    @Synchronized
    override fun close() {
        if (closed) return
        lastTimestampNanos = Long.MIN_VALUE
        latestResult = null
        diagnostics.clear()
        closed = true
    }

    private fun scoreObservations(
        observations: List<Observation>,
        pairOnly: Boolean,
    ): List<ObservationScore> {
        if (observations.size == 3) {
            return scoreThreeViewConsensus(observations)
        }

        val scores = ArrayList<ObservationScore>(
            observations.size,
        )

        for (index in observations.indices) {
            val observation = observations[index]
            val peers =
                observations.filterIndexed { peerIndex, _ ->
                    peerIndex != index
                }

            if (peers.size < 2) {
                val symmetric = pairConsistency(
                    a = observation,
                    b = peers.firstOrNull(),
                )
                scores += symmetric
                continue
            }

            val consensusError =
                globalConsensusReprojectionError(
                    candidate = observation,
                    observations = observations,
                )
            val geometryErrors = buildList {
                consensusError?.let { add(it) }
                addAll(
                    leaveOneOutErrors(
                        candidate = observation,
                        peers = peers,
                    ),
                )
            }

            if (geometryErrors.isEmpty()) {
                scores += weakScore(
                    observation = observation,
                    pairOnly = pairOnly,
                )
                continue
            }

            val p50 = percentile(geometryErrors, 0.50)
            val p95 = percentile(geometryErrors, 0.95)

            val geometryScore =
                residualScore(
                    p50 = p50,
                    p95 = p95,
                )

            val accepted =
                geometryScore >= config.minimumGeometryScore &&
                    p95 <= config.hardReprojectionErrorPx &&
                    (
                        observations.size < 3 ||
                            consensusError == null ||
                            consensusError <=
                                config.warningReprojectionErrorPx
                    )

            val confidenceScale =
                (
                    geometryScore.coerceIn(
                        config.leaveOneOutConfidenceFloor,
                        1.0,
                    ) *
                        (
                            1.0 -
                                config.duplicatePenalty *
                                    if (pairOnly) 1.0 else 0.0
                            )
                    ).coerceIn(
                        config.leaveOneOutConfidenceFloor,
                        1.0,
                    )

            scores += ObservationScore(
                cameraId = observation.cameraId,
                geometryScore = geometryScore,
                reprojectionP50Px = p50,
                reprojectionP95Px = p95,
                leaveOneOutSupported = true,
                accepted = accepted,
                confidenceScale = confidenceScale,
                warnings = buildList {
                    if (p50 > config.warningReprojectionErrorPx) {
                        add("landmark identity residual elevated")
                    }
                    if (p95 > config.hardReprojectionErrorPx) {
                        add("landmark identity residual hard-failed")
                    }
                },
            )
        }

        return scores
    }

    private fun scoreThreeViewConsensus(
        observations: List<Observation>,
    ): List<ObservationScore> {
        /*
         * For exactly three cameras, score each observation against a triangulation
         * built only from the other two. The candidate must not help construct the
         * model that is used to validate itself.
         */
        return observations.mapIndexed { index, observation ->
            val peers = observations.filterIndexed { peerIndex, _ -> peerIndex != index }
            val solve =
                triangulation.robust(
                    peers.map(::toTriangulationObservation),
                )
            val residual =
                solve?.let {
                    project(it.point, observation)
                        ?.distanceTo(Vec2(observation.u, observation.v))
                }?.takeIf(Double::isFinite) ?: Double.POSITIVE_INFINITY
            val geometry =
                residualScore(
                    p50 = residual,
                    p95 = residual,
                )
            ObservationScore(
                cameraId = observation.cameraId,
                geometryScore = geometry,
                reprojectionP50Px = residual,
                reprojectionP95Px = residual,
                leaveOneOutSupported = true,
                accepted =
                    geometry >= config.minimumGeometryScore &&
                        residual <= config.hardReprojectionErrorPx,
                confidenceScale =
                    geometry.coerceIn(
                        config.leaveOneOutConfidenceFloor,
                        1.0,
                    ),
                warnings = buildList {
                    if (residual > config.warningReprojectionErrorPx) {
                        add("leave-one-out-consensus-residual-elevated")
                    }
                    if (residual > config.hardReprojectionErrorPx) {
                        add("leave-one-out-consensus-residual-hard-failed")
                    }
                },
            )
        }
    }

    /*
     * Legacy three-view consensus implementation retained below for deterministic reference
     * comparison during replay.
     */
    private fun legacyScoreThreeViewConsensus(
        observations: List<Observation>,
    ): List<ObservationScore> {
        data class Candidate(
            val first: Int,
            val second: Int,
            val point: Vec3,
            val totalResidual: Double,
        )

        val candidates =
            buildList {
                for (first in 0 until observations.lastIndex) {
                    for (second in first + 1 until observations.size) {
                        val solve =
                            triangulation.robust(
                                listOf(
                                    toTriangulationObservation(observations[first]),
                                    toTriangulationObservation(observations[second]),
                                ),
                            ) ?: continue

                        val residuals =
                            observations.mapNotNull { observation ->
                                project(
                                    point = solve.point,
                                    observation = observation,
                                )?.distanceTo(
                                    Vec2(
                                        observation.u,
                                        observation.v,
                                    ),
                                )?.takeIf(Double::isFinite)
                            }

                        if (residuals.size == observations.size) {
                            add(
                                Candidate(
                                    first = first,
                                    second = second,
                                    point = solve.point,
                                    totalResidual = residuals.sum(),
                                ),
                            )
                        }
                    }
                }
            }

        val best =
            candidates.minWithOrNull(
                compareBy<Candidate> { it.totalResidual }
                    .thenBy { it.first }
                    .thenBy { it.second },
            )

        if (best == null) {
            return observations.map {
                weakScore(
                    observation = it,
                    pairOnly = false,
                ).copy(
                    warnings =
                        listOf(
                            "three-view-consensus-reconstruction-unavailable",
                        ),
                )
            }
        }

        return observations.mapIndexed { index, observation ->
            val residual =
                project(
                    point = best.point,
                    observation = observation,
                )?.distanceTo(
                    Vec2(
                        observation.u,
                        observation.v,
                    ),
                )?.takeIf(Double::isFinite)
                    ?: Double.POSITIVE_INFINITY

            val geometry =
                residualScore(
                    p50 = residual,
                    p95 = residual,
                )

            ObservationScore(
                cameraId = observation.cameraId,
                geometryScore = geometry,
                reprojectionP50Px = residual,
                reprojectionP95Px = residual,
                leaveOneOutSupported = true,
                accepted =
                    geometry >= config.minimumGeometryScore &&
                        residual <= config.hardReprojectionErrorPx,
                confidenceScale =
                    geometry.coerceIn(
                        config.leaveOneOutConfidenceFloor,
                        1.0,
                    ),
                warnings = buildList {
                    if (residual > config.warningReprojectionErrorPx) {
                        add("leave-one-out-consensus-residual-elevated")
                    }
                    if (residual > config.hardReprojectionErrorPx) {
                        add("leave-one-out-consensus-residual-hard-failed")
                    }
                },
            )
        }
    }

    private fun globalConsensusReprojectionError(
        candidate: Observation,
        observations: List<Observation>,
    ): Double? {
        if (observations.size < 3) {
            return pairReprojectionError(
                first = candidate,
                second = observations.firstOrNull {
                    it.cameraId != candidate.cameraId
                },
                measured = candidate,
            )
        }

        val solve =
            triangulation.robust(
                observations.map(::toTriangulationObservation),
            ) ?: return null

        return project(
            point = solve.point,
            observation = candidate,
        )?.distanceTo(
            Vec2(candidate.u, candidate.v),
        )?.takeIf(Double::isFinite)
    }

    private fun pairReprojectionError(
        first: Observation,
        second: Observation?,
        measured: Observation,
    ): Double? {
        if (second == null) return null
        val solve = triangulation.robust(
            listOf(
                toTriangulationObservation(first),
                toTriangulationObservation(second),
            ),
        ) ?: return null
        return project(
            point = solve.point,
            observation = measured,
        )?.distanceTo(
            Vec2(
                measured.u,
                measured.v,
            ),
        )?.takeIf(Double::isFinite)
    }

    private fun leaveOneOutErrors(
        candidate: Observation,
        peers: List<Observation>,
    ): List<Double> {
        val errors = ArrayList<Double>()
        val peerCount = peers.size

        for (heldOutPeer in peers.indices) {
            val support = peers.filterIndexed {
                index, _ -> index != heldOutPeer
            }

            if (support.size < 2) {
                continue
            }

            val solveObservations =
                support.map {
                    TriangulationEngine.Observation(
                        cameraPose = it.cameraPose,
                        intrinsics = it.intrinsics,
                        cameraId = it.cameraId,
                        u = it.u,
                        v = it.v,
                        weight = it.confidenceWeight(),
                        observationUncertainty = it.uncertainty,
                        visibility = it.visibility,
                        sourceQuality = it.sourceQuality,
                        temporalStability = it.temporalStability,
                    )
                }

            val reconstructed =
                triangulation.robust(solveObservations)
                    ?: continue

            val heldOut = peers[heldOutPeer]
            project(
                point = reconstructed.point,
                observation = heldOut,
            )?.let { projection ->
                val error =
                    projection.distanceTo(
                        Vec2(
                            heldOut.u,
                            heldOut.v,
                        ),
                    )
                if (error.isFinite()) {
                    errors += error
                }
            }
        }

        return errors
    }

    private fun pairConsistency(
        a: Observation,
        b: Observation?,
    ): ObservationScore {
        if (b == null) {
            return ObservationScore(
                cameraId = a.cameraId,
                geometryScore = 0.0,
                reprojectionP50Px = Double.POSITIVE_INFINITY,
                reprojectionP95Px = Double.POSITIVE_INFINITY,
                leaveOneOutSupported = false,
                accepted = false,
                confidenceScale = 0.0,
                warnings = listOf(
                    "insufficient pair support",
                ),
            )
        }

        val solve = triangulation.robust(
            listOf(
                toTriangulationObservation(a),
                toTriangulationObservation(b),
            ),
        )

        if (solve == null) {
            return weakScore(
                observation = a,
                pairOnly = true,
            )
        }

        val errorA =
            project(
                point = solve.point,
                observation = a,
            )?.distanceTo(
                Vec2(a.u, a.v),
            ) ?: Double.POSITIVE_INFINITY

        val errorB =
            project(
                point = solve.point,
                observation = b,
            )?.distanceTo(
                Vec2(b.u, b.v),
            ) ?: Double.POSITIVE_INFINITY

        val p50 =
            listOf(errorA, errorB)
                .filter(Double::isFinite)
                .sorted()
                .let {
                    if (it.isEmpty()) {
                        Double.POSITIVE_INFINITY
                    } else {
                        it[it.size / 2]
                    }
                }

        val p95 =
            listOf(errorA, errorB)
                .filter(Double::isFinite)
                .maxOrNull()
                ?: Double.POSITIVE_INFINITY

        val geometry =
            residualScore(
                p50 = p50,
                p95 = p95,
            ) * config.pairOnlyConfidencePenalty

        return ObservationScore(
            cameraId = a.cameraId,
            geometryScore = geometry,
            reprojectionP50Px = p50,
            reprojectionP95Px = p95,
            leaveOneOutSupported = false,
            accepted =
                geometry >= config.minimumGeometryScore &&
                    p95 <= config.hardReprojectionErrorPx,
            confidenceScale =
                geometry.coerceIn(
                    config.leaveOneOutConfidenceFloor,
                    1.0,
                ),
            warnings = buildList {
                if (p95 > config.warningReprojectionErrorPx) {
                    add("pairwise cross-view residual elevated")
                }
                add("pairwise identity gate carries limited redundancy")
            },
        )
    }

    private fun weakScore(
        observation: Observation,
        pairOnly: Boolean,
    ): ObservationScore =
        ObservationScore(
            cameraId = observation.cameraId,
            geometryScore = (
                observation.confidence *
                    observation.visibility *
                    observation.temporalStability *
                    observation.sourceQuality
                ).coerceIn(0.0, 1.0),
            reprojectionP50Px = config.warningReprojectionErrorPx * 2.0,
            reprojectionP95Px = config.warningReprojectionErrorPx * 2.5,
            leaveOneOutSupported = false,
            accepted = !pairOnly &&
                observation.confidence >= config.minimumObservationConfidence,
            confidenceScale =
                (
                    observation.confidence *
                        config.leaveOneOutConfidenceFloor
                    ).coerceIn(0.0, 1.0),
            warnings = listOf(
                "insufficient leave-one-out geometry support",
            ),
        )

    private fun toTriangulationObservation(
        observation: Observation,
    ): TriangulationEngine.Observation =
        TriangulationEngine.Observation(
            cameraPose = observation.cameraPose,
            intrinsics = observation.intrinsics,
            cameraId = observation.cameraId,
            u = observation.u,
            v = observation.v,
            weight = observation.confidenceWeight(),
            observationUncertainty =
                observation.uncertainty.coerceIn(0.0, 1.0),
            visibility = observation.visibility.coerceIn(0.0, 1.0),
            sourceQuality = observation.sourceQuality.coerceIn(0.0, 1.0),
            temporalStability =
                observation.temporalStability.coerceIn(0.0, 1.0),
        )

    private fun Observation.confidenceWeight(): Double =
        (
            confidence.coerceIn(0.0, 1.0) *
                (
                    1.0 -
                        0.70 *
                            uncertainty.coerceIn(0.0, 1.0)
                    ).coerceIn(0.20, 1.0) *
                (
                    0.30 +
                        0.70 *
                            visibility.coerceIn(0.0, 1.0)
                    ) *
                (
                    0.30 +
                        0.70 *
                            temporalStability.coerceIn(
                                0.0,
                                1.0,
                            )
                    ) *
                (
                    0.40 +
                        0.60 *
                            sourceQuality.coerceIn(
                                0.0,
                                1.0,
                            )
                    )
            ).coerceIn(0.001, 1.0)

    private fun observationWeight(
        observation: Observation,
    ): Double =
        observation.confidenceWeight()

    private fun residualScore(
        p50: Double,
        p95: Double,
    ): Double {
        if (!p50.isFinite() || !p95.isFinite()) return 0.0

        val medianScore =
            exp(
                -(
                    p50.coerceAtLeast(0.0) /
                        config.robustReprojectionScalePx
                    ).powSafe(
                        config.residualExponent,
                    ),
            ).coerceIn(0.0, 1.0)

        val tailScore =
            exp(
                -(
                    p95.coerceAtLeast(0.0) /
                        (
                            config.robustReprojectionScalePx * 1.8
                        )
                    ).powSafe(
                        config.residualExponent,
                    ),
            ).coerceIn(0.0, 1.0)

        return (
            medianScore * 0.58 +
                tailScore * 0.42
            ).coerceIn(0.0, 1.0)
    }

    private fun project(
        point: Vec3,
        observation: Observation,
    ): Vec2? {
        val cameraPoint =
            observation.cameraPose.rotation
                .conjugate()
                .normalized()
                .rotate(
                    point -
                        observation.cameraPose.position,
                )

        if (!cameraPoint.z.isFinite() || cameraPoint.z <= 1e-6) {
            return null
        }

        var x = cameraPoint.x / cameraPoint.z
        var y = cameraPoint.y / cameraPoint.z

        val r2 = x * x + y * y

        val radial =
            1.0 +
                observation.intrinsics.k1 * r2 +
                observation.intrinsics.k2 * r2 * r2 +
                observation.intrinsics.k3 * r2 * r2 * r2

        val dx =
            2.0 *
                observation.intrinsics.p1 *
                x * y +
                observation.intrinsics.p2 *
                (
                    r2 +
                        2.0 * x * x
                    )

        val dy =
            observation.intrinsics.p1 *
                (
                    r2 +
                        2.0 * y * y
                    ) +
                2.0 *
                observation.intrinsics.p2 *
                x * y

        x = x * radial + dx
        y = y * radial + dy

        val u =
            x *
                observation.intrinsics.fx +
                observation.intrinsics.cx

        val v =
            y *
                observation.intrinsics.fy +
                observation.intrinsics.cy

        if (!u.isFinite() || !v.isFinite()) return null

        return Vec2(u, v)
    }

    private fun chooseDuplicate(
        first: PreparedObservation,
        second: PreparedObservation,
    ): PreparedObservation {
        val firstScore =
            first.confidenceWeight *
                (1.0 - config.duplicatePenalty)

        val secondScore =
            second.confidenceWeight

        return if (secondScore >= firstScore) {
            second
        } else {
            first
        }
    }

    private fun sanitize(
        observation: Observation,
    ): Observation =
        observation.copy(
            u = observation.u.coerceIn(-1_000_000.0, 1_000_000.0),
            v = observation.v.coerceIn(-1_000_000.0, 1_000_000.0),
            confidence =
                observation.confidence.coerceIn(0.0, 1.0),
            uncertainty =
                observation.uncertainty.coerceIn(0.0, 1.0),
            visibility =
                observation.visibility.coerceIn(0.0, 1.0),
            temporalStability =
                observation.temporalStability.coerceIn(
                    0.0,
                    1.0,
                ),
            sourceQuality =
                observation.sourceQuality.coerceIn(0.0, 1.0),
        )

    private fun percentile(
        values: List<Double>,
        quantile: Double,
    ): Double {
        val sorted =
            values
                .filter(Double::isFinite)
                .sorted()
        if (sorted.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }
        if (sorted.size == 1) return sorted.first()

        val position =
            quantile.coerceIn(0.0, 1.0) *
                (sorted.size - 1).toDouble()

        val lower = position.toInt()
        val upper =
            min(
                sorted.lastIndex,
                lower + 1,
            )
        val fraction = position - lower

        return (
            sorted[lower] +
                (
                    sorted[upper] -
                        sorted[lower]
                    ) *
                    fraction
            )
    }

    private fun Double.powSafe(exponent: Double): Double {
        if (!isFinite()) return Double.POSITIVE_INFINITY
        return this.coerceAtLeast(0.0).let {
            it.pow(exponent)
        }
    }

    private fun metric(value: Double): String =
        "%.4f".format(
            value.coerceIn(-99.0, 99.0),
        )

    private fun appendDiagnostic(message: String) {
        diagnostics.addLast(message)
        while (diagnostics.size > config.diagnosticsLimit) {
            diagnostics.removeFirst()
        }
    }

    private fun emptyResult(
        timestampNanos: Long,
        hardFailure: Boolean,
        warning: String,
    ): Result =
        Result(
            timestampNanos = timestampNanos,
            acceptedObservations = emptyList(),
            rejectedObservations = emptyList(),
            scores = emptyList(),
            identityConsistency = 0.0,
            medianReprojectionErrorPx = Double.POSITIVE_INFINITY,
            p95ReprojectionErrorPx = Double.POSITIVE_INFINITY,
            uniqueCameraCount = 0,
            duplicateObservationCount = 0,
            staleObservationCount = 0,
            nonFiniteObservationCount = 0,
            warningCount = 1,
            hardFailure = hardFailure,
            pairOnlyGeometry = false,
            warnings = listOf(warning),
            diagnostics = diagnostics.toList(),
        )
}

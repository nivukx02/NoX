package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Vec3
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min

/**
 * Causal camera-selection policy for expensive multi-view reconstruction.
 *
 * It consumes already measured per-camera trust plus the geometry carried by calibrated camera
 * observations. The planner prefers reliable cameras with complementary viewpoints and preserves
 * a configurable minimum number of views. It never fabricates a missing view and never hides a
 * camera failure as success.
 */
class MultiViewCameraContributionPlanner(
    private val config: Config = Config(),
    private val anisotropic: AnisotropicViewContributionEngine =
        AnisotropicViewContributionEngine(AnisotropicViewContributionEngine.Config.precision()),
) {

    data class Config(
        val minimumCameras: Int = 2,
        // Dense reconstruction remains bounded, but benefits from a larger high-quality view envelope.
        val maximumCameras: Int = 32,
        val maximumWitnessCameras: Int = 96,
        val minimumObservationConfidence: Double = 0.10,
        val reliabilityWeight: Double = 0.52,
        val diversityWeight: Double = 0.30,
        val coverageWeight: Double = 0.18,
        val informationGainWeight: Double = 0.16,
        val anisotropicConditioningWeight: Double = 0.18,
        val holdStateBonus: Double = -0.30,
        val degradedStateBonus: Double = -0.10,
        val minimumSelectionScore: Double = 0.26,
        val selectionHysteresisBonus: Double = 0.07,
        val maximumSelectionHysteresis: Double = 0.14,
        val selectionSwitchPenalty: Double = 0.04,
        val selectionHistoryFrames: Int = 6,
        val targetViewDirection: Vec3? = null,
        val targetViewWeight: Double = 0.0,
    ) {
        init {
            require(minimumCameras >= 1)
            require(maximumCameras >= minimumCameras)
            require(maximumWitnessCameras >= 0)
            require(minimumObservationConfidence in 0.0..1.0)
            require(reliabilityWeight >= 0.0)
            require(diversityWeight >= 0.0)
            require(coverageWeight >= 0.0)
            require(informationGainWeight >= 0.0)
            require(anisotropicConditioningWeight in 0.0..0.35)
            require(
                reliabilityWeight +
                    diversityWeight +
                    coverageWeight +
                    informationGainWeight +
                    anisotropicConditioningWeight >
                    0.0,
            )
            require(holdStateBonus <= 0.0)
            require(degradedStateBonus <= 0.0)
            require(minimumSelectionScore in 0.0..1.0)
            require(selectionHysteresisBonus in 0.0..0.25)
            require(maximumSelectionHysteresis in 0.0..0.30)
            require(maximumSelectionHysteresis >= selectionHysteresisBonus)
            require(selectionSwitchPenalty in 0.0..0.25)
            require(selectionHistoryFrames in 1..32)
            require(targetViewWeight in 0.0..0.35)
            targetViewDirection?.let {
                require(
                    it.x.isFinite() &&
                        it.y.isFinite() &&
                        it.z.isFinite(),
                )
                require(it.norm() > 1e-6)
            }
        }
    }

    data class CameraContribution(
        val deviceId: DeviceId,
        val observationConfidence: Double,
        val trust: Double,
        val reliabilityScore: Double,
        val geometricDiversityScore: Double,
        val coverageScore: Double,
        val informationGainScore: Double,
        val anisotropicConditioningScore: Double = 0.0,
        val targetViewAlignmentScore: Double = 0.0,
        val selectionScore: Double,
        val excluded: Boolean,
        val reason: String,
    )

    fun configuration(): Config = config

    /**
     * Reconfigure only the target-view bias at runtime.
     * Reliability, diversity and minimum-view protections remain unchanged.
     */
    fun configureTargetViewBias(
        direction: Vec3?,
        weight: Double,
    ) {
        val normalized =
            direction?.normalized()

        require(
            normalized == null ||
                normalized.norm() > 1e-6,
        )
        require(weight in 0.0..0.35)

        targetViewDirectionOverride = normalized
        targetViewWeightOverride = weight
    }

    data class Plan(
        val selectedDeviceIds: Set<DeviceId>,
        val excludedDeviceIds: Set<DeviceId>,
        val rankedCameras: List<CameraContribution>,
        val diversitySatisfied: Boolean,
        val globalInformationGainScore: Double = 0.0,
        val minimumPairwiseDiversity: Double = 0.0,
        val viewStarvationRisk: Double = 0.0,
        val witnessDeviceIds: Set<DeviceId> = emptySet(),
        val reason: String,
    )

    private val previousSelection = LinkedHashMap<DeviceId, Int>()

    @Volatile
    private var targetViewDirectionOverride: Vec3? = null

    @Volatile
    private var targetViewWeightOverride: Double? = null

    private data class CameraCandidate(
        val deviceId: DeviceId,
        val observations: List<CameraObservation>,
        val observationConfidence: Double,
        val trust: Double,
        val state:
            AdaptiveMultiViewEvidenceWeightEngine.State,
        val meanResidualPx: Double,
        var diversityScore: Double = 0.0,
        var informationGainScore: Double = 0.0,
        var anisotropicConditioningScore: Double = 0.0,
        var targetViewAlignmentScore: Double = 0.0,
        var selectionScore: Double = 0.0,
    )

    fun plan(
        frames: List<SkeletonReconstructionEngine.CameraFrame>,
        cameraStatuses:
            List<AdaptiveMultiViewEvidenceWeightEngine.CameraStatus>,
    ): Plan {
        if (frames.isEmpty()) {
            previousSelection.clear()
            return Plan(
                selectedDeviceIds = emptySet(),
                excludedDeviceIds = emptySet(),
                rankedCameras = emptyList(),
                diversitySatisfied = false,
                globalInformationGainScore = 0.0,
                witnessDeviceIds = emptySet(),
                reason = "no-camera-frames",
            )
        }

        val statusByDevice =
            cameraStatuses.associateBy { it.deviceId }

        val candidates =
            frames
                .map { frame ->
                    val validObservations =
                        frame.observations.filter {
                            it.confidence >=
                                config.minimumObservationConfidence
                        }

                    val observationConfidence =
                        validObservations
                            .map { it.confidence }
                            .averageOrZero()

                    val status =
                        statusByDevice[frame.deviceId]

                    CameraCandidate(
                        deviceId = frame.deviceId,
                        observations = validObservations,
                        observationConfidence = observationConfidence,
                        trust =
                            status?.trust?.coerceIn(0.0, 1.0)
                                ?: observationConfidence,
                        state =
                            status?.state
                                ?: AdaptiveMultiViewEvidenceWeightEngine.State.UNKNOWN,
                        meanResidualPx =
                            status?.meanResidualPx?.coerceAtLeast(0.0)
                                ?: 0.0,
                    )
                }
                .filter { it.observations.isNotEmpty() }

        if (candidates.size <= config.maximumCameras) {
            return buildPlan(
                candidates = candidates,
                selected = select(
                    candidates = candidates,
                    limit = candidates.size,
                ),
            )
        }

        val selected =
            select(
                candidates = candidates,
                limit = config.maximumCameras,
            )

        val plan = buildPlan(
            candidates = candidates,
            selected = selected,
        )

        val nextStreaks =
            LinkedHashMap<DeviceId, Int>().apply {
                plan.selectedDeviceIds.forEach { id ->
                    put(
                        id,
                        min(
                            config.selectionHistoryFrames,
                            (previousSelection[id] ?: 0) + 1,
                        ),
                    )
                }
            }

        val decayedSelections =
            previousSelection
                .filterKeys { it !in plan.selectedDeviceIds }
                .mapValues { (_, streak) ->
                    (streak - 1).coerceAtLeast(0)
                }
                .filterValues { it > 0 }

        previousSelection.clear()
        previousSelection.putAll(decayedSelections)
        previousSelection.putAll(nextStreaks)

        val staleSelectionIds =
            previousSelection.keys
                .filter { id ->
                    id !in candidates.map { it.deviceId }
                }
        staleSelectionIds.forEach {
            previousSelection.remove(it)
        }

        while (
            previousSelection.size >
                config.maximumCameras * 2
        ) {
            previousSelection.remove(
                previousSelection.keys.first(),
            )
        }

        return plan
    }

    private fun select(
        candidates: List<CameraCandidate>,
        limit: Int,
    ): Set<DeviceId> {
        if (candidates.isEmpty()) return emptySet()

        val normalizedReliability =
            candidates.maxOfOrNull {
                reliability(
                    candidate = it,
                )
            }?.coerceAtLeast(1e-9)
                ?: 1.0

        val anisotropicByCamera = anisotropicScores(candidates)

        val weightSum =
            (
                config.reliabilityWeight +
                    config.diversityWeight +
                    config.coverageWeight +
                    config.informationGainWeight +
                    config.anisotropicConditioningWeight +
                    (
                        targetViewWeightOverride
                            ?: config.targetViewWeight
                    )
                ).coerceAtLeast(1e-9)

        candidates.forEach { candidate ->
            val baseReliability =
                reliability(candidate)
            candidate.informationGainScore =
                informationGain(
                    candidate = candidate,
                    selected = emptySet(),
                    candidates = candidates,
                )
            candidate.anisotropicConditioningScore =
                anisotropicByCamera[candidate.deviceId] ?: 0.0
            candidate.targetViewAlignmentScore =
                targetViewAlignment(candidate)
            candidate.selectionScore =
                (
                    (
                        baseReliability / normalizedReliability *
                            config.reliabilityWeight +
                            candidate.diversityScore *
                                config.diversityWeight +
                            coverage(candidate) *
                                config.coverageWeight +
                            candidate.informationGainScore *
                                config.informationGainWeight +
                            candidate.anisotropicConditioningScore *
                                config.anisotropicConditioningWeight +
                            candidate.targetViewAlignmentScore *
                                (
                                    targetViewWeightOverride
                                        ?: config.targetViewWeight
                                )
                        ) / weightSum
                    ).coerceIn(0.0, 1.0)
        }

        val selected = LinkedHashSet<DeviceId>()

        val first =
            candidates
                .sortedByDescending {
                    initialScore(it)
                }
                .first()

        first.selectionScore =
            initialScore(first)
        selected += first.deviceId

        while (
            selected.size <
                min(
                    limit,
                    candidates.size,
                )
        ) {
            val next =
                candidates
                    .filterNot {
                        selected.contains(it.deviceId)
                    }
                    .maxByOrNull { candidate ->
                        if (
                            selected.size <
                                min(config.minimumCameras, candidates.size)
                        ) {
                            reliability(candidate)
                        } else {
                            marginalScore(
                                candidate = candidate,
                                selected = selected,
                                candidates = candidates,
                            )
                        }
                    }
                    ?: break

            if (
                selected.size >= config.minimumCameras &&
                    marginalScore(
                        candidate = next,
                        selected = selected,
                        candidates = candidates,
                    ) <
                    config.minimumSelectionScore
            ) {
                break
            }

            selected += next.deviceId
        }

        if (
            selected.size <
                min(
                    config.minimumCameras,
                    candidates.size,
                )
        ) {
            candidates
                .filterNot {
                    selected.contains(it.deviceId)
                }
                .sortedByDescending { reliability(it) }
                .take(
                    min(
                        config.minimumCameras - selected.size,
                        candidates.size - selected.size,
                    ),
                )
                .forEach {
                    selected += it.deviceId
                }
        }

        return selected
    }

    private fun anisotropicScores(
        candidates: List<CameraCandidate>,
    ): Map<DeviceId, Double> {
        val representatives =
            candidates.mapNotNull { candidate ->
                val observation =
                    candidate.observations.maxByOrNull {
                        it.confidence
                    } ?: return@mapNotNull null
                candidate.deviceId to
                    AnisotropicViewContributionEngine.Candidate(
                        observation = observation,
                        semanticConfidence = candidate.observationConfidence,
                        visibility = candidate.trust,
                        sourceDiversity =
                            candidate.observations.size
                                .coerceAtMost(8)
                                .toDouble()
                                .div(8.0)
                                .coerceIn(0.0, 1.0),
                        appearanceEvidence =
                            observation.appearanceConsistency
                                .coerceIn(0.0, 1.0),
                    )
            }

        if (representatives.size < 2) return emptyMap()

        val rank =
            anisotropic.rank(
                candidates = representatives.map { it.second },
                limit = representatives.size,
            )

        return rank.selected.associate {
            it.cameraId to
                (
                    it.totalScore *
                        (0.55 + 0.45 * rank.conditionScore)
                    ).coerceIn(0.0, 1.0)
        }
    }

    private fun initialScore(
        candidate: CameraCandidate,
    ): Double {
        val stateAdjustment =
            when (candidate.state) {
                AdaptiveMultiViewEvidenceWeightEngine.State.HOLD ->
                    config.holdStateBonus
                AdaptiveMultiViewEvidenceWeightEngine.State.DEGRADED ->
                    config.degradedStateBonus
                else -> 0.0
            }

        val persistence =
            if (
                previousSelection.containsKey(candidate.deviceId) &&
                    candidate.state !=
                    AdaptiveMultiViewEvidenceWeightEngine.State.HOLD
            ) {
                (
                    config.selectionHysteresisBonus +
                        min(
                            config.maximumSelectionHysteresis,
                            (previousSelection[candidate.deviceId] ?: 0) *
                                config.selectionHysteresisBonus *
                                0.12,
                        )
                    ).coerceIn(
                        0.0,
                        config.maximumSelectionHysteresis,
                    )
            } else {
                0.0
            }

        val score = (
            reliability(candidate) * config.reliabilityWeight +
                coverage(candidate) * config.coverageWeight +
                targetViewAlignment(candidate) *
                    (
                        targetViewWeightOverride
                            ?: config.targetViewWeight
                    ) +
                stateAdjustment +
                persistence
            ).coerceIn(0.0, 1.0)

        return score
    }

    private fun marginalScore(
        candidate: CameraCandidate,
        selected: Set<DeviceId>,
        candidates: List<CameraCandidate>,
    ): Double {
        val selectedCandidates =
            candidates.filter {
                selected.contains(it.deviceId)
            }

        val diversity =
            if (selectedCandidates.isEmpty()) {
                1.0
            } else {
                selectedCandidates.map { other ->
                    pairDiversity(
                        candidate = candidate,
                        other = other,
                    )
                }.average()
            }

        val informationGain =
            informationGain(
                candidate = candidate,
                selected = selected,
                candidates = candidates,
            )

        val stateAdjustment =
            when (candidate.state) {
                AdaptiveMultiViewEvidenceWeightEngine.State.HOLD ->
                    config.holdStateBonus
                AdaptiveMultiViewEvidenceWeightEngine.State.DEGRADED ->
                    config.degradedStateBonus
                else -> 0.0
            }

        val persistenceAdjustment =
            when {
                candidate.state ==
                    AdaptiveMultiViewEvidenceWeightEngine.State.HOLD ->
                    -config.selectionSwitchPenalty
                previousSelection.containsKey(candidate.deviceId) ->
                    min(
                        config.maximumSelectionHysteresis,
                        config.selectionHysteresisBonus +
                            (previousSelection[candidate.deviceId] ?: 0) *
                                config.selectionHysteresisBonus *
                                0.12,
                    )
                else ->
                    -config.selectionSwitchPenalty
            }

        candidate.diversityScore =
            diversity.coerceIn(0.0, 1.0)

        candidate.informationGainScore =
            informationGain.coerceIn(0.0, 1.0)

        val weightSum =
            (
                config.reliabilityWeight +
                    config.diversityWeight +
                    config.coverageWeight +
                    config.informationGainWeight +
                    config.anisotropicConditioningWeight
                ).coerceAtLeast(1e-9)

        return (
            (
                reliability(candidate) *
                    config.reliabilityWeight +
                    diversity *
                    config.diversityWeight +
                    coverage(candidate) *
                    config.coverageWeight +
                    informationGain *
                    config.informationGainWeight +
                    targetViewAlignment(candidate) *
                    (
                        targetViewWeightOverride
                            ?: config.targetViewWeight
                    )
                ) / weightSum +
                    stateAdjustment +
                    persistenceAdjustment
            ).coerceIn(0.0, 1.0)
    }

    private fun informationGain(
        candidate: CameraCandidate,
        selected: Set<DeviceId>,
        candidates: List<CameraCandidate>,
    ): Double {
        val temporalAlignment =
            candidate.observations
                .map {
                    it.temporalAlignmentConfidence
                }
                .averageOrZero()
                .coerceIn(0.0, 1.0)

        val reprojectionQuality =
            candidates.firstOrNull {
                it.deviceId == candidate.deviceId
            }?.let {
                statusQuality(
                    observationConfidence =
                        it.observationConfidence,
                    trust = it.trust,
                    state = it.state,
                    meanResidualPx = candidate.meanResidualPx,
                )
            } ?: 0.0

        val novelty =
            if (selected.isEmpty()) {
                1.0
            } else {
                selected.mapNotNull { id ->
                    candidates.firstOrNull {
                        it.deviceId == id
                    }
                }.map { other ->
                    pairDiversity(
                        candidate = candidate,
                        other = other,
                    )
                }.maxOrNull() ?: 0.0
            }

        val landmarkCoverage =
            coverage(candidate)

        val targetViewAlignment =
            targetViewAlignment(candidate)

        /*
         * This is an information-gain proxy, not a fabricated Bayesian posterior. It rewards
         * measured temporal alignment, reliable observations, broad landmark coverage and novel
         * calibrated viewpoints. A separate probabilistic surface model can consume this score
         * later without changing the camera-selection contract.
         */
        val targetWeight =
            (
                targetViewWeightOverride
                    ?: config.targetViewWeight
            ).coerceIn(0.0, 0.35)
        val baseWeight =
            1.0 - targetWeight
        val measuredEvidence =
            (
                temporalAlignment * 0.24 +
                    reprojectionQuality * 0.26 +
                    landmarkCoverage * 0.20 +
                    novelty * 0.30
            ).coerceIn(0.0, 1.0)

        return (
            measuredEvidence * baseWeight +
                targetViewAlignment * targetWeight
        ).coerceIn(0.0, 1.0)
    }

    private fun targetViewAlignment(
        candidate: CameraCandidate,
    ): Double {
        val target =
            (
                targetViewDirectionOverride
                    ?: config.targetViewDirection
            )?.normalized()
                ?: return 0.0

        val forward =
            candidate.observations
                .firstOrNull()
                ?.pose
                ?.rotation
                ?.rotate(
                    Vec3(0.0, 0.0, 1.0),
                )
                ?.normalized()
                ?: return 0.0

        return (
            forward.dot(target) + 1.0
        ).coerceIn(0.0, 2.0) * 0.5
    }

    private fun statusQuality(
        observationConfidence: Double,
        trust: Double,
        state:
            AdaptiveMultiViewEvidenceWeightEngine.State,
        meanResidualPx: Double,
    ): Double {
        val statePenalty =
            when (state) {
                AdaptiveMultiViewEvidenceWeightEngine.State.HOLD -> 0.46
                AdaptiveMultiViewEvidenceWeightEngine.State.DEGRADED -> 0.72
                else -> 1.0
            }
        val residualQuality =
            kotlin.math.exp(
                -meanResidualPx / 5.0,
            ).coerceIn(0.0, 1.0)

        return (
            observationConfidence * 0.36 +
                trust * 0.44 +
                residualQuality * 0.20
            ).coerceIn(0.0, 1.0) *
            statePenalty
    }

    private fun reliability(
        candidate: CameraCandidate,
    ): Double =
        (
            candidate.trust * 0.62 +
                candidate.observationConfidence * 0.38
            ).coerceIn(0.0, 1.0)

    private fun coverage(
        candidate: CameraCandidate,
    ): Double {
        val distinctLandmarks =
            candidate.observations
                .map { it.landmarkId }
                .filter { it >= 0 }
                .distinct()
                .size

        return (
            min(
                1.0,
                distinctLandmarks / 17.0,
            ) * 0.72 +
                candidate.observations.size
                    .coerceAtMost(32)
                    .toDouble() /
                    32.0 * 0.28
            ).coerceIn(0.0, 1.0)
    }

    private fun pairDiversity(
        candidate: CameraCandidate,
        other: CameraCandidate,
    ): Double {
        val a = candidate.observations.firstOrNull()
            ?: return 0.0
        val b = other.observations.firstOrNull()
            ?: return 0.0

        val pa = a.pose.position
        val pb = b.pose.position

        val baseline =
            pa.distanceTo(pb)

        val forwardA =
            a.pose.rotation.rotate(
                com.nivukx.nox.core.model.Vec3(
                    0.0,
                    0.0,
                    1.0,
                ),
            ).normalized()

        val forwardB =
            b.pose.rotation.rotate(
                com.nivukx.nox.core.model.Vec3(
                    0.0,
                    0.0,
                    1.0,
                ),
            ).normalized()

        val orientationAngle =
            acos(
                forwardA.dot(forwardB)
                    .coerceIn(-1.0, 1.0),
            ) / Math.PI

        val normalizedBaseline =
            (
                baseline /
                    (baseline + 0.50)
                ).coerceIn(0.0, 1.0)

        return (
            normalizedBaseline * 0.58 +
                orientationAngle * 0.42
            ).coerceIn(0.0, 1.0)
    }

    private fun buildPlan(
        candidates: List<CameraCandidate>,
        selected: Set<DeviceId>,
    ): Plan {
        val ranked =
            candidates
                .map { candidate ->
                    val selectedFlag =
                        selected.contains(candidate.deviceId)

                    CameraContribution(
                        deviceId = candidate.deviceId,
                        observationConfidence =
                            candidate.observationConfidence,
                        trust = candidate.trust,
                        reliabilityScore =
                            reliability(candidate),
                        geometricDiversityScore =
                            candidate.diversityScore,
                        coverageScore =
                            coverage(candidate),
                        informationGainScore =
                            candidate.informationGainScore,
                        targetViewAlignmentScore =
                            candidate.targetViewAlignmentScore,
                        selectionScore =
                            candidate.selectionScore,
                        excluded = !selectedFlag,
                        reason =
                            when {
                                selectedFlag ->
                                    if (candidate.diversityScore >= 0.55) {
                                        "selected:reliable-and-geometrically-diverse"
                                    } else {
                                        "selected:reliability"
                                    }

                                candidate.state ==
                                    AdaptiveMultiViewEvidenceWeightEngine.State.HOLD ->
                                    "excluded:camera-trust-held"
                                else ->
                                    "excluded:lower-marginal-contribution"
                            },
                    )
                }
                .sortedByDescending {
                    it.selectionScore
                }

        val selectedInformationGain =
            candidates
                .filter { selected.contains(it.deviceId) }
                .map { it.informationGainScore }
                .averageOrZero()

        val selectedCandidates =
            candidates.filter { selected.contains(it.deviceId) }
        val pairwiseDiversity =
            if (selectedCandidates.size < 2) {
                0.0
            } else {
                selectedCandidates.flatMapIndexed { i, first ->
                    selectedCandidates.drop(i + 1).map { second ->
                        pairDiversity(first, second)
                    }
                }.minOrNull() ?: 0.0
            }

        val diversitySatisfied =
            if (selected.size <= 1) {
                false
            } else {
                pairwiseDiversity >= 0.20
            }
        val reliabilities =
            selectedCandidates
                .map(::reliability)
                .filter { it.isFinite() && it > 0.0 }
        val reliabilitySum = reliabilities.sum().coerceAtLeast(1e-9)
        val concentration =
            (reliabilities.maxOrNull() ?: 0.0) /
                reliabilitySum
        val starvationRisk =
            (
                concentration * 0.70 +
                    (1.0 - pairwiseDiversity) * 0.30
                ).coerceIn(0.0, 1.0)

        return Plan(
            selectedDeviceIds = selected,
            excludedDeviceIds =
                candidates
                    .map { it.deviceId }
                    .filterNot { selected.contains(it) }
                    .toSet(),
            rankedCameras = ranked,
            diversitySatisfied = diversitySatisfied,
            globalInformationGainScore =
                selectedInformationGain.coerceIn(0.0, 1.0),
            minimumPairwiseDiversity =
                pairwiseDiversity.coerceIn(0.0, 1.0),
            viewStarvationRisk = starvationRisk,
            witnessDeviceIds =
                ranked.asSequence()
                    .filter { it.deviceId !in selected }
                    .take(config.maximumWitnessCameras)
                    .map { it.deviceId }
                    .toSet(),
            reason =
                if (selected.isEmpty()) {
                    "no-valid-camera-contributions"
                } else if (
                    !diversitySatisfied &&
                        selected.size >= 2
                ) {
                    "minimum-view-count-met-but-geometric-diversity-limited"
                } else if (selected.size == 1) {
                    "single-camera-fallback"
                } else {
                    "adaptive-reliable-diverse-view-selection"
                },
        )
    }

    private fun DoubleArray.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private fun List<CameraObservation>.averageConfidence():
        Double =
        map { it.confidence }
            .averageOrZero()
}

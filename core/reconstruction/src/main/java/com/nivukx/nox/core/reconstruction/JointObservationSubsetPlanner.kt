package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.max

/**
 * Selects the most useful observations for one 3D landmark.
 *
 * Camera-level selection protects the global reconstruction budget. Joint-level selection protects
 * the geometry of the actual triangulation problem. The planner is conservative: it never invents
 * an observation, and it keeps a minimum available evidence set even when diversity is poor.
 */
class JointObservationSubsetPlanner(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumObservations: Int = 2,
        val maximumObservations: Int = 6,
        val minimumConfidence: Double = 0.05,
        val reliabilityWeight: Double = 0.44,
        val diversityWeight: Double = 0.34,
        val timingWeight: Double = 0.22,
        val uncertaintyWeight: Double = 0.14,
        val minimumSelectionScore: Double = 0.24,
        val minimumDiversityScore: Double = 0.14,
    ) {
        init {
            require(minimumObservations >= 1)
            require(maximumObservations >= minimumObservations)
            require(minimumConfidence in 0.0..1.0)
            require(reliabilityWeight >= 0.0)
            require(diversityWeight >= 0.0)
            require(timingWeight >= 0.0)
            require(uncertaintyWeight in 0.0..1.0)
            require(
                reliabilityWeight +
                    diversityWeight +
                    timingWeight >
                    0.0,
            )
            require(minimumSelectionScore in 0.0..1.0)
            require(minimumDiversityScore in 0.0..1.0)
        }
    }

    data class Selection(
        val selectedObservations: List<CameraObservation>,
        val excludedDeviceIds: Set<DeviceId>,
        val selectedDeviceIds: Set<DeviceId>,
        val diversitySatisfied: Boolean,
        val score: Double,
        val reason: String,
    )

    private data class Candidate(
        val observation: CameraObservation,
        val reliability: Double,
        val timing: Double,
        val uncertainty: Double,
        var diversity: Double = 0.0,
        var score: Double = 0.0,
    )

    fun plan(
        observations: List<CameraObservation>,
    ): Selection {
        val unique =
            observations
                .asSequence()
                .filter {
                    it.confidence >= config.minimumConfidence
                }
                .groupBy { it.deviceId }
                .mapNotNull { (_, candidates) ->
                    candidates.maxByOrNull {
                        observationBaseScore(it)
                    }
                }
                .map {
                    Candidate(
                        observation = it,
                        reliability =
                            it.confidence.coerceIn(0.0, 1.0),
                        timing =
                            it.temporalAlignmentConfidence
                                .coerceIn(0.0, 1.0),
                        uncertainty =
                            observationUncertaintyScore(it),
                    )
                }
                .toList()

        if (unique.isEmpty()) {
            return Selection(
                selectedObservations = emptyList(),
                excludedDeviceIds = emptySet(),
                selectedDeviceIds = emptySet(),
                diversitySatisfied = false,
                score = 0.0,
                reason = "no-observations-above-confidence-floor",
            )
        }

        if (unique.size <= config.maximumObservations) {
            unique.forEach {
                it.diversity = if (unique.size == 1) {
                    0.0
                } else {
                    unique
                        .asSequence()
                        .filterNot { other ->
                            other.observation.deviceId ==
                                it.observation.deviceId
                        }
                        .map { other ->
                            pairDiversity(
                                first = it.observation,
                                second = other.observation,
                            )
                        }
                        .maxOrNull()
                        ?: 0.0
                }
            }

            unique.forEach {
                it.score = weightedScore(it)
            }

            return buildSelection(
                candidates = unique,
                selected = unique.toSet(),
            )
        }

        val selected =
            LinkedHashSet<Candidate>()

        val first =
            unique.maxBy {
                initialScore(it)
            }

        first.score = initialScore(first)
        selected += first

        while (selected.size < config.maximumObservations) {
            val next =
                unique
                    .filterNot { selected.contains(it) }
                    .maxByOrNull { candidate ->
                        marginalScore(
                            candidate = candidate,
                            selected = selected,
                        )
                    }
                    ?: break

            val marginal =
                marginalScore(
                    candidate = next,
                    selected = selected,
                )

            if (
                selected.size >=
                    config.minimumObservations &&
                    marginal <
                    config.minimumSelectionScore
            ) {
                break
            }

            next.score = marginal
            selected += next
        }

        if (selected.size < config.minimumObservations) {
            unique
                .filterNot { selected.contains(it) }
                .sortedByDescending {
                    weightedScore(it)
                }
                .take(
                    (
                        config.minimumObservations -
                            selected.size
                        ).coerceAtLeast(0),
                )
                .forEach {
                    it.score = weightedScore(it)
                    selected += it
                }
        }

        return buildSelection(
            candidates = unique,
            selected = selected,
        )
    }

    private fun buildSelection(
        candidates: List<Candidate>,
        selected: Set<Candidate>,
    ): Selection {
        val selectedList =
            candidates
                .filter { selected.contains(it) }
                .sortedByDescending {
                    it.score
                }

        val diversitySatisfied =
            selectedList.size >= 2 &&
                selectedList
                    .pairwiseDiversity()
                    .averageOrZero() >=
                config.minimumDiversityScore

        val finalScore =
            selectedList
                .map {
                    weightedScore(it)
                }
                .averageOrZero()

        val selectedIds =
            selectedList
                .map { it.observation.deviceId }
                .toSet()

        return Selection(
            selectedObservations =
                selectedList.map { it.observation },
            excludedDeviceIds =
                candidates
                    .map { it.observation.deviceId }
                    .filterNot(selectedIds::contains)
                    .toSet(),
            selectedDeviceIds = selectedIds,
            diversitySatisfied = diversitySatisfied,
            score = finalScore.coerceIn(0.0, 1.0),
            reason =
                when {
                    selectedList.isEmpty() ->
                        "no-valid-observation-selection"
                    selectedList.size == 1 ->
                        "single-observation-fallback"
                    diversitySatisfied ->
                        "reliable-diverse-joint-evidence"
                    else ->
                        "minimum-joint-evidence-met-diversity-limited"
                },
        )
    }

    private fun initialScore(
        candidate: Candidate,
    ): Double =
        (
            candidate.reliability * config.reliabilityWeight +
                candidate.timing * config.timingWeight
        ).coerceIn(0.0, 1.0)

    private fun marginalScore(
        candidate: Candidate,
        selected: Collection<Candidate>,
    ): Double {
        val diversity =
            selected
                .map {
                    pairDiversity(
                        first = candidate.observation,
                        second = it.observation,
                    )
                }
                .maxOrNull()
                ?: 0.0

        candidate.diversity = diversity

        return weightedScore(candidate)
    }

    private fun weightedScore(
        candidate: Candidate,
    ): Double {
        val geometricScore =
            (
                candidate.reliability * config.reliabilityWeight +
                    candidate.diversity * config.diversityWeight +
                    candidate.timing * config.timingWeight
            ).coerceIn(0.0, 1.0)

        return (
            geometricScore *
                (1.0 - config.uncertaintyWeight) +
                candidate.uncertainty *
                config.uncertaintyWeight
            ).coerceIn(0.0, 1.0)
    }

    private fun observationBaseScore(
        observation: CameraObservation,
    ): Double =
        (
            observation.confidence.coerceIn(0.0, 1.0) * 0.58 +
                observation.temporalAlignmentConfidence
                    .coerceIn(0.0, 1.0) * 0.24 +
                observationUncertaintyScore(observation) * 0.18
        )

    private fun observationUncertaintyScore(
        observation: CameraObservation,
    ): Double {
        val calibration =
            observation.calibrationUncertainty
        val calibrationScore =
            if (calibration.hasEvidence) {
                val translation =
                    (
                        1.0 -
                            calibration.translationSigmaMeters /
                                0.020
                    ).coerceIn(0.0, 1.0)
                val rotation =
                    (
                        1.0 -
                            calibration.rotationSigmaRadians /
                                0.015
                    ).coerceIn(0.0, 1.0)
                val intrinsic =
                    (
                        1.0 -
                            calibration.intrinsicRelativeSigma /
                                0.02
                    ).coerceIn(0.0, 1.0)

                (
                    translation * 0.48 +
                        rotation * 0.28 +
                        intrinsic * 0.24
                ) *
                    calibration.evidenceConfidence
            } else {
                0.65
            }

        val temporalScore =
            if (observation.temporalUncertaintyNanos > 0L) {
                exp(
                    -observation.temporalUncertaintyNanos.toDouble() /
                        4_000_000.0,
                ).coerceIn(0.0, 1.0)
            } else {
                1.0
            }

        val rollingScore =
            (
                1.0 -
                    observation.rollingShutterFraction
            ).coerceIn(0.0, 1.0)

        return (
            calibrationScore * 0.55 +
                temporalScore * 0.25 +
                rollingScore * 0.20
            ).coerceIn(0.0, 1.0)
    }

    private fun pairDiversity(
        first: CameraObservation,
        second: CameraObservation,
    ): Double {
        val baseline =
            first.pose.position.distanceTo(
                second.pose.position,
            )

        val forwardFirst =
            first.pose.rotation.rotate(
                com.nivukx.nox.core.model.Vec3(
                    0.0,
                    0.0,
                    1.0,
                ),
            ).normalized()

        val forwardSecond =
            second.pose.rotation.rotate(
                com.nivukx.nox.core.model.Vec3(
                    0.0,
                    0.0,
                    1.0,
                ),
            ).normalized()

        val angular =
            acos(
                forwardFirst.dot(
                    forwardSecond,
                ).coerceIn(-1.0, 1.0),
            ) / Math.PI

        val normalizedBaseline =
            (
                baseline /
                    (baseline + 0.35)
                ).coerceIn(0.0, 1.0)

        return (
            normalizedBaseline * 0.64 +
                angular * 0.36
            ).coerceIn(0.0, 1.0)
    }

    private fun List<Candidate>.pairwiseDiversity():
        List<Double> {
        if (size < 2) return emptyList()

        val values = ArrayList<Double>()
        for (i in indices) {
            for (j in (i + 1) until size) {
                values += pairDiversity(
                    first = this[i].observation,
                    second = this[j].observation,
                )
            }
        }
        return values
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}

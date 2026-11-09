package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec3
import kotlin.math.acos
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Deterministic active-view optimizer for a single metric 3D reconstruction target.
 *
 * The optimizer never deletes admissible observations. It orders them so the first passes through
 * robust reconstruction see views that add the most independent geometric information, while all
 * remaining views stay available for consensus and covariance refinement.
 *
 * Signals:
 *  - angular ray novelty (parallax diversity),
 *  - camera-center baseline novelty,
 *  - observation confidence and visibility,
 *  - calibration confidence,
 *  - temporal alignment confidence,
 *  - source quality,
 *  - predicted metric uncertainty.
 *
 * This is deliberately model-independent. It can sit in front of a classical triangulator or a
 * learned downstream reconstructor without changing the underlying measurement contract.
 */
class ActiveViewInformationGainEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumRankedViews: Int = 32,
        val anchorViews: Int = 4,
        val baselineSaturationMeters: Double = 1.5,
        val angularSaturationDegrees: Double = 25.0,
        val triangulationPeakAngleDegrees: Double = 90.0,
        val uncertaintyScaleMeters: Double = 0.020,
        val minimumCandidateScore: Double = 0.08,
        val reliabilityWeight: Double = 0.34,
        val angularNoveltyWeight: Double = 0.28,
        val baselineNoveltyWeight: Double = 0.16,
        val uncertaintyWeight: Double = 0.10,
        val temporalWeight: Double = 0.05,
        val sourceWeight: Double = 0.05,
        val calibrationWeight: Double = 0.07,
        val timingWeight: Double = 0.06,
        val opticalWeight: Double = 0.07,
    ) {
        init {
            require(maximumRankedViews >= 2)
            require(anchorViews >= 2)
            require(baselineSaturationMeters > 0.0)
            require(angularSaturationDegrees > 0.0)
            require(
                triangulationPeakAngleDegrees in 45.0..120.0
            )
            require(uncertaintyScaleMeters > 0.0)
            require(minimumCandidateScore in 0.0..1.0)
            require(reliabilityWeight >= 0.0)
            require(angularNoveltyWeight >= 0.0)
            require(baselineNoveltyWeight >= 0.0)
            require(uncertaintyWeight >= 0.0)
            require(temporalWeight >= 0.0)
            require(sourceWeight >= 0.0)
            require(calibrationWeight >= 0.0)
            require(timingWeight >= 0.0)
            require(opticalWeight >= 0.0)
            require(
                reliabilityWeight +
                    angularNoveltyWeight +
                    baselineNoveltyWeight +
                    uncertaintyWeight +
                    temporalWeight +
                    sourceWeight +
                    calibrationWeight +
                    timingWeight +
                    opticalWeight > 0.0,
            )
        }
    }

    data class Candidate(
        val cameraId: String,
        val pose: Pose,
        val confidence: Double,
        val visibility: Double = 1.0,
        val calibrationConfidence: Double = 1.0,
        val temporalAlignmentConfidence: Double = 1.0,
        val sourceQuality: Double = 1.0,
        val uncertaintyMeters: Double = 0.0,
        val calibrationTranslationUncertaintyMeters: Double = 0.0,
        val calibrationRotationUncertaintyRadians: Double = 0.0,
        val intrinsicRelativeUncertainty: Double = 0.0,
        val timingUncertaintyMillis: Double = 0.0,
        val rollingShutterRisk: Double = 0.0,
        val motionBlurRisk: Double = 0.0,
        val admissible: Boolean = true,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(confidence in 0.0..1.0)
            require(visibility in 0.0..1.0)
            require(calibrationConfidence in 0.0..1.0)
            require(temporalAlignmentConfidence in 0.0..1.0)
            require(sourceQuality in 0.0..1.0)
            require(uncertaintyMeters.isFinite() && uncertaintyMeters >= 0.0)
            require(calibrationTranslationUncertaintyMeters.isFinite() && calibrationTranslationUncertaintyMeters >= 0.0)
            require(calibrationRotationUncertaintyRadians.isFinite() && calibrationRotationUncertaintyRadians >= 0.0)
            require(intrinsicRelativeUncertainty.isFinite() && intrinsicRelativeUncertainty >= 0.0)
            require(timingUncertaintyMillis.isFinite() && timingUncertaintyMillis >= 0.0)
            require(rollingShutterRisk in 0.0..1.0)
            require(motionBlurRisk in 0.0..1.0)
        }
    }

    data class RankedView(
        val originalIndex: Int,
        val cameraId: String,
        val score: Double,
        val reliabilityScore: Double,
        val angularNoveltyScore: Double,
        val baselineNoveltyScore: Double,
        val uncertaintyScore: Double,
        val selectedAsAnchor: Boolean,
    )

    data class Result(
        val orderedIndices: List<Int>,
        val rankedViews: List<RankedView>,
        val anchorIndices: List<Int>,
        val informationGain: Double,
        val redundancyScore: Double,
        val selectedCameraIds: List<String>,
        val warnings: List<String>,
    )

    fun order(
        point: Vec3,
        candidates: List<Candidate>,
    ): Result {
        require(point.isFinite())

        if (candidates.isEmpty()) {
            return Result(
                orderedIndices = emptyList(),
                rankedViews = emptyList(),
                anchorIndices = emptyList(),
                informationGain = 0.0,
                redundancyScore = 0.0,
                selectedCameraIds = emptyList(),
                warnings = listOf("ACTIVE_VIEW_INPUT_EMPTY"),
            )
        }

        val valid = candidates
            .withIndex()
            .filter { it.value.admissible }
            .take(config.maximumRankedViews)

        if (valid.isEmpty()) {
            return Result(
                orderedIndices = emptyList(),
                rankedViews = emptyList(),
                anchorIndices = emptyList(),
                informationGain = 0.0,
                redundancyScore = 0.0,
                selectedCameraIds = emptyList(),
                warnings = listOf("ACTIVE_VIEW_NO_ADMISSIBLE_CANDIDATES"),
            )
        }

        val remaining = valid.toMutableList()
        val selected = mutableListOf<Int>()
        val ranked = mutableListOf<RankedView>()
        var information = 0.0

        while (remaining.isNotEmpty()) {
            val best = remaining.maxWithOrNull(
                compareBy<IndexedValue<Candidate>> { candidate ->
                    score(
                        candidate = candidate.value,
                        point = point,
                        selectedIndices = selected,
                        all = candidates,
                    )
                }.thenByDescending { it.value.confidence }
                    .thenBy { it.value.cameraId },
            ) ?: break

            val candidate = best.value
            val candidateScore =
                score(
                    candidate = candidate,
                    point = point,
                    selectedIndices = selected,
                    all = candidates,
                )

            if (
                selected.isNotEmpty() &&
                candidateScore < config.minimumCandidateScore
            ) {
                break
            }

            selected += best.index
            remaining.remove(best)
            information += candidateScore

            val rankedView =
                buildRankedView(
                    index = best.index,
                    candidate = candidate,
                    point = point,
                    selectedBefore = selected.dropLast(1),
                    selectedAsAnchor = selected.size <= config.anchorViews,
                    all = candidates,
                )
            ranked += rankedView
        }

        val redundancy =
            if (ranked.size <= 1) {
                0.0
            } else {
                (
                    1.0 -
                        (
                            ranked.drop(1).map {
                                1.0 -
                                    it.angularNoveltyScore
                            }.average()
                        )
                    ).coerceIn(0.0, 1.0)
            }

        val warnings = buildList {
            if (valid.size < 2) {
                add("ACTIVE_VIEW_ONLY_ONE_VALID_VIEW")
            }
            if (valid.size >= 2 && selected.size < min(config.anchorViews, valid.size)) {
                add("ACTIVE_VIEW_ANCHOR_COVERAGE_LIMITED")
            }
            if (redundancy > 0.72) {
                add("ACTIVE_VIEW_REMAINING_VIEWS_HIGHLY_REDUNDANT")
            }
            if (valid.size == config.maximumRankedViews) {
                add("ACTIVE_VIEW_INPUT_BOUNDED")
            }
        }

        return Result(
            orderedIndices = selected,
            rankedViews = ranked,
            anchorIndices = selected.take(
                min(config.anchorViews, selected.size),
            ),
            informationGain = (
                information /
                    max(1, selected.size)
            ).coerceIn(0.0, 1.0),
            redundancyScore = redundancy,
            selectedCameraIds = selected.mapNotNull {
                candidates.getOrNull(it)?.cameraId
            },
            warnings = warnings.distinct(),
        )
    }

    private fun buildRankedView(
        index: Int,
        candidate: Candidate,
        point: Vec3,
        selectedBefore: List<Int>,
        selectedAsAnchor: Boolean,
        all: List<Candidate>,
    ): RankedView {
        val reliability = reliability(candidate)
        val angular = angularNovelty(
            candidate = candidate,
            point = point,
            selectedIndices = selectedBefore,
            all = all,
        )
        val baseline = baselineNovelty(
            candidate = candidate,
            point = point,
            selectedIndices = selectedBefore,
            all = all,
        )
        val uncertainty = uncertaintyScore(candidate)
        val calibration = calibrationIntegrity(candidate)
        val timing = timingIntegrity(candidate)
        val optical = opticalIntegrity(candidate)

        val score = weighted(
            reliability = reliability,
            angular = angular,
            baseline = baseline,
            uncertainty = uncertainty,
            temporal = candidate.temporalAlignmentConfidence,
            source = candidate.sourceQuality,
            calibration = calibration,
            timing = timing,
            optical = optical,
        )

        return RankedView(
            originalIndex = index,
            cameraId = candidate.cameraId,
            score = score,
            reliabilityScore = reliability,
            angularNoveltyScore = angular,
            baselineNoveltyScore = baseline,
            uncertaintyScore = uncertainty,
            selectedAsAnchor = selectedAsAnchor,
        )
    }

    private fun score(
        candidate: Candidate,
        point: Vec3,
        selectedIndices: List<Int>,
        all: List<Candidate>,
    ): Double {
        val reliability = reliability(candidate)
        val angular = angularNovelty(
            candidate = candidate,
            point = point,
            selectedIndices = selectedIndices,
            all = all,
        )
        val baseline = baselineNovelty(
            candidate = candidate,
            point = point,
            selectedIndices = selectedIndices,
            all = all,
        )
        val uncertainty = uncertaintyScore(candidate)
        val calibration = calibrationIntegrity(candidate)
        val timing = timingIntegrity(candidate)
        val optical = opticalIntegrity(candidate)

        return weighted(
            reliability = reliability,
            angular = angular,
            baseline = baseline,
            uncertainty = uncertainty,
            temporal = candidate.temporalAlignmentConfidence,
            source = candidate.sourceQuality,
            calibration = calibration,
            timing = timing,
            optical = optical,
        )
    }

    private fun weighted(
        reliability: Double,
        angular: Double,
        baseline: Double,
        uncertainty: Double,
        temporal: Double,
        source: Double,
        calibration: Double,
        timing: Double,
        optical: Double,
    ): Double {
        val total =
            config.reliabilityWeight +
                config.angularNoveltyWeight +
                config.baselineNoveltyWeight +
                config.uncertaintyWeight +
                config.temporalWeight +
                config.sourceWeight +
                config.calibrationWeight +
                config.timingWeight +
                config.opticalWeight

        return (
            reliability * config.reliabilityWeight +
                angular * config.angularNoveltyWeight +
                baseline * config.baselineNoveltyWeight +
                uncertainty * config.uncertaintyWeight +
                temporal * config.temporalWeight +
                source * config.sourceWeight +
                calibration * config.calibrationWeight +
                timing * config.timingWeight +
                optical * config.opticalWeight
            ) / total.coerceAtLeast(1e-9)
    }

    private fun reliability(candidate: Candidate): Double =
        (
            candidate.confidence * 0.46 +
                candidate.visibility * 0.18 +
                candidate.calibrationConfidence * 0.18 +
                candidate.temporalAlignmentConfidence * 0.10 +
                candidate.sourceQuality * 0.08
            ).coerceIn(0.0, 1.0)

    private fun calibrationIntegrity(candidate: Candidate): Double {
        val translation =
            exp(
                -candidate.calibrationTranslationUncertaintyMeters /
                    0.010,
            ).coerceIn(0.0, 1.0)

        val rotation =
            exp(
                -candidate.calibrationRotationUncertaintyRadians /
                    0.010,
            ).coerceIn(0.0, 1.0)

        val intrinsic =
            (
                1.0 /
                    (
                        1.0 +
                            candidate.intrinsicRelativeUncertainty * 6.0
                    )
                ).coerceIn(0.0, 1.0)

        return (
            0.42 * translation +
                0.36 * rotation +
                0.22 * intrinsic
            ).coerceIn(0.0, 1.0)
    }

    private fun timingIntegrity(candidate: Candidate): Double {
        val timing =
            exp(
                -candidate.timingUncertaintyMillis /
                    18.0,
            ).coerceIn(0.0, 1.0)

        val rolling =
            (
                1.0 -
                    0.65 *
                        candidate.rollingShutterRisk
                ).coerceIn(0.15, 1.0)

        return (timing * rolling).coerceIn(0.0, 1.0)
    }

    private fun opticalIntegrity(candidate: Candidate): Double =
        (
            1.0 -
                0.72 *
                    candidate.motionBlurRisk
            ).coerceIn(0.0, 1.0)

    private fun uncertaintyScore(candidate: Candidate): Double {
        if (!candidate.uncertaintyMeters.isFinite()) return 0.0
        return exp(
            -candidate.uncertaintyMeters /
                config.uncertaintyScaleMeters,
        ).coerceIn(0.0, 1.0)
    }

    private fun angularNovelty(
        candidate: Candidate,
        point: Vec3,
        selectedIndices: List<Int>,
        all: List<Candidate>,
    ): Double {
        val ray = viewingRay(candidate, point)
            ?: return 0.0

        if (selectedIndices.isEmpty()) return 1.0

        val minimumAngle = selectedIndices
            .mapNotNull { index ->
                val other = all.getOrNull(index) ?: return@mapNotNull null
                viewingRay(other, point)
                    ?.let { otherRay ->
                        angleDegrees(ray, otherRay)
                    }
            }
            .minOrNull()
            ?: config.angularSaturationDegrees

        val noveltyRamp =
            (
                minimumAngle /
                    config.angularSaturationDegrees
            ).coerceIn(0.0, 1.0)

        val conditioning =
            abs(
                kotlin.math.sin(
                    Math.toRadians(
                        minimumAngle.coerceIn(0.0, 180.0),
                    ),
                ),
            )

        val peakPenalty =
            abs(
                minimumAngle -
                    config.triangulationPeakAngleDegrees,
            ) /
                90.0

        return (
            0.62 * conditioning +
                0.30 * noveltyRamp +
                0.08 *
                    (
                        1.0 -
                            peakPenalty.coerceIn(0.0, 1.0)
                    )
        ).coerceIn(0.0, 1.0)
    }

    private fun baselineNovelty(
        candidate: Candidate,
        point: Vec3,
        selectedIndices: List<Int>,
        all: List<Candidate>,
    ): Double {
        if (selectedIndices.isEmpty()) return 1.0

        val minimumBaseline =
            selectedIndices
                .mapNotNull { index ->
                    all.getOrNull(index)
                        ?.pose
                        ?.position
                        ?.distanceTo(candidate.pose.position)
                }
                .minOrNull()
                ?: config.baselineSaturationMeters

        return (
            minimumBaseline /
                config.baselineSaturationMeters
            ).coerceIn(0.0, 1.0)
    }

    private fun viewingRay(
        candidate: Candidate,
        point: Vec3,
    ): Vec3? {
        val ray =
            point -
                candidate.pose.position
        if (!ray.x.isFinite() || !ray.y.isFinite() || !ray.z.isFinite()) {
            return null
        }
        return ray.normalized()
            .takeIf { it.norm() > 1e-9 }
    }

    private fun angleDegrees(
        a: Vec3,
        b: Vec3,
    ): Double {
        val cosine =
            a.dot(b) /
                (
                    a.norm() *
                        b.norm()
                    ).coerceAtLeast(1e-12)
        return Math.toDegrees(
            acos(
                cosine.coerceIn(-1.0, 1.0),
            ),
        )
    }

    private fun Vec3.isFinite(): Boolean =
        x.isFinite() &&
            y.isFinite() &&
            z.isFinite()
}

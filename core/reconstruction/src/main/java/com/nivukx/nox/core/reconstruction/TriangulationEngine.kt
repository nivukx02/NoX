
package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Production multi-view point solver.
 *
 * Stage 1 uses the calibrated ray intersection to get a metric seed.
 * Stage 2 performs robust non-linear reprojection refinement with a damped Gauss-Newton solve.
 * Stage 3 performs an additional cross-view reprojection-consensus refinement pass, inspired by
 * recent sparse-RGB metric reconstruction research.
 * Stage 4 computes explicit geometry/cheirality/uncertainty diagnostics.
 *
 * The result is accepted only when the observations have enough parallax, a positive camera-depth
 * ratio, and a finite reprojection solution. This avoids turning degenerate camera geometry into
 * apparently stable 3D motion.
 */
class TriangulationEngine(
    private val minParallaxDegrees: Double = 0.75,
    private val huberDeltaPx: Double = 3.0,
    private val cauchyScalePx: Double = 6.0,
    private val iterations: Int = 10,
    private val damping: Double = 1e-4,
    private val numericJacobianStepMeters: Double = 1e-4,
    private val minimumTemporalConfidence: Double = 0.10,
    private val maximumTemporalUncertaintyNanos: Long = 40_000_000L,
    private val maximumDominantViewWeightRatio: Double = 0.82,
    private val reprojectionConsensusScalePx: Double = 2.5,
    private val minimumConsensusWeight: Double = 0.15,
    private val consensusRefinementIterations: Int = 3,
    private val maximumMotionCompensationMeters: Double = 0.20,
    private val directionalUncertaintyEngine:
        TriangulationUncertaintyEngine =
        TriangulationUncertaintyEngine(),
    private val activeViewInformationGainEngine:
        ActiveViewInformationGainEngine =
        ActiveViewInformationGainEngine(),
    private val symmetricEpipolarConsistencyEngine:
        SymmetricEpipolarConsistencyEngine =
        SymmetricEpipolarConsistencyEngine(),
    private val minimumEpipolarEvidenceScore: Double = 0.18,
    private val maximumNormalMatrixConditionNumber: Double = 1.0e10,
) {
    data class PrecisionTuning(
        val tailResidualQuantile: Double = 0.95,
        val residualInflation: Double = 1.15,
        val minimumPixelNoisePx: Double = 0.25,
    ) {
        init {
            require(tailResidualQuantile in 0.50..0.999)
            require(residualInflation in 1.0..3.0)
            require(minimumPixelNoisePx in 0.05..2.0)
        }
    }

    @Volatile
    private var precisionTuning = PrecisionTuning()

    init {
        require(minimumTemporalConfidence in 0.0..1.0)
        require(maximumTemporalUncertaintyNanos > 0L)
        require(reprojectionConsensusScalePx > 0.0)
        require(minimumConsensusWeight in 0.0..1.0)
        require(consensusRefinementIterations in 0..8)
        require(maximumMotionCompensationMeters > 0.0)
        require(
            maximumNormalMatrixConditionNumber.isFinite() &&
                maximumNormalMatrixConditionNumber > 1.0,
        )
    }

    @Synchronized
    fun configurePrecisionTuning(tuning: PrecisionTuning) {
        precisionTuning = tuning
    }

    fun precisionTuning(): PrecisionTuning = precisionTuning

    /**
     * A bounded spatial probability mode for a 2D keypoint.
     *
     * Upstream detectors may provide multiple plausible locations instead of collapsing a
     * multimodal response map to a single peak. The triangulator can then score 3D candidates
     * against the preserved probability mass. Empty mode lists retain legacy point-only behavior.
     */
    data class ProbabilityMode(
        val u: Double,
        val v: Double,
        val sigmaX: Double,
        val sigmaY: Double,
        val correlation: Double = 0.0,
        val weight: Double = 1.0,
    ) {
        init {
            require(u.isFinite() && v.isFinite())
            require(sigmaX.isFinite() && sigmaX > 0.05)
            require(sigmaY.isFinite() && sigmaY > 0.05)
            require(correlation.isFinite() && correlation in -0.98..0.98)
            require(weight.isFinite() && weight >= 0.0)
        }
    }

    data class Observation(
        val cameraPose: Pose,
        val intrinsics: CameraIntrinsics,
        val cameraId: String = "unknown",
        val u: Double,
        val v: Double,
        val weight: Double = 1.0,
        val rollingShutterFraction: Double = 0.0,
        val temporalAlignmentConfidence: Double = 1.0,
        val temporalUncertaintyNanos: Long = 0L,
        val rowTimeOffsetNanos: Long = 0L,
        val observationUncertainty: Double = 0.0,
        val visibility: Double = 1.0,
        val sourceQuality: Double = 1.0,
        val crossViewAgreement: Double = 1.0,
        val temporalStability: Double = 1.0,
        val contactProbability: Double = 0.0,
        val probabilityModes: List<ProbabilityMode> = emptyList(),
        val distributionCalibrationScore: Double = 1.0,
        val calibrationUncertainty:
            CalibrationUncertaintyProfile = CalibrationUncertaintyProfile(),
        /**
         * Optical integration blur estimate in [0, 1]. Zero means no explicit blur estimate
         * was supplied by the source.
         */
        val motionBlurRisk: Double = 0.0,
        val motionVelocityMetersPerSecond: Vec3 =
            Vec3(0.0, 0.0, 0.0),
    ) {
        init {
            require(temporalAlignmentConfidence in 0.0..1.0)
            require(temporalUncertaintyNanos >= 0L)
            require(observationUncertainty in 0.0..1.0)
            require(visibility in 0.0..1.0)
            require(sourceQuality in 0.0..1.0)
            require(crossViewAgreement in 0.0..1.0)
            require(temporalStability in 0.0..1.0)
            require(contactProbability in 0.0..1.0)
            require(probabilityModes.size <= 12)
            require(distributionCalibrationScore in 0.0..1.0)
            require(motionBlurRisk in 0.0..1.0)
            require(
                motionVelocityMetersPerSecond.x.isFinite() &&
                    motionVelocityMetersPerSecond.y.isFinite() &&
                    motionVelocityMetersPerSecond.z.isFinite(),
            )
        }

        val effectiveTemporalUncertaintyNanos: Long
            get() {
                val row =
                    if (rowTimeOffsetNanos == Long.MIN_VALUE) {
                        Long.MAX_VALUE
                    } else {
                        abs(rowTimeOffsetNanos)
                    }

                return if (
                    temporalUncertaintyNanos >
                        Long.MAX_VALUE - row
                ) {
                    Long.MAX_VALUE
                } else {
                    temporalUncertaintyNanos + row
                }
            }
        constructor(value: CameraObservation) : this(
            cameraPose = value.pose,
            intrinsics = value.intrinsics,
            cameraId = value.deviceId.value,
            u = value.point.x,
            v = value.point.y,
            weight = value.confidence,
            rollingShutterFraction = value.rollingShutterFraction,
            temporalAlignmentConfidence =
                value.temporalAlignmentConfidence,
            temporalUncertaintyNanos =
                value.temporalUncertaintyNanos,
            rowTimeOffsetNanos =
                value.rowTimeOffsetNanos,
            calibrationUncertainty =
                value.calibrationUncertainty,
            motionBlurRisk =
                value.estimatedMotionBlurRisk,
            motionVelocityMetersPerSecond =
                Vec3(0.0, 0.0, 0.0),
        )
    }

    data class Result(
        val point: Vec3,
        val rmsReprojectionPx: Double,
        val medianReprojectionPx: Double,
        val inliers: Int,
        val observations: Int,
        val parallaxDegrees: Double,
        val cheiralityRatio: Double = 1.0,
        val uncertaintyMeters: Double = 0.0,
        val minDepthMeters: Double = 0.0,
        val maxDepthMeters: Double = 0.0,
        val conditionScore: Double = 0.0,
        val robustCost: Double = 0.0,
        val uncertaintyMajorMeters: Double = 0.0,
        val uncertaintyMiddleMeters: Double = 0.0,
        val uncertaintyMinorMeters: Double = 0.0,
        val uncertaintyDepthMeters: Double = 0.0,
        val uncertainty95MajorMeters: Double = 0.0,
        val uncertaintyAnisotropyRatio: Double = 1.0,
        val uncertaintyDepthDominance: Double = 0.0,
        val uncertaintyConfidence: Double = 1.0,
        val uncertaintyWarnings: List<String> = emptyList(),
        val independentViewCount: Int = 0,
        val dominantViewWeightRatio: Double = 0.0,
        val viewDiversityScore: Double = 0.0,
        val viewDiversityWarnings: List<String> = emptyList(),
        val evidenceCoverage: Double = 1.0,
        val temporalStability: Double = 1.0,
        val contactSupport: Double = 0.0,
        val pairwiseReprojectionConsensus: Double = 1.0,
        val pairwiseConsensusMinimum: Double = 1.0,
        val pairwiseConsensusWarnings: List<String> = emptyList(),
        val consensusRefinementIterationsAccepted: Int = 0,
        val effectiveBaselineMeters: Double = 0.0,
        val firstOrderDepthSensitivityMetersPerPixel: Double =
            Double.POSITIVE_INFINITY,
        val firstOrderDepthErrorMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val reprojectionP95Px: Double = Double.POSITIVE_INFINITY,
        val tailAdjustedFirstOrderDepthErrorMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val tailAdjustedSubMillimeterFeasibilityScore: Double = 0.0,
        val subMillimeterFeasibilityScore: Double = 0.0,
        val subMillimeterWarnings: List<String> = emptyList(),
        val probabilisticEvidenceScore: Double = 1.0,
        val probabilisticEvidenceWarnings: List<String> = emptyList(),
        val calibrationUncertaintyMeters: Double = 0.0,
        val calibrationUncertaintyApplied: Boolean = false,
        val activeViewInformationGainScore: Double = 0.0,
        val activeViewRedundancyScore: Double = 0.0,
        /**
         * Conservative metric evidence authority for this 3D point.
         * This is an evidence score, not a claimed positional accuracy.
         */
        val metricEvidenceAuthority: Double = 0.0,
        val symmetricEpipolarConsistencyScore: Double = 0.0,
        val symmetricEpipolarEvaluatedPairCount: Int = 0,
        val symmetricEpipolarWarnings: List<String> = emptyList(),
        val activeViewAnchorCameraIds: List<String> = emptyList(),
        val activeViewWarnings: List<String> = emptyList(),
        val normalMatrixConditionEstimate: Double = 1.0,
        val normalMatrixConditionScore: Double = 1.0,
    )

    fun triangulate(observations: List<Observation>): Vec3? =
        robust(observations)?.point

    fun robust(observations: List<Observation>): Result? {
        val valid = observations.filter {
            it.weight > 1e-4 &&
                it.visibility >= 0.05 &&
                it.sourceQuality >= 0.05 &&
                it.crossViewAgreement >= 0.05 &&
                it.temporalStability >= 0.05 &&
                it.observationUncertainty <= 0.99 &&
                it.temporalAlignmentConfidence >=
                    minimumTemporalConfidence &&
                it.effectiveTemporalUncertaintyNanos <=
                    maximumTemporalUncertaintyNanos &&
                it.intrinsics.fx > 1e-6 &&
                it.intrinsics.fy > 1e-6
        }

        if (valid.size < 2) return null

        val independent = selectIndependentViews(valid)
        if (independent.size < 2) return null

        val epipolarEvidence =
            symmetricEpipolarEvidence(independent)

        val viewWeights =
            independent.mapIndexed { index, observation ->
                val support =
                    epipolarEvidence.perObservationSupport
                        .getOrNull(index)
                        ?.coerceIn(0.0, 1.0)
                        ?: 1.0
                effectiveWeight(observation) *
                    (
                        0.55 +
                            0.45 * support
                        )
            }
        val totalViewWeight =
            viewWeights.sum().coerceAtLeast(1e-9)
        val dominantViewWeightRatio =
            viewWeights.maxOrNull()?.div(totalViewWeight) ?: 1.0

        val knownCameraIds =
            independent.map { it.cameraId }
                .filter { it.isNotBlank() && it != "unknown" }
                .toSet()

        val viewDiversityScore =
            when {
                knownCameraIds.isEmpty() ->
                    (independent.size / 4.0).coerceIn(0.0, 1.0)
                else ->
                    (
                        knownCameraIds.size / 4.0
                    ).coerceIn(0.0, 1.0) *
                        (
                            1.0 -
                                (
                                    dominantViewWeightRatio -
                                        0.50
                                    ).coerceAtLeast(0.0)
                                )
                            .coerceIn(0.0, 1.0)
            }

        if (
            epipolarEvidence.evaluatedPairs >= 1 &&
            epipolarEvidence.score <
                minimumEpipolarEvidenceScore
        ) {
            return null
        }

        val viewDiversityWarnings = buildList {
            if (knownCameraIds.size >= 2 &&
                dominantViewWeightRatio > maximumDominantViewWeightRatio
            ) {
                add("one camera dominates the metric triangulation weight")
            }
            if (knownCameraIds.size == 2 &&
                dominantViewWeightRatio > 0.70
            ) {
                add("two-view solve is strongly asymmetric")
            }
            if (knownCameraIds.size >= 3 &&
                knownCameraIds.size < 4
            ) {
                add("metric solve has limited independent-view redundancy")
            }
        }

        if (knownCameraIds.size >= 2 &&
            dominantViewWeightRatio > maximumDominantViewWeightRatio
        ) {
            return null
        }

        val normalMatrixConditionEstimate =
            estimateRayNormalMatrixCondition(
                observations = independent,
                weights = viewWeights,
            )
        val normalMatrixConditionScore =
            conditionScore(normalMatrixConditionEstimate)

        if (
            !normalMatrixConditionEstimate.isFinite() ||
            normalMatrixConditionEstimate >
                maximumNormalMatrixConditionNumber
        ) {
            return null
        }

        val parallax = maxParallax(independent)
        if (!parallax.isFinite() || parallax < minParallaxDegrees) {
            return null
        }

        var point = solveRayIntersection(
            independent,
            viewWeights,
        ) ?: return null

        val activeViewOrdering =
            activeViewInformationGainEngine.order(
                point = point,
                candidates = independent.map {
                    ActiveViewInformationGainEngine.Candidate(
                        cameraId =
                            it.cameraId.ifBlank {
                                "unknown-" + independent.indexOf(it)
                            },
                        pose = it.cameraPose,
                        confidence =
                            effectiveWeight(it)
                                .coerceIn(0.0, 1.0),
                        visibility =
                            it.visibility.coerceIn(0.0, 1.0),
                        calibrationConfidence =
                            it.calibrationUncertainty.evidenceConfidence
                                .takeIf { confidence -> confidence > 0.0 }
                                ?: it.sourceQuality.coerceIn(0.0, 1.0),
                        temporalAlignmentConfidence =
                            it.temporalAlignmentConfidence
                                .coerceIn(0.0, 1.0),
                        sourceQuality =
                            it.sourceQuality.coerceIn(0.0, 1.0),
                        uncertaintyMeters =
                            if (it.observationUncertainty.isFinite()) {
                                it.observationUncertainty
                            } else {
                                1.0
                            },
                        calibrationTranslationUncertaintyMeters =
                            it.calibrationUncertainty
                                .translationSigmaMeters
                                .coerceAtLeast(0.0),
                        calibrationRotationUncertaintyRadians =
                            it.calibrationUncertainty
                                .rotationSigmaRadians
                                .coerceAtLeast(0.0),
                        intrinsicRelativeUncertainty =
                            it.calibrationUncertainty
                                .intrinsicRelativeSigma
                                .coerceAtLeast(0.0),
                        timingUncertaintyMillis =
                            it.effectiveTemporalUncertaintyNanos
                                .coerceAtLeast(0L)
                                .toDouble() / 1_000_000.0,
                        rollingShutterRisk =
                            abs(it.rollingShutterFraction)
                                .coerceIn(0.0, 1.0),
                        motionBlurRisk =
                            it.motionBlurRisk.coerceIn(0.0, 1.0),
                    )
                },
            )
        val processingIndependent =
            activeViewOrdering.orderedIndices.mapNotNull {
                independent.getOrNull(it)
            }.ifEmpty {
                independent
            }

        var bestPoint = point
        var bestCost = robustCost(point, independent)
        var accepted = 0
        var adaptiveDamping =
            damping.coerceIn(1e-9, 1e2)

        repeat(iterations) {
            val normal = Array(3) { DoubleArray(3) }
            val gradient = DoubleArray(3)

            processingIndependent.forEachIndexed { index, observation ->
                val projected = project(point, observation) ?: return@forEachIndexed
                val residualX = observation.u - projected.x
                val residualY = observation.v - projected.y
                val error = sqrt(
                    residualX * residualX +
                        residualY * residualY,
                )

                val robustWeight = robustWeight(
                    errorPx = error,
                    deltaPx = huberDeltaPx,
                    cauchyScalePx = cauchyScalePx,
                )

                val rowWeight =
                    effectiveWeight(observation) *
                        probabilisticResidualWeight(
                            observation = observation,
                            projectedU = projected.x,
                            projectedV = projected.y,
                        ) *
                        robustWeight.coerceIn(0.02, 1.0)

                val jacobian = numericJacobian(
                    point = point,
                    observation = observation,
                )

                accumulateNormalEquation(
                    normal = normal,
                    gradient = gradient,
                    jacobian = jacobian,
                    residualX = residualX,
                    residualY = residualY,
                    weight = rowWeight,
                )

            }

            val dampingScale =
                (
                    1.0 +
                        bestCost.coerceAtLeast(0.0) * 0.08
                ).coerceIn(1.0, 8.0)
            for (axis in 0..2) {
                normal[axis][axis] +=
                    adaptiveDamping * dampingScale
            }

            val delta =
                solve3(normal, gradient)
                    ?: run {
                        adaptiveDamping =
                            (
                                adaptiveDamping * 4.0
                            ).coerceAtMost(1e2)
                        return@repeat
                    }
            val stepNorm = delta.norm()

            if (!stepNorm.isFinite() || stepNorm < 1e-8) {
                accepted = max(accepted, 1)
                return@repeat
            }

            val clampedStep = delta * (
                min(1.0, 0.18 / stepNorm.coerceAtLeast(1e-9))
            )

            val candidate = point + clampedStep
            if (!candidate.isFinite()) return@repeat

            val candidateCost = robustCost(candidate, independent)
            val cheirality = positiveDepthRatio(candidate, independent)

            if (
                candidateCost.isFinite() &&
                candidateCost <= bestCost &&
                cheirality >= 0.5
            ) {
                val improvement =
                    (
                        bestCost -
                            candidateCost
                    ) /
                    bestCost.coerceAtLeast(1e-9)

                point = candidate
                bestPoint = candidate
                bestCost = candidateCost
                accepted += 1

                adaptiveDamping =
                    if (improvement >= 0.10) {
                        (
                            adaptiveDamping * 0.55
                        ).coerceAtLeast(1e-9)
                    } else {
                        (
                            adaptiveDamping * 0.82
                        ).coerceAtLeast(1e-9)
                    }
            } else {
                adaptiveDamping =
                    (
                        adaptiveDamping * 2.25
                    ).coerceAtMost(1e2)
            }
        }

        var consensusIterationsAccepted = 0

        repeat(consensusRefinementIterations) {
            val currentResiduals =
                processingIndependent.map { reprojectionError(point, it) }

            val consensusWeights =
                pairwiseConsensusWeights(currentResiduals)

            if (consensusWeights.size != independent.size) {
                return@repeat
            }

            val normal = Array(3) { DoubleArray(3) }
            val gradient = DoubleArray(3)

            processingIndependent.forEachIndexed { index, observation ->
                val projected =
                    project(point, observation)
                        ?: return@forEachIndexed

                val residualX =
                    observation.u - projected.x
                val residualY =
                    observation.v - projected.y
                val error =
                    sqrt(
                        residualX * residualX +
                            residualY * residualY,
                    )

                val baseWeight =
                    effectiveWeight(observation) *
                        consensusWeights[index]
                    .coerceAtLeast(minimumConsensusWeight)

                val rowWeight =
                    baseWeight *
                        probabilisticResidualWeight(
                            observation = observation,
                            projectedU = projected.x,
                            projectedV = projected.y,
                        ) *
                        robustWeight(
                            errorPx = error,
                            deltaPx = huberDeltaPx,
                            cauchyScalePx = cauchyScalePx,
                        ).coerceIn(0.02, 1.0)

                val jacobian =
                    numericJacobian(
                        point = point,
                        observation = observation,
                    )

                accumulateNormalEquation(
                    normal = normal,
                    gradient = gradient,
                    jacobian = jacobian,
                    residualX = residualX,
                    residualY = residualY,
                    weight = rowWeight,
                )
            }

            val consensusDampingScale =
                (
                    1.0 +
                        bestCost.coerceAtLeast(0.0) * 0.05
                ).coerceIn(1.0, 6.0)
            for (axis in 0..2) {
                normal[axis][axis] +=
                    adaptiveDamping *
                        consensusDampingScale
            }

            val delta =
                solve3(normal, gradient)
                    ?: run {
                        adaptiveDamping =
                            (
                                adaptiveDamping * 3.0
                            ).coerceAtMost(1e2)
                        return@repeat
                    }

            val stepNorm = delta.norm()
            if (!stepNorm.isFinite() || stepNorm < 1e-8) {
                return@repeat
            }

            val clampedStep =
                delta *
                    min(
                        1.0,
                        0.12 /
                            stepNorm.coerceAtLeast(1e-9),
                    )

            val candidate = point + clampedStep
            if (!candidate.isFinite()) return@repeat

            val candidateCost =
                robustCost(
                    candidate,
                    independent,
                )
            val cheirality =
                positiveDepthRatio(
                    candidate,
                    independent,
                )

            if (
                candidateCost <= bestCost &&
                cheirality >= 0.5
            ) {
                point = candidate
                bestPoint = candidate
                bestCost = candidateCost
                consensusIterationsAccepted++
            }
        }

        val residualVector =
            processingIndependent.map {
                reprojectionError(bestPoint, it)
            }

        val residuals =
            residualVector
                .filter { it.isFinite() }
                .sorted()

        if (residuals.size < 2) return null

        val rms = sqrt(
            residuals.sumOf { it * it } /
                residuals.size.coerceAtLeast(1),
        )
        val reprojectionP95 =
            percentile(
                residuals,
                precisionTuning.tailResidualQuantile,
            )
        val median = residuals[residuals.size / 2]
        val cheirality = positiveDepthRatio(bestPoint, independent)

        if (cheirality < 0.75) return null

        val pairwiseConsensus =
            pairwiseConsensusSummary(
                residualVector,
            )

        val uncertainty = estimateUncertaintyMeters(
            point = bestPoint,
            observations = processingIndependent,
            rmsPx = rms,
            parallaxDegrees = parallax,
        )

        val depths = processingIndependent.mapNotNull {
            cameraDepth(bestPoint, it)
                .takeIf(Double::isFinite)
                ?.takeIf { depth -> depth > 0.0 }
        }

        val geometricConditionScore = geometryConditionScore(
            observations = processingIndependent,
            parallaxDegrees = parallax,
            point = bestPoint,
        )
        val conditionScore =
            (
                geometricConditionScore *
                    normalMatrixConditionScore
            ).coerceIn(0.0, 1.0)

        val precisionEvidence =
            firstOrderPrecisionEvidence(
                point = bestPoint,
                observations = processingIndependent,
                rmsReprojectionPx = rms,
                reprojectionP95Px = reprojectionP95,
                depthsMeters = depths,
                tuning = precisionTuning,
            )

        val probabilisticEvidence =
            probabilisticEvidence(
                point = bestPoint,
                observations = processingIndependent,
            )

        val probabilisticUncertaintyInflation =
            (
                1.0 +
                    (
                        1.0 -
                            probabilisticEvidence.score
                    ) * 0.35
                ).coerceIn(1.0, 1.35)

        val uncertaintyAfterProbability =
            uncertainty *
                probabilisticUncertaintyInflation

        val directionalUncertainty =
            directionalUncertaintyEngine.estimate(
                TriangulationUncertaintyEngine.Input(
                    point = bestPoint,
                    observations = processingIndependent.map {
                        TriangulationUncertaintyEngine.Observation(
                            cameraPose = it.cameraPose,
                            intrinsics = it.intrinsics,
                            u = it.u,
                            v = it.v,
                            weight = it.weight,
                            confidence =
                                it.weight.coerceIn(0.0, 1.0),
                            temporalAlignmentConfidence =
                                it.temporalAlignmentConfidence,
                            calibrationUncertainty =
                                it.calibrationUncertainty,
                        )
                    },
                    residualsPx = residuals,
                    robustResidualScalePx =
                        (
                            rms /
                                (residuals.size + 1).toDouble().sqrtSafe()
                            ).coerceAtLeast(0.25),
                ),
            )

        val reprojectionAuthority =
            (
                1.0 -
                    min(
                        1.0,
                        reprojectionP95 /
                            8.0,
                    )
                ).coerceIn(0.0, 1.0)
        val uncertaintyAuthority =
            (
                1.0 -
                    min(
                        1.0,
                        uncertaintyAfterProbability /
                            0.10,
                    )
                ).coerceIn(0.0, 1.0)
        val calibrationAuthority =
            (
                1.0 /
                    (
                        1.0 +
                            directionalUncertainty
                                .calibrationUncertaintyMeters
                                .coerceAtLeast(0.0) /
                            0.005
                    )
                ).coerceIn(0.0, 1.0)
        val opticalEvidenceAuthority =
            processingIndependent
                .map { observation ->
                    (
                        1.0 -
                            observation.motionBlurRisk
                                .coerceIn(0.0, 1.0)
                    ) *
                        (
                            1.0 -
                                0.72 *
                                    abs(
                                        observation.rollingShutterFraction,
                                    ).coerceIn(0.0, 1.0)
                        )
                }
                .average()
                .coerceIn(0.0, 1.0)

        val metricEvidenceAuthority =
            weightedGeometricMeanEvidence(
                values =
                    listOf(
                        conditionScore to 0.17,
                        reprojectionAuthority to 0.14,
                        uncertaintyAuthority to 0.14,
                        calibrationAuthority to 0.09,
                        cheirality to 0.07,
                        viewDiversityScore to 0.09,
                        pairwiseConsensus.score to 0.07,
                        epipolarEvidence.score to 0.10,
                        probabilisticEvidence.score to 0.06,
                        opticalEvidenceAuthority to 0.07,
                    ),
            )

        return Result(
            point = bestPoint,
            rmsReprojectionPx = rms,
            medianReprojectionPx = median,
            inliers = residuals.count {
                it <= huberDeltaPx * 2.5
            },
            observations = processingIndependent.size,
            parallaxDegrees = parallax,
            cheiralityRatio = cheirality,
            uncertaintyMeters =
                uncertaintyAfterProbability,
            minDepthMeters = depths.minOrNull() ?: 0.0,
            maxDepthMeters = depths.maxOrNull() ?: 0.0,
            conditionScore = conditionScore,
            robustCost = bestCost,
            uncertaintyMajorMeters =
                directionalUncertainty.sigmaMajorMeters,
            uncertaintyMiddleMeters =
                directionalUncertainty.sigmaMiddleMeters,
            uncertaintyMinorMeters =
                directionalUncertainty.sigmaMinorMeters,
            uncertaintyDepthMeters =
                directionalUncertainty.sigmaDepthMeters,
            uncertainty95MajorMeters =
                directionalUncertainty.uncertainty95MajorMeters *
                    probabilisticUncertaintyInflation,
            uncertaintyAnisotropyRatio =
                directionalUncertainty.anisotropyRatio,
            uncertaintyDepthDominance =
                directionalUncertainty.depthDominance,
            uncertaintyConfidence =
                directionalUncertainty.confidence,
            uncertaintyWarnings =
                (
                    directionalUncertainty.warnings +
                        probabilisticEvidence.warnings
                    ).distinct(),
            independentViewCount =
                knownCameraIds.size.takeIf { it > 0 }
                    ?: processingIndependent.size,
            dominantViewWeightRatio =
                dominantViewWeightRatio.coerceIn(0.0, 1.0),
            viewDiversityScore =
                viewDiversityScore,
            viewDiversityWarnings =
                viewDiversityWarnings,
            evidenceCoverage =
                (
                    processingIndependent.sumOf {
                        effectiveWeight(it)
                    } /
                        processingIndependent.size.toDouble().coerceAtLeast(1.0)
                    ).coerceIn(0.0, 1.0),
            temporalStability =
                processingIndependent.map {
                    it.temporalStability
                }.average().coerceIn(0.0, 1.0),
            contactSupport =
                processingIndependent.map {
                    it.contactProbability
                }.average().coerceIn(0.0, 1.0),
            pairwiseReprojectionConsensus =
                pairwiseConsensus.score,
            pairwiseConsensusMinimum =
                pairwiseConsensus.minimum,
            pairwiseConsensusWarnings =
                pairwiseConsensus.warnings,
            consensusRefinementIterationsAccepted =
                consensusIterationsAccepted,
            effectiveBaselineMeters =
                precisionEvidence.effectiveBaselineMeters,
            firstOrderDepthSensitivityMetersPerPixel =
                precisionEvidence.depthSensitivityMetersPerPixel,
            firstOrderDepthErrorMillimeters =
                precisionEvidence.firstOrderErrorMillimeters,
            reprojectionP95Px = reprojectionP95,
            tailAdjustedFirstOrderDepthErrorMillimeters =
                precisionEvidence.tailAdjustedFirstOrderErrorMillimeters,
            tailAdjustedSubMillimeterFeasibilityScore =
                precisionEvidence.tailAdjustedFeasibilityScore,
            subMillimeterFeasibilityScore =
                precisionEvidence.feasibilityScore,
            subMillimeterWarnings =
                precisionEvidence.warnings,
            probabilisticEvidenceScore =
                probabilisticEvidence.score,
            probabilisticEvidenceWarnings =
                probabilisticEvidence.warnings,
            calibrationUncertaintyMeters =
                directionalUncertainty.calibrationUncertaintyMeters,
            calibrationUncertaintyApplied =
                directionalUncertainty.calibrationUncertaintyApplied,
            activeViewInformationGainScore =
                activeViewOrdering.informationGain,
            activeViewRedundancyScore =
                activeViewOrdering.redundancyScore,
            activeViewAnchorCameraIds =
                activeViewOrdering.rankedViews
                    .filter { it.selectedAsAnchor }
                    .map { it.cameraId },
            activeViewWarnings =
                activeViewOrdering.warnings,
            metricEvidenceAuthority =
                metricEvidenceAuthority,
            opticalEvidenceAuthority =
                opticalEvidenceAuthority,
            symmetricEpipolarConsistencyScore =
                epipolarEvidence.score,
            symmetricEpipolarEvaluatedPairCount =
                epipolarEvidence.evaluatedPairs,
            symmetricEpipolarWarnings =
                epipolarEvidence.warnings,
            normalMatrixConditionEstimate =
                normalMatrixConditionEstimate,
            normalMatrixConditionScore =
                normalMatrixConditionScore,
        )
    }

    private fun weightedGeometricMeanEvidence(
        values: List<Pair<Double, Double>>,
    ): Double {
        if (values.isEmpty()) return 0.0
        var weightedLog = 0.0
        var totalWeight = 0.0
        values.forEach { (rawValue, rawWeight) ->
            if (!rawValue.isFinite() ||
                !rawWeight.isFinite() ||
                rawWeight <= 0.0
            ) {
                return@forEach
            }
            weightedLog +=
                rawWeight *
                    kotlin.math.ln(
                        rawValue.coerceIn(
                            1.0e-6,
                            1.0,
                        ),
                    )
            totalWeight += rawWeight
        }
        if (totalWeight <= 0.0) return 0.0
        return kotlin.math.exp(
            weightedLog /
                totalWeight,
        ).coerceIn(0.0, 1.0)
    }

    private data class SymmetricEpipolarEvidence(
        val score: Double,
        val evaluatedPairs: Int,
        val perObservationSupport: List<Double>,
        val warnings: List<String>,
    )

    /**
     * Bounded two-view geometry preflight.
     *
     * The strongest weighted view is used as a deterministic hub and compared with the next
     * highest-weight independent views. Every comparison is symmetric in ray geometry, while the
     * bounded fan keeps dense 128-camera rigs from turning a single landmark into an O(N²) hot path.
     */
    private fun symmetricEpipolarEvidence(
        observations: List<Observation>,
    ): SymmetricEpipolarEvidence {
        if (observations.size < 2) {
            return SymmetricEpipolarEvidence(
                score = 1.0,
                evaluatedPairs = 0,
                perObservationSupport =
                    List(observations.size) { 1.0 },
                warnings = emptyList(),
            )
        }

        val order =
            observations.indices.sortedByDescending {
                effectiveWeight(observations[it])
            }
        val hubIndex = order.first()
        val hub = observations[hubIndex]
        val boundedTargets =
            order
                .drop(1)
                .take(31)

        val supports =
            DoubleArray(observations.size) { 1.0 }
        var sum = 0.0
        var evaluated = 0

        boundedTargets.forEach { targetIndex ->
            val target = observations[targetIndex]
            val pair =
                runCatching {
                    symmetricEpipolarConsistencyEngine.evaluate(
                        left =
                            SymmetricEpipolarConsistencyEngine.Observation(
                                cameraPose = hub.cameraPose,
                                intrinsics = hub.intrinsics,
                                u = hub.u,
                                v = hub.v,
                                confidence =
                                    effectiveWeight(hub)
                                        .coerceIn(0.0, 1.0),
                            ),
                        right =
                            SymmetricEpipolarConsistencyEngine.Observation(
                                cameraPose = target.cameraPose,
                                intrinsics = target.intrinsics,
                                u = target.u,
                                v = target.v,
                                confidence =
                                    effectiveWeight(target)
                                        .coerceIn(0.0, 1.0),
                            ),
                    )
                }.getOrNull()
                ?: return@forEach

            val pairScore =
                pair.score.coerceIn(0.0, 1.0)
            supports[targetIndex] =
                minOf(
                    supports[targetIndex],
                    pairScore,
                )
            supports[hubIndex] =
                minOf(
                    supports[hubIndex],
                    pairScore,
                )
            sum += pairScore
            evaluated += 1
        }

        val score =
            if (evaluated == 0) {
                1.0
            } else {
                (sum / evaluated.toDouble())
                    .coerceIn(0.0, 1.0)
            }

        val warnings =
            buildList {
                if (evaluated > 0 && score < 0.45) {
                    add("symmetric epipolar evidence degraded")
                }
                if (evaluated > 0 && score < 0.30) {
                    add("symmetric epipolar evidence strongly degraded")
                }
                if (evaluated > 0 && score < minimumEpipolarEvidenceScore) {
                    add("symmetric epipolar evidence rejected")
                }
                if (observations.size > 32) {
                    add("symmetric epipolar fan bounded to 31 target comparisons")
                }
            }

        return SymmetricEpipolarEvidence(
            score = score,
            evaluatedPairs = evaluated,
            perObservationSupport =
                supports.toList(),
            warnings = warnings.distinct(),
        )
    }

    private data class PrecisionEvidence(
        val effectiveBaselineMeters: Double,
        val depthSensitivityMetersPerPixel: Double,
        val firstOrderErrorMillimeters: Double,
        val feasibilityScore: Double,
        val tailAdjustedFirstOrderErrorMillimeters: Double,
        val tailAdjustedFeasibilityScore: Double,
        val calibrationEvidenceCoverage: Double = 0.0,
        val warnings: List<String>,
    )

    private data class ProbabilisticEvidence(
        val score: Double,
        val warnings: List<String>,
    )

    private fun probabilisticEvidence(
        point: Vec3,
        observations: List<Observation>,
    ): ProbabilisticEvidence {
        val distributions =
            observations.filter {
                it.probabilityModes.isNotEmpty()
            }

        if (distributions.isEmpty()) {
            return ProbabilisticEvidence(
                score = 1.0,
                warnings = emptyList(),
            )
        }

        val scores =
            distributions.map { observation ->
                val projected =
                    project(
                        point,
                        observation,
                    ) ?: return@map 0.0

                val densities =
                    observation.probabilityModes.map {
                        gaussianDensity(
                            x = projected.x,
                            y = projected.y,
                            mode = it,
                        )
                    }

                val weighted =
                    densities.zip(
                        observation.probabilityModes,
                    ).sumOf { (density, mode) ->
                        density *
                            mode.weight.coerceAtLeast(0.0)
                    }

                val peak =
                    densities.maxOrNull()
                        ?.coerceAtLeast(1e-12)
                        ?: 1e-12

                (
                    (weighted / peak).coerceIn(0.0, 1.0) * 0.72 +
                        observation.distributionCalibrationScore * 0.28
                ).coerceIn(0.0, 1.0)
            }

        val score =
            scores.average()
                .coerceIn(0.0, 1.0)

        val warnings =
            buildList {
                if (score < 0.45) {
                    add(
                        "2d-probability-mass-disagrees-with-reprojection",
                    )
                }
                if (distributions.any {
                        it.probabilityModes.size > 1
                    }) {
                    add(
                        "multimodal-2d-keypoint-evidence-preserved",
                    )
                }
                if (distributions.any {
                        it.distributionCalibrationScore < 0.50
                    }) {
                    add(
                        "2d-probability-calibration-is-weak",
                    )
                }
            }.distinct()

        return ProbabilisticEvidence(
            score = score,
            warnings = warnings,
        )
    }

    private fun firstOrderPrecisionEvidence(
        point: Vec3,
        observations: List<Observation>,
        rmsReprojectionPx: Double,
        reprojectionP95Px: Double,
        depthsMeters: List<Double>,
        tuning: PrecisionTuning,
    ): PrecisionEvidence {
        if (observations.size < 2 || depthsMeters.isEmpty()) {
            return PrecisionEvidence(
                effectiveBaselineMeters = 0.0,
                depthSensitivityMetersPerPixel =
                    Double.POSITIVE_INFINITY,
                firstOrderErrorMillimeters =
                    Double.POSITIVE_INFINITY,
                feasibilityScore = 0.0,
                tailAdjustedFirstOrderErrorMillimeters =
                    Double.POSITIVE_INFINITY,
                tailAdjustedFeasibilityScore = 0.0,
                calibrationEvidenceCoverage = 0.0,
                warnings = listOf(
                    "submillimeter-geometry-evidence-insufficient",
                ),
            )
        }

        var maximumBaseline = 0.0
        for (i in 0 until observations.lastIndex) {
            for (j in (i + 1) until observations.size) {
                maximumBaseline =
                    max(
                        maximumBaseline,
                        observations[i].cameraPose.position.distanceTo(
                            observations[j].cameraPose.position,
                        ),
                    )
            }
        }

        val depth =
            medianFinite(depthsMeters)
                .coerceAtLeast(1e-6)

        val focal =
            observations
                .map {
                    sqrt(
                        (
                            it.intrinsics.fx *
                                it.intrinsics.fy
                            ).coerceAtLeast(1e-9),
                    )
                }
                .filter(Double::isFinite)
                .minOrNull()
                ?.coerceAtLeast(1.0)
                ?: 0.0

        if (
            !maximumBaseline.isFinite() ||
            maximumBaseline <= 1e-6 ||
            !focal.isFinite() ||
            focal <= 1.0
        ) {
            return PrecisionEvidence(
                effectiveBaselineMeters =
                    maximumBaseline.coerceAtLeast(0.0),
                depthSensitivityMetersPerPixel =
                    Double.POSITIVE_INFINITY,
                firstOrderErrorMillimeters =
                    Double.POSITIVE_INFINITY,
                feasibilityScore = 0.0,
                tailAdjustedFirstOrderErrorMillimeters =
                    Double.POSITIVE_INFINITY,
                tailAdjustedFeasibilityScore = 0.0,
                warnings = listOf(
                    "submillimeter-baseline-or-focal-evidence-insufficient",
                ),
            )
        }

        // Stereo/SfM first-order depth sensitivity:
        // Z ≈ fB/d  ->  dZ ≈ Z²/(fB) * dd.
        // Pixel residuals are combined with the measured/declared camera-calibration envelope.
        // Calibration error is correlated across views and therefore is not divided by sqrt(N).
        val sensitivity =
            depth * depth /
                (focal * maximumBaseline)
        val pixelEvidence =
            max(
                rmsReprojectionPx,
                tuning.minimumPixelNoisePx,
            )
        val tailPixelEvidence =
            max(
                max(
                    rmsReprojectionPx,
                    reprojectionP95Px * tuning.residualInflation,
                ),
                tuning.minimumPixelNoisePx,
            )
        val firstOrderErrorMm =
            sensitivity *
                pixelEvidence *
                1000.0
        val tailAdjustedFirstOrderErrorMm =
            sensitivity *
                tailPixelEvidence *
                1000.0

        val calibrationSigmaMeters =
            calibrationUncertaintyEnvelopeMeters(
                point = point,
                observations = observations,
            )

        val calibrationEvidenceCoverage =
            observations.count {
                it.calibrationUncertainty.hasEvidence
            }.toDouble() /
                observations.size.coerceAtLeast(1).toDouble()
            .coerceIn(0.0, 1.0)

        val calibrationSigmaEquivalentPx =
            if (sensitivity <= 1e-12) {
                Double.POSITIVE_INFINITY
            } else {
                calibrationSigmaMeters /
                    sensitivity
            }

        val effectiveFirstOrderErrorMeters =
            sqrt(
                (
                    firstOrderErrorMm / 1000.0
                ) * (
                    firstOrderErrorMm / 1000.0
                ) +
                    calibrationSigmaMeters *
                        calibrationSigmaMeters
            )

        val effectiveTailFirstOrderErrorMeters =
            sqrt(
                (
                    tailAdjustedFirstOrderErrorMm / 1000.0
                ) * (
                    tailAdjustedFirstOrderErrorMm / 1000.0
                ) +
                    calibrationSigmaMeters *
                        calibrationSigmaMeters
            )

        val effectiveFirstOrderErrorMm =
            effectiveFirstOrderErrorMeters * 1000.0

        val effectiveTailFirstOrderErrorMm =
            effectiveTailFirstOrderErrorMeters * 1000.0

        val targetMeters = 0.001
        val requiredBaseline =
            depth * depth *
                pixelEvidence /
                (focal * targetMeters)

        val baselineScore =
            (
                maximumBaseline /
                    requiredBaseline.coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)

        val calibrationScore =
            if (calibrationEvidenceCoverage <= 0.0) {
                0.54
            } else {
                (
                    targetMeters /
                        calibrationSigmaMeters
                            .coerceAtLeast(1e-9)
                    ).coerceIn(0.0, 1.0)
            } *
                (
                    0.54 +
                        0.46 * calibrationEvidenceCoverage
                )
                .coerceIn(0.0, 1.0)

        val residualScore =
            (
                targetMeters /
                    effectiveFirstOrderErrorMeters
                        .coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)

        val tailResidualScore =
            (
                targetMeters /
                    effectiveTailFirstOrderErrorMeters
                        .coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)

        val parallaxSupport =
            (
                1.0 -
                    kotlin.math.exp(
                        -maxParallax(observations) /
                            1.75,
                    )
            ).coerceIn(0.0, 1.0)

        val feasibility =
            (
                baselineScore * 0.34 +
                    residualScore * 0.36 +
                    calibrationScore * 0.15 +
                    parallaxSupport * 0.15
                ).coerceIn(0.0, 1.0)
        val tailAdjustedFeasibility =
            (
                baselineScore * 0.34 +
                    tailResidualScore * 0.36 +
                    calibrationScore * 0.15 +
                    parallaxSupport * 0.15
                ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (firstOrderErrorMm > 1.0) {
                add(
                    "first-order-depth-error-exceeds-one-millimeter",
                )
            }
            if (maximumBaseline < requiredBaseline) {
                add(
                    "baseline-insufficient-for-one-millimeter-first-order-target",
                )
            }
            if (rmsReprojectionPx > 1.0) {
                add(
                    "reprojection-residual-limits-submillimeter-feasibility",
                )
            }
            if (reprojectionP95Px > max(rmsReprojectionPx * 1.5, 1.0)) {
                add(
                    "reprojection-tail-limits-submillimeter-feasibility",
                )
            }
            if (effectiveFirstOrderErrorMm > 1.0) {
                add(
                    "calibration-aware-first-order-depth-error-exceeds-one-millimeter",
                )
            }
            if (effectiveTailFirstOrderErrorMm > 1.0) {
                add(
                    "calibration-aware-tail-adjusted-depth-error-exceeds-one-millimeter",
                )
            }
            if (calibrationEvidenceCoverage < 1.0) {
                add(
                    "calibration-uncertainty-evidence-incomplete-across-views",
                )
            }
            if (calibrationEvidenceCoverage == 0.0) {
                add(
                    "calibration-uncertainty-profile-unavailable; feasibility-remains-provisional",
                )
            }
            if (calibrationSigmaMeters > 0.001) {
                add(
                    "calibration-uncertainty-alone-exceeds-one-millimeter-target",
                )
            }
            if (calibrationSigmaEquivalentPx > 0.0 &&
                calibrationSigmaEquivalentPx.isFinite() &&
                calibrationSigmaEquivalentPx > tailPixelEvidence
            ) {
                add(
                    "camera-calibration-envelope-dominates-pixel-error-budget",
                )
            }
            if (depth > 5.0) {
                add(
                    "depth-range-amplifies-first-order-depth-sensitivity",
                )
            }
            if (observations.size < 3) {
                add(
                    "two-view-geometry-has-limited-redundancy-for-submillimeter-evidence",
                )
            }
            add(
                "submillimeter-feasibility-is-a-first-order-geometry-estimate-not-a-measured-accuracy-claim",
            )
        }

        return PrecisionEvidence(
            effectiveBaselineMeters = maximumBaseline,
            depthSensitivityMetersPerPixel = sensitivity,
            firstOrderErrorMillimeters = firstOrderErrorMm,
            feasibilityScore = feasibility,
            tailAdjustedFirstOrderErrorMillimeters = effectiveTailFirstOrderErrorMm,
            tailAdjustedFeasibilityScore = tailAdjustedFeasibility,
            warnings = warnings.distinct(),
        )
    }

    private fun calibrationUncertaintyEnvelopeMeters(
        point: Vec3,
        observations: List<Observation>,
    ): Double {
        return observations.maxOfOrNull { observation ->
            val profile = observation.calibrationUncertainty
            if (!profile.hasEvidence) {
                0.0
            } else {
                val range =
                    (
                        point -
                            observation.cameraPose.position
                        ).norm()
                        .coerceAtLeast(0.0)
                sqrt(
                    profile.translationSigmaMeters *
                        profile.translationSigmaMeters +
                        (
                            range *
                                profile.rotationSigmaRadians
                        ) * (
                            range *
                                profile.rotationSigmaRadians
                            ) +
                        (
                            range *
                                profile.intrinsicRelativeSigma
                        ) * (
                            range *
                                profile.intrinsicRelativeSigma
                            ),
                )
            }
        }?.coerceAtMost(3.0) ?: 0.0
    }

    private fun percentile(
        values: List<Double>,
        quantile: Double,
    ): Double {
        if (values.isEmpty()) return Double.POSITIVE_INFINITY
        if (values.size == 1) return values.first()
        val position = quantile.coerceIn(0.0, 1.0) * (values.size - 1).toDouble()
        val lower = position.toInt().coerceIn(0, values.lastIndex)
        val upper = (lower + 1).coerceAtMost(values.lastIndex)
        val fraction = position - lower.toDouble()
        return values[lower] + (values[upper] - values[lower]) * fraction
    }

    private fun medianFinite(values: List<Double>): Double {
        val finite = values.filter(Double::isFinite).sorted()
        if (finite.isEmpty()) return Double.POSITIVE_INFINITY
        val middle = finite.size / 2
        return if (finite.size % 2 == 0) {
            (finite[middle - 1] + finite[middle]) * 0.5
        } else {
            finite[middle]
        }
    }

    private data class PairwiseConsensusSummary(
        val score: Double,
        val minimum: Double,
        val warnings: List<String>,
    )

    private fun pairwiseConsensusWeights(
        residuals: List<Double>,
    ): List<Double> {
        if (residuals.isEmpty()) return emptyList()
        if (residuals.size == 1) return listOf(1.0)

        return residuals.mapIndexed { index, residual ->
            if (!residual.isFinite()) {
                minimumConsensusWeight
            } else {
                val pairwise =
                    residuals.indices
                        .asSequence()
                        .filter { it != index }
                        .map { other ->
                            residuals[other]
                                .takeIf(Double::isFinite)
                                ?.let {
                                    exp(
                                        -abs(residual - it) /
                                            reprojectionConsensusScalePx
                                                .coerceAtLeast(1e-6),
                                    )
                                }
                                ?: minimumConsensusWeight
                        }
                        .average()

                max(
                    minimumConsensusWeight,
                    pairwise.coerceIn(0.0, 1.0),
                )
            }
        }
    }

    private fun pairwiseConsensusSummary(
        residuals: List<Double>,
    ): PairwiseConsensusSummary {
        val weights =
            pairwiseConsensusWeights(residuals)

        if (weights.isEmpty()) {
            return PairwiseConsensusSummary(
                score = 0.0,
                minimum = 0.0,
                warnings = listOf("pairwise-consensus-unavailable"),
            )
        }

        val finiteResiduals =
            residuals.filter(Double::isFinite)

        val score =
            weights.average().coerceIn(0.0, 1.0)

        val minimum =
            weights.minOrNull()
                ?.coerceIn(0.0, 1.0)
                ?: 0.0

        val spread =
            if (finiteResiduals.size < 2) {
                0.0
            } else {
                val center =
                    finiteResiduals.sorted()[
                        finiteResiduals.size / 2
                    ]
                finiteResiduals.map {
                    abs(it - center)
                }.average()
            }

        val warnings = buildList {
            if (score < 0.55) {
                add("cross-view-reprojection-consensus-low")
            }
            if (minimum < 0.35) {
                add("one-or-more-views-disagree-with-reprojection-consensus")
            }
            if (spread > reprojectionConsensusScalePx * 2.0) {
                add("reprojection-consensus-residual-spread-high")
            }
        }

        return PairwiseConsensusSummary(
            score = score,
            minimum = minimum,
            warnings = warnings.distinct(),
        )
    }

    private fun selectIndependentViews(
        observations: List<Observation>,
    ): List<Observation> {
        if (
            observations.isEmpty() ||
            observations.all {
                it.cameraId.isBlank() || it.cameraId == "unknown"
            }
        ) {
            return observations
        }

        val grouped = observations
            .groupBy { observation ->
                observation.cameraId
                    .takeIf { it.isNotBlank() && it != "unknown" }
                    ?: "unknown@" + observations.indexOf(observation)
            }

        return grouped.values.mapNotNull { candidates ->
            candidates.maxByOrNull(::effectiveWeight)
        }
    }

    private fun effectiveWeight(o: Observation): Double {
        val temporalUncertaintyRatio =
            (
                o.effectiveTemporalUncertaintyNanos.toDouble() /
                    maximumTemporalUncertaintyNanos.toDouble()
                ).coerceIn(0.0, 1.0)

        val temporalUncertaintyWeight =
            (
                1.0 -
                    0.55 * temporalUncertaintyRatio
                ).coerceIn(0.35, 1.0)

        val uncertaintyWeight =
            (
                1.0 -
                    0.72 * o.observationUncertainty
                ).coerceIn(0.28, 1.0)

        val visibilityWeight =
            (
                0.35 +
                    0.65 * o.visibility
                ).coerceIn(0.35, 1.0)

        val sourceQualityWeight =
            (
                0.40 +
                    0.60 * o.sourceQuality
                ).coerceIn(0.40, 1.0)

        val crossViewWeight =
            (
                0.35 +
                    0.65 * o.crossViewAgreement
                ).coerceIn(0.35, 1.0)

        val temporalStabilityWeight =
            (
                0.35 +
                    0.65 * o.temporalStability
                ).coerceIn(0.35, 1.0)

        val calibrationEvidenceWeight =
            (
                0.82 +
                    0.18 *
                        o.calibrationUncertainty.evidenceConfidence
                            .coerceIn(0.0, 1.0)
                ).coerceIn(0.82, 1.0)

        val calibrationDriftRisk =
            calibrationDriftRisk(
                o.calibrationUncertainty,
            )

        val calibrationDriftWeight =
            (
                1.0 -
                    0.70 * calibrationDriftRisk
                ).coerceIn(0.30, 1.0)

        val motionBlurWeight =
            (
                1.0 -
                    0.72 *
                        o.motionBlurRisk
                ).coerceIn(0.22, 1.0)

        val distributionCalibrationWeight =
            (
                0.40 +
                    0.60 * o.distributionCalibrationScore
                ).coerceIn(0.40, 1.0)

        val probabilitySpreadWeight =
            if (o.probabilityModes.isEmpty()) {
                1.0
            } else {
                val meanSigma =
                    o.probabilityModes
                        .map { it.sigmaX + it.sigmaY }
                        .average()
                        .coerceAtLeast(0.10)

                (
                    1.0 /
                        (
                            1.0 +
                                meanSigma / 12.0
                        )
                    ).coerceIn(0.55, 1.0)
            }

        val contactSupportWeight =
            (
                1.0 +
                    0.08 * o.contactProbability
                ).coerceIn(1.0, 1.08)

        return (
            o.weight.coerceIn(1e-4, 1.0) *
                o.temporalAlignmentConfidence.coerceIn(0.0, 1.0) *
                temporalUncertaintyWeight *
                uncertaintyWeight *
                visibilityWeight *
                sourceQualityWeight *
                crossViewWeight *
                temporalStabilityWeight *
                calibrationEvidenceWeight *
                calibrationDriftWeight *
                motionBlurWeight *
                distributionCalibrationWeight *
                probabilitySpreadWeight *
                contactSupportWeight *
                (1.0 - 0.5 * abs(o.rollingShutterFraction))
                    .coerceIn(0.25, 1.0)
            ).coerceIn(1e-4, 1.0)
    }

    private fun calibrationDriftRisk(
        profile: CalibrationUncertaintyProfile,
    ): Double {
        if (!profile.hasEvidence) return 0.0

        val translationRisk =
            (
                profile.translationSigmaMeters / 0.010
            ).coerceIn(0.0, 1.0)

        val rotationRisk =
            (
                profile.rotationSigmaRadians /
                    Math.toRadians(0.50)
            ).coerceIn(0.0, 1.0)

        val intrinsicRisk =
            (
                profile.intrinsicRelativeSigma / 0.010
            ).coerceIn(0.0, 1.0)

        return (
            0.50 * translationRisk +
                0.35 * rotationRisk +
                0.15 * intrinsicRisk
        ).coerceIn(0.0, 1.0)
    }

$marker
        observation: Observation,
        projectedU: Double,
        projectedV: Double,
    ): Double {
        val modes = observation.probabilityModes
        if (modes.isEmpty()) return 1.0

        val densities = modes.map {
            gaussianDensity(
                x = projectedU,
                y = projectedV,
                mode = it,
            )
        }

        val weighted =
            densities.zip(modes).sumOf { (density, mode) ->
                density * mode.weight.coerceAtLeast(0.0)
            }

        val peak =
            densities.maxOrNull()
                ?.coerceAtLeast(1e-12)
                ?: return 0.45

        val normalized =
            (weighted / peak).coerceIn(0.0, 1.0)

        return (
            0.45 +
                0.55 *
                    normalized *
                    (
                        0.50 +
                            0.50 *
                                observation.distributionCalibrationScore
                    )
        ).coerceIn(0.20, 1.0)
    }

    private fun gaussianDensity(
        x: Double,
        y: Double,
        mode: ProbabilityMode,
    ): Double {
        val dx = x - mode.u
        val dy = y - mode.v
        val sx2 = mode.sigmaX * mode.sigmaX
        val sy2 = mode.sigmaY * mode.sigmaY
        val rho = mode.correlation.coerceIn(-0.98, 0.98)
        val oneMinusRho2 =
            (1.0 - rho * rho).coerceAtLeast(1e-4)

        val quadratic =
            (
                dx * dx / sx2 +
                    dy * dy / sy2 -
                    2.0 * rho * dx * dy /
                        (
                            mode.sigmaX *
                                mode.sigmaY
                            ).coerceAtLeast(1e-6)
                ) / oneMinusRho2

        val determinantRoot =
            mode.sigmaX *
                mode.sigmaY *
                sqrt(oneMinusRho2)

        val normalization =
            1.0 /
                (
                    2.0 *
                        Math.PI *
                        determinantRoot.coerceAtLeast(1e-6)
                    )

        return (
            normalization *
                exp(-0.5 * quadratic)
        ).takeIf(Double::isFinite)
            ?.coerceAtLeast(0.0)
            ?: 0.0
    }

    private fun robustWeight(
        errorPx: Double,
        deltaPx: Double,
        cauchyScalePx: Double,
    ): Double {
        val finiteError =
            errorPx.takeIf { it.isFinite() }
                ?.coerceAtLeast(0.0)
                ?: Double.POSITIVE_INFINITY

        val safeDelta =
            deltaPx.coerceAtLeast(1e-6)
        val safeCauchy =
            cauchyScalePx.coerceAtLeast(1e-6)

        val huber =
            if (finiteError <= safeDelta) {
                1.0
            } else {
                safeDelta /
                    finiteError.coerceAtLeast(safeDelta)
            }

        val normalized =
            finiteError / safeCauchy

        val cauchy =
            1.0 /
                (1.0 + normalized * normalized)

        // Tukey-style redescending tail. It preserves the high-quality core of a
        // correspondence cloud while sharply reducing leverage from grossly mismatched views.
        val tukeyRadius =
            (finiteError / (2.5 * safeDelta))
                .coerceAtLeast(0.0)
        val tukey =
            if (tukeyRadius < 1.0) {
                val oneMinus =
                    1.0 - tukeyRadius * tukeyRadius
                oneMinus * oneMinus
            } else {
                0.0
            }

        // Retain a non-zero floor so an observation can recover after a temporary
        // correspondence failure instead of becoming permanently dead weight.
        val tailFloor =
            if (finiteError.isFinite()) {
                0.005
            } else {
                0.0
            }

        val combined =
            huber *
                cauchy *
                max(tukey, tailFloor)

        return combined.coerceIn(
            0.0,
            1.0,
        )
    }

    private fun robustCost(
        point: Vec3,
        observations: List<Observation>,
    ): Double =
        observations.sumOf { observation ->
            val error = reprojectionError(point, observation)
            val delta = huberDeltaPx.coerceAtLeast(1e-6)
            val normalized =
                error / cauchyScalePx.coerceAtLeast(1e-6)
            val cauchyLoss =
                cauchyScalePx.coerceAtLeast(1e-6) *
                    cauchyScalePx.coerceAtLeast(1e-6) *
                    kotlin.math.ln(1.0 + normalized * normalized)
            val quadratic = min(
                error * error,
                2.0 * delta * error - delta * delta,
            )
            val robustLoss =
                0.70 * quadratic +
                    0.30 * cauchyLoss
            effectiveWeight(observation) * robustLoss
        }

    private fun numericJacobian(
        point: Vec3,
        observation: Observation,
    ): Array<DoubleArray> {
        projectionJacobian(
            point = point,
            observation = observation,
        )?.let { return it }

        val step = numericJacobianStepMeters.coerceIn(1e-6, 1e-2)

        fun derivative(axis: Int): Pair<Vec2, Vec2> {
            val plus = when (axis) {
                0 -> point + Vec3(step, 0.0, 0.0)
                1 -> point + Vec3(0.0, step, 0.0)
                else -> point + Vec3(0.0, 0.0, step)
            }
            val minus = when (axis) {
                0 -> point - Vec3(step, 0.0, 0.0)
                1 -> point - Vec3(0.0, step, 0.0)
                else -> point - Vec3(0.0, 0.0, step)
            }

            val plusProjection =
                project(plus, observation)
                    ?: Vec2(observation.u, observation.v)

            val minusProjection =
                project(minus, observation)
                    ?: Vec2(observation.u, observation.v)

            return (
                (plusProjection - minusProjection) * (0.5 / step)
            ) to Vec2(0.0, 0.0)
        }

        val dx = derivative(0).first
        val dy = derivative(1).first
        val dz = derivative(2).first

        return arrayOf(
            doubleArrayOf(dx.x, dy.x, dz.x),
            doubleArrayOf(dx.y, dy.y, dz.y),
        )
    }

    /**
     * Exact local projection Jacobian for the calibrated Brown-Conrady camera model.
     *
     * The derivative is taken with respect to world-space XYZ. Camera translation has no
     * derivative term; the inverse camera rotation maps each world basis vector into camera space.
     * The method is intentionally finite-checked and returns null for degenerate depth so the
     * numerical fallback remains available for unusual calibration states.
     */
    private fun projectionJacobian(
        point: Vec3,
        observation: Observation,
    ): Array<DoubleArray>? {
        val inverse =
            observation.cameraPose.rotation
                .conjugate()
                .normalized()

        val cameraPoint =
            inverse.rotate(
                point - observation.cameraPose.position,
            )

        val z = cameraPoint.z
        if (!z.isFinite() || z <= 1e-6) return null

        val x = cameraPoint.x / z
        val y = cameraPoint.y / z
        if (!x.isFinite() || !y.isFinite()) return null

        val r2 = x * x + y * y
        val r4 = r2 * r2
        val r6 = r4 * r2
        val k1 = observation.intrinsics.k1
        val k2 = observation.intrinsics.k2
        val k3 = observation.intrinsics.k3
        val p1 = observation.intrinsics.p1
        val p2 = observation.intrinsics.p2

        val radial =
            1.0 +
                k1 * r2 +
                k2 * r4 +
                k3 * r6
        if (!radial.isFinite()) return null

        val radialSlope =
            k1 +
                2.0 * k2 * r2 +
                3.0 * k3 * r4
        val dRadialDx = 2.0 * x * radialSlope
        val dRadialDy = 2.0 * y * radialSlope

        // Brown-Conrady tangential terms:
        // tx = 2 p1 x y + p2 (3 x² + y²)
        // ty = p1 (x² + 3 y²) + 2 p2 x y
        val tx =
            2.0 * p1 * x * y +
                p2 * (3.0 * x * x + y * y)
        val ty =
            p1 * (x * x + 3.0 * y * y) +
                2.0 * p2 * x * y

        val dTxDx = 2.0 * p1 * y + 6.0 * p2 * x
        val dTxDy = 2.0 * p1 * x + 2.0 * p2 * y
        val dTyDx = 2.0 * p1 * x + 2.0 * p2 * y
        val dTyDy = 6.0 * p1 * y + 2.0 * p2 * x

        val distortedX = x * radial + tx
        val distortedY = y * radial + ty
        if (!distortedX.isFinite() || !distortedY.isFinite()) return null

        val dDistortedXDx =
            radial +
                x * dRadialDx +
                dTxDx
        val dDistortedXDy =
            x * dRadialDy +
                dTxDy
        val dDistortedYDx =
            y * dRadialDx +
                dTyDx
        val dDistortedYDy =
            radial +
                y * dRadialDy +
                dTyDy

        val invZ = 1.0 / z
        val invZ2 = invZ * invZ

        val worldToCameraX = inverse.rotate(
            Vec3(1.0, 0.0, 0.0),
        )
        val worldToCameraY = inverse.rotate(
            Vec3(0.0, 1.0, 0.0),
        )
        val worldToCameraZ = inverse.rotate(
            Vec3(0.0, 0.0, 1.0),
        )

        fun normalizedDerivative(
            dCamera: Vec3,
        ): Vec2 =
            Vec2(
                (dCamera.x * z - cameraPoint.x * dCamera.z) *
                    invZ2,
                (dCamera.y * z - cameraPoint.y * dCamera.z) *
                    invZ2,
            )

        val dNormX = listOf(
            normalizedDerivative(worldToCameraX),
            normalizedDerivative(worldToCameraY),
            normalizedDerivative(worldToCameraZ),
        )

        val duDx =
            dDistortedXDx * dNormX[0].x +
                dDistortedXDy * dNormX[0].y
        val duDy =
            dDistortedXDx * dNormX[1].x +
                dDistortedXDy * dNormX[1].y
        val duDz =
            dDistortedXDx * dNormX[2].x +
                dDistortedXDy * dNormX[2].y

        val dvDx =
            dDistortedYDx * dNormX[0].x +
                dDistortedYDy * dNormX[0].y
        val dvDy =
            dDistortedYDx * dNormX[1].x +
                dDistortedYDy * dNormX[1].y
        val dvDz =
            dDistortedYDx * dNormX[2].x +
                dDistortedYDy * dNormX[2].y

        val jacobian = arrayOf(
            doubleArrayOf(
                observation.intrinsics.fx * duDx,
                observation.intrinsics.fx * duDy,
                observation.intrinsics.fx * duDz,
            ),
            doubleArrayOf(
                observation.intrinsics.fy * dvDx,
                observation.intrinsics.fy * dvDy,
                observation.intrinsics.fy * dvDz,
            ),
        )

        return if (
            jacobian.all { row ->
                row.all { it.isFinite() }
            }
        ) {
            jacobian
        } else {
            null
        }
    }

    private fun accumulateNormalEquation(
        normal: Array<DoubleArray>,
        gradient: DoubleArray,
        jacobian: Array<DoubleArray>,
        residualX: Double,
        residualY: Double,
        weight: Double,
    ) {
        val residual = doubleArrayOf(residualX, residualY)

        for (r in 0..2) {
            for (c in 0..2) {
                normal[r][c] += weight * (
                    jacobian[0][r] * jacobian[0][c] +
                        jacobian[1][r] * jacobian[1][c]
                    )
            }

            gradient[r] += weight * (
                jacobian[0][r] * residual[0] +
                    jacobian[1][r] * residual[1]
                )
        }
    }

    private fun maxParallax(observations: List<Observation>): Double {
        var best = 0.0

        for (i in 0 until observations.lastIndex) {
            val a = ray(observations[i]).direction

            for (j in i + 1 until observations.size) {
                val b = ray(observations[j]).direction
                val radians =
                    acos(a.dot(b).coerceIn(-1.0, 1.0))

                best = max(
                    best,
                    radians * 180.0 / Math.PI,
                )
            }
        }

        return best
    }

    private data class Ray(
        val origin: Vec3,
        val direction: Vec3,
    )

    private fun ray(o: Observation): Ray {
        val normalized =
            undistort(o.u, o.v, o.intrinsics)

        val cameraRay =
            Vec3(
                normalized.x,
                normalized.y,
                1.0,
            ).normalized()

        val motionDelta = motionCompensationMeters(o)
        return Ray(
            origin = o.cameraPose.position - motionDelta,
            direction = o.cameraPose.rotation
                .rotate(cameraRay)
                .normalized(),
        )
    }

    private fun undistort(
        u: Double,
        v: Double,
        k: CameraIntrinsics,
    ): Vec2 {
        val xd = (u - k.cx) / k.fx
        val yd = (v - k.cy) / k.fy

        var x = xd
        var y = yd

        repeat(10) {
            val r2 = x * x + y * y
            val radial =
                1.0 +
                    k.k1 * r2 +
                    k.k2 * r2 * r2 +
                    k.k3 * r2 * r2 * r2

            val tx =
                2.0 * k.p1 * x * y +
                    k.p2 * (r2 + 2.0 * x * x)

            val ty =
                k.p1 * (r2 + 2.0 * y * y) +
                    2.0 * k.p2 * x * y

            if (abs(radial) > 1e-9) {
                x = (xd - tx) / radial
                y = (yd - ty) / radial
            }
        }

        return Vec2(x, y)
    }

    private fun project(
        point: Vec3,
        o: Observation,
    ): Vec2? {
        val inverse =
            o.cameraPose.rotation
                .conjugate()
                .normalized()

        val cameraPoint =
            inverse.rotate(
                temporalObservationPoint(point, o) -
                    o.cameraPose.position,
            )

        if (cameraPoint.z <= 1e-6) return null

        var x = cameraPoint.x / cameraPoint.z
        var y = cameraPoint.y / cameraPoint.z

        val r2 = x * x + y * y
        val radial =
            1.0 +
                o.intrinsics.k1 * r2 +
                o.intrinsics.k2 * r2 * r2 +
                o.intrinsics.k3 * r2 * r2 * r2

        val dx =
            2.0 * o.intrinsics.p1 * x * y +
                o.intrinsics.p2 *
                (r2 + 2.0 * x * x)

        val dy =
            o.intrinsics.p1 *
                (r2 + 2.0 * y * y) +
                2.0 * o.intrinsics.p2 * x * y

        x = x * radial + dx
        y = y * radial + dy

        return Vec2(
            x * o.intrinsics.fx + o.intrinsics.cx,
            y * o.intrinsics.fy + o.intrinsics.cy,
        )
    }

    private fun motionCompensationMeters(
        observation: Observation,
    ): Vec3 {
        val dtSeconds =
            observation.rowTimeOffsetNanos
                .coerceIn(
                    -40_000_000L,
                    40_000_000L,
                )
                .toDouble() / 1e9
        return (
            observation.motionVelocityMetersPerSecond *
                dtSeconds
            ).clampNorm(
                maximumMotionCompensationMeters,
            )
    }

    private fun temporalObservationPoint(
        point: Vec3,
        observation: Observation,
    ): Vec3 =
        point +
            motionCompensationMeters(observation)

    private fun reprojectionError(
        point: Vec3,
        observation: Observation,
    ): Double {
        val projected =
            project(point, observation)
                ?: return 1e6

        return (
            projected -
                Vec2(
                    observation.u,
                    observation.v,
                )
            ).norm()
    }

    private fun Vec3.clampNorm(maximum: Double): Vec3 {
        val magnitude = norm()
        return if (!magnitude.isFinite() || magnitude <= maximum || magnitude < 1e-12) {
            this
        } else {
            this * (maximum / magnitude)
        }
    }

    private fun solveRayIntersection(
        observations: List<Observation>,
        weights: List<Double>,
    ): Vec3? {
        val a = Array(3) { DoubleArray(3) }
        val b = DoubleArray(3)

        observations.indices.forEach { i ->
            val r = ray(observations[i])
            val p = r.origin
            val d = r.direction

            val projector = arrayOf(
                doubleArrayOf(
                    1 - d.x * d.x,
                    -d.x * d.y,
                    -d.x * d.z,
                ),
                doubleArrayOf(
                    -d.y * d.x,
                    1 - d.y * d.y,
                    -d.y * d.z,
                ),
                doubleArrayOf(
                    -d.z * d.x,
                    -d.z * d.y,
                    1 - d.z * d.z,
                ),
            )

            val weight =
                weights[i].coerceAtLeast(1e-8)

            for (row in 0..2) {
                for (col in 0..2) {
                    a[row][col] +=
                        weight * projector[row][col]
                }

                b[row] +=
                    weight * (
                        projector[row][0] * p.x +
                            projector[row][1] * p.y +
                            projector[row][2] * p.z
                        )
            }
        }

        return solve3(a, b)
    }

    private fun cameraDepth(
        point: Vec3,
        observation: Observation,
    ): Double =
        observation.cameraPose.rotation
            .conjugate()
            .normalized()
            .rotate(point - observation.cameraPose.position)
            .z

    private fun positiveDepthRatio(
        point: Vec3,
        observations: List<Observation>,
    ): Double {
        val positive =
            observations.count {
                cameraDepth(point, it) > 1e-6
            }

        return positive.toDouble() /
            observations.size.coerceAtLeast(1)
    }

    private fun estimateUncertaintyMeters(
        point: Vec3,
        observations: List<Observation>,
        rmsPx: Double,
        parallaxDegrees: Double,
    ): Double {
        val focal =
            observations.map {
                (it.intrinsics.fx + it.intrinsics.fy) * 0.5
            }.average().coerceAtLeast(1.0)

        val depth =
            observations.mapNotNull {
                cameraDepth(point, it)
                    .takeIf(Double::isFinite)
                    ?.takeIf { value -> value > 0.0 }
            }.average().coerceAtLeast(0.05)

        val angular =
            Math.toRadians(
                parallaxDegrees.coerceAtLeast(0.1),
            ).coerceAtLeast(1e-4)

        return (
            depth *
                (rmsPx / focal) /
                sinSafe(angular)
            ).coerceIn(1e-4, 2.0)
    }

    private fun geometryConditionScore(
        observations: List<Observation>,
        parallaxDegrees: Double,
        point: Vec3,
    ): Double {
        val baselines =
            observations.map {
                point.distanceTo(it.cameraPose.position)
            }

        if (baselines.isEmpty()) return 0.0

        val mean = baselines.average().coerceAtLeast(1e-6)
        val spread =
            baselines.map {
                abs(it - mean) / mean
            }.average().coerceIn(0.0, 1.0)

        return (
            0.55 * (
                parallaxDegrees / 30.0
            ).coerceIn(0.0, 1.0) +
                0.25 * (
                    observations.size / 6.0
                ).coerceIn(0.0, 1.0) +
                0.20 * (1.0 - spread)
            ).coerceIn(0.0, 1.0)
    }

    private fun Double.sqrtSafe(): Double =
        kotlin.math.sqrt(
            coerceAtLeast(1e-9),
        )

    private fun sinSafe(radians: Double): Double =
        kotlin.math.sin(radians).coerceAtLeast(1e-5)

    private fun Vec3.isFinite(): Boolean =
        x.isFinite() && y.isFinite() && z.isFinite()

    private fun conditionScore(
        conditionNumber: Double,
    ): Double {
        if (!conditionNumber.isFinite() || conditionNumber <= 0.0) {
            return 0.0
        }
        val logCondition =
            kotlin.math.log10(
                conditionNumber.coerceAtLeast(1.0),
            )
        val logLimit =
            kotlin.math.log10(
                maximumNormalMatrixConditionNumber,
            ).coerceAtLeast(1.0)
        return (
            1.0 -
                logCondition / logLimit
            ).coerceIn(0.0, 1.0)
    }

    private fun estimateRayNormalMatrixCondition(
        observations: List<Observation>,
        weights: List<Double>,
    ): Double {
        if (
            observations.size < 2 ||
            observations.size != weights.size
        ) {
            return Double.POSITIVE_INFINITY
        }

        val matrix = Array(3) { DoubleArray(3) }
        observations.indices.forEach { index ->
            val direction =
                ray(observations[index])
                    .direction
                    .normalized()
            val weight =
                weights[index].coerceAtLeast(1.0e-8)

            matrix[0][0] +=
                weight * (1.0 - direction.x * direction.x)
            matrix[0][1] +=
                weight * (-direction.x * direction.y)
            matrix[0][2] +=
                weight * (-direction.x * direction.z)
            matrix[1][0] +=
                weight * (-direction.y * direction.x)
            matrix[1][1] +=
                weight * (1.0 - direction.y * direction.y)
            matrix[1][2] +=
                weight * (-direction.y * direction.z)
            matrix[2][0] +=
                weight * (-direction.z * direction.x)
            matrix[2][1] +=
                weight * (-direction.z * direction.y)
            matrix[2][2] +=
                weight * (1.0 - direction.z * direction.z)
        }

        repeat(8) {
            var p = 0
            var q = 1
            var largest = abs(matrix[0][1])

            if (abs(matrix[0][2]) > largest) {
                p = 0
                q = 2
                largest = abs(matrix[0][2])
            }
            if (abs(matrix[1][2]) > largest) {
                p = 1
                q = 2
                largest = abs(matrix[1][2])
            }

            if (!largest.isFinite()) {
                return Double.POSITIVE_INFINITY
            }
            if (largest < 1.0e-12) {
                return@repeat
            }

            val app = matrix[p][p]
            val aqq = matrix[q][q]
            val apq = matrix[p][q]
            val angle =
                0.5 *
                    kotlin.math.atan2(
                        2.0 * apq,
                        aqq - app,
                    )
            val cosine = kotlin.math.cos(angle)
            val sine = kotlin.math.sin(angle)

            for (k in 0..2) {
                if (k == p || k == q) continue
                val mkp = matrix[k][p]
                val mkq = matrix[k][q]
                val rotatedKp =
                    cosine * mkp -
                        sine * mkq
                val rotatedKq =
                    sine * mkp +
                        cosine * mkq

                matrix[k][p] = rotatedKp
                matrix[p][k] = rotatedKp
                matrix[k][q] = rotatedKq
                matrix[q][k] = rotatedKq
            }

            matrix[p][p] =
                cosine * cosine * app -
                    2.0 * sine * cosine * apq +
                    sine * sine * aqq
            matrix[q][q] =
                sine * sine * app +
                    2.0 * sine * cosine * apq +
                    cosine * cosine * aqq
            matrix[p][q] = 0.0
            matrix[q][p] = 0.0
        }

        val eigenvalues =
            doubleArrayOf(
                matrix[0][0],
                matrix[1][1],
                matrix[2][2],
            )
        if (eigenvalues.any { !it.isFinite() }) {
            return Double.POSITIVE_INFINITY
        }

        val maximum = eigenvalues.maxOrNull() ?: return Double.POSITIVE_INFINITY
        val minimum = eigenvalues.minOrNull() ?: return Double.POSITIVE_INFINITY

        if (maximum <= 0.0 || minimum <= 1.0e-12) {
            return Double.POSITIVE_INFINITY
        }

        return (maximum / minimum).coerceAtLeast(1.0)
    }

    private fun solve3(
        a: Array<DoubleArray>,
        b: DoubleArray,
    ): Vec3? {
        val m =
            Array(3) { row ->
                doubleArrayOf(
                    a[row][0],
                    a[row][1],
                    a[row][2],
                    b[row],
                )
            }

        for (col in 0 until 3) {
            var pivot = col

            for (row in col + 1 until 3) {
                if (
                    abs(m[row][col]) >
                    abs(m[pivot][col])
                ) {
                    pivot = row
                }
            }

            if (
                !m[pivot][col].isFinite() ||
                abs(m[pivot][col]) < 1e-12
            ) {
                return null
            }

            val tmp = m[col]
            m[col] = m[pivot]
            m[pivot] = tmp

            val divisor = m[col][col]

            for (c in col until 4) {
                m[col][c] /= divisor
            }

            for (row in 0 until 3) {
                if (row == col) continue

                val factor = m[row][col]
                if (!factor.isFinite()) return null

                for (c in col until 4) {
                    m[row][c] -= factor * m[col][c]
                }
            }
        }

        val result =
            Vec3(
                m[0][3],
                m[1][3],
                m[2][3],
            )

        return result.takeIf { it.isFinite() }
    }
}

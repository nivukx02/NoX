package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * High-end dense calibrated multi-view surface reconstruction.
 *
 * This engine is a dense point/surface companion to the skeleton reconstruction stack. It consumes
 * calibrated TriangulationEngine observations for arbitrary point identities and turns them into
 * temporally coherent metric 3D points with explicit uncertainty and topology evidence.
 *
 * The engine is intentionally independent of a particular neural model. Any upstream detector that
 * can emit calibrated 2D observations can feed it. This makes the dense layer useful for body
 * surface landmarks, hands, face regions, clothing deformation proxies, props, or category-specific
 * geometric points.
 *
 * Production guarantees:
 * - all observations are finite-checked before geometry work;
 * - camera identities are de-duplicated per point;
 * - triangulation remains fail-closed on weak parallax or invalid cheirality;
 * - temporal prediction is bounded and confidence-decayed;
 * - stale state is pruned and memory is capped;
 * - topology metrics never promote geometry that failed point-level quality gates;
 * - P50/P95/P99 uncertainty is exposed separately from confidence;
 * - no physical accuracy claim is inferred from architecture alone.
 */
class DenseMultiViewSurfaceReconstructionEngine(
    private val config: Config = Config(),
    private val triangulation: TriangulationEngine =
        TriangulationEngine(),
    private val spatialDetailAuthority: SpatialMultiScaleDetailAuthority =
        SpatialMultiScaleDetailAuthority(
            SpatialMultiScaleDetailAuthority.Config(
                maximumRegions = 16_384,
            ),
        )
) : AutoCloseable {

    data class Config(
        val maximumPoints: Int = 16_384,
        val maximumSubjects: Int = 64,
        val maximumObservationsPerPoint: Int = 48,
        val maximumGeometricViewsPerPoint: Int = 12,
        val minimumGeometricViewsPerPoint: Int = 2,
        val minimumViewSelectionScore: Double = 0.28,
        val viewSelectionBaselineScaleMeters: Double = 0.35,
        val viewSelectionOrientationWeight: Double = 0.24,
        val viewSelectionBaselineWeight: Double = 0.30,
        val viewSelectionEvidenceWeight: Double = 0.46,
        val maximumEdgesPerPoint: Int = 12,
        val maximumHistoryFrames: Int = 16,
        val maximumDiagnostics: Int = 256,
        val stalePointMillis: Double = 400.0,
        val temporalHalfLifeMillis: Double = 120.0,
        val predictionHorizonMillis: Double = 100.0,
        val minimumPointConfidence: Double = 0.42,
        val minimumPointUncertaintyMeters: Double = 0.12,
        val hardPointUncertaintyMeters: Double = 0.50,
        val minimumParallaxDegrees: Double = 0.75,
        val preferredParallaxDegrees: Double = 2.5,
        val hardReprojectionErrorPx: Double = 8.0,
        val preferredReprojectionErrorPx: Double = 2.5,
        val minimumCheiralityRatio: Double = 0.75,
        val hardCheiralityRatio: Double = 0.50,
        val topologyWarningRatio: Double = 0.16,
        val topologyAuthorityRatio: Double = 0.72,
        val topologyHardRatio: Double = 0.35,
        val topologyTemporalWeight: Double = 0.42,
        val pointTemporalWeight: Double = 0.24,
        val geometricWeight: Double = 0.24,
        val confidenceWeight: Double = 0.10,
        val appearanceFusionWeight: Double = 0.10,
        val appearanceSelectionWeight: Double = 0.08,
        val minimumAppearanceTrust: Double = 0.42,
        val maximumPredictionSpeedMetersPerSecond: Double = 8.0,
        val maximumPredictionAccelerationMetersPerSecond2: Double = 80.0,
        val predictionConfidenceFloor: Double = 0.18,
        val minimumTrustedViewCount: Int = 2,
        val minimumTrustedPointRatio: Double = 0.35,
        val maximumKinematicTracks: Int = 4_096,
        val maximumKinematicInnovationMeters: Double = 0.045,
        val precisionMinimumFeasibility: Double = 0.50,
        val precisionMaximumFirstOrderDepthErrorMillimeters: Double = 1.0,
        val precisionMinimumMotionStability: Double = 0.55,
        val minimumActiveViewInformationGain: Double = 0.20,
        val minimumMetricEvidenceAuthorityForPrecision: Double = 0.55,
        val precisionMinimumAnisotropicWeakAxisConstraint: Double = 0.35,
        val precisionMinimumAnisotropicViewConditionScore: Double = 0.45,
    ) {
        init {
            require(maximumPoints > 0)
            require(maximumSubjects > 0)
            require(maximumObservationsPerPoint >= 2)
            require(maximumGeometricViewsPerPoint >= 2)
            require(maximumGeometricViewsPerPoint >= minimumGeometricViewsPerPoint)
            require(minimumGeometricViewsPerPoint >= 2)
            require(minimumViewSelectionScore in 0.0..1.0)
            require(viewSelectionBaselineScaleMeters > 0.0)
            require(viewSelectionOrientationWeight >= 0.0)
            require(viewSelectionBaselineWeight >= 0.0)
            require(viewSelectionEvidenceWeight >= 0.0)
            require(
                viewSelectionOrientationWeight +
                    viewSelectionBaselineWeight +
                    viewSelectionEvidenceWeight > 0.0,
            )
            require(maximumEdgesPerPoint > 0)
            require(maximumHistoryFrames >= 2)
            require(maximumDiagnostics >= 16)
            require(stalePointMillis > 0.0)
            require(temporalHalfLifeMillis > 0.0)
            require(predictionHorizonMillis >= 0.0)
            require(minimumPointConfidence in 0.0..1.0)
            require(minimumPointUncertaintyMeters > 0.0)
            require(hardPointUncertaintyMeters > minimumPointUncertaintyMeters)
            require(minimumParallaxDegrees > 0.0)
            require(preferredParallaxDegrees >= minimumParallaxDegrees)
            require(hardReprojectionErrorPx > preferredReprojectionErrorPx)
            require(preferredReprojectionErrorPx > 0.0)
            require(minimumCheiralityRatio in 0.0..1.0)
            require(hardCheiralityRatio in 0.0..1.0)
            require(hardCheiralityRatio <= minimumCheiralityRatio)
            require(topologyWarningRatio > 0.0)
            require(topologyAuthorityRatio in 0.0..1.0)
            require(topologyAuthorityRatio >= topologyWarningRatio)
            require(topologyHardRatio > topologyWarningRatio)
            require(
                topologyTemporalWeight +
                    pointTemporalWeight +
                    geometricWeight +
                    confidenceWeight > 0.0,
            )
            require(appearanceFusionWeight in 0.0..1.0)
            require(appearanceSelectionWeight in 0.0..1.0)
            require(minimumAppearanceTrust in 0.0..1.0)
            require(maximumPredictionSpeedMetersPerSecond > 0.0)
            require(maximumPredictionAccelerationMetersPerSecond2 > 0.0)
            require(predictionConfidenceFloor in 0.0..1.0)
            require(minimumTrustedViewCount >= 2)
            require(minimumTrustedPointRatio in 0.0..1.0)
            require(maximumKinematicTracks > 0)
            require(maximumKinematicInnovationMeters > 0.0)
            require(precisionMinimumFeasibility in 0.0..1.0)
            require(precisionMaximumFirstOrderDepthErrorMillimeters > 0.0)
            require(precisionMinimumMotionStability in 0.0..1.0)
            require(minimumActiveViewInformationGain in 0.0..1.0)
            require(minimumMetricEvidenceAuthorityForPrecision in 0.0..1.0)
            require(precisionMinimumAnisotropicWeakAxisConstraint in 0.0..1.0)
            require(precisionMinimumAnisotropicViewConditionScore in 0.0..1.0)
        }
    }

    data class PointKey(
        val subjectId: Int,
        val pointId: Int,
    ) {
        init {
            require(subjectId >= 0)
            require(pointId >= 0)
        }
    }

    data class PointObservation(
        val key: PointKey,
        val observation: TriangulationEngine.Observation,
        val semanticConfidence: Double = 1.0,
        val visibility: Double = 1.0,
        val contactProbability: Double = 0.0,
        val sourceDiversity: Double = 1.0,
        val photometricConsistency: Double = 1.0,
        val silhouetteConsistency: Double = 1.0,
        val motionBoundaryConfidence: Double = 1.0,
        val lightingStability: Double = 1.0,
    ) {
        init {
            require(semanticConfidence in 0.0..1.0)
            require(visibility in 0.0..1.0)
            require(contactProbability in 0.0..1.0)
            require(sourceDiversity in 0.0..1.0)
            require(photometricConsistency in 0.0..1.0)
            require(silhouetteConsistency in 0.0..1.0)
            require(motionBoundaryConfidence in 0.0..1.0)
            require(lightingStability in 0.0..1.0)
        }
    }

    data class TopologyEdge(
        val a: PointKey,
        val b: PointKey,
        val restLengthMeters: Double,
        val confidence: Double = 1.0,
        val rigidity: Double = 0.6,
    ) {
        init {
            require(a != b)
            require(restLengthMeters > 0.0)
            require(confidence in 0.0..1.0)
            require(rigidity in 0.0..1.0)
        }
    }

    data class FrameInput(
        val timestampNanos: Long,
        val points: List<PointObservation>,
        val topologyEdges: List<TopologyEdge> = emptyList(),
    ) {
        init {
            require(timestampNanos > 0L)
        }
    }

    enum class PointState {
        TRUSTED,
        OBSERVED,
        TEMPORALLY_BRIDGED,
        LOW_CONFIDENCE,
        GEOMETRY_REJECTED,
        STALE,
    }

    data class PointResult(
        val key: PointKey,
        val timestampNanos: Long,
        val positionMeters: com.nivukx.nox.core.model.Vec3,
        val velocityMetersPerSecond: com.nivukx.nox.core.model.Vec3,
        val accelerationMetersPerSecond2:
            com.nivukx.nox.core.model.Vec3,
        val confidence: Double,
        val uncertaintyMeters: Double,
        val uncertainty95Meters: Double,
        val reprojectionErrorPx: Double,
        val parallaxDegrees: Double,
        val cheiralityRatio: Double,
        val viewCount: Int,
        val independentViewCount: Int,
        val viewDiversityScore: Double,
        val temporalStability: Double,
        val topologyStability: Double,
        val contactSupport: Double,
        val appearanceConsistency: Double = 1.0,
        val silhouetteConsistency: Double = 1.0,
        val motionBoundaryConfidence: Double = 1.0,
        val lightingStability: Double = 1.0,
        val adaptiveViewSelectionScore: Double = 0.0,
        val adaptiveViewDiversityScore: Double = 0.0,
        /** Actual triangulator-reported information quality of the selected metric views. */
        val activeViewInformationGainScore: Double = 0.0,
        /** Redundancy-aware geometric diversity of the active-view ordering. */
        val activeViewRedundancyScore: Double = 0.0,
        /** Combined metric evidence authority used for precision/detail gating. */
        val metricEvidenceAuthority: Double = 0.0,
        val spatialDetailBand:
            SpatialMultiScaleDetailAuthority.Band =
                SpatialMultiScaleDetailAuthority.Band.COARSE,
        val spatialDetailPriority: Double = 0.0,
        val spatialDetailComputeShare: Double = 0.0,
        val excludedWeakViewCount: Int = 0,
        val anisotropicWeakAxisConstraint: Double = 0.0,
        val anisotropicViewConditionScore: Double = 0.0,
        val probabilisticEvidenceScore: Double = 1.0,
        val state: PointState,
        val predicted: Boolean,
        val staleObservation: Boolean,
        val warnings: List<String>,
        val jerkMetersPerSecond3:
            com.nivukx.nox.core.model.Vec3 = com.nivukx.nox.core.model.Vec3(0.0, 0.0, 0.0),
        val motionStability: Double = 0.0,
        val kinematicInnovationMeters: Double = 0.0,
        val kinematicModelConfidence: Double = 0.0,
        val subMillimeterFeasibilityScore: Double = 0.0,
        val firstOrderDepthErrorMillimeters: Double =
            Double.NaN,
        val reprojectionP95Px: Double = Double.POSITIVE_INFINITY,
        val tailAdjustedFirstOrderDepthErrorMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val tailAdjustedSubMillimeterFeasibilityScore: Double = 0.0,
        val precisionQualified: Boolean = false,
    ) {
        val trusted: Boolean
            get() =
                state == PointState.TRUSTED &&
                    confidence >= 0.65 &&
                    uncertainty95Meters <= 0.16 &&
                    cheiralityRatio >= 0.80 &&
                    reprojectionErrorPx <= 4.0

        val finite: Boolean
            get() =
                positionMeters.isFinite() &&
                    velocityMetersPerSecond.isFinite() &&
                    accelerationMetersPerSecond2.isFinite() &&
                    confidence.isFinite() &&
                    uncertaintyMeters.isFinite() &&
                    uncertainty95Meters.isFinite() &&
                    jerkMetersPerSecond3.isFinite() &&
                    motionStability.isFinite() &&
                    kinematicInnovationMeters.isFinite() &&
                    kinematicModelConfidence.isFinite() &&
                    subMillimeterFeasibilityScore.isFinite()
    }

    data class Result(
        val timestampNanos: Long,
        val points: List<PointResult>,
        val topologyEdgesEvaluated: Int,
        val topologyEdgesTrusted: Int,
        val topologyWarningCount: Int,
        val coverage: Double,
        val trustedPointRatio: Double,
        val meanConfidence: Double,
        val uncertaintyP50Meters: Double,
        val uncertaintyP95Meters: Double,
        val uncertaintyP99Meters: Double,
        val meanReprojectionErrorPx: Double,
        val p95ReprojectionErrorPx: Double,
        val meanParallaxDegrees: Double,
        val temporalStability: Double,
        val topologyStability: Double,
        val viewDiversity: Double,
        val adaptiveViewSelectionScore: Double,
        val adaptiveViewDiversityScore: Double,
        val excludedWeakViewCount: Int,
        val predictedPointRatio: Double,
        val stalePointRatio: Double,
        val contactSupport: Double,
        val meanAppearanceConsistency: Double,
        val meanSilhouetteConsistency: Double,
        val meanMotionBoundaryConfidence: Double,
        val meanLightingStability: Double,
        val acceptedPointCount: Int,
        val rejectedPointCount: Int,
        val hardFailure: Boolean,
        val warnings: List<String>,
        val diagnostics: List<String>,
        val meanJerkMetersPerSecond3: Double = 0.0,
        val meanMotionStability: Double = 0.0,
        val meanKinematicInnovationMillimeters: Double = 0.0,
        val meanSubMillimeterFeasibilityScore: Double = 0.0,
        val meanFirstOrderDepthErrorMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val p95ReprojectionTailAdjustedDepthErrorMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val meanTailAdjustedSubMillimeterFeasibilityScore: Double = 0.0,
        val meanAnisotropicWeakAxisConstraint: Double = 0.0,
        val meanAnisotropicViewConditionScore: Double = 0.0,
        val meanProbabilisticEvidenceScore: Double = 1.0,
        val meanActiveViewInformationGainScore: Double = 0.0,
        val meanActiveViewRedundancyScore: Double = 0.0,
        val meanMetricEvidenceAuthority: Double = 0.0,
        val meanSpatialDetailPriority: Double = 0.0,
        val spatialDetailBudgetUtilization: Double = 0.0,
        val ultraDetailPointRatio: Double = 0.0,
        val fineOrHigherDetailPointRatio: Double = 0.0,
        val spatialDetailHold: Boolean = false,
        val precisionQualified: Boolean = false,
        val authoritative: Boolean = false,
    )

    data class Telemetry(
        val activePointCount: Int,
        val latestTimestampNanos: Long?,
        val coverage: Double,
        val trustedPointRatio: Double,
        val uncertaintyP95Meters: Double,
        val uncertaintyP99Meters: Double,
        val meanReprojectionErrorPx: Double,
        val temporalStability: Double,
        val topologyStability: Double,
        val viewDiversity: Double,
        val meanAppearanceConsistency: Double = 1.0,
        val meanSilhouetteConsistency: Double = 1.0,
        val meanMotionBoundaryConfidence: Double = 1.0,
        val meanLightingStability: Double = 1.0,
        val adaptiveViewSelectionScore: Double = 0.0,
        val adaptiveViewDiversityScore: Double = 0.0,
        val excludedWeakViewCount: Int = 0,
        val predictedPointRatio: Double,
        val stalePointRatio: Double,
        val hardFailure: Boolean,
        val cumulativeRejectedPoints: Long,
        val cumulativeStalePoints: Long,
        val cumulativeGeometryRejects: Long,
        val continuousTimeFieldCount: Int = 0,
        val meanJerkMetersPerSecond3: Double = 0.0,
        val meanMotionStability: Double = 0.0,
        val meanKinematicInnovationMillimeters: Double = 0.0,
        val meanSubMillimeterFeasibilityScore: Double = 0.0,
        val meanFirstOrderDepthErrorMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val p95ReprojectionTailAdjustedDepthErrorMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val meanTailAdjustedSubMillimeterFeasibilityScore: Double = 0.0,
        val meanAnisotropicWeakAxisConstraint: Double = 0.0,
        val meanAnisotropicViewConditionScore: Double = 0.0,
        val meanProbabilisticEvidenceScore: Double = 1.0,
        val meanActiveViewInformationGainScore: Double = 0.0,
        val meanActiveViewRedundancyScore: Double = 0.0,
        val meanMetricEvidenceAuthority: Double = 0.0,
        val precisionQualified: Boolean = false,
    )

    private data class PointTrackState(
        val key: PointKey,
        var timestampNanos: Long = Long.MIN_VALUE,
        var observedTimestampNanos: Long = Long.MIN_VALUE,
        var position: com.nivukx.nox.core.model.Vec3 =
            com.nivukx.nox.core.model.Vec3(0.0, 0.0, 0.0),
        var velocity: com.nivukx.nox.core.model.Vec3 =
            com.nivukx.nox.core.model.Vec3(0.0, 0.0, 0.0),
        var acceleration: com.nivukx.nox.core.model.Vec3 =
            com.nivukx.nox.core.model.Vec3(0.0, 0.0, 0.0),
        var jerk: com.nivukx.nox.core.model.Vec3 =
            com.nivukx.nox.core.model.Vec3(0.0, 0.0, 0.0),
        var confidence: Double = 0.0,
        var uncertaintyMeters: Double = 1.0,
        var uncertainty95Meters: Double = 1.0,
        var observedFrames: Long = 0L,
        var predictedFrames: Long = 0L,
        var contactSupport: Double = 0.0,
        var appearanceConsistency: Double = 1.0,
        var silhouetteConsistency: Double = 1.0,
        var motionBoundaryConfidence: Double = 1.0,
        var lightingStability: Double = 1.0,
        var topologyStability: Double = 0.0,
        var lastParallaxDegrees: Double = 0.0,
        var lastViewDiversity: Double = 0.0,
        var lastReprojectionErrorPx: Double = Double.POSITIVE_INFINITY,
        var state: PointState = PointState.LOW_CONFIDENCE,
    )

    private data class EdgeEvaluation(
        val edge: TopologyEdge,
        val actualLengthMeters: Double,
        val normalizedStrain: Double,
        val temporalWeight: Double,
        val trusted: Boolean,
        val warning: Boolean,
    )

    private val states =
        LinkedHashMap<PointKey, PointTrackState>()

    private val kinematicEngines =
        LinkedHashMap<PointKey, KinematicStabilityEngine>()

    private val anisotropicViewContributionEngine =
        AnisotropicViewContributionEngine(
            AnisotropicViewContributionEngine.Config.precision(),
        )

    private val motionAwareRecoveryEngine =
        MotionAwareRecoveryEngine(
            MotionAwareRecoveryEngine.Config.precision(),
        )

    private val continuousTimeSurfaceEngine =
        ContinuousTimeGaussianSurfaceEngine()

    private val history =
        ArrayDeque<Result>()

    private val diagnostics =
        ArrayDeque<String>()

    private var latest: Result? = null
    private var lastFrameTimestampNanos = Long.MIN_VALUE
    private var closed = false

    private var cumulativeRejectedPoints = 0L
    private var cumulativeStalePoints = 0L
    private var cumulativeGeometryRejects = 0L

    @Synchronized
    fun process(input: FrameInput): Result {
        check(!closed) {
            "DenseMultiViewSurfaceReconstructionEngine is closed"
        }

        if (
            lastFrameTimestampNanos != Long.MIN_VALUE &&
            input.timestampNanos < lastFrameTimestampNanos
        ) {
            cumulativeRejectedPoints += input.points.size.toLong()
            val failed =
                emptyResult(
                    timestampNanos = input.timestampNanos,
                    warning = "dense reconstruction timestamp reversal rejected",
                )
            latest = failed
            appendDiagnostic(
                "timestamp reversal: previous=" +
                    lastFrameTimestampNanos +
                    " current=" +
                    input.timestampNanos,
            )
            return failed
        }

        if (
            lastFrameTimestampNanos != Long.MIN_VALUE &&
            input.timestampNanos == lastFrameTimestampNanos
        ) {
            cumulativeRejectedPoints += input.points.size.toLong()
            appendDiagnostic(
                "duplicate timestamp ignored: timestamp=" +
                    input.timestampNanos,
            )
            return latest
                ?: emptyResult(
                    timestampNanos = input.timestampNanos,
                    warning = "duplicate timestamp ignored before first dense result",
                )
        }

        lastFrameTimestampNanos = input.timestampNanos
        prune(input.timestampNanos)

        val grouped =
            LinkedHashMap<PointKey, MutableList<PointObservation>>()

        for (item in input.points) {
            if (grouped.size >= config.maximumPoints &&
                !grouped.containsKey(item.key)
            ) {
                cumulativeRejectedPoints++
                continue
            }

            if (
                item.key.subjectId >= config.maximumSubjects ||
                !item.observation.u.isFinite() ||
                !item.observation.v.isFinite() ||
                !item.observation.weight.isFinite()
            ) {
                cumulativeRejectedPoints++
                continue
            }

            val bucket = grouped.getOrPut(item.key) {
                mutableListOf()
            }

            if (
                bucket.size >=
                    config.maximumObservationsPerPoint
            ) {
                cumulativeRejectedPoints++
                continue
            }

            bucket += item
        }

        val reconstructed =
            ArrayList<PointResult>(grouped.size)

        for ((key, observations) in grouped) {
            reconstructed +=
                reconstructPoint(
                    timestampNanos = input.timestampNanos,
                    key = key,
                    observations = observations,
                )
        }

        val groupedKeys = grouped.keys.toSet()

        for ((key, state) in states) {
            if (key !in groupedKeys) {
                bridgeShortOcclusion(
                    timestampNanos = input.timestampNanos,
                    state = state,
                )?.let(reconstructed::add)
            }
        }

        reconstructed.sortWith(
            compareBy<PointResult> { it.key.subjectId }
                .thenBy { it.key.pointId },
        )

        reconstructed.forEach { point ->
            if (
                point.finite &&
                point.state != PointState.GEOMETRY_REJECTED &&
                point.state != PointState.STALE
            ) {
                continuousTimeSurfaceEngine.observe(
                    fieldId = continuousFieldId(point.key),
                    sample = ContinuousTimeGaussianSurfaceEngine.Sample(
                        timestampNanos = point.timestampNanos,
                        positionMeters = point.positionMeters,
                        velocityMetersPerSecond = point.velocityMetersPerSecond,
                        accelerationMetersPerSecond2 = point.accelerationMetersPerSecond2,
                        covariance = UncertaintyEnvelope(
                            xxMeters2 = point.uncertaintyMeters * point.uncertaintyMeters,
                            yyMeters2 = point.uncertaintyMeters * point.uncertaintyMeters,
                            zzMeters2 = point.uncertaintyMeters * point.uncertaintyMeters,
                            confidence = point.confidence,
                        ),
                        confidence = point.confidence,
                        origin = pointOrigin(point.state),
                    ),
                )
            }
        }

        val topology =
            evaluateTopology(
                timestampNanos = input.timestampNanos,
                points = reconstructed,
                edges = input.topologyEdges,
            )

        val result =
            summarize(
                timestampNanos = input.timestampNanos,
                points = reconstructed,
                topology = topology,
            )

        latest = result

        history.addLast(result)
        while (history.size > config.maximumHistoryFrames) {
            history.removeFirst()
        }

        return result
    }

    @Synchronized
    fun latest(): Result? = latest

    @Synchronized
    fun telemetry(): Telemetry =
        Telemetry(
            activePointCount = states.size,
            latestTimestampNanos = latest?.timestampNanos,
            coverage = latest?.coverage ?: 0.0,
            trustedPointRatio = latest?.trustedPointRatio ?: 0.0,
            uncertaintyP95Meters = latest?.uncertaintyP95Meters
                ?: Double.POSITIVE_INFINITY,
            uncertaintyP99Meters = latest?.uncertaintyP99Meters
                ?: Double.POSITIVE_INFINITY,
            meanReprojectionErrorPx = latest?.meanReprojectionErrorPx
                ?: Double.POSITIVE_INFINITY,
            temporalStability = latest?.temporalStability ?: 0.0,
            topologyStability = latest?.topologyStability ?: 0.0,
            viewDiversity = latest?.viewDiversity ?: 0.0,
            meanAppearanceConsistency = latest?.meanAppearanceConsistency ?: 0.0,
            meanSilhouetteConsistency = latest?.meanSilhouetteConsistency ?: 0.0,
            meanMotionBoundaryConfidence = latest?.meanMotionBoundaryConfidence ?: 0.0,
            meanLightingStability = latest?.meanLightingStability ?: 0.0,
            meanJerkMetersPerSecond3 = latest?.meanJerkMetersPerSecond3 ?: 0.0,
            meanMotionStability = latest?.meanMotionStability ?: 0.0,
            meanKinematicInnovationMillimeters =
                latest?.meanKinematicInnovationMillimeters ?: 0.0,
            meanSubMillimeterFeasibilityScore =
                latest?.meanSubMillimeterFeasibilityScore ?: 0.0,
            meanFirstOrderDepthErrorMillimeters =
                latest?.meanFirstOrderDepthErrorMillimeters
                    ?: Double.POSITIVE_INFINITY,
            meanProbabilisticEvidenceScore =
                latest?.meanProbabilisticEvidenceScore ?: 1.0,
            meanActiveViewInformationGainScore =
                latest?.meanActiveViewInformationGainScore ?: 0.0,
            meanActiveViewRedundancyScore =
                latest?.meanActiveViewRedundancyScore ?: 0.0,
            meanMetricEvidenceAuthority =
                latest?.meanMetricEvidenceAuthority ?: 0.0,
            precisionQualified = latest?.precisionQualified ?: false,
            adaptiveViewSelectionScore = latest?.adaptiveViewSelectionScore ?: 0.0,
            adaptiveViewDiversityScore = latest?.adaptiveViewDiversityScore ?: 0.0,
            excludedWeakViewCount = latest?.excludedWeakViewCount ?: 0,
            predictedPointRatio = latest?.predictedPointRatio ?: 0.0,
            stalePointRatio = latest?.stalePointRatio ?: 0.0,
            hardFailure = latest?.hardFailure ?: false,
            cumulativeRejectedPoints = cumulativeRejectedPoints,
            cumulativeStalePoints = cumulativeStalePoints,
            cumulativeGeometryRejects = cumulativeGeometryRejects,
            continuousTimeFieldCount = continuousTimeSurfaceEngine.activeFieldCount(),
        )

    @Synchronized
    fun queryContinuousTime(
        subjectId: Int,
        pointId: Int,
        timestampNanos: Long,
    ): ContinuousTimeGaussianSurfaceEngine.Query? {
        require(subjectId >= 0)
        require(pointId >= 0)
        return continuousTimeSurfaceEngine.query(
            continuousFieldId(PointKey(subjectId, pointId)),
            timestampNanos,
        )
    }

    @Synchronized
    fun reset() {
        check(!closed) {
            "DenseMultiViewSurfaceReconstructionEngine is closed"
        }
        states.clear()
        kinematicEngines.values.forEach { it.close() }
        kinematicEngines.clear()
        motionAwareRecoveryEngine.reset()
        continuousTimeSurfaceEngine.reset()
        history.clear()
        diagnostics.clear()
        latest = null
        lastFrameTimestampNanos = Long.MIN_VALUE
        cumulativeRejectedPoints = 0L
        cumulativeStalePoints = 0L
        cumulativeGeometryRejects = 0L
    }

    @Synchronized
    fun diagnosticHistory(): List<String> =
        diagnostics.toList()

    @Synchronized
    override fun close() {
        if (closed) return
        reset()
        motionAwareRecoveryEngine.close()
        continuousTimeSurfaceEngine.close()
        closed = true
    }

    private fun continuousFieldId(key: PointKey): Long =
        (key.subjectId.toLong() shl 32) xor
            (key.pointId.toLong() and 0xffffffffL)

    private fun pointOrigin(state: PointState): EvidenceOrigin =
        when (state) {
            PointState.TRUSTED,
            PointState.OBSERVED -> EvidenceOrigin.OBSERVED
            PointState.TEMPORALLY_BRIDGED -> EvidenceOrigin.TEMPORALLY_PROPAGATED
            PointState.LOW_CONFIDENCE,
            PointState.GEOMETRY_REJECTED -> EvidenceOrigin.GEOMETRICALLY_CONSTRAINED
            PointState.STALE -> EvidenceOrigin.TEMPORALLY_PROPAGATED
        }

    private fun reconstructPoint(
        timestampNanos: Long,
        key: PointKey,
        observations: List<PointObservation>,
    ): PointResult {
        val state =
            states.getOrPut(key) {
                PointTrackState(key = key)
            }

        val usable =
            observations.filter {
                it.semanticConfidence >= config.minimumPointConfidence &&
                    it.visibility >= 0.10 &&
                    it.observation.temporalAlignmentConfidence >=
                        0.10 &&
                    it.observation.temporalUncertaintyNanos <=
                        40_000_000L
            }

        if (usable.size < 2) {
            cumulativeGeometryRejects++
            return (
                bridgeShortOcclusion(
                    timestampNanos = timestampNanos,
                    state = state,
                )
                    ?: rejectedPoint(
                        timestampNanos = timestampNanos,
                        key = key,
                        warning = "fewer than two trustworthy calibrated views",
                    )
                )
        }

        val deduplicated =
            usable
                .groupBy {
                    it.observation.cameraId
                }
                .values
                .mapNotNull { candidates ->
                    candidates.maxByOrNull {
                        effectiveSemanticWeight(it)
                    }
                }

        val viewSelection = selectGeometricViews(deduplicated)
        val selectedViews = viewSelection.first
        val excludedWeakViews = viewSelection.second

        if (selectedViews.size < 2) {
            cumulativeGeometryRejects++
            return (
                bridgeShortOcclusion(
                    timestampNanos = timestampNanos,
                    state = state,
                )
                    ?: rejectedPoint(
                        timestampNanos = timestampNanos,
                        key = key,
                        warning = "insufficient independent camera views",
                    )
                )
        }

        val anisotropicSelection =
            anisotropicViewContributionEngine.rank(
                candidates =
                    selectedViews.map {
                        AnisotropicViewContributionEngine.Candidate(
                            observation = it.observation,
                            semanticConfidence =
                                it.semanticConfidence,
                            visibility = it.visibility,
                            sourceDiversity = it.sourceDiversity,
                            appearanceEvidence =
                                (
                                    it.photometricConsistency * 0.36 +
                                        it.silhouetteConsistency * 0.26 +
                                        it.motionBoundaryConfidence * 0.20 +
                                        it.lightingStability * 0.18
                                    ).coerceIn(0.0, 1.0),
                        )
                    },
                limit = selectedViews.size,
            )

        val anisotropicWeakAxisConstraint =
            anisotropicSelection.selected
                .map { it.principalAxisConstraint }
                .average()
                .coerceIn(0.0, 1.0)

        val anisotropicConditionScore =
            anisotropicSelection.conditionScore
                .coerceIn(0.0, 1.0)

        val motionAwareViews =
            selectedViews.map { candidate ->
                candidate.copy(
                    observation = candidate.observation.copy(
                        motionVelocityMetersPerSecond =
                            state.velocity,
                    ),
                )
            }

        val result =
            triangulation.robust(
                motionAwareViews.map {
                    it.observation
                },
            )

        if (result == null) {
            cumulativeGeometryRejects++
            return (
                bridgeShortOcclusion(
                    timestampNanos = timestampNanos,
                    state = state,
                )
                    ?: rejectedPoint(
                        timestampNanos = timestampNanos,
                        key = key,
                        warning = "robust triangulation rejected the point",
                    )
                )
        }

        val geometryConfidence =
            geometryConfidence(result)

        val semanticConfidence =
            selectedViews.map {
                it.semanticConfidence
            }.average().coerceIn(0.0, 1.0)

        val visibility =
            selectedViews.map {
                it.visibility
            }.average().coerceIn(0.0, 1.0)

        val temporalConfidence =
            selectedViews.map {
                it.observation.temporalStability *
                    it.observation.temporalAlignmentConfidence
            }.average().coerceIn(0.0, 1.0)

        val appearanceConsistency =
            selectedViews.map { it.photometricConsistency }
                .average().coerceIn(0.0, 1.0)
        val silhouetteConsistency =
            selectedViews.map { it.silhouetteConsistency }
                .average().coerceIn(0.0, 1.0)
        val motionBoundaryConfidence =
            selectedViews.map { it.motionBoundaryConfidence }
                .average().coerceIn(0.0, 1.0)
        val lightingStability =
            selectedViews.map { it.lightingStability }
                .average().coerceIn(0.0, 1.0)

        val appearanceEvidence =
            (
                appearanceConsistency * 0.36 +
                    silhouetteConsistency * 0.26 +
                    motionBoundaryConfidence * 0.20 +
                    lightingStability * 0.18
                ).coerceIn(0.0, 1.0)

        val geometricTemporalConfidence =
            (
                geometryConfidence * config.geometricWeight +
                    semanticConfidence * config.confidenceWeight +
                    visibility * 0.12 +
                    temporalConfidence * config.pointTemporalWeight +
                    stateContinuityScore(
                        timestampNanos,
                        state,
                        result.point,
                    ) * config.topologyTemporalWeight
                ).coerceIn(0.0, 1.0)

        val activeViewInformationGain =
            result.activeViewInformationGainScore
                .coerceIn(0.0, 1.0)
        val activeViewRedundancy =
            result.activeViewRedundancyScore
                .coerceIn(0.0, 1.0)
        val metricEvidenceAuthority =
            (
                activeViewInformationGain * 0.60 +
                    activeViewRedundancy * 0.40
                ).coerceIn(0.0, 1.0)

        val basePointConfidence =
            (
                (
                    geometricTemporalConfidence *
                        (1.0 - config.appearanceFusionWeight) +
                        appearanceEvidence *
                            config.appearanceFusionWeight
                ) *
                    (
                        0.78 +
                            0.22 *
                                metricEvidenceAuthority
                    )
                ).coerceIn(0.0, 1.0)

        val motionRecoveryObservation =
            motionAwareRecoveryEngine.update(
                MotionAwareRecoveryEngine.Observation(
                    trackId = motionTrackId(key),
                    positionMeters = result.point,
                    confidence =
                        (
                            geometryConfidence *
                                result.uncertaintyConfidence
                        ).coerceIn(0.0, 1.0),
                    reprojectionErrorPx =
                        result.rmsReprojectionPx,
                    timestampNanos = timestampNanos,
                    occlusionRisk =
                        (1.0 - visibility).coerceIn(0.0, 1.0),
                    temporalConfidence =
                        temporalConfidence,
                ),
            )

        val provisionalConfidence =
            basePointConfidence *
                (
                    0.88 +
                        0.12 *
                            motionRecoveryObservation.confidence
                    ).coerceIn(0.88, 1.0)

        val pointConfidence =
            provisionalConfidence.coerceIn(0.0, 1.0)

        val kinematic =
            kinematicEngineFor(key).observe(
                KinematicStabilityEngine.Observation(
                    timestampNanos = timestampNanos,
                    positionMeters = result.point,
                    confidence = pointConfidence,
                ),
            )

        val stateBand =
            when {
                pointConfidence >= 0.72 &&
                    metricEvidenceAuthority >= 0.55 &&
                    kinematic.motionStability >=
                    config.precisionMinimumMotionStability &&
                    result.cheiralityRatio >=
                    config.minimumCheiralityRatio ->
                    PointState.TRUSTED

                pointConfidence >= 0.42 ->
                    PointState.OBSERVED

                else ->
                    PointState.LOW_CONFIDENCE
            }

        val warningList =
            buildPointWarnings(

                result = result,
                pointConfidence = pointConfidence,
                temporalConfidence = temporalConfidence,
                appearanceConsistency = appearanceConsistency,
                lightingStability = lightingStability,
            )

        state.timestampNanos = timestampNanos
        state.observedTimestampNanos = timestampNanos
        state.position = result.point
        state.velocity = kinematic.velocityMetersPerSecond
        state.acceleration = kinematic.accelerationMetersPerSecond2
        state.jerk = kinematic.jerkMetersPerSecond3
        state.confidence =
            state.confidence * 0.36 +
                pointConfidence * 0.64
        state.uncertaintyMeters =
            state.uncertaintyMeters * 0.35 +
                result.uncertaintyMeters * 0.65
        state.uncertainty95Meters =
            state.uncertainty95Meters * 0.35 +
                result.uncertainty95MajorMeters * 0.65
        state.observedFrames++
        state.contactSupport =
            state.contactSupport * 0.60 +
                result.contactSupport * 0.40
        state.appearanceConsistency =
            state.appearanceConsistency * 0.36 +
                appearanceConsistency * 0.64
        state.silhouetteConsistency =
            state.silhouetteConsistency * 0.36 +
                silhouetteConsistency * 0.64
        state.motionBoundaryConfidence =
            state.motionBoundaryConfidence * 0.36 +
                motionBoundaryConfidence * 0.64
        state.lightingStability =
            state.lightingStability * 0.36 +
                lightingStability * 0.64
        state.topologyStability =
            state.topologyStability * 0.35 +
                result.conditionScore * 0.65
        state.lastParallaxDegrees =
            result.parallaxDegrees
        state.lastViewDiversity =
            result.viewDiversityScore
        state.lastReprojectionErrorPx =
            result.rmsReprojectionPx
        state.state = stateBand

        return PointResult(
            key = key,
            timestampNanos = timestampNanos,
            positionMeters = result.point,
            velocityMetersPerSecond = state.velocity,
            accelerationMetersPerSecond2 = state.acceleration,
            confidence = state.confidence.coerceIn(0.0, 1.0),
            uncertaintyMeters = result.uncertaintyMeters,
            uncertainty95Meters =
                result.uncertainty95MajorMeters,
            reprojectionErrorPx =
                result.rmsReprojectionPx,
            parallaxDegrees =
                result.parallaxDegrees,
            cheiralityRatio =
                result.cheiralityRatio,
            viewCount =
                selectedViews.size,
            independentViewCount =
                result.independentViewCount,
            viewDiversityScore =
                result.viewDiversityScore,
            temporalStability =
                temporalConfidence,
            topologyStability =
                state.topologyStability,
            contactSupport =
                result.contactSupport,
            appearanceConsistency = appearanceConsistency,
            silhouetteConsistency = silhouetteConsistency,
            motionBoundaryConfidence = motionBoundaryConfidence,
            lightingStability = lightingStability,
            adaptiveViewSelectionScore =
                selectedViews.map(::viewSelectionScore).averageOrZeroDense(),
            adaptiveViewDiversityScore =
                geometricDiversity(selectedViews),
            activeViewInformationGainScore =
                activeViewInformationGain,
            activeViewRedundancyScore =
                activeViewRedundancy,
            metricEvidenceAuthority =
                metricEvidenceAuthority,
            excludedWeakViewCount =
                excludedWeakViews,
            anisotropicWeakAxisConstraint =
                anisotropicWeakAxisConstraint,
            anisotropicViewConditionScore =
                anisotropicConditionScore,
            probabilisticEvidenceScore =
                result.probabilisticEvidenceScore
                    .coerceIn(0.0, 1.0),
            state = stateBand,
            predicted = false,
            staleObservation = false,
            warnings = (
                warningList +
                    anisotropicSelection.warnings +
                    result.probabilisticEvidenceWarnings +
                    motionRecoveryObservation.warnings +
                    if (anisotropicWeakAxisConstraint < 0.35) {
                        listOf("anisotropic-weak-axis-underconstrained")
                    } else {
                        emptyList()
                    } +
                    if (
                        activeViewInformationGain <
                            config.minimumActiveViewInformationGain
                    ) {
                        listOf("active-view-information-gain-below-production-band")
                    } else {
                        emptyList()
                    } +
                    if (metricEvidenceAuthority < 0.55) {
                        listOf("metric-evidence-authority-limited")
                    } else {
                        emptyList()
                    } +
                    kinematic.warnings +
                    if (
                        kinematic.innovationMeters >
                            config.maximumKinematicInnovationMeters
                    ) {
                        listOf("kinematic-innovation-exceeds-production-envelope")
                    } else {
                        emptyList()
                    }
                ).distinct().take(16),
            jerkMetersPerSecond3 = state.jerk,
            motionStability = kinematic.motionStability,
            kinematicInnovationMeters = kinematic.innovationMeters,
            kinematicModelConfidence = kinematic.modelConfidence,
            subMillimeterFeasibilityScore =
                result.subMillimeterFeasibilityScore,
            firstOrderDepthErrorMillimeters =
                result.firstOrderDepthErrorMillimeters,
            reprojectionP95Px =
                result.reprojectionP95Px,
            tailAdjustedFirstOrderDepthErrorMillimeters =
                result.tailAdjustedFirstOrderDepthErrorMillimeters,
            tailAdjustedSubMillimeterFeasibilityScore =
                result.tailAdjustedSubMillimeterFeasibilityScore,
            precisionQualified =
                result.subMillimeterFeasibilityScore >=
                    config.precisionMinimumFeasibility &&
                    result.tailAdjustedSubMillimeterFeasibilityScore >=
                        config.precisionMinimumFeasibility &&
                    result.firstOrderDepthErrorMillimeters.isFinite() &&
                    result.firstOrderDepthErrorMillimeters <=
                        config.precisionMaximumFirstOrderDepthErrorMillimeters &&
                    result.tailAdjustedFirstOrderDepthErrorMillimeters.isFinite() &&
                    result.tailAdjustedFirstOrderDepthErrorMillimeters <=
                        config.precisionMaximumFirstOrderDepthErrorMillimeters &&
                    result.reprojectionP95Px.isFinite() &&
                    result.reprojectionP95Px <=
                        config.preferredReprojectionErrorPx * 1.5 &&
                    result.independentViewCount >=
                        config.minimumTrustedViewCount &&
                    result.viewDiversityScore >= 0.50 &&
                    activeViewInformationGain >=
                        config.minimumMetricEvidenceAuthorityForPrecision &&
                    metricEvidenceAuthority >=
                        config.minimumMetricEvidenceAuthorityForPrecision &&
                    result.anisotropicWeakAxisConstraint >=
                        config.precisionMinimumAnisotropicWeakAxisConstraint &&
                    result.anisotropicViewConditionScore >=
                        config.precisionMinimumAnisotropicViewConditionScore &&
                    kinematic.motionStability >=
                        config.precisionMinimumMotionStability,
        )
    }

    private fun selectGeometricViews(
        candidates: List<PointObservation>,
    ): Pair<List<PointObservation>, Int> {
        if (candidates.size < 2) {
            return candidates to 0
        }

        val anisotropic =
            anisotropicViewContributionEngine.rank(
                candidates =
                    candidates.map {
                        AnisotropicViewContributionEngine.Candidate(
                            observation = it.observation,
                            semanticConfidence =
                                it.semanticConfidence,
                            visibility = it.visibility,
                            sourceDiversity = it.sourceDiversity,
                            appearanceEvidence =
                                (
                                    it.photometricConsistency * 0.36 +
                                        it.silhouetteConsistency * 0.26 +
                                        it.motionBoundaryConfidence * 0.20 +
                                        it.lightingStability * 0.18
                                    ).coerceIn(0.0, 1.0),
                        )
                    },
                limit = candidates.size,
            )

        val anisotropicScoreByCamera =
            anisotropic.selected.associate {
                it.cameraId to it.totalScore
            }

        val anisotropicAxisScoreByCamera =
            anisotropic.selected.associate {
                it.cameraId to it.principalAxisConstraint
            }

        fun score(candidate: PointObservation): Double =
            (
                viewSelectionScore(candidate) * 0.58 +
                    (
                        anisotropicScoreByCamera[
                            candidate.observation.cameraId
                        ] ?: 0.0
                    ) * 0.42
            ).coerceIn(0.0, 1.0)

        if (candidates.size <= config.maximumGeometricViewsPerPoint) {
            return candidates.sortedByDescending(::score) to 0
        }

        val ranked =
            candidates.sortedByDescending(::score)
        val selected =
            LinkedHashSet<PointObservation>()

        ranked.firstOrNull()?.let {
            selected.add(it)
        }

        while (
            selected.size <
                min(
                    config.maximumGeometricViewsPerPoint,
                    candidates.size,
                )
        ) {
            val next =
                candidates
                    .filterNot {
                        selected.contains(it)
                    }
                    .maxByOrNull {
                        val anisotropicAxis =
                            anisotropicAxisScoreByCamera[
                                it.observation.cameraId
                            ] ?: 0.0
                        val baseMarginal =
                            marginalViewSelectionScore(
                                candidate = it,
                                selected = selected,
                            )
                        (
                            baseMarginal * 0.60 +
                                anisotropicAxis * 0.40
                        ).coerceIn(0.0, 1.0)
                    }
                    ?: break

            val marginal =
                (
                    marginalViewSelectionScore(
                        candidate = next,
                        selected = selected,
                    ) * 0.60 +
                        (
                            anisotropicAxisScoreByCamera[
                                next.observation.cameraId
                            ] ?: 0.0
                        ) * 0.40
                ).coerceIn(0.0, 1.0)

            if (
                selected.size >=
                    config.minimumGeometricViewsPerPoint &&
                marginal < config.minimumViewSelectionScore
            ) {
                break
            }

            selected.add(next)
        }

        if (
            selected.size <
                config.minimumGeometricViewsPerPoint
        ) {
            ranked
                .filterNot {
                    selected.contains(it)
                }
                .take(
                    config.minimumGeometricViewsPerPoint -
                        selected.size,
                )
                .forEach {
                    selected.add(it)
                }
        }

        val selectedList = selected.toList()
        return selectedList to (
            candidates.size -
                selectedList.size
            ).coerceAtLeast(0)
    }

    private fun viewSelectionScore(
        candidate: PointObservation,
    ): Double {
        val geometryEvidence =
            (
                candidate.semanticConfidence * 0.24 +
                    candidate.visibility * 0.15 +
                    candidate.observation.temporalAlignmentConfidence * 0.17 +
                    candidate.observation.temporalStability * 0.13 +
                    candidate.observation.crossViewAgreement * 0.11 +
                    (1.0 - candidate.observation.observationUncertainty) * 0.08 +
                    candidate.sourceDiversity * 0.04
                ).coerceIn(0.0, 1.0)
        val appearanceEvidence =
            (
                candidate.photometricConsistency * 0.36 +
                    candidate.silhouetteConsistency * 0.26 +
                    candidate.motionBoundaryConfidence * 0.20 +
                    candidate.lightingStability * 0.18
                ).coerceIn(0.0, 1.0)
        return (
            geometryEvidence * (1.0 - config.appearanceSelectionWeight) +
                appearanceEvidence * config.appearanceSelectionWeight
            ).coerceIn(0.0, 1.0)
    }

    private fun marginalViewSelectionScore(
        candidate: PointObservation,
        selected: Set<PointObservation>,
    ): Double {
        if (selected.isEmpty()) {
            return viewSelectionScore(candidate)
        }

        val diversity = selected.maxOfOrNull {
            pairwiseViewDiversity(
                candidate.observation,
                it.observation,
            )
        } ?: 0.0

        val anchor = selected.first()
        val baselineNovelty = pairwiseViewDiversity(
            candidate.observation,
            anchor.observation,
        )

        return (
            viewSelectionScore(candidate) *
                config.viewSelectionEvidenceWeight +
                baselineNovelty *
                config.viewSelectionBaselineWeight +
                diversity *
                config.viewSelectionOrientationWeight
            ).coerceIn(0.0, 1.0)
    }

    private fun pairwiseViewDiversity(
        a: TriangulationEngine.Observation,
        b: TriangulationEngine.Observation,
    ): Double {
        val baseline =
            a.cameraPose.position.distanceTo(
                b.cameraPose.position,
            )

        val baselineScore =
            (
                baseline /
                    (baseline + config.viewSelectionBaselineScaleMeters)
                ).coerceIn(0.0, 1.0)

        val forwardA =
            a.cameraPose.rotation
                .rotate(
                    com.nivukx.nox.core.model.Vec3(
                        0.0,
                        0.0,
                        1.0,
                    ),
                )
                .normalized()

        val forwardB =
            b.cameraPose.rotation
                .rotate(
                    com.nivukx.nox.core.model.Vec3(
                        0.0,
                        0.0,
                        1.0,
                    ),
                )
                .normalized()

        val orientationScore =
            (
                kotlin.math.acos(
                    forwardA.dot(forwardB)
                        .coerceIn(-1.0, 1.0),
                ) / Math.PI
            ).coerceIn(0.0, 1.0)

        return (
            baselineScore * 0.62 +
                orientationScore * 0.38
            ).coerceIn(0.0, 1.0)
    }

    private fun geometricDiversity(
        candidates: List<PointObservation>,
    ): Double {
        if (candidates.size < 2) return 0.0
        var total = 0.0
        var count = 0
        for (i in candidates.indices) {
            for (j in (i + 1) until candidates.size) {
                total += pairwiseViewDiversity(
                    candidates[i].observation,
                    candidates[j].observation,
                )
                count++
            }
        }
        return (total / count.coerceAtLeast(1)).coerceIn(0.0, 1.0)
    }

    private fun List<Double>.averageOrZeroDense(): Double =
        if (isEmpty()) 0.0 else average().coerceIn(0.0, 1.0)

    private fun bridgeShortOcclusion(
        timestampNanos: Long,
        state: PointTrackState,
    ): PointResult? {
        if (state.observedTimestampNanos == Long.MIN_VALUE) return null

        val ageMillis =
            (timestampNanos - state.observedTimestampNanos)
                .coerceAtLeast(0L)
                .toDouble() / 1_000_000.0

        if (ageMillis > config.predictionHorizonMillis) {
            state.state = PointState.STALE
            cumulativeStalePoints++
            return null
        }

        val motionRecoveryPrediction =
            motionAwareRecoveryEngine.predict(
                trackId = motionTrackId(state.key),
                timestampNanos = timestampNanos,
            )

        val kinematicPrediction =
            kinematicEngines[state.key]
                ?.predict(timestampNanos)

        val prediction =
            when {
                motionRecoveryPrediction != null &&
                    motionRecoveryPrediction.positionMeters.isFinite() ->
                    motionRecoveryPrediction.positionMeters to
                        motionRecoveryPrediction.velocityMetersPerSecond

                kinematicPrediction != null &&
                    kinematicPrediction.positionMeters.isFinite() ->
                    kinematicPrediction.positionMeters to
                        kinematicPrediction.velocityMetersPerSecond

                else ->
                    predictPosition(timestampNanos, state)
            }

        val decay =
            temporalFreshness(ageMillis)

        val confidence =
            max(
                config.predictionConfidenceFloor,
                state.confidence * decay * 0.78,
            )

        val uncertainty =
            min(
                config.hardPointUncertaintyMeters,
                state.uncertainty95Meters +
                    (
                        ageMillis /
                            config.predictionHorizonMillis
                                .coerceAtLeast(1.0)
                        ) * 0.12,
            )

        state.timestampNanos = timestampNanos
        state.predictedFrames++
        state.state = PointState.TEMPORALLY_BRIDGED
        state.uncertainty95Meters = uncertainty

        return PointResult(
            key = state.key,
            timestampNanos = timestampNanos,
            positionMeters = prediction.first,
            velocityMetersPerSecond = state.velocity,
            accelerationMetersPerSecond2 = state.acceleration,
            confidence = confidence,
            uncertaintyMeters = uncertainty * 0.65,
            uncertainty95Meters = uncertainty,
            reprojectionErrorPx =
                state.lastReprojectionErrorPx
                    .takeIf(Double::isFinite)
                    ?: Double.POSITIVE_INFINITY,
            parallaxDegrees = state.lastParallaxDegrees,
            cheiralityRatio = 1.0,
            viewCount = 0,
            independentViewCount = 0,
            viewDiversityScore =
                state.lastViewDiversity * 0.72,
            temporalStability = decay,
            topologyStability =
                state.topologyStability * decay,
            contactSupport =
                state.contactSupport * decay,
            state = PointState.TEMPORALLY_BRIDGED,
            predicted = true,
            staleObservation = false,
            warnings = (
                listOf(
                    "dense point temporarily bridged by bounded temporal prediction",
                ) +
                    listOfNotNull(
                        motionRecoveryPrediction
                            ?.warnings
                            ?.firstOrNull(),
                        kinematicEngines[state.key]
                            ?.latest()
                            ?.warnings
                            ?.firstOrNull(),
                    )
                ).distinct(),
            jerkMetersPerSecond3 = state.jerk,
            motionStability = state.confidence * decay,
            kinematicInnovationMeters = 0.0,
            kinematicModelConfidence = confidence,
        )
    }

    private fun rejectedPoint(
        timestampNanos: Long,
        key: PointKey,
        warning: String,
    ): PointResult =
        PointResult(
            key = key,
            timestampNanos = timestampNanos,
            positionMeters =
                com.nivukx.nox.core.model.Vec3(
                    0.0,
                    0.0,
                    0.0,
                ),
            velocityMetersPerSecond =
                com.nivukx.nox.core.model.Vec3(
                    0.0,
                    0.0,
                    0.0,
                ),
            accelerationMetersPerSecond2 =
                com.nivukx.nox.core.model.Vec3(
                    0.0,
                    0.0,
                    0.0,
                ),
            confidence = 0.0,
            uncertaintyMeters =
                Double.POSITIVE_INFINITY,
            uncertainty95Meters =
                Double.POSITIVE_INFINITY,
            reprojectionErrorPx =
                Double.POSITIVE_INFINITY,
            parallaxDegrees = 0.0,
            cheiralityRatio = 0.0,
            viewCount = 0,
            independentViewCount = 0,
            viewDiversityScore = 0.0,
            temporalStability = 0.0,
            topologyStability = 0.0,
            contactSupport = 0.0,
            state = PointState.GEOMETRY_REJECTED,
            predicted = false,
            staleObservation = false,
            warnings = listOf(warning),
        )

    private fun geometryConfidence(
        result: TriangulationEngine.Result,
    ): Double {
        val parallaxScore =
            (
                (result.parallaxDegrees -
                    config.minimumParallaxDegrees) /
                    (
                        config.preferredParallaxDegrees -
                            config.minimumParallaxDegrees
                        ).coerceAtLeast(1e-6)
                ).coerceIn(0.0, 1.0)

        val reprojectionScore =
            exp(
                -result.rmsReprojectionPx /
                    config.preferredReprojectionErrorPx
                        .coerceAtLeast(0.1),
            ).coerceIn(0.0, 1.0)

        val uncertaintyScore =
            exp(
                -result.uncertainty95MajorMeters /
                    config.minimumPointUncertaintyMeters,
            ).coerceIn(0.0, 1.0)

        val cheiralityScore =
            (
                (
                    result.cheiralityRatio -
                        config.hardCheiralityRatio
                    ) /
                    (
                        config.minimumCheiralityRatio -
                            config.hardCheiralityRatio
                        ).coerceAtLeast(1e-6)
                ).coerceIn(0.0, 1.0)

        val diversityScore =
            result.viewDiversityScore.coerceIn(0.0, 1.0)

        val uncertaintyConfidence =
            result.uncertaintyConfidence.coerceIn(0.0, 1.0)

        val probabilisticEvidence =
            result.probabilisticEvidenceScore
                .coerceIn(0.0, 1.0)

        val reprojectionConsensus =
            result.pairwiseReprojectionConsensus
                .coerceIn(0.0, 1.0)

        return (
            parallaxScore * 0.20 +
                reprojectionScore * 0.20 +
                reprojectionConsensus * 0.10 +
                uncertaintyScore * 0.20 +
                cheiralityScore * 0.15 +
                diversityScore * 0.05 +
                uncertaintyConfidence * 0.05 +
                probabilisticEvidence * 0.05
            ).coerceIn(0.0, 1.0)
    }

    private fun effectiveSemanticWeight(
        observation: PointObservation,
    ): Double =
        (
            observation.semanticConfidence.coerceIn(0.0, 1.0) *
                (0.35 + 0.65 * observation.visibility) *
                (0.40 + 0.60 * observation.sourceDiversity)
            ).coerceIn(0.0, 1.0)

    private fun stateContinuityScore(
        timestampNanos: Long,
        state: PointTrackState,
        candidate:
            com.nivukx.nox.core.model.Vec3,
    ): Double {
        if (state.timestampNanos == Long.MIN_VALUE) return 0.55

        val prediction =
            predictPosition(
                timestampNanos,
                state,
            )

        val predicted =
            prediction?.first ?: state.position

        val error =
            candidate.distanceTo(predicted)

        val scale =
            (
                config.minimumPointUncertaintyMeters *
                    3.0
                ).coerceAtLeast(0.01)

        return exp(-error / scale)
            .coerceIn(0.0, 1.0)
    }

    private fun updateKinematics(
        state: PointTrackState,
        timestampNanos: Long,
        position:
            com.nivukx.nox.core.model.Vec3,
    ): Pair<
        com.nivukx.nox.core.model.Vec3,
        com.nivukx.nox.core.model.Vec3,
    > {
        if (state.timestampNanos == Long.MIN_VALUE) {
            return com.nivukx.nox.core.model.Vec3(
                0.0,
                0.0,
                0.0,
            ) to com.nivukx.nox.core.model.Vec3(
                0.0,
                0.0,
                0.0,
            )
        }

        val dt =
            (
                timestampNanos -
                    state.timestampNanos
                ).coerceAtLeast(1L)
                .toDouble() / 1_000_000_000.0

        val rawVelocity =
            (position - state.position) /
                dt.coerceAtLeast(1e-6)

        val speed =
            rawVelocity.length()

        val velocityScale =
            min(
                1.0,
                config.maximumPredictionSpeedMetersPerSecond /
                    speed.coerceAtLeast(1e-9),
            )

        val velocity =
            state.velocity * 0.54 +
                rawVelocity * velocityScale * 0.46

        val rawAcceleration =
            (velocity - state.velocity) /
                dt.coerceAtLeast(1e-6)

        val accelerationMagnitude =
            rawAcceleration.length()

        val accelerationScale =
            min(
                1.0,
                config.maximumPredictionAccelerationMetersPerSecond2 /
                    accelerationMagnitude.coerceAtLeast(1e-9),
            )

        val acceleration =
            state.acceleration * 0.64 +
                rawAcceleration * accelerationScale * 0.36

        return velocity to acceleration
    }

    private fun predictPosition(
        timestampNanos: Long,
        state: PointTrackState,
    ): Pair<
        com.nivukx.nox.core.model.Vec3,
        com.nivukx.nox.core.model.Vec3,
    > {
        val dt =
            if (state.timestampNanos == Long.MIN_VALUE) {
                0.0
            } else {
                (
                    timestampNanos -
                        state.timestampNanos
                    ).coerceAtLeast(0L)
                    .toDouble() / 1_000_000_000.0
            }

        val limitedDt =
            min(
                dt,
                config.predictionHorizonMillis / 1000.0,
            )

        val predicted =
            state.position +
                state.velocity * limitedDt +
                state.acceleration *
                (0.5 * limitedDt * limitedDt) +
                state.jerk *
                (limitedDt * limitedDt * limitedDt / 6.0)

        return predicted to state.velocity
    }

    private fun temporalFreshness(ageMillis: Double): Double =
        if (ageMillis <= 0.0) {
            1.0
        } else {
            exp(
                -ln(2.0) *
                    ageMillis /
                    config.temporalHalfLifeMillis,
            ).coerceIn(0.0, 1.0)
        }

    private fun evaluateTopology(
        timestampNanos: Long,
        points: List<PointResult>,
        edges: List<TopologyEdge>,
    ): List<EdgeEvaluation> {
        if (edges.isEmpty()) return emptyList()

        val pointMap =
            points.associateBy { it.key }

        return edges.take(
            config.maximumPoints *
                config.maximumEdgesPerPoint,
        ).mapNotNull { edge ->
            val a = pointMap[edge.a]
            val b = pointMap[edge.b]
            if (
                a == null ||
                b == null ||
                !a.positionMeters.isFinite() ||
                !b.positionMeters.isFinite()
            ) {
                return@mapNotNull null
            }

            val length =
                a.positionMeters.distanceTo(
                    b.positionMeters,
                )

            val normalizedStrain =
                abs(
                    length - edge.restLengthMeters,
                ) /
                    edge.restLengthMeters
                        .coerceAtLeast(1e-6)

            val confidence =
                (
                    a.confidence *
                        b.confidence *
                        edge.confidence *
                        (
                            0.45 +
                                0.55 * edge.rigidity
                            )
                    ).coerceIn(0.0, 1.0)

            val temporalWeight =
                (
                    a.temporalStability *
                        b.temporalStability
                    ).coerceIn(0.0, 1.0)

            val trusted =
                normalizedStrain <=
                    config.topologyWarningRatio &&
                    confidence >= 0.50 &&
                    temporalWeight >= 0.50

            val warning =
                normalizedStrain >
                    config.topologyWarningRatio

            EdgeEvaluation(
                edge = edge,
                actualLengthMeters = length,
                normalizedStrain = normalizedStrain,
                temporalWeight = temporalWeight,
                trusted = trusted,
                warning = warning,
            )
        }
    }

    private fun summarize(
        timestampNanos: Long,
        points: List<PointResult>,
        topology: List<EdgeEvaluation>,
    ): Result {
        if (points.isEmpty()) {
            return emptyResult(
                timestampNanos = timestampNanos,
                warning = "no dense metric points survived reconstruction",
            )
        }

        val finitePoints =
            points.filter { it.finite }

        val visiblePoints =
            finitePoints.filter {
                it.state != PointState.GEOMETRY_REJECTED &&
                    it.state != PointState.STALE
            }

        val trustedPoints =
            finitePoints.filter { it.trusted }

        val coverage =
            visiblePoints.size.toDouble() /
                points.size.toDouble().coerceAtLeast(1.0)

        val trustedRatio =
            trustedPoints.size.toDouble() /
                points.size.toDouble().coerceAtLeast(1.0)

        val confidence =
            finitePoints.map {
                it.confidence
            }.average().coerceIn(0.0, 1.0)

        val uncertainties =
            finitePoints.mapNotNull {
                it.uncertainty95Meters.takeIf(Double::isFinite)
            }

        val reprojection =
            finitePoints.mapNotNull {
                it.reprojectionErrorPx.takeIf(Double::isFinite)
            }

        val parallaxes =
            finitePoints.mapNotNull {
                it.parallaxDegrees.takeIf(Double::isFinite)
            }

        val temporal =
            finitePoints.map {
                it.temporalStability
            }.average().coerceIn(0.0, 1.0)

        val jerk =
            finitePoints.mapNotNull {
                it.jerkMetersPerSecond3
                    .length()
                    .takeIf(Double::isFinite)
            }.averageOrZero()

        val motionStability =
            finitePoints.map {
                it.motionStability
            }.average().coerceIn(0.0, 1.0)

        val kinematicInnovationMillimeters =
            finitePoints.mapNotNull {
                (
                    it.kinematicInnovationMeters * 1000.0
                ).takeIf(Double::isFinite)
            }.averageOrZero()

        val subMillimeterFeasibility =
            finitePoints.map {
                it.subMillimeterFeasibilityScore
            }.averageOrZero()

        val firstOrderDepthError =
            finitePoints.mapNotNull {
                it.firstOrderDepthErrorMillimeters
                    .takeIf(Double::isFinite)
            }.averageOrZero()

        val tailAdjustedDepthErrorP95 =
            finitePoints.mapNotNull {
                it.tailAdjustedFirstOrderDepthErrorMillimeters
                    .takeIf(Double::isFinite)
            }.let { values ->
                percentileOr(
                    values,
                    Double.POSITIVE_INFINITY,
                    0.95,
                )
            }

        val meanTailAdjustedFeasibility =
            finitePoints.map {
                it.tailAdjustedSubMillimeterFeasibilityScore
            }.averageOrZero()

        val topologyStable =
            topology.map {
                if (it.trusted) {
                    1.0 -
                        (
                            it.normalizedStrain /
                                config.topologyWarningRatio
                            ).coerceIn(0.0, 1.0) * 0.35
                } else {
                    0.35
                }
            }.takeIf { it.isNotEmpty() }
                ?.average()
                ?.coerceIn(0.0, 1.0)
                ?: 1.0

        val meanViewDiversity =
            finitePoints.map {
                it.viewDiversityScore
            }.average().coerceIn(0.0, 1.0)

        val predictedRatio =
            finitePoints.count {
                it.predicted
            }.toDouble() /
                points.size.toDouble().coerceAtLeast(1.0)

        val staleRatio =
            points.count {
                it.staleObservation ||
                    it.state == PointState.STALE
            }.toDouble() /
                points.size.toDouble().coerceAtLeast(1.0)

        val contact =
            finitePoints.map {
                it.contactSupport
            }.average().coerceIn(0.0, 1.0)

        val topologyWarnings =
            topology.count { it.warning }

        val acceptedCount =
            visiblePoints.count {
                it.state ==
                    PointState.TRUSTED ||
                    it.state ==
                    PointState.OBSERVED ||
                    it.state ==
                    PointState.TEMPORALLY_BRIDGED
            }

        val rejectedCount =
            points.count {
                it.state ==
                    PointState.GEOMETRY_REJECTED ||
                    it.state ==
                    PointState.STALE
            }

        val spatialRegions =
            finitePoints.map { point ->
                SpatialMultiScaleDetailAuthority.RegionEvidence(
                    regionId =
                        point.key.subjectId.toLong() * 1_000_000L +
                            point.key.pointId.toLong(),
                    geometricComplexity =
                        (
                            (1.0 - point.topologyStability) * 0.60 +
                                (
                                    1.0 -
                                        point.anisotropicViewConditionScore
                                ) * 0.40
                            ).coerceIn(0.0, 1.0),
                    motionBoundary =
                        point.motionBoundaryConfidence,
                    semanticImportance =
                        (
                            point.contactSupport * 0.45 +
                                point.appearanceConsistency * 0.30 +
                                point.silhouetteConsistency * 0.25
                            ).coerceIn(0.0, 1.0),
                    uncertaintyMeters =
                        point.uncertainty95Meters.coerceAtLeast(0.0),
                    viewInformationGain =
                        (
                            point.activeViewInformationGainScore * 0.42 +
                                point.activeViewRedundancyScore * 0.18 +
                                point.viewDiversityScore * 0.24 +
                                point.anisotropicWeakAxisConstraint * 0.16
                            ).coerceIn(0.0, 1.0),
                    temporalStability =
                        point.temporalStability,
                    metricEvidenceAuthority =
                        point.metricEvidenceAuthority,
                )
            }

        val spatialAllocation =
            spatialDetailAuthority.allocate(
                regions = spatialRegions,
                availableBudgetUnits =
                    max(
                        4_096,
                        finitePoints.size * 2,
                    ),
            )

        val detailByRegion =
            spatialAllocation.assignments.associateBy {
                it.regionId
            }

        val detailedPoints =
            points.map { point ->
                val regionId =
                    point.key.subjectId.toLong() * 1_000_000L +
                        point.key.pointId.toLong()
                val detail =
                    detailByRegion[regionId]
                point.copy(
                    spatialDetailBand =
                        detail?.band
                            ?: SpatialMultiScaleDetailAuthority.Band.COARSE,
                    spatialDetailPriority =
                        detail?.priority ?: 0.0,
                    spatialDetailComputeShare =
                        detail?.computeShare ?: 0.0,
                )
            }

        val warningList =
            buildList {
                if (coverage < 0.50) add(
                    "dense reconstruction coverage is below 50%",
                )
                if (trustedRatio < config.minimumTrustedPointRatio) add(
                    "trusted dense point ratio is below the production band",
                )
                if (
                    uncertainties.isNotEmpty() &&
                    percentile(
                        uncertainties,
                        0.95,
                    ) > 0.20
                ) {
                    add(
                        "dense point uncertainty P95 is elevated",
                    )
                }
                if (
                    reprojection.isNotEmpty() &&
                    percentile(
                        reprojection,
                        0.95,
                    ) > config.preferredReprojectionErrorPx * 1.8
                ) {
                    add(
                        "dense point reprojection P95 is elevated",
                    )
                }
                if (meanViewDiversity < 0.50) add(
                    "dense point view diversity is limited",
                )
                if (temporal < 0.55) add(
                    "dense point temporal stability is degraded",
                )
                if (tailAdjustedDepthErrorP95.isFinite() &&
                    tailAdjustedDepthErrorP95 >
                        config.precisionMaximumFirstOrderDepthErrorMillimeters
                ) {
                    add(
                        "dense reprojection tail limits first-order precision evidence",
                    )
                }
                if (meanTailAdjustedFeasibility <
                    config.precisionMinimumFeasibility * 0.80
                ) {
                    add(
                        "dense tail-adjusted precision feasibility is degraded",
                    )
                }
                if (topologyWarnings > 0) add(
                    "dense topology strain warnings are active",
                )
                if (
                    predictedRatio >
                        0.25
                ) {
                    add(
                        "a material fraction of dense points are temporarily predicted",
                    )
                }
                addAll(
                    spatialAllocation.warnings,
                )
            }

        val hardFailure =
            finitePoints.isEmpty() ||
                (
                    coverage < 0.20 &&
                        trustedRatio < 0.10
                    ) ||
                (
                    uncertainties.isNotEmpty() &&
                        percentile(
                            uncertainties,
                            0.99,
                        ) >
                        config.hardPointUncertaintyMeters
                    ) ||
                (
                    topology.any {
                        it.normalizedStrain >
                            config.topologyHardRatio
                    }
                )

        val precisionQualifiedPointRatio =
            finitePoints.count { it.precisionQualified }.toDouble() /
                finitePoints.size.toDouble().coerceAtLeast(1.0)

        val precisionQualified =
            !hardFailure &&
                precisionQualifiedPointRatio >=
                    config.minimumTrustedPointRatio &&
                subMillimeterFeasibility >=
                    config.precisionMinimumFeasibility &&
                meanTailAdjustedFeasibility >=
                    config.precisionMinimumFeasibility &&
                firstOrderDepthError.isFinite() &&
                firstOrderDepthError <=
                    config.precisionMaximumFirstOrderDepthErrorMillimeters &&
                tailAdjustedDepthErrorP95.isFinite() &&
                tailAdjustedDepthErrorP95 <=
                    config.precisionMaximumFirstOrderDepthErrorMillimeters &&
                motionStability >=
                    config.precisionMinimumMotionStability

        val diagnostics =
            listOf(
                "densePoints=" + points.size,
                "finitePoints=" + finitePoints.size,
                "trustedPoints=" + trustedPoints.size,
                "topologyEdges=" + topology.size,
                "topologyWarnings=" + topologyWarnings,
                "coverage=" + metric(coverage),
                "trustedPointRatio=" + metric(trustedRatio),
                "meanConfidence=" + metric(confidence),
                "uncertaintyP95=" +
                    metric(
                        percentileOr(
                            uncertainties,
                            Double.POSITIVE_INFINITY,
                        ),
                    ),
                "uncertaintyP99=" +
                    metric(
                        percentileOr(
                            uncertainties,
                            Double.POSITIVE_INFINITY,
                        ),
                    ),
                "reprojectionP95=" +
                    metric(
                        percentileOr(
                            reprojection,
                            Double.POSITIVE_INFINITY,
                        ),
                    ),
                "meanParallax=" +
                    metric(
                        parallaxes.averageOrZero(),
                    ),
                "temporalStability=" + metric(temporal),
                "motionStability=" + metric(motionStability),
                "meanJerk=" + metric(jerk),
                "kinematicInnovationMm=" +
                    metric(kinematicInnovationMillimeters),
                "subMillimeterFeasibility=" +
                    metric(subMillimeterFeasibility),
                "firstOrderDepthErrorMm=" +
                    metric(firstOrderDepthError),
                "tailAdjustedDepthErrorP95Mm=" +
                    metric(tailAdjustedDepthErrorP95),
                "tailAdjustedFeasibility=" +
                    metric(meanTailAdjustedFeasibility),
                "precisionQualifiedPointRatio=" +
                    metric(precisionQualifiedPointRatio),
                "topologyStability=" + metric(topologyStable),
                "viewDiversity=" + metric(meanViewDiversity),
                "adaptiveViewSelection=" +
                    metric(
                        points
                            .map { it.adaptiveViewSelectionScore }
                            .averageOrZero(),
                    ),
                "adaptiveViewDiversity=" +
                    metric(
                        points
                            .map { it.adaptiveViewDiversityScore }
                            .averageOrZero(),
                    ),
                "anisotropicWeakAxisConstraint=" +
                    metric(
                        points
                            .map { it.anisotropicWeakAxisConstraint }
                            .averageOrZero(),
                    ),
                "anisotropicViewCondition=" +
                    metric(
                        points
                            .map { it.anisotropicViewConditionScore }
                            .averageOrZero(),
                    ),
                "probabilisticEvidence=" +
                    metric(
                        points
                            .map { it.probabilisticEvidenceScore }
                            .averageOrZero(),
                    ),
                "activeViewInformationGain=" +
                    metric(
                        points
                            .map { it.activeViewInformationGainScore }
                            .averageOrZero(),
                    ),
                "activeViewRedundancy=" +
                    metric(
                        points
                            .map { it.activeViewRedundancyScore }
                            .averageOrZero(),
                    ),
                "metricEvidenceAuthority=" +
                    metric(
                        points
                            .map { it.metricEvidenceAuthority }
                            .averageOrZero(),
                    ),
                "excludedWeakViews=" +
                    points.sumOf { it.excludedWeakViewCount },
                "predictedRatio=" + metric(predictedRatio),
                "meanAppearanceConsistency=" +
                    metric(points.map { it.appearanceConsistency }.averageOrZero()),
                "meanSilhouetteConsistency=" +
                    metric(points.map { it.silhouetteConsistency }.averageOrZero()),
                "meanMotionBoundaryConfidence=" +
                    metric(points.map { it.motionBoundaryConfidence }.averageOrZero()),
                "meanLightingStability=" +
                    metric(points.map { it.lightingStability }.averageOrZero()),
                "spatialDetailPriority=" +
                    metric(spatialAllocation.meanPriority),
                "spatialDetailBudgetUtilization=" +
                    metric(spatialAllocation.budgetUtilization),
                "spatialDetailUltraRatio=" +
                    metric(spatialAllocation.ultraRatio),
                "spatialDetailFineOrHigherRatio=" +
                    metric(spatialAllocation.fineOrHigherRatio),
            )

        diagnostics.forEach(::appendDiagnostic)

        return Result(
            timestampNanos = timestampNanos,
            points = detailedPoints,
            topologyEdgesEvaluated = topology.size,
            topologyEdgesTrusted =
                topology.count { it.trusted },
            topologyWarningCount =
                topologyWarnings,
            coverage = coverage,
            trustedPointRatio = trustedRatio,
            meanConfidence = confidence,
            uncertaintyP50Meters =
                percentileOr(
                    uncertainties,
                    Double.POSITIVE_INFINITY,
                    0.50,
                ),
            uncertaintyP95Meters =
                percentileOr(
                    uncertainties,
                    Double.POSITIVE_INFINITY,
                    0.95,
                ),
            uncertaintyP99Meters =
                percentileOr(
                    uncertainties,
                    Double.POSITIVE_INFINITY,
                    0.99,
                ),
            meanReprojectionErrorPx =
                reprojection.averageOrZero(),
            p95ReprojectionErrorPx =
                percentileOr(
                    reprojection,
                    Double.POSITIVE_INFINITY,
                    0.95,
                ),
            meanParallaxDegrees =
                parallaxes.averageOrZero(),
            temporalStability = temporal,
            topologyStability = topologyStable,
            viewDiversity = meanViewDiversity,
            adaptiveViewSelectionScore =
                points.map { it.adaptiveViewSelectionScore }.averageOrZero(),
            adaptiveViewDiversityScore =
                points.map { it.adaptiveViewDiversityScore }.averageOrZero(),
            excludedWeakViewCount =
                points.sumOf { it.excludedWeakViewCount },
            predictedPointRatio = predictedRatio,
            stalePointRatio = staleRatio,
            contactSupport = contact,
            meanAppearanceConsistency =
                points.map { it.appearanceConsistency }.averageOrZero(),
            meanSilhouetteConsistency =
                points.map { it.silhouetteConsistency }.averageOrZero(),
            meanMotionBoundaryConfidence =
                points.map { it.motionBoundaryConfidence }.averageOrZero(),
            meanLightingStability =
                points.map { it.lightingStability }.averageOrZero(),
            acceptedPointCount = acceptedCount,
            rejectedPointCount = rejectedCount,
            hardFailure = hardFailure,
            warnings = warningList.distinct(),
            diagnostics = diagnostics,
            meanJerkMetersPerSecond3 = jerk,
            meanMotionStability = motionStability,
            meanKinematicInnovationMillimeters =
                kinematicInnovationMillimeters,
            meanSubMillimeterFeasibilityScore =
                subMillimeterFeasibility,
            meanFirstOrderDepthErrorMillimeters =
                firstOrderDepthError,
            p95ReprojectionTailAdjustedDepthErrorMillimeters =
                tailAdjustedDepthErrorP95,
            meanTailAdjustedSubMillimeterFeasibilityScore =
                meanTailAdjustedFeasibility,
            meanAnisotropicWeakAxisConstraint =
                points
                    .map { it.anisotropicWeakAxisConstraint }
                    .averageOrZero(),
            meanAnisotropicViewConditionScore =
                points
                    .map { it.anisotropicViewConditionScore }
                    .averageOrZero(),
            meanProbabilisticEvidenceScore =
                points
                    .map { it.probabilisticEvidenceScore }
                    .averageOrZero(),
            meanActiveViewInformationGainScore =
                points
                    .map { it.activeViewInformationGainScore }
                    .averageOrZero(),
            meanActiveViewRedundancyScore =
                points
                    .map { it.activeViewRedundancyScore }
                    .averageOrZero(),
            meanMetricEvidenceAuthority =
                points
                    .map { it.metricEvidenceAuthority }
                    .averageOrZero(),
            meanSpatialDetailPriority =
                spatialAllocation.meanPriority,
            spatialDetailBudgetUtilization =
                spatialAllocation.budgetUtilization,
            ultraDetailPointRatio =
                spatialAllocation.ultraRatio,
            fineOrHigherDetailPointRatio =
                spatialAllocation.fineOrHigherRatio,
            spatialDetailHold =
                spatialAllocation.holdOutput,
            precisionQualified =
                precisionQualified,
            authoritative =
                !hardFailure &&
                    coverage >= 0.50 &&
                    trustedRatio >=
                        config.minimumTrustedPointRatio &&
                    percentileOr(
                        uncertainties,
                        Double.POSITIVE_INFINITY,
                        0.95,
                    ) <=
                        config.hardPointUncertaintyMeters &&
                    points
                        .map { it.metricEvidenceAuthority }
                        .averageOrZero() >=
                        config.minimumMetricEvidenceAuthorityForPrecision &&
                    points
                        .map { it.anisotropicWeakAxisConstraint }
                        .averageOrZero() >=
                        config.precisionMinimumAnisotropicWeakAxisConstraint &&
                    points
                        .map { it.anisotropicViewConditionScore }
                        .averageOrZero() >=
                        config.precisionMinimumAnisotropicViewConditionScore &&
                    topologyStable >=
                        config.topologyAuthorityRatio,
        )
    }

    private fun prune(timestampNanos: Long) {
        val cutoff =
            timestampNanos -
                (
                    config.stalePointMillis *
                        1_000_000.0
                    ).toLong().coerceAtLeast(0L)

        val iterator =
            states.entries.iterator()

        while (iterator.hasNext()) {
            val state =
                iterator.next().value
            if (
                state.observedTimestampNanos !=
                    Long.MIN_VALUE &&
                state.observedTimestampNanos < cutoff
            ) {
                kinematicEngines.remove(state.key)?.close()
                iterator.remove()
            }
        }

        while (states.size > config.maximumPoints) {
            val oldest =
                states.entries.minByOrNull {
                    it.value.observedTimestampNanos
                        .takeIf { value ->
                            value != Long.MIN_VALUE
                        }
                        ?: Long.MIN_VALUE
                }?.key ?: break
            kinematicEngines.remove(oldest)?.close()
            states.remove(oldest)
        }

        while (kinematicEngines.size > config.maximumKinematicTracks) {
            val oldestKey =
                kinematicEngines.entries.firstOrNull()?.key
                    ?: break
            kinematicEngines.remove(oldestKey)?.close()
        }
    }

    private fun motionTrackId(
        key: PointKey,
    ): Int =
        (
            key.subjectId.coerceIn(0, 31) * 100_000 +
                key.pointId.coerceIn(0, 4_095)
            ).coerceIn(0, 3_199_999)

    private fun kinematicEngineFor(
        key: PointKey,
    ): KinematicStabilityEngine {
        val existing = kinematicEngines[key]
        if (existing != null) return existing

        while (kinematicEngines.size >= config.maximumKinematicTracks) {
            val oldestKey =
                kinematicEngines.entries.firstOrNull()?.key
                    ?: break
            if (oldestKey == key) break
            kinematicEngines.remove(oldestKey)?.close()
        }

        return KinematicStabilityEngine(
            KinematicStabilityEngine.Config(
                maximumHistorySamples = 12,
                minimumRegressionSamples = 4,
                maximumRegressionWindowSeconds = 0.16,
                maximumPredictionHorizonSeconds =
                    config.predictionHorizonMillis / 1000.0,
                maximumSpeedMetersPerSecond =
                    config.maximumPredictionSpeedMetersPerSecond,
                maximumAccelerationMetersPerSecond2 =
                    config.maximumPredictionAccelerationMetersPerSecond2,
                maximumJerkMetersPerSecond3 = 1_200.0,
            ),
        ).also {
            kinematicEngines[key] = it
        }
    }

    private fun buildPointWarnings(
        result: TriangulationEngine.Result,
        pointConfidence: Double,
        temporalConfidence: Double,
        appearanceConsistency: Double,
        lightingStability: Double,
    ): List<String> =
        buildList {
            if (
                result.parallaxDegrees <
                    config.preferredParallaxDegrees
            ) {
                add("point parallax is below the preferred dense-surface band")
            }
            if (
                result.rmsReprojectionPx >
                    config.preferredReprojectionErrorPx
            ) {
                add("point reprojection error is above the preferred band")
            }
            if (
                result.uncertainty95MajorMeters >
                    config.minimumPointUncertaintyMeters
            ) {
                add("point directional uncertainty is elevated")
            }
            if (
                result.viewDiversityScore < 0.50
            ) {
                add("point view diversity is limited")
            }
            if (
                result.uncertaintyDepthDominance > 0.80
            ) {
                add("depth-axis uncertainty dominates the point estimate")
            }
            if (
                temporalConfidence < 0.50
            ) {
                add("point temporal evidence is degraded")
            }
            if (
                pointConfidence < 0.65
            ) {
                add("point confidence is below the trusted threshold")
            }
            if (appearanceConsistency < config.minimumAppearanceTrust) {
                add("appearance consistency is below the trust band")
            }
            if (lightingStability < 0.50) {
                add("lighting stability is degrading surface evidence")
            }
            if (result.subMillimeterFeasibilityScore < 0.50) {
                result.subMillimeterWarnings
                    .take(3)
                    .forEach(::add)
            }
        }

    private fun emptyResult(
        timestampNanos: Long,
        warning: String,
    ): Result =
        Result(
            timestampNanos = timestampNanos,
            points = emptyList(),
            topologyEdgesEvaluated = 0,
            topologyEdgesTrusted = 0,
            topologyWarningCount = 0,
            coverage = 0.0,
            trustedPointRatio = 0.0,
            meanConfidence = 0.0,
            uncertaintyP50Meters = Double.POSITIVE_INFINITY,
            uncertaintyP95Meters = Double.POSITIVE_INFINITY,
            uncertaintyP99Meters = Double.POSITIVE_INFINITY,
            meanReprojectionErrorPx = Double.POSITIVE_INFINITY,
            p95ReprojectionErrorPx = Double.POSITIVE_INFINITY,
            meanParallaxDegrees = 0.0,
            temporalStability = 0.0,
            topologyStability = 0.0,
            viewDiversity = 0.0,
            adaptiveViewSelectionScore = 0.0,
            adaptiveViewDiversityScore = 0.0,
            excludedWeakViewCount = 0,
            predictedPointRatio = 0.0,
            stalePointRatio = 0.0,
            contactSupport = 0.0,
            meanAppearanceConsistency = 0.0,
            meanSilhouetteConsistency = 0.0,
            meanMotionBoundaryConfidence = 0.0,
            meanLightingStability = 0.0,
            acceptedPointCount = 0,
            rejectedPointCount = 0,
            hardFailure = true,
            warnings = listOf(warning),
            diagnostics = diagnostics.toList(),
            meanJerkMetersPerSecond3 = 0.0,
            meanMotionStability = 0.0,
            meanKinematicInnovationMillimeters = 0.0,
            meanSubMillimeterFeasibilityScore = 0.0,
            meanFirstOrderDepthErrorMillimeters = Double.POSITIVE_INFINITY,
            p95ReprojectionTailAdjustedDepthErrorMillimeters = Double.POSITIVE_INFINITY,
            meanTailAdjustedSubMillimeterFeasibilityScore = 0.0,
            meanAnisotropicWeakAxisConstraint = 0.0,
            meanAnisotropicViewConditionScore = 0.0,
            meanProbabilisticEvidenceScore = 0.0,
            meanSpatialDetailPriority = 0.0,
            spatialDetailBudgetUtilization = 0.0,
            ultraDetailPointRatio = 0.0,
            fineOrHigherDetailPointRatio = 0.0,
            spatialDetailHold = true,
        )

    private fun appendDiagnostic(message: String) {
        if (message.isBlank()) return
        diagnostics.addLast(message)
        while (diagnostics.size > config.maximumDiagnostics) {
            diagnostics.removeFirst()
        }
    }

    private fun metric(value: Double): String =
        "%.4f".format(
            if (value.isFinite()) value else 99.0,
        )

    private fun percentileOr(
        values: List<Double>,
        fallback: Double,
        quantile: Double = 0.95,
    ): Double {
        if (values.isEmpty()) return fallback
        val sorted = values.filter { it.isFinite() }.sorted()
        if (sorted.isEmpty()) return fallback
        if (sorted.size == 1) return sorted.first()

        val position =
            quantile.coerceIn(0.0, 1.0) *
                (sorted.size - 1).toDouble()

        val lower = position.toInt()
        val upper = min(
            sorted.lastIndex,
            lower + 1,
        )
        val fraction =
            position - lower

        return sorted[lower] +
            (
                sorted[upper] -
                    sorted[lower]
                ) * fraction
    }

    private fun percentile(
        values: List<Double>,
        quantile: Double,
    ): Double =
        percentileOr(
            values = values,
            fallback = 1.0,
            quantile = quantile,
        )

    private fun List<Double>.averageOrZero(): Double =
        filter { it.isFinite() }
            .takeIf { it.isNotEmpty() }
            ?.average()
            ?.let { it }
            ?: 0.0

    internal fun com.nivukx.nox.core.model.Vec3.isFinite(): Boolean =
        x.isFinite() && y.isFinite() && z.isFinite()

    private fun com.nivukx.nox.core.model.Vec3.length(): Double =
        sqrt(
            x * x +
                y * y +
                z * z,
        )

    private fun com.nivukx.nox.core.model.Vec3.distanceTo(
        other: com.nivukx.nox.core.model.Vec3,
    ): Double =
        (this - other).length()
}

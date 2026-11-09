package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

/**
 * Production articulated 3D reconstruction.
 *
 * The engine treats every joint as a probabilistic measurement:
 * 1. synchronize camera observations,
 * 2. robustly triangulate,
 * 3. reject weak geometry,
 * 4. estimate depth uncertainty,
 * 5. run temporal filtering,
 * 6. apply learned skeleton constraints,
 * 7. recompute confidence from the resulting state.
 *
 * A smooth point is not automatically a correct point, so uncertainty and geometry quality are
 * carried all the way to the final SkeletonFrame.
 */
class SkeletonReconstructionEngine(
    private val triangulator: TriangulationEngine = TriangulationEngine(),
    private val filter: TemporalMotionFilter = TemporalMotionFilter(),
    private val kalman: AdaptiveKalman3DFilter = AdaptiveKalman3DFilter(),
    private val constraints: SkeletonConstraintSolver = SkeletonConstraintSolver(),
    private val qualityModel: JointQualityModel = JointQualityModel(),
    private val trajectoryOptimizer: KinematicTrajectoryOptimizer =
        KinematicTrajectoryOptimizer(),
    private val bundleRefiner: BundleAdjustmentRefiner =
        BundleAdjustmentRefiner(),
    private val evidenceFusion: MultiEvidence3DReconstructionEngine =
        MultiEvidence3DReconstructionEngine(),
    private val bodySpacePriorAlignmentEngine: BodySpacePriorAlignmentEngine =
        BodySpacePriorAlignmentEngine(),
    @Volatile
    private var jointObservationSubsetPlanner: JointObservationSubsetPlanner =
        JointObservationSubsetPlanner(),
    private val maxTemporalSkewNanos: Long = 20_000_000L,
    private val preferredTemporalSkewNanos: Long = 5_000_000L,
    private val minimumParallaxDegrees: Double = 0.55,
    private val maximumReprojectionErrorPx: Double = 8.0,
    private val maximumUncertaintyMeters: Double = 0.75,
) {
    init {
        require(maxTemporalSkewNanos > 0L)
        require(
            preferredTemporalSkewNanos > 0L &&
                preferredTemporalSkewNanos < maxTemporalSkewNanos,
        )
        require(minimumParallaxDegrees > 0.0)
        require(maximumReprojectionErrorPx > 0.0)
        require(maximumUncertaintyMeters > 0.0)
    }

    data class ObservabilityTuning(
        val maximumTemporalSkewMillis: Double = 20.0,
        val preferredTemporalSkewMillis: Double = 5.0,
        val minimumParallaxDegrees: Double = 0.55,
        val maximumReprojectionErrorPx: Double = 8.0,
        val maximumUncertaintyMeters: Double = 0.75,
        val maximumDirectionalUncertaintyMeters: Double = 0.60,
        val maximumDepthUncertaintyMeters: Double = 0.52,
        val maximumUncertaintyAnisotropyRatio: Double = 10.0,
        val minimumDirectionalUncertaintyConfidence: Double = 0.18,
        val rollingShutterCompensationTrust: Double = 0.88,
    ) {
        init {
            require(maximumTemporalSkewMillis > preferredTemporalSkewMillis)
            require(preferredTemporalSkewMillis > 0.0)
            require(minimumParallaxDegrees > 0.0)
            require(maximumReprojectionErrorPx > 0.0)
            require(maximumUncertaintyMeters > 0.0)
            require(maximumDirectionalUncertaintyMeters > 0.0)
            require(maximumDepthUncertaintyMeters > 0.0)
            require(maximumUncertaintyAnisotropyRatio >= 1.0)
            require(minimumDirectionalUncertaintyConfidence in 0.0..1.0)
            require(
                rollingShutterCompensationTrust in 0.0..1.0
            )
        }
    }

    @Volatile
    private var observabilityTuning =
        ObservabilityTuning()

    @Volatile
    private var lastDirectionalUncertainty =
        DirectionalUncertaintySummary(
            major95Meters = 0.0,
            depthMeters = 0.0,
            maxAnisotropyRatio = 1.0,
            confidence = 1.0,
            degradedJointCount = 0,
            warnings = emptyList(),
        )

    @Volatile
    private var lastMeanMetricEvidenceAuthority: Double = 0.0

    @Volatile
    private var lastMeanSymmetricEpipolarConsistencyScore: Double = 0.0

    fun metricEvidenceAuthoritySummary(): Double =
        lastMeanMetricEvidenceAuthority

    fun symmetricEpipolarConsistencySummary(): Double =
        lastMeanSymmetricEpipolarConsistencyScore

    fun directionalUncertaintySummary():
        DirectionalUncertaintySummary =
        lastDirectionalUncertainty

    @Synchronized
    fun configureObservability(
        tuning: ObservabilityTuning,
    ) {
        observabilityTuning = tuning
        filter.clear()
        kalman.clear()
        constraints.reset()
        trajectoryOptimizer.reset()
    }

    fun observabilityTuning(): ObservabilityTuning =
        observabilityTuning

    @Synchronized
    fun configurePrecisionTuning(
        tuning: TriangulationEngine.PrecisionTuning,
    ) {
        triangulator.configurePrecisionTuning(tuning)
    }

    fun precisionTuning(): TriangulationEngine.PrecisionTuning =
        triangulator.precisionTuning()

    @Synchronized
    fun configureTrajectoryTuning(
        tuning: KinematicTrajectoryOptimizer.Tuning,
    ) {
        trajectoryOptimizer.configure(tuning)
        trajectoryOptimizer.reset()
    }

    fun trajectoryTuning(): KinematicTrajectoryOptimizer.Tuning =
        trajectoryOptimizer.currentTuning()

    @Synchronized
    fun configureEvidenceFusion(
        tuning: MultiEvidence3DReconstructionEngine.Config,
    ) {
        evidenceFusion.configure(tuning)
    }

    fun evidenceFusionTuning(): MultiEvidence3DReconstructionEngine.Config =
        evidenceFusion.currentConfig()

    @Synchronized
    fun configureJointObservationTuning(
        tuning: JointObservationSubsetPlanner.Config,
    ) {
        jointObservationSubsetPlanner =
            JointObservationSubsetPlanner(tuning)
    }

        data class CameraFrame(
        val deviceId: DeviceId,
        val timestampNanos: Long,
        val observations: List<CameraObservation>,
        val frameDurationNanos: Long = 0L,
        val exposureTimeNanos: Long = 0L,
        val rollingShutterSkewNanos: Long = 0L,
        val captureTimingConfidence: Double = 1.0,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(frameDurationNanos >= 0L)
            require(exposureTimeNanos >= 0L)
            require(rollingShutterSkewNanos >= 0L)
            require(captureTimingConfidence in 0.0..1.0)
        }

        val estimatedMotionBlurRisk: Double
            get() = estimateOpticalMotionBlurRisk(
                captureTimingConfidence = captureTimingConfidence,
                rollingShutterSkewNanos = rollingShutterSkewNanos,
                frameDurationNanos = frameDurationNanos,
                exposureTimeNanos = exposureTimeNanos,
            )
    }

    data class JointDiagnostic(
        val landmarkId: Int,
        val accepted: Boolean,
        val parallaxDegrees: Double,
        val reprojectionErrorPx: Double,
        val uncertaintyMeters: Double,
        val inlierRatio: Double,
        val conditionScore: Double,
        val temporalConfidence: Double,
        val selectedObservationCount: Int = 0,
        val excludedObservationCount: Int = 0,
        val jointViewDiversitySatisfied: Boolean = false,
        val jointViewSelectionScore: Double = 0.0,
        val jointViewSelectionReason: String = "UNAVAILABLE",
        val directionalUncertaintyMeters: Double = 0.0,
        val directionalDepthUncertaintyMeters: Double = 0.0,
        val directionalUncertaintyAnisotropy: Double = 1.0,
        val directionalUncertaintyConfidence: Double = 1.0,
        val directionalUncertaintyWarnings: List<String> = emptyList(),
        val triangulationIndependentViewCount: Int = 0,
        val triangulationDominantViewWeightRatio: Double = 0.0,
        val triangulationViewDiversityScore: Double = 0.0,
        val triangulationViewDiversityWarnings: List<String> = emptyList(),
        val triangulationMetricEvidenceAuthority: Double = 0.0,
        val triangulationEpipolarConsistencyScore: Double = 1.0,
        val triangulationEpipolarWarnings: List<String> = emptyList(),
    )

    data class ReconstructionResult(
        val frame: SkeletonFrame,
        val diagnostics: List<JointDiagnostic>,
        /** Mean conservative metric authority across triangulated joints. */
        val meanMetricEvidenceAuthority: Double = 0.0,
        /** Mean symmetric epipolar consistency across triangulated joints. */
        val meanSymmetricEpipolarConsistencyScore: Double = 0.0,
        val epipolarWarnings: List<String> = emptyList(),
    )

    data class DirectionalUncertaintySummary(
        val major95Meters: Double,
        val depthMeters: Double,
        val maxAnisotropyRatio: Double,
        val confidence: Double,
        val degradedJointCount: Int,
        val warnings: List<String>,
    )

    fun reconstruct(
        timestampNanos: Long,
        subjectType: SubjectType,
        trackId: Long,
        frames: List<CameraFrame>,
        bodySpacePriors:
            List<BodySpacePriorAlignmentEngine.Prior> = emptyList(),
    ): SkeletonFrame =
        reconstructDetailed(
            timestampNanos = timestampNanos,
            subjectType = subjectType,
            trackId = trackId,
            frames = frames,
            bodySpacePriors = bodySpacePriors,
        ).frame

    fun reconstructDetailed(
        timestampNanos: Long,
        subjectType: SubjectType,
        trackId: Long,
        frames: List<CameraFrame>,
        bodySpacePriors:
            List<BodySpacePriorAlignmentEngine.Prior> = emptyList(),
    ): ReconstructionResult {
        val joints = mutableListOf<ReconstructionJoint>()
        val diagnostics = mutableListOf<JointDiagnostic>()
        val grouped = HashMap<Int, MutableList<CameraObservation>>()
        val temporalConfidenceByLandmark =
            HashMap<Int, MutableList<Double>>()
        val triangulationByLandmark =
            HashMap<Int, TriangulationEngine.Result>()
        val jointObservationSelectionByLandmark =
            HashMap<Int, JointObservationSubsetPlanner.Selection>()

        frames.forEach { frame ->
            val temporalSkewNanos =
                abs(frame.timestampNanos - timestampNanos)

            if (temporalSkewNanos > observabilityTuning.maximumTemporalSkewMillis * 1_000_000.0) {
                return@forEach
            }

            val temporalWeight =
                temporalConfidence(temporalSkewNanos)

            if (temporalWeight <= 0.01) {
                return@forEach
            }

            frame.observations.forEach { observation ->
                if (observation.confidence <= 0.01) return@forEach

                val rollingShutterTrust =
                    (
                        1.0 -
                            0.50 *
                                abs(
                                    observation
                                        .rollingShutterFraction
                                ).coerceIn(0.0, 1.0) *
                                (
                                    1.0 -
                                        observabilityTuning
                                            .rollingShutterCompensationTrust
                                )
                        ).coerceIn(0.50, 1.0)

                val temporallyWeightedObservation =
                    observation.copy(
                        confidence =
                            (
                                observation.confidence *
                                    temporalWeight *
                                    rollingShutterTrust
                                ).coerceIn(0.0, 1.0),
                    )

                grouped.getOrPut(observation.landmarkId) {
                    mutableListOf()
                }.add(temporallyWeightedObservation)
                temporalConfidenceByLandmark
                    .getOrPut(observation.landmarkId) {
                        mutableListOf()
                    }
                    .add(temporalWeight)
            }
        }

        grouped.forEach { (landmarkId, observations) ->
            if (observations.size < 2) {
                diagnostics += JointDiagnostic(
                    landmarkId = landmarkId,
                    accepted = false,
                    parallaxDegrees = 0.0,
                    reprojectionErrorPx = Double.POSITIVE_INFINITY,
                    uncertaintyMeters = Double.POSITIVE_INFINITY,
                    inlierRatio = 0.0,
                    conditionScore = 0.0,
                    temporalConfidence =
                        temporalConfidenceByLandmark[landmarkId]
                            ?.average()
                            ?.coerceIn(0.0, 1.0)
                            ?: 0.0,
                )
                return@forEach
            }

            val selection =
                jointObservationSubsetPlanner.plan(
                    observations = observations,
                )
            jointObservationSelectionByLandmark[landmarkId] =
                selection

            val result = triangulator.robust(
                selection.selectedObservations.map(
                    TriangulationEngine::Observation
                ),
            )

            if (result != null) {
                triangulationByLandmark[landmarkId] = result
            }

            if (result == null) {
                diagnostics += JointDiagnostic(
                    landmarkId = landmarkId,
                    accepted = false,
                    parallaxDegrees = 0.0,
                    reprojectionErrorPx = Double.POSITIVE_INFINITY,
                    uncertaintyMeters = Double.POSITIVE_INFINITY,
                    inlierRatio = 0.0,
                    conditionScore = 0.0,
                    temporalConfidence =
                        temporalConfidenceByLandmark[landmarkId]
                            ?.average()
                            ?.coerceIn(0.0, 1.0)
                            ?: 0.0,
                )
                return@forEach
            }

            if (result.parallaxDegrees < observabilityTuning.minimumParallaxDegrees) {
                diagnostics += JointDiagnostic(
                    landmarkId = landmarkId,
                    accepted = false,
                    parallaxDegrees = result.parallaxDegrees,
                    reprojectionErrorPx = result.rmsReprojectionPx,
                    uncertaintyMeters = maxOf(
                        result.uncertaintyMeters,
                        result.uncertaintyMajorMeters,
                    ),
                    inlierRatio =
                        result.inliers.toDouble() /
                            result.observations.coerceAtLeast(1),
                    conditionScore = result.conditionScore,
                    triangulationMetricEvidenceAuthority =
                        result.metricEvidenceAuthority,
                    triangulationEpipolarConsistencyScore =
                        result.symmetricEpipolarConsistencyScore,
                    triangulationEpipolarWarnings =
                        result.symmetricEpipolarWarnings,
                    temporalConfidence =
                        temporalConfidenceByLandmark[landmarkId]
                            ?.average()
                            ?.coerceIn(0.0, 1.0)
                            ?: 0.0,
                )
                return@forEach
            }

            val directionalHold =
                result.uncertaintyMajorMeters >
                    observabilityTuning.maximumDirectionalUncertaintyMeters ||
                    result.uncertaintyDepthMeters >
                    observabilityTuning.maximumDepthUncertaintyMeters ||
                    result.uncertaintyAnisotropyRatio >
                    observabilityTuning.maximumUncertaintyAnisotropyRatio ||
                    result.uncertaintyConfidence <
                    observabilityTuning.minimumDirectionalUncertaintyConfidence

            if (
                result.rmsReprojectionPx >
                    observabilityTuning.maximumReprojectionErrorPx ||
                result.uncertaintyMeters >
                    observabilityTuning.maximumUncertaintyMeters ||
                directionalHold
            ) {
                diagnostics += JointDiagnostic(
                    landmarkId = landmarkId,
                    accepted = false,
                    parallaxDegrees = result.parallaxDegrees,
                    reprojectionErrorPx = result.rmsReprojectionPx,
                    uncertaintyMeters = result.uncertaintyMeters,
                    inlierRatio =
                        result.inliers.toDouble() /
                            result.observations.coerceAtLeast(1),
                    conditionScore = result.conditionScore,
                    temporalConfidence =
                        temporalConfidenceByLandmark[landmarkId]
                            ?.average()
                            ?.coerceIn(0.0, 1.0)
                            ?: 0.0,
                )
                return@forEach
            }
            val inlierRatio = result.inliers.toDouble() /
                result.observations.coerceAtLeast(1)

            val directionalConfidence =
                result.uncertaintyConfidence.coerceIn(0.0, 1.0)
            val directionalEvidencePenalty =
                (
                    0.74 +
                        directionalConfidence * 0.26
                    ).coerceIn(0.74, 1.0)

            val detectorConfidence =
                selection.selectedObservations
                    .map { it.confidence }
                    .average()
                    .coerceIn(0.0, 1.0) *
                    directionalEvidencePenalty
            val temporalConfidence =
                temporalConfidenceByLandmark[landmarkId]
                    ?.average()
                    ?.coerceIn(0.0, 1.0)
                    ?: 0.0

            val geometricConfidence =
                (
                    qualityModel.score(
                        parallaxDegrees = result.parallaxDegrees,
                reprojectionErrorPx = result.rmsReprojectionPx,
                inliers = result.inliers,
                observations = result.observations,
                uncertaintyMeters = result.uncertaintyMeters,
                detectorConfidence = detectorConfidence,
                        cheiralityRatio = result.cheiralityRatio,
                        conditionScore = result.conditionScore,
                    ) *
                        (
                            0.70 +
                                0.30 *
                                    result.metricEvidenceAuthority
                        ) *
                        (
                            0.78 +
                                0.22 *
                                    result.symmetricEpipolarConsistencyScore
                        )
                    ).coerceIn(0.0, 1.0)

            val state = kalman.update(
                id = landmarkId,
                measurement = result.point,
                measurementConfidence = geometricConfidence,
                reprojectionErrorPx = result.rmsReprojectionPx,
                timestampNanos = timestampNanos,
            )

            val refinedConfidence =
                (
                    qualityModel.score(
                        parallaxDegrees = result.parallaxDegrees,
                        reprojectionErrorPx = result.rmsReprojectionPx,
                        inliers = result.inliers,
                        observations = result.observations,
                        uncertaintyMeters = maxOf(
                            result.uncertaintyMeters,
                            result.uncertaintyMajorMeters,
                            result.uncertaintyDepthMeters,
                            state.uncertaintyMeters,
                        ),
                        detectorConfidence = detectorConfidence,
                        cheiralityRatio = result.cheiralityRatio,
                        conditionScore = result.conditionScore,
                    ) *
                        (
                            0.70 +
                                0.30 *
                                    result.metricEvidenceAuthority
                        ) *
                        (
                            0.78 +
                                0.22 *
                                    result.symmetricEpipolarConsistencyScore
                        ) *
                        (0.55 + 0.45 * state.confidence) *
                        (0.45 + 0.55 * temporalConfidence)
                    ).coerceIn(0.0, 1.0)

            val temporal = filter.update(
                id = landmarkId,
                measurement = state.position,
                confidence = refinedConfidence,
                timestampNanos = timestampNanos,
            )

            val finalConfidence =
                (
                    refinedConfidence *
                        (0.72 + 0.28 * temporal.confidence)
                    ).coerceIn(0.0, 1.0)

            joints += ReconstructionJoint(
                landmarkId = landmarkId,
                position = temporal.position,
                velocity = temporal.velocity,
                confidence = finalConfidence,
                reprojectionErrorPx = result.rmsReprojectionPx,
                observationCount = result.observations,
                positionUncertaintyMeters =
                    maxOf(
                        result.uncertaintyMeters,
                        result.uncertaintyMajorMeters,
                        result.uncertaintyDepthMeters,
                        state.uncertaintyMeters,
                        0.002 + temporal.acceleration.norm() * 0.0005,
                    ),
            )
        }

        val bodyPriorAlignment =
            if (bodySpacePriors.isEmpty()) {
                BodySpacePriorAlignmentEngine.Result.rejected(
                    "BODY_PRIOR_NOT_PROVIDED",
                )
            } else {
                bodySpacePriorAlignmentEngine.align(
                    priors = bodySpacePriors,
                    metricAnchors =
                        joints.map { joint ->
                            BodySpacePriorAlignmentEngine.MetricAnchor(
                                landmarkId = joint.landmarkId,
                                positionMeters = joint.position,
                                confidence = joint.confidence,
                                uncertaintyMeters =
                                    joint.positionUncertaintyMeters,
                            )
                        },
                )
            }

        val bodyPriorById =
            bodyPriorAlignment.aligned
                .associateBy { it.landmarkId }

        val priorOnlyJoints =
            bodyPriorById.values
                .filter { prior ->
                    joints.none {
                        it.landmarkId == prior.landmarkId
                    }
                }
                .map { prior ->
                    ReconstructionJoint(
                        landmarkId = prior.landmarkId,
                        position = prior.positionMeters,
                        velocity = Vec3(0.0, 0.0, 0.0),
                        confidence =
                            prior.confidence.coerceIn(0.05, 0.34),
                        reprojectionErrorPx =
                            Double.POSITIVE_INFINITY,
                        observationCount = 0,
                        positionUncertaintyMeters =
                            prior.uncertaintyMeters
                                .coerceAtLeast(0.025),
                    )
                }

        val constrained =
            constraints.apply(
                joints + priorOnlyJoints,
            )

        val refined = mutableListOf<ReconstructionJoint>()
        constrained.forEach { joint ->
            val observations = grouped[joint.landmarkId].orEmpty()
            val bundle =
                if (
                    (
                        jointObservationSelectionByLandmark[
                            joint.landmarkId
                        ]?.selectedObservations?.size
                            ?: 0
                        ) < 2
                ) {
                    null
                } else {
                    bundleRefiner.refine(
                        joint = joint,
                        observations =
                            jointObservationSelectionByLandmark[
                                joint.landmarkId
                            ]!!.selectedObservations,
                        priorPosition = joint.position,
                    )
                }

            val temporalPriorJoint =
                joints.firstOrNull {
                    it.landmarkId == joint.landmarkId
                }

            val evidenceCandidates =
                buildList {
                    temporalPriorJoint?.let { prior ->
                        if (prior.position.distanceTo(joint.position) > 1e-6) {
                            add(
                                MultiEvidence3DReconstructionEngine.Candidate(
                                    joint = prior,
                                    source =
                                        MultiEvidence3DReconstructionEngine.Source
                                            .TEMPORAL_PRIOR,
                                    correlationGroup = "temporal",
                                ),
                            )
                        }
                    }
                    if (bundle?.accepted == true) {
                        add(
                            MultiEvidence3DReconstructionEngine.Candidate(
                                joint = bundle.joint,
                                source =
                                    MultiEvidence3DReconstructionEngine.Source
                                        .MULTI_VIEW,
                                correlationGroup = "geometry",
                            ),
                        )
                    }

                    if (joint.observationCount > 0) {
                        bodyPriorById[joint.landmarkId]?.let { prior ->
                            add(
                                MultiEvidence3DReconstructionEngine.Candidate(
                                    joint =
                                        ReconstructionJoint(
                                            landmarkId = prior.landmarkId,
                                            position = prior.positionMeters,
                                            velocity = Vec3(0.0, 0.0, 0.0),
                                            confidence =
                                                prior.confidence
                                                    .coerceIn(0.0, 0.34),
                                            reprojectionErrorPx =
                                                Double.POSITIVE_INFINITY,
                                            observationCount = 0,
                                            positionUncertaintyMeters =
                                                prior.uncertaintyMeters
                                                    .coerceAtLeast(0.025),
                                        ),
                                    source =
                                        MultiEvidence3DReconstructionEngine.Source
                                            .BODY_SPACE_PRIOR,
                                    correlationGroup =
                                        "learned-body-space",
                                ),
                            )
                        }
                    }

                    triangulationByLandmark[joint.landmarkId]?.let { triangulated ->
                        add(
                            MultiEvidence3DReconstructionEngine.Candidate(
                                joint =
                                    joint.copy(
                                        position = triangulated.point,
                                        confidence =
                                            (
                                                joint.confidence *
                                                    (
                                                        0.55 +
                                                            0.45 *
                                                                triangulated.conditionScore
                                                    )
                                                ).coerceIn(0.0, 1.0),
                                        positionUncertaintyMeters =
                                            maxOf(
                                                triangulated.uncertaintyMeters,
                                                0.001,
                                            ),
                                    ),
                                source =
                                    MultiEvidence3DReconstructionEngine.Source
                                        .MULTI_VIEW,
                                correlationGroup = "geometry",
                            ),
                        )
                    }
                }

            val fused =
                evidenceFusion.fuse(
                    timestampNanos = timestampNanos,
                    primary = joint,
                    candidates = evidenceCandidates,
                )

            val acceptedJoint =
                joint.copy(
                    position = fused.position,
                    velocity = fused.velocity,
                    confidence =
                        minOf(
                            joint.confidence,
                            fused.confidence,
                        ),
                    positionUncertaintyMeters =
                        maxOf(
                            joint.positionUncertaintyMeters,
                            fused.positionUncertaintyMeters,
                        ),
                )

            refined += acceptedJoint

            val diagnosticResult = bundle
            diagnostics += JointDiagnostic(
                landmarkId = joint.landmarkId,
                accepted =
                    bundle == null ||
                        diagnosticResult?.accepted == true,
                parallaxDegrees =
                    triangulationByLandmark[joint.landmarkId]
                        ?.parallaxDegrees
                        ?: 0.0,
                reprojectionErrorPx =
                    diagnosticResult?.finalErrorPx
                        ?: joint.reprojectionErrorPx,
                uncertaintyMeters =
                    diagnosticResult?.uncertaintyMeters
                        ?: joint.positionUncertaintyMeters,
                inlierRatio =
                    diagnosticResult?.let {
                        it.inliers.toDouble() /
                            it.observationsUsed.coerceAtLeast(1)
                    }
                        ?: triangulationByLandmark[joint.landmarkId]
                            ?.let {
                                it.inliers.toDouble() /
                                    it.observations.coerceAtLeast(1)
                            }
                            ?: 1.0,
                conditionScore =
                    diagnosticResult?.conditionScore
                        ?: triangulationByLandmark[joint.landmarkId]
                            ?.conditionScore
                        ?: 1.0,
                triangulationMetricEvidenceAuthority =
                    triangulationByLandmark[joint.landmarkId]
                        ?.metricEvidenceAuthority
                        ?: 0.0,
                triangulationEpipolarConsistencyScore =
                    triangulationByLandmark[joint.landmarkId]
                        ?.symmetricEpipolarConsistencyScore
                        ?: 1.0,
                triangulationEpipolarWarnings =
                    triangulationByLandmark[joint.landmarkId]
                        ?.symmetricEpipolarWarnings
                        ?: emptyList(),
                temporalConfidence =
                    temporalConfidenceByLandmark[joint.landmarkId]
                        ?.average()
                        ?.coerceIn(0.0, 1.0)
                        ?: 0.0,
                selectedObservationCount =
                    jointObservationSelectionByLandmark[
                        joint.landmarkId
                    ]?.selectedObservations?.size
                        ?: 0,
                excludedObservationCount =
                    jointObservationSelectionByLandmark[
                        joint.landmarkId
                    ]?.excludedDeviceIds?.size
                        ?: 0,
                jointViewDiversitySatisfied =
                    jointObservationSelectionByLandmark[
                        joint.landmarkId
                    ]?.diversitySatisfied
                        ?: false,
                jointViewSelectionScore =
                    jointObservationSelectionByLandmark[
                        joint.landmarkId
                    ]?.score
                        ?: 0.0,
                jointViewSelectionReason =
                    jointObservationSelectionByLandmark[
                        joint.landmarkId
                    ]?.reason
                        ?: "UNAVAILABLE",
                directionalUncertaintyMeters =
                    triangulationByLandmark[joint.landmarkId]
                        ?.uncertaintyMajorMeters
                        ?: 0.0,
                directionalDepthUncertaintyMeters =
                    triangulationByLandmark[joint.landmarkId]
                        ?.uncertaintyDepthMeters
                        ?: 0.0,
                directionalUncertaintyAnisotropy =
                    triangulationByLandmark[joint.landmarkId]
                        ?.uncertaintyAnisotropyRatio
                        ?: 1.0,
                directionalUncertaintyConfidence =
                    triangulationByLandmark[joint.landmarkId]
                        ?.uncertaintyConfidence
                        ?: 1.0,
                directionalUncertaintyWarnings =
                    triangulationByLandmark[joint.landmarkId]
                        ?.uncertaintyWarnings
                        ?: emptyList(),
                triangulationIndependentViewCount =
                    triangulationByLandmark[joint.landmarkId]
                        ?.independentViewCount
                        ?: 0,
                triangulationDominantViewWeightRatio =
                    triangulationByLandmark[joint.landmarkId]
                        ?.dominantViewWeightRatio
                        ?: 0.0,
                triangulationViewDiversityScore =
                    triangulationByLandmark[joint.landmarkId]
                        ?.viewDiversityScore
                        ?: 0.0,
                triangulationViewDiversityWarnings =
                    triangulationByLandmark[joint.landmarkId]
                        ?.viewDiversityWarnings
                        ?: emptyList(),
            )
        }

        val finalJoints = refined
            .sortedBy { it.landmarkId }

        val globalConfidence = if (finalJoints.isEmpty()) {
            0.0
        } else {
            val weighted = finalJoints.sumOf {
                val uncertaintyWeight = exp(
                    -it.positionUncertaintyMeters /
                        0.10.coerceAtLeast(1e-3),
                )

                it.confidence * uncertaintyWeight
            }

            (
                weighted /
                    finalJoints.size
                ).coerceIn(0.0, 1.0)
        }

        val rawFrame = SkeletonFrame(
            timestampNanos = timestampNanos,
            subjectType = subjectType,
            trackId = trackId,
            joints = finalJoints,
            globalConfidence = globalConfidence,
        )

        val optimizedFrame =
            trajectoryOptimizer.optimize(rawFrame)

        val directionalResults =
            triangulationByLandmark.values.toList()

        lastDirectionalUncertainty =
            if (directionalResults.isEmpty()) {
                DirectionalUncertaintySummary(
                    major95Meters = 0.0,
                    depthMeters = 0.0,
                    maxAnisotropyRatio = 1.0,
                    confidence = 0.0,
                    degradedJointCount =
                        diagnostics.count { !it.accepted },
                    warnings = listOf(
                        "directional-uncertainty-no-triangulated-joints",
                    ),
                )
            } else {
                DirectionalUncertaintySummary(
                    major95Meters =
                        percentile(
                            directionalResults.map {
                                it.uncertainty95MajorMeters
                            },
                            0.95,
                        ),
                    depthMeters =
                        directionalResults
                            .map {
                                it.uncertaintyDepthMeters
                            }
                            .average()
                            .coerceAtLeast(0.0),
                    maxAnisotropyRatio =
                        directionalResults
                            .maxOf {
                                it.uncertaintyAnisotropyRatio
                            }
                            .coerceAtLeast(1.0),
                    confidence =
                        directionalResults
                            .map {
                                it.uncertaintyConfidence
                            }
                            .average()
                            .coerceIn(0.0, 1.0),
                    degradedJointCount =
                        directionalResults.count {
                            it.uncertaintyConfidence < 0.30 ||
                                it.uncertaintyAnisotropyRatio >
                                observabilityTuning
                                    .maximumUncertaintyAnisotropyRatio
                        },
                    warnings =
                        directionalResults
                            .flatMap {
                                it.uncertaintyWarnings
                            }
                            .distinct()
                            .take(12),
                )
            }

        val triangulatedResults =
            directionalResults
                .filter { it.observations >= 2 }

        val meanMetricEvidenceAuthority =
            triangulatedResults
                .map { it.metricEvidenceAuthority }
                .averageOrZero()
                .coerceIn(0.0, 1.0)

        val meanSymmetricEpipolarConsistencyScore =
            triangulatedResults
                .map {
                    it.symmetricEpipolarConsistencyScore
                }
                .averageOrZero()
                .coerceIn(0.0, 1.0)

        val epipolarWarnings =
            triangulatedResults
                .flatMap {
                    it.symmetricEpipolarWarnings
                }
                .distinct()
                .take(16)

        lastMeanMetricEvidenceAuthority =
            meanMetricEvidenceAuthority
        lastMeanSymmetricEpipolarConsistencyScore =
            meanSymmetricEpipolarConsistencyScore

        return ReconstructionResult(
            frame = optimizedFrame,
            diagnostics = diagnostics,
            meanMetricEvidenceAuthority =
                meanMetricEvidenceAuthority,
            meanSymmetricEpipolarConsistencyScore =
                meanSymmetricEpipolarConsistencyScore,
            epipolarWarnings =
                epipolarWarnings,
        )
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) {
            0.0
        } else {
            average().takeIf { it.isFinite() } ?: 0.0
        }

    private fun percentile(
        values: List<Double>,
        fraction: Double,
    ): Double {
        if (values.isEmpty()) return 0.0
        val sorted =
            values
                .filter { it.isFinite() }
                .sorted()
        if (sorted.isEmpty()) return 0.0
        if (sorted.size == 1) return sorted.first()

        val position =
            fraction.coerceIn(0.0, 1.0) *
                (sorted.size - 1)
        val lower = position.toInt()
        val upper =
            minOf(
                lower + 1,
                sorted.lastIndex,
            )
        val blend =
            position - lower

        return (
            sorted[lower] *
                (1.0 - blend) +
                sorted[upper] * blend
            )
    }

    private fun temporalConfidence(
        skewNanos: Long,
    ): Double {
        val preferredNanos =
            (
                observabilityTuning.preferredTemporalSkewMillis *
                    1_000_000.0
                ).toLong()
        val maximumNanos =
            (
                observabilityTuning.maximumTemporalSkewMillis *
                    1_000_000.0
                ).toLong()

        if (skewNanos <= preferredNanos) {
            return 1.0
        }
        if (skewNanos >= maximumNanos) {
            return 0.0
        }

        val span =
            (maximumNanos - preferredNanos)
                .coerceAtLeast(1L)

        return (
            1.0 -
                (skewNanos - preferredNanos).toDouble() /
                    span.toDouble()
            ).coerceIn(0.0, 1.0)
    }

    fun reset() {
        filter.clear()
        kalman.clear()
        constraints.reset()
        trajectoryOptimizer.reset()
        lastDirectionalUncertainty =
            DirectionalUncertaintySummary(
                major95Meters = 0.0,
                depthMeters = 0.0,
                maxAnisotropyRatio = 1.0,
                confidence = 1.0,
                degradedJointCount = 0,
                warnings = emptyList(),
            )
    }
}

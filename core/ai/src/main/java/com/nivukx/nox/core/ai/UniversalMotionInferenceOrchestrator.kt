package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.InferenceBudget
import com.nivukx.nox.core.model.MotionCaptureMode
import com.nivukx.nox.core.model.MotionCaptureCustomization
import com.nivukx.nox.core.model.MotionInferenceDecision
import com.nivukx.nox.core.model.MotionPipelineTelemetry
import com.nivukx.nox.core.model.SubjectLockPhase
import com.nivukx.nox.core.model.SubjectType
import kotlin.math.max
import kotlin.math.min

/**
 * Adaptive orchestration layer for the RGB-only motion-capture pipeline.
 *
 * Heavy detection is not allowed to disappear permanently after initial lock. The state machine
 * intentionally performs periodic re-localization and immediate recovery when confidence falls.
 * Tracking remains cheap between detector passes, while depth/topology/reconstruction are scheduled
 * independently according to scene complexity and device headroom.
 */
class UniversalMotionInferenceOrchestrator(
    private val config: Config = Config(),
) {

    data class Config(
        val acquireFrames: Int = 4,
        val lockFrames: Int = 5,
        val recheckIntervalFrames: Int = 18,
        val lostTimeoutNanos: Long = 650_000_000L,
        val trackingConfidenceFloor: Double = 0.44,
        val detectorConfidenceFloor: Double = 0.24,
        val emergencyConfidenceFloor: Double = 0.16,
        val defaultBudget: InferenceBudget = InferenceBudget(),
    )

    data class FrameInput(
        val timestampNanos: Long,
        val frameIndex: Long,
        val motionQuality: MotionCaptureEngine.Quality,
        val assessment: AdvancedMotionFusionEngine.Assessment,
        val subjectCount: Int,
        val dominantTrackId: Long?,
        val dominantSubjectType: SubjectType,
        val lockConfidence: Double,
        val thermalScale: Double = 1.0,
        val memoryScale: Double = 1.0,
        val gpuScale: Double = 1.0,
        val npuScale: Double = 1.0,
        val captureFps: Double = 30.0,
        val viewpointChallenge: Double = 0.0,
        val leftRightAmbiguity: Double = 0.0,
    )

    private var phase = SubjectLockPhase.SEARCH
    private var candidateFrames = 0
    private var lockedFrames = 0
    private var framesSinceDetector = Int.MAX_VALUE
    private var framesSinceDepth = Int.MAX_VALUE
    private var framesSinceTopology = Int.MAX_VALUE
    private var framesSinceReconstruction = Int.MAX_VALUE
    private var lostSinceNanos = Long.MIN_VALUE
    private var activeTrackId: Long? = null
    private var activeSubjectType = SubjectType.UNKNOWN
    private var previousTimestampNanos = Long.MIN_VALUE
    private var previousConfidence = 0.0
    private var inferenceEmaMs = 0.0
    private var droppedFrames = 0L
    private var customization =
        MotionCaptureCustomization.production()

    private val viewRefinementController =
        AdaptiveViewRefinementController()

    private val identityContinuityEngine =
        TemporalSubjectIdentityContinuityEngine()

    @Synchronized
    fun configureCustomization(
        value: MotionCaptureCustomization,
    ) {
        customization = value
    }

    @Synchronized
    fun customization(): MotionCaptureCustomization = customization

    @Synchronized
    fun update(input: FrameInput): MotionInferenceDecision {
        val identityContinuity = identityContinuityEngine.evaluate(
            TemporalSubjectIdentityContinuityEngine.FrameEvidence(
                timestampNanos = input.timestampNanos,
                candidateTrackId = input.dominantTrackId,
                candidateConfidence = (
                    input.assessment.dominantSubject?.confidence
                        ?: input.lockConfidence
                    ),
                subjectCount = input.subjectCount,
                occlusionRisk = input.assessment.occlusionRisk,
                temporalStability = input.motionQuality.temporalStability,
                leftRightAmbiguity = input.leftRightAmbiguity,
                viewpointChallenge = input.viewpointChallenge,
            ),
        )

        updateState(input, identityContinuity)
        val budget = resolveBudget(input)
        val confidence = aggregateConfidence(input)

        val viewRefinement = viewRefinementController.evaluate(
            AdaptiveViewRefinementController.FrameEvidence(
                frameIndex = input.frameIndex,
                timestampNanos = input.timestampNanos,
                viewpointChallenge = input.viewpointChallenge.coerceIn(0.0, 1.0),
                leftRightAmbiguity = input.leftRightAmbiguity.coerceIn(0.0, 1.0),
                occlusionRisk = input.assessment.occlusionRisk.coerceIn(0.0, 1.0),
                confidence = confidence,
                temporalStability = input.motionQuality.temporalStability.coerceIn(0.0, 1.0),
                motionEnergy = input.motionQuality.motionScore.coerceIn(0.0, 1.0),
            ),
        )

        framesSinceDetector += 1
        framesSinceDepth += 1
        framesSinceTopology += 1
        framesSinceReconstruction += 1

        val detectorRequired = when (phase) {
            SubjectLockPhase.SEARCH -> true
            SubjectLockPhase.DETECT -> true
            SubjectLockPhase.LOCK -> {
                framesSinceDetector >= max(2, budget.detectorIntervalFrames / 2)
            }
            SubjectLockPhase.TRACK -> {
                framesSinceDetector >=
                    min(
                        config.recheckIntervalFrames,
                        budget.detectorIntervalFrames * 3,
                    ) ||
                    confidence < config.trackingConfidenceFloor ||
                    viewRefinement.requestRefinement
            }
            SubjectLockPhase.RECHECK -> true
            SubjectLockPhase.LOST -> true
        }

        val depthRequired = when {
            confidence < config.emergencyConfidenceFloor -> true
            input.motionQuality.motionClass == MotionCaptureEngine.MotionClass.EXTREME -> true
            framesSinceDepth >= budget.depthIntervalFrames -> true
            else -> false
        }

        val topologyRequired = when {
            phase == SubjectLockPhase.SEARCH -> true
            phase == SubjectLockPhase.DETECT -> true
            framesSinceTopology >= budget.topologyIntervalFrames -> true
            input.assessment.occlusionRisk > 0.58 -> true
            else -> false
        }

        val reconstructionRequired = when {
            phase == SubjectLockPhase.LOST -> false
            framesSinceReconstruction >= budget.reconstructionIntervalFrames -> true
            input.motionQuality.motionClass == MotionCaptureEngine.MotionClass.EXTREME -> true
            else -> false
        }

        val baseReason = buildReason(
            detectorRequired = detectorRequired,
            depthRequired = depthRequired,
            topologyRequired = topologyRequired,
            reconstructionRequired = reconstructionRequired,
            confidence = confidence,
            input = input,
        )
        val reason = if (
            identityContinuity.action == TemporalSubjectIdentityContinuityEngine.Action.KEEP ||
                identityContinuity.action == TemporalSubjectIdentityContinuityEngine.Action.ACQUIRE
        ) {
            baseReason
        } else {
            baseReason + ",identity-" + identityContinuity.reason
        }

        if (detectorRequired) framesSinceDetector = 0
        if (depthRequired) framesSinceDepth = 0
        if (topologyRequired) framesSinceTopology = 0
        if (reconstructionRequired) framesSinceReconstruction = 0

        previousTimestampNanos = input.timestampNanos
        previousConfidence = confidence

        return MotionInferenceDecision(
            timestampNanos = input.timestampNanos,
            subjectPhase = phase,
            activeTrackId = activeTrackId,
            dominantSubjectType = activeSubjectType,
            detectorRequired = detectorRequired,
            depthRequired = depthRequired,
            topologyRequired = topologyRequired,
            reconstructionRequired = reconstructionRequired,
            reason = reason,
            confidence = confidence,
            budget = budget,
            viewRefinementRequired = viewRefinement.requestRefinement,
            viewRefinementScore = viewRefinement.score,
            viewRefinementReason = viewRefinement.reason,
            viewRefinementForced = viewRefinement.forced,
            identityContinuityScore = identityContinuity.continuityScore,
            identitySwapRisk = identityContinuity.identitySwapRisk,
            identityAction = identityContinuity.action.name,
            identityReason = identityContinuity.reason,
            identitySwitchEvidenceFrames = identityContinuity.switchEvidenceFrames,
        )
    }

    @Synchronized
    fun recordInferenceLatency(latencyMillis: Double) {
        val sample = latencyMillis.coerceAtLeast(0.0)
        inferenceEmaMs =
            if (inferenceEmaMs <= 0.0) {
                sample
            } else {
                inferenceEmaMs * 0.86 + sample * 0.14
            }
    }

    @Synchronized
    fun recordDroppedFrame() {
        droppedFrames += 1
    }

    @Synchronized
    fun recordDetectorCompleted() {
        framesSinceDetector = 0
    }

    @Synchronized
    fun recordDepthCompleted() {
        framesSinceDepth = 0
    }

    @Synchronized
    fun recordTopologyCompleted() {
        framesSinceTopology = 0
    }

    @Synchronized
    fun recordReconstructionCompleted() {
        framesSinceReconstruction = 0
    }

    @Synchronized
    fun telemetry(
        timestampNanos: Long,
        captureFps: Double,
        mode: MotionCaptureMode,
        visibleJoints: Int,
        predictedJoints: Int,
        thermalScale: Double,
        memoryScale: Double,
        gpuScale: Double,
        npuScale: Double,
        trackedSubjects: Int,
        meanConfidence: Double,
        temporalStability: Double,
        warnings: List<String> = emptyList(),
    ): MotionPipelineTelemetry =
        MotionPipelineTelemetry(
            timestampNanos = timestampNanos,
            captureFps = captureFps,
            inferenceFps = if (inferenceEmaMs <= 0.0) {
                0.0
            } else {
                min(
                    captureFps.coerceAtLeast(1.0),
                    1000.0 / inferenceEmaMs,
                )
            },
            detectorLatencyMs = inferenceEmaMs * 0.46,
            depthLatencyMs = inferenceEmaMs * 0.20,
            reconstructionLatencyMs = inferenceEmaMs * 0.31,
            totalLatencyMs = inferenceEmaMs,
            droppedFrames = droppedFrames,
            trackedSubjects = trackedSubjects,
            visibleJoints = visibleJoints,
            predictedJoints = predictedJoints,
            meanConfidence = meanConfidence,
            temporalStability = temporalStability,
            thermalScale = thermalScale,
            memoryScale = memoryScale,
            gpuScale = gpuScale,
            npuScale = npuScale,
            currentMode = mode,
            warnings = warnings.distinct(),
        )

    @Synchronized
    fun reset() {
        phase = SubjectLockPhase.SEARCH
        candidateFrames = 0
        lockedFrames = 0
        framesSinceDetector = Int.MAX_VALUE
        framesSinceDepth = Int.MAX_VALUE
        framesSinceTopology = Int.MAX_VALUE
        framesSinceReconstruction = Int.MAX_VALUE
        lostSinceNanos = Long.MIN_VALUE
        activeTrackId = null
        activeSubjectType = SubjectType.UNKNOWN
        previousTimestampNanos = Long.MIN_VALUE
        previousConfidence = 0.0
        inferenceEmaMs = 0.0
        droppedFrames = 0L
        customization = MotionCaptureCustomization.production()
        viewRefinementController.reset()
        identityContinuityEngine.reset()
    }

    @Synchronized
    fun phase(): SubjectLockPhase = phase

    @Synchronized
    fun activeTrack(): Long? = activeTrackId

    private fun updateState(
        input: FrameInput,
        identity: TemporalSubjectIdentityContinuityEngine.Result,
    ) {
        val lock = input.lockConfidence.coerceIn(0.0, 1.0)
        val dominantConfidence =
            input.assessment.dominantSubject?.confidence ?: 0.0
        val combined = (
            lock * 0.45 +
                dominantConfidence * 0.20 +
                input.assessment.attentionConfidence * 0.20 +
                input.motionQuality.meanConfidence * 0.15
            ).coerceIn(0.0, 1.0)

        val identityContinuityActive =
            identity.selectedTrackId != null &&
                identity.action != TemporalSubjectIdentityContinuityEngine.Action.SEARCH

        val hasSubject =
            (
                input.subjectCount > 0 &&
                    combined >= config.detectorConfidenceFloor
                ) ||
                (
                    identityContinuityActive &&
                        combined >= config.emergencyConfidenceFloor
                    )

        when (phase) {
            SubjectLockPhase.SEARCH -> {
                if (hasSubject) {
                    phase = SubjectLockPhase.DETECT
                    candidateFrames = 1
                    activeTrackId = identity.selectedTrackId ?: input.dominantTrackId
                    activeSubjectType = input.dominantSubjectType
                }
            }

            SubjectLockPhase.DETECT -> {
                if (!hasSubject) {
                    candidateFrames = 0
                    return
                }

                activeTrackId = identity.selectedTrackId ?: input.dominantTrackId ?: activeTrackId
                activeSubjectType = input.dominantSubjectType
                candidateFrames += 1

                if (
                    candidateFrames >= config.acquireFrames &&
                    lock >= 0.45
                ) {
                    phase = SubjectLockPhase.LOCK
                    lockedFrames = 0
                }
            }

            SubjectLockPhase.LOCK -> {
                if (!hasSubject || lock < config.detectorConfidenceFloor) {
                    phase = SubjectLockPhase.RECHECK
                    return
                }

                lockedFrames += 1
                activeTrackId = identity.selectedTrackId ?: input.dominantTrackId ?: activeTrackId
                activeSubjectType = input.dominantSubjectType

                if (
                    lockedFrames >= config.lockFrames &&
                    combined >= config.trackingConfidenceFloor
                ) {
                    phase = SubjectLockPhase.TRACK
                    lostSinceNanos = Long.MIN_VALUE
                }
            }

            SubjectLockPhase.TRACK -> {
                activeTrackId = identity.selectedTrackId ?: input.dominantTrackId ?: activeTrackId
                activeSubjectType = input.dominantSubjectType

                if (!hasSubject || combined < config.trackingConfidenceFloor) {
                    if (lostSinceNanos == Long.MIN_VALUE) {
                        lostSinceNanos = input.timestampNanos
                    }
                    phase = SubjectLockPhase.RECHECK
                }
            }

            SubjectLockPhase.RECHECK -> {
                activeTrackId = identity.selectedTrackId ?: input.dominantTrackId ?: activeTrackId

                if (
                    hasSubject &&
                    combined >= config.trackingConfidenceFloor
                ) {
                    phase = SubjectLockPhase.TRACK
                    lostSinceNanos = Long.MIN_VALUE
                    return
                }

                if (lostSinceNanos == Long.MIN_VALUE) {
                    lostSinceNanos = input.timestampNanos
                }

                if (
                    input.timestampNanos - lostSinceNanos >=
                        config.lostTimeoutNanos
                ) {
                    phase = SubjectLockPhase.LOST
                }
            }

            SubjectLockPhase.LOST -> {
                if (hasSubject) {
                    phase = SubjectLockPhase.DETECT
                    candidateFrames = 0
                    activeTrackId = identity.selectedTrackId ?: input.dominantTrackId
                    activeSubjectType = input.dominantSubjectType
                    lostSinceNanos = Long.MIN_VALUE
                }
            }
        }
    }

    private fun resolveBudget(input: FrameInput): InferenceBudget {
        val headroom = min(
            min(
                input.thermalScale.coerceIn(0.20, 1.0),
                input.memoryScale.coerceIn(0.20, 1.0),
            ),
            min(
                input.gpuScale.coerceIn(0.20, 1.0),
                input.npuScale.coerceIn(0.20, 1.0),
            ),
        )

        var budget = if (headroom < 0.48) {
            config.defaultBudget.degraded(0.50)
        } else if (headroom < 0.68) {
            config.defaultBudget.degraded(0.72)
        } else {
            config.defaultBudget
        }

        val detailBias =
            (
                customization.posePriorStrength * 0.24 +
                    customization.landmarkConsistencyStrictness * 0.18 +
                    customization.semanticMotionSensitivity * 0.18 +
                    customization.occlusionRecoveryConfidence * 0.16 +
                    customization.temporalSmoothing * 0.12 +
                    customization.depthTemporalFusion * 0.12
                ).coerceIn(0.0, 1.0)

        val thermalProtection =
            if (headroom < customization.thermalQualityFloor.coerceIn(0.20, 1.0)) {
                1.35
            } else if (
                headroom <
                (customization.thermalQualityFloor + 0.12).coerceAtMost(1.0)
            ) {
                1.16
            } else {
                1.0
            }

        val occlusionRecoveryBias =
            (
                1.0 -
                    customization.occlusionRecoveryConfidence * 0.35
                ).coerceIn(0.55, 1.0)

        budget = budget.copy(
            detectorIntervalFrames =
                (
                    customization.adaptiveDetectorIntervalFrames /
                        thermalProtection
                    ).toInt().coerceAtLeast(1),
            depthIntervalFrames =
                detailInterval(
                    detail = (
                        customization.depthDetail * 0.72 +
                            customization.depthTemporalFusion * 0.28
                        ).coerceIn(0.0, 1.0),
                    high = 1,
                    low = 8,
                ),

            faceIntervalFrames =
                detailInterval(
                    detail = customization.faceDetail,
                    high = 1,
                    low = 8,
                ),
            eyeIntervalFrames =
                detailInterval(
                    detail = customization.eyeDetail,
                    high = 1,
                    low = 6,
                ),
            handIntervalFrames =
                detailInterval(
                    detail = (
                        customization.handDetail * 0.70 +
                            customization.handGeometryStrictness * 0.30
                        ).coerceIn(0.0, 1.0),
                    high = 1,
                    low = 6,
                ),
            objectIntervalFrames =
                detailInterval(
                    detail = customization.objectDetail,
                    high = 1,
                    low = 10,
                ),
            denseMeshIntervalFrames =
                detailInterval(
                    detail = (
                        customization.denseMeshDetail * 0.62 +
                            customization.meshAdaptiveDetail * 0.38
                        ).coerceIn(0.0, 1.0),
                    high = 1,
                    low = 12,
                ),
            contactIntervalFrames =
                detailInterval(
                    detail = (
                        customization.contactDetail * 0.64 +
                            customization.footContactStrictness * 0.36
                        ).coerceIn(0.0, 1.0),
                    high = 1,
                    low = 6,
                ),

        )

        val complexity = (
            input.assessment.motionComplexity * 0.40 +
                input.assessment.occlusionRisk * 0.22 +
                input.motionQuality.meanJerkNormalizedPerSecond3
                    .div(80.0)
                    .coerceIn(0.0, 1.0) * 0.12 +
                (1.0 - input.motionQuality.temporalStability) * 0.16 +
                (1.0 - input.assessment.attentionConfidence) * 0.10
            ).coerceIn(0.0, 1.0)

        val qualityBoost =
            (
                detailBias *
                    (
                        0.62 +
                            customization.headPoseStrictness * 0.20 +
                            customization.gazePriority * 0.08 +
                            customization.retargetingStrictness * 0.10
                        )
                ).coerceIn(0.0, 1.0)

        if (complexity > 0.68 || qualityBoost > 0.82) {
            budget = budget.copy(
                detectorIntervalFrames =
                    max(1, budget.detectorIntervalFrames / 2),
                depthIntervalFrames =
                    max(1, budget.depthIntervalFrames / 2),
                reconstructionIntervalFrames = 1,
            )
        } else if (
            complexity < 0.24 &&
            phase == SubjectLockPhase.TRACK &&
            customization.semanticMotionSensitivity < 0.76
        ) {
            budget = budget.copy(
                detectorIntervalFrames =
                    min(
                        24,
                        budget.detectorIntervalFrames + 2,
                    ),
                depthIntervalFrames =
                    min(
                        8,
                        budget.depthIntervalFrames + 1,
                    ),
            )
        }

        if (
            input.motionQuality.motionClass ==
            MotionCaptureEngine.MotionClass.EXTREME ||
            customization.microMotionPreservation >= 0.96
        ) {
            budget = budget.copy(
                detectorIntervalFrames = 1,
                depthIntervalFrames = 1,
                topologyIntervalFrames = 1,
                reconstructionIntervalFrames = 1,
            )
        }

        return budget.degraded(
            factor = (
                (
                    1.0 -
                        (
                            customization.outputIntegrityStrictness * 0.10 +
                                customization.cameraDriftResponse * 0.08
                            )
                    ) *
                    occlusionRecoveryBias *
                    thermalProtection
                ).coerceIn(0.55, 1.0),
        )
    }


    private fun detailInterval(
        detail: Double,
        high: Int,
        low: Int,
    ): Int =
        (
            low -
                (low - high) *
                detail.coerceIn(0.0, 1.0)
            )
            .toInt()
            .coerceIn(high, low)

    private fun aggregateConfidence(
        input: FrameInput,
    ): Double {
        val lock = input.lockConfidence.coerceIn(0.0, 1.0)
        val dominant = (
            input.assessment.dominantSubject?.confidence ?: 0.0
            ).coerceIn(0.0, 1.0)
        val motion = input.motionQuality.meanConfidence
        val temporal = input.motionQuality.temporalStability
        val attention = input.assessment.attentionConfidence
        val occlusionPenalty = input.assessment.occlusionRisk * 0.18

        return (
            0.25 * lock +
                0.20 * dominant +
                0.20 * motion +
                0.18 * temporal +
                0.17 * attention -
                occlusionPenalty
            ).coerceIn(0.0, 1.0)
    }

    private fun buildReason(
        detectorRequired: Boolean,
        depthRequired: Boolean,
        topologyRequired: Boolean,
        reconstructionRequired: Boolean,
        confidence: Double,
        input: FrameInput,
    ): String {
        val reasons = ArrayList<String>(6)

        if (detectorRequired) reasons += when (phase) {
            SubjectLockPhase.TRACK -> "scheduled_relocalization"
            SubjectLockPhase.RECHECK -> "reacquisition"
            SubjectLockPhase.LOST -> "recovery_search"
            else -> "subject_search"
        }

        if (depthRequired) reasons += "depth_update"
        if (topologyRequired) reasons += "topology_update"
        if (reconstructionRequired) reasons += "3d_update"

        if (input.assessment.occlusionRisk > 0.55) {
            reasons += "occlusion_recovery"
        }

        if (input.assessment.semanticMotion.cameraMotionRisk > 0.55) {
            reasons += "camera_motion_compensation"
        }

        if (confidence < config.trackingConfidenceFloor) {
            reasons += "confidence_recovery"
        }

        return if (reasons.isEmpty()) "steady_tracking" else reasons.joinToString(",")
    }
}

package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Production motion-evidence fusion.
 *
 * A pose model can report motion when the camera moves, when a subject is briefly occluded, or when
 * a track changes identity. This engine treats model motion as one observation among several and
 * produces a stateful subject-motion decision with hysteresis.
 */
class MotionEvidenceFusionEngine(
    private val config: Config = Config(),
) {

    enum class State {
        UNKNOWN,
        STATIC,
        TRACKING,
        ACTIVE,
        CAMERA_DOMINANT,
        DEGRADED,
    }

    data class Config(
        val staticThreshold: Double = 0.24,
        val activeThreshold: Double = 0.60,
        val strongThreshold: Double = 0.82,
        val cameraDominantThreshold: Double = 0.60,
        val occlusionDegradeThreshold: Double = 0.62,
        val minimumSubjectConfidence: Double = 0.32,
        val minimumTemporalStability: Double = 0.28,
        val historyAlpha: Double = 0.18,
        val stateAlpha: Double = 0.24,
        val enterActiveFrames: Int = 3,
        val exitActiveFrames: Int = 4,
        val degradedFrames: Int = 3,
        val recoveryFrames: Int = 5,
        val ambiguityPenaltyScale: Double = 0.72,
        val cameraPenaltyScale: Double = 0.58,
        val occlusionPenaltyScale: Double = 0.46,
        val minimumTimingConfidence: Double = 0.55,
        val minimumSyncConfidence: Double = 0.55,
        val denseConsensusWeight: Double = 0.05,
        val viewDiversityWeight: Double = 0.035,
        val continuousTimeWeight: Double = 0.035,
        val uncertaintyPenaltyScale: Double = 0.18,
    ) {
        init {
            require(staticThreshold > 0.0)
            require(activeThreshold > staticThreshold)
            require(strongThreshold > activeThreshold)
            require(cameraDominantThreshold in 0.0..1.0)
            require(occlusionDegradeThreshold in 0.0..1.0)
            require(minimumSubjectConfidence in 0.0..1.0)
            require(minimumTemporalStability in 0.0..1.0)
            require(historyAlpha in 0.0..1.0)
            require(stateAlpha in 0.0..1.0)
            require(enterActiveFrames >= 1)
            require(exitActiveFrames >= 1)
            require(degradedFrames >= 1)
            require(recoveryFrames >= 1)
            require(ambiguityPenaltyScale in 0.0..1.0)
            require(cameraPenaltyScale in 0.0..1.0)
            require(occlusionPenaltyScale in 0.0..1.0)
            require(minimumTimingConfidence in 0.0..1.0)
            require(minimumSyncConfidence in 0.0..1.0)
            require(denseConsensusWeight >= 0.0)
            require(viewDiversityWeight >= 0.0)
            require(continuousTimeWeight >= 0.0)
            require(uncertaintyPenaltyScale in 0.0..1.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val modelMotionScore: Double,
        val modelTemporalStability: Double,
        val semanticMotionEnergy: Double,
        val cameraMotionRisk: Double,
        val occlusionRisk: Double,
        val subjectLockConfidence: Double,
        val poseConfidence: Double,
        val reconstructionConfidence: Double,
        val captureTimingConfidence: Double,
        val synchronizationConfidence: Double,
        val lightingConfidence: Double,
        val denseConsensus: Double = 1.0,
        val viewDiversity: Double = 1.0,
        val continuousTimeConfidence: Double = 1.0,
        val reconstructionUncertaintyMeters: Double = 0.0,
    ) {
        init {
            require(timestampNanos > 0L)
            require(modelMotionScore in 0.0..1.0)
            require(modelTemporalStability in 0.0..1.0)
            require(semanticMotionEnergy in 0.0..1.0)
            require(cameraMotionRisk in 0.0..1.0)
            require(occlusionRisk in 0.0..1.0)
            require(subjectLockConfidence in 0.0..1.0)
            require(poseConfidence in 0.0..1.0)
            require(reconstructionConfidence in 0.0..1.0)
            require(captureTimingConfidence in 0.0..1.0)
            require(synchronizationConfidence in 0.0..1.0)
            require(lightingConfidence in 0.0..1.0)
            require(denseConsensus in 0.0..1.0)
            require(viewDiversity in 0.0..1.0)
            require(continuousTimeConfidence in 0.0..1.0)
            require(reconstructionUncertaintyMeters.isFinite() && reconstructionUncertaintyMeters >= 0.0)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val state: State,
        val score: Double,
        val confidence: Double,
        val subjectEvidence: Double,
        val cameraInterference: Double,
        val occlusionInterference: Double,
        val temporalEvidence: Double,
        val degraded: Boolean,
        val enterActiveFrames: Int,
        val exitActiveFrames: Int,
        val degradedFrames: Int,
        val recoveryFrames: Int,
        val warnings: List<String>,
        val recommendations: List<String>,
        val denseGeometryEvidence: Double = 1.0,
        val continuousTimeEvidence: Double = 1.0,
        val uncertaintyPenalty: Double = 0.0,
    ) {
        val active: Boolean
            get() = state == State.ACTIVE

        val cameraDominant: Boolean
            get() = state == State.CAMERA_DOMINANT
    }

    private data class History(
        var timestampNanos: Long,
        var scoreEma: Double,
        var confidenceEma: Double,
        var subjectEvidenceEma: Double,
        var temporalEvidenceEma: Double,
        var lastState: State,
        var enterActiveFrames: Int,
        var exitActiveFrames: Int,
        var degradedFrames: Int,
        var recoveryFrames: Int,
    )

    private var history: History? = null

    @Synchronized
    fun evaluate(input: Input): Result {
        val motionQuality = (
            0.45 * input.modelMotionScore +
                0.35 * input.semanticMotionEnergy +
                0.10 * input.modelTemporalStability +
                0.05 * input.poseConfidence +
                0.05 * input.subjectLockConfidence
        ).coerceIn(0.0, 1.0)

        val supportQuality = (
            0.45 * input.reconstructionConfidence +
                0.25 * input.captureTimingConfidence +
                0.20 * input.synchronizationConfidence +
                0.10 * input.lightingConfidence
        ).coerceIn(0.0, 1.0)

        val evidenceCoverage = (
            config.denseConsensusWeight * input.denseConsensus +
                config.viewDiversityWeight * input.viewDiversity +
                config.continuousTimeWeight * input.continuousTimeConfidence
        ).coerceIn(0.0, 0.12)

        val subjectEvidence = (
            motionQuality * 0.86 +
                supportQuality * 0.09 +
                evidenceCoverage * 0.05
        ).coerceIn(0.0, 1.0)

        val uncertaintyPenalty = (
            1.0 -
                exp(
                    -input.reconstructionUncertaintyMeters /
                        0.030.coerceAtLeast(1e-6)
                )
            ).coerceIn(0.0, 1.0) *
            config.uncertaintyPenaltyScale

        val cameraInterference = (
            input.cameraMotionRisk *
                config.cameraPenaltyScale
        ).coerceIn(0.0, 1.0)

        val occlusionInterference = (
            input.occlusionRisk *
                config.occlusionPenaltyScale
        ).coerceIn(0.0, 1.0)

        val timingInterference = (
            1.0 -
                input.captureTimingConfidence
        ).coerceIn(0.0, 1.0)

        val syncInterference = (
            1.0 -
                input.synchronizationConfidence
        ).coerceIn(0.0, 1.0)

        val ambiguity = (
            cameraInterference * 0.36 +
                occlusionInterference * 0.20 +
                timingInterference * 0.15 +
                syncInterference * 0.13 +
                uncertaintyPenalty * 0.16
        ).coerceIn(0.0, 1.0)

        val rawScore = (
            subjectEvidence *
                (1.0 - ambiguity * config.ambiguityPenaltyScale)
        ).coerceIn(0.0, 1.0)

        val temporalEvidence = (
            0.38 * input.modelTemporalStability +
                0.22 * input.captureTimingConfidence +
                0.15 * input.synchronizationConfidence +
                0.10 * input.reconstructionConfidence +
                0.15 * input.continuousTimeConfidence
        ).coerceIn(0.0, 1.0)

        val stateHistory = history
        if (stateHistory == null) {
            history = History(
                timestampNanos = input.timestampNanos,
                scoreEma = rawScore,
                confidenceEma = evidenceConfidence(
                    subjectEvidence,
                    temporalEvidence,
                    ambiguity,
                ) * (1.0 - uncertaintyPenalty),
                subjectEvidenceEma = subjectEvidence,
                temporalEvidenceEma = temporalEvidence,
                lastState = State.UNKNOWN,
                enterActiveFrames = if (
                    rawScore >= config.activeThreshold &&
                        subjectEvidence >= config.minimumSubjectConfidence &&
                        temporalEvidence >= config.minimumTemporalStability
                ) 1 else 0,
                exitActiveFrames = if (
                    rawScore <= config.staticThreshold &&
                        0.65 * input.modelMotionScore +
                        0.35 * input.semanticMotionEnergy <= 0.20
                ) config.exitActiveFrames else 0,
                degradedFrames = 0,
                recoveryFrames = if (ambiguity < 0.45) 1 else 0,
            )
        } else {
            updateHistory(
                stateHistory,
                input.timestampNanos,
                rawScore,
                subjectEvidence,
                temporalEvidence,
                ambiguity,
                input.modelMotionScore,
                input.semanticMotionEnergy,
            )
        }

        val current = history!!

        val state = resolveState(
            history = current,
            score = current.scoreEma,
            confidence = current.confidenceEma,
            subjectEvidence = current.subjectEvidenceEma,
            temporalEvidence = current.temporalEvidenceEma,
            cameraInterference = cameraInterference,
            occlusionInterference = occlusionInterference,
            input = input,
        )

        current.lastState = state

        val confidence = evidenceConfidence(
            subjectEvidence = current.subjectEvidenceEma,
            temporalEvidence = current.temporalEvidenceEma,
            ambiguity = ambiguity,
        ) * (1.0 - uncertaintyPenalty)

        current.confidenceEma =
            current.confidenceEma * (1.0 - config.stateAlpha) +
                confidence * config.stateAlpha

        val degraded =
            current.confidenceEma < config.minimumSubjectConfidence ||
                current.temporalEvidenceEma <
                    config.minimumTemporalStability ||
                input.captureTimingConfidence <
                    config.minimumTimingConfidence ||
                input.synchronizationConfidence <
                    config.minimumSyncConfidence

        val warnings = buildWarnings(
            state = state,
            score = current.scoreEma,
            confidence = current.confidenceEma,
            subjectEvidence = current.subjectEvidenceEma,
            temporalEvidence = current.temporalEvidenceEma,
            cameraInterference = cameraInterference,
            occlusionInterference = occlusionInterference,
            timingInterference = timingInterference,
            syncInterference = syncInterference,
            denseConsensus = input.denseConsensus,
            viewDiversity = input.viewDiversity,
            continuousTimeConfidence = input.continuousTimeConfidence,
            uncertaintyPenalty = uncertaintyPenalty,
            degraded = degraded,
            history = current,
        )

        val recommendations = buildRecommendations(
            state = state,
            confidence = current.confidenceEma,
            cameraInterference = cameraInterference,
            occlusionInterference = occlusionInterference,
            timingInterference = timingInterference,
            syncInterference = syncInterference,
            degraded = degraded,
        )

        return Result(
            timestampNanos = input.timestampNanos,
            state = state,
            score = current.scoreEma,
            confidence = current.confidenceEma,
            subjectEvidence = current.subjectEvidenceEma,
            cameraInterference = cameraInterference,
            occlusionInterference = occlusionInterference,
            temporalEvidence = current.temporalEvidenceEma,
            degraded = degraded,
            enterActiveFrames = current.enterActiveFrames,
            exitActiveFrames = current.exitActiveFrames,
            degradedFrames = current.degradedFrames,
            recoveryFrames = current.recoveryFrames,
            warnings = warnings,
            recommendations = recommendations,
            denseGeometryEvidence = (
                input.denseConsensus * input.viewDiversity
            ).coerceIn(0.0, 1.0),
            continuousTimeEvidence = input.continuousTimeConfidence,
            uncertaintyPenalty = uncertaintyPenalty,
        )
    }

    @Synchronized
    fun reset() {
        history = null
    }

    @Synchronized
    fun state(): State =
        history?.lastState ?: State.UNKNOWN

    private fun updateHistory(
        history: History,
        timestampNanos: Long,
        rawScore: Double,
        subjectEvidence: Double,
        temporalEvidence: Double,
        ambiguity: Double,
        modelMotionScore: Double,
        semanticMotionEnergy: Double,
    ) {
        val alpha = config.historyAlpha
        history.timestampNanos = timestampNanos
        history.scoreEma =
            history.scoreEma * (1.0 - alpha) +
                rawScore * alpha
        history.subjectEvidenceEma =
            history.subjectEvidenceEma * (1.0 - alpha) +
                subjectEvidence * alpha
        history.temporalEvidenceEma =
            history.temporalEvidenceEma * (1.0 - alpha) +
                temporalEvidence * alpha

        val possibleActive =
            history.scoreEma >= config.activeThreshold &&
                history.subjectEvidenceEma >=
                    config.minimumSubjectConfidence &&
                history.temporalEvidenceEma >=
                    config.minimumTemporalStability

        val motionPresence =
            (
                0.65 * modelMotionScore +
                    0.35 * semanticMotionEnergy
            ).coerceIn(0.0, 1.0)

        val possibleStatic =
            history.scoreEma <= config.staticThreshold ||
                (
                    motionPresence <= 0.20 &&
                        history.scoreEma <=
                            config.activeThreshold * 0.60
                    )

        val possibleDegraded =
            history.confidenceEma <
                config.minimumSubjectConfidence ||
                history.temporalEvidenceEma <
                    config.minimumTemporalStability ||
                ambiguity > 0.72

        if (possibleActive) {
            history.enterActiveFrames += 1
            history.exitActiveFrames = 0
        } else {
            history.enterActiveFrames =
                max(0, history.enterActiveFrames - 1)
            history.exitActiveFrames += 1
        }

        if (possibleDegraded) {
            history.degradedFrames += 1
            history.recoveryFrames = 0
        } else {
            history.degradedFrames =
                max(0, history.degradedFrames - 1)
            history.recoveryFrames += 1
        }

        if (possibleStatic) {
            history.exitActiveFrames =
                max(
                    history.exitActiveFrames,
                    config.exitActiveFrames,
                )
        }

        if (ambiguity < 0.45) {
            history.recoveryFrames =
                max(
                    history.recoveryFrames,
                    1,
                )
        }
    }

    private fun resolveState(
        history: History,
        score: Double,
        confidence: Double,
        subjectEvidence: Double,
        temporalEvidence: Double,
        cameraInterference: Double,
        occlusionInterference: Double,
        input: Input,
    ): State {
        val cameraDominant =
            input.cameraMotionRisk >=
                config.cameraDominantThreshold &&
                input.cameraMotionRisk >
                    input.semanticMotionEnergy * 0.95

        if (cameraDominant) {
            return State.CAMERA_DOMINANT
        }

        if (
            history.degradedFrames >=
                config.degradedFrames
        ) {
            if (
                history.recoveryFrames <
                    config.recoveryFrames
            ) {
                return State.DEGRADED
            }
        }

        if (
            history.lastState == State.ACTIVE &&
                history.exitActiveFrames <
                    config.exitActiveFrames &&
                score > config.staticThreshold
        ) {
            return State.ACTIVE
        }

        if (
            history.lastState == State.ACTIVE &&
                history.exitActiveFrames >=
                    config.exitActiveFrames
        ) {
            if (
                history.recoveryFrames >=
                    config.recoveryFrames
            ) {
                return State.TRACKING
            }
        }

        if (
            history.enterActiveFrames >=
                config.enterActiveFrames &&
                score >= config.activeThreshold &&
                subjectEvidence >=
                    config.minimumSubjectConfidence
        ) {
            return State.ACTIVE
        }

        if (
            confidence < config.minimumSubjectConfidence ||
                temporalEvidence <
                    config.minimumTemporalStability
        ) {
            return State.DEGRADED
        }

        if (
            (
                score <= config.staticThreshold ||
                    (
                        0.65 * input.modelMotionScore +
                            0.35 * input.semanticMotionEnergy
                        ) <= 0.20
                ) &&
                subjectEvidence < config.activeThreshold
        ) {
            return State.STATIC
        }

        return State.TRACKING
    }

    private fun evidenceConfidence(
        subjectEvidence: Double,
        temporalEvidence: Double,
        ambiguity: Double,
    ): Double = (
        0.54 * subjectEvidence +
            0.32 * temporalEvidence +
            0.14 * (1.0 - ambiguity)
    ).coerceIn(0.0, 1.0)

    private fun buildWarnings(
        state: State,
        score: Double,
        confidence: Double,
        subjectEvidence: Double,
        temporalEvidence: Double,
        cameraInterference: Double,
        occlusionInterference: Double,
        timingInterference: Double,
        syncInterference: Double,
        denseConsensus: Double,
        viewDiversity: Double,
        continuousTimeConfidence: Double,
        uncertaintyPenalty: Double,
        degraded: Boolean,
        history: History,
    ): List<String> = buildList {
        when (state) {
            State.UNKNOWN -> add("motion-evidence-awaiting-temporal-history")
            State.STATIC -> Unit
            State.TRACKING -> {
                if (score < config.activeThreshold) {
                    add("subject-motion-below-active-threshold")
                }
            }
            State.ACTIVE -> {
                if (config.strongThreshold <= score) {
                    add("high-motion-energy")
                }
            }
            State.CAMERA_DOMINANT -> {
                add("camera-motion-dominates-observed-frame-change")
                add("subject-motion-confidence-reduced-until-camera-stabilizes")
            }
            State.DEGRADED -> {
                add("motion-evidence-quality-degraded")
                if (temporalEvidence < config.minimumTemporalStability) {
                    add("temporal-motion-evidence-is-unstable")
                }
            }
        }

        if (cameraInterference > 0.35) {
            add("camera-motion-interference-elevated")
        }
        if (occlusionInterference > 0.35) {
            add("occlusion-interference-elevated")
        }
        if (timingInterference > 0.35) {
            add("capture-timing-quality-reduces-motion-confidence")
        }
        if (syncInterference > 0.35) {
            add("multi-camera-synchronization-reduces-motion-confidence")
        }
        if (denseConsensus < 0.55) {
            add("dense-cross-view-consensus-reduces-motion-confidence")
        }
        if (viewDiversity < 0.55) {
            add("multi-view-diversity-is-insufficient-for-robust-motion-evidence")
        }
        if (continuousTimeConfidence < 0.55) {
            add("continuous-time-observation-confidence-is-degraded")
        }
        if (uncertaintyPenalty > 0.10) {
            add("reconstruction-uncertainty-reduces-motion-evidence-confidence")
        }
        if (confidence < config.minimumSubjectConfidence) {
            add("motion-evidence-confidence-below-production-floor")
        }
        if (history.enterActiveFrames >= config.enterActiveFrames) {
            add("active-motion-state-persisted")
        }
        if (degraded && history.degradedFrames >= config.degradedFrames) {
            add("persistent-motion-evidence-degradation")
        }
    }.distinct()

    private fun buildRecommendations(
        state: State,
        confidence: Double,
        cameraInterference: Double,
        occlusionInterference: Double,
        timingInterference: Double,
        syncInterference: Double,
        degraded: Boolean,
    ): List<String> = buildList {
        if (state == State.CAMERA_DOMINANT) {
            add("stabilize the camera rig before interpreting large frame changes as subject motion")
        }
        if (cameraInterference > 0.35) {
            add("reduce camera motion or increase inertial compensation confidence")
        }
        if (occlusionInterference > 0.35) {
            add("improve subject visibility or use a more complete multi-view configuration")
        }
        if (timingInterference > 0.35) {
            add("restore stable capture cadence before trusting high-frequency motion")
        }
        if (syncInterference > 0.35) {
            add("tighten inter-camera clock synchronization before metric motion analysis")
        }
        if (degraded || confidence < config.minimumSubjectConfidence) {
            add("hold the motion event state until temporal evidence recovers")
        }
    }.distinct()
}

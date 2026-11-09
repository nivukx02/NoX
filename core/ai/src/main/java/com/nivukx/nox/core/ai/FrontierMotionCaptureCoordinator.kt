package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.WorldSpaceProposal

/** Composes the frontier motion stages without replacing existing perception implementations. */
class FrontierMotionCaptureCoordinator(
    private val proposalEngine: WorldSpaceMotionProposalEngine = WorldSpaceMotionProposalEngine(),
    private val motionEngine: HierarchicalMotionInferenceEngine = HierarchicalMotionInferenceEngine(),
    private val correspondenceEngine: DenseWorldPointCorrespondenceEngine = DenseWorldPointCorrespondenceEngine(),
    private val identityEngine: MultiActorHypothesisAssociationEngine = MultiActorHypothesisAssociationEngine(),
    private val scheduler: FrontierActorDetailScheduler = FrontierActorDetailScheduler(),
    private val evidenceAuthority: FrontierEvidenceAuthorityEngine = FrontierEvidenceAuthorityEngine(),
    private val crowdEvidence: ProductionCrowdMotionEvidenceEngine = ProductionCrowdMotionEvidenceEngine(),
    private val occlusionRecovery: MultiHypothesisOcclusionRecoveryEngine =
        MultiHypothesisOcclusionRecoveryEngine(),
    private val higherOrderIdentityEngine:
        MultiViewActorHypothesisConsensusEngine =
        MultiViewActorHypothesisConsensusEngine(),
) : AutoCloseable {

    data class Input(
        val timestampNanos: Long,
        val rayObservations: List<WorldSpaceMotionProposalEngine.RayObservation>,
        val learnedProposals: Map<Pair<Int, Int>, WorldSpaceMotionProposalEngine.LearnedProposal> = emptyMap(),
        val denseObservations: List<DenseWorldPointCorrespondenceEngine.Observation> = emptyList(),
        val actorObservations: List<MultiActorHypothesisAssociationEngine.ActorObservation> = emptyList(),
        val actorPredictions: Map<Int, com.nivukx.nox.core.model.Vec3> = emptyMap(),
        val actorDemands: List<FrontierActorDetailScheduler.ActorDemand> = emptyList(),
        val runtimePressure: Double = 0.0,
    )

    data class Result(
        val proposals: List<WorldSpaceProposal>,
        val motion: HierarchicalMotionInferenceEngine.Result,
        val denseTracks: DenseWorldPointCorrespondenceEngine.Result,
        val identities: MultiActorHypothesisAssociationEngine.Result,
        val actorAllocation: FrontierActorDetailScheduler.Result,
        val evidenceAuthority: FrontierEvidenceAuthorityEngine.Result,
        val crowdEvidence: ProductionCrowdMotionEvidenceEngine.FrameResult? = null,
        val occlusionRecovery:
            List<MultiHypothesisOcclusionRecoveryEngine.Result> = emptyList(),
        val higherOrderIdentity:
            Map<Int, MultiViewActorHypothesisConsensusEngine.Result> = emptyMap(),
    )

    @Synchronized
    fun process(input: Input): Result {
        require(input.runtimePressure in 0.0..1.0)
        val grouped = input.rayObservations.groupBy { it.actorId to it.pointId }
        val proposals = grouped.entries.mapNotNull { (key, observations) ->
            proposalEngine.propose(input.timestampNanos, observations, input.learnedProposals[key]).proposal
        }
        val motion = motionEngine.process(input.timestampNanos, proposals)

        val occlusionResults =
            motion.states
                .asSequence()
                .filter {
                    it.origin ==
                        com.nivukx.nox.core.model.EvidenceOrigin.TEMPORALLY_PROPAGATED &&
                        it.hypothesisRank == 0 &&
                        it.confidence >= 0.05
                }
                .sortedWith(
                    compareByDescending<HierarchicalMotionInferenceEngine.State> {
                        it.uncertaintyMeters
                    }.thenBy {
                        it.actorId
                    }.thenBy {
                        it.pointId
                    },
                )
                .take(2_048)
                .map { state ->
                    occlusionRecovery.predict(
                        track =
                            MultiHypothesisOcclusionRecoveryEngine.TrackState(
                                actorId = state.actorId.toLong(),
                                timestampNanos = state.timestampNanos.coerceAtLeast(1L),
                                positionMeters = state.positionMeters,
                                velocityMetersPerSecond = state.velocityMetersPerSecond,
                                accelerationMetersPerSecond2 =
                                    state.accelerationMetersPerSecond2,
                                confidence = state.confidence,
                                uncertaintyMeters =
                                    state.uncertaintyMeters.coerceAtLeast(1e-6),
                            ),
                        timestampNanos =
                            input.timestampNanos.coerceAtLeast(
                                state.timestampNanos,
                            ),
                    )
                }
                .toList()

        val denseTracks = correspondenceEngine.update(input.denseObservations, input.timestampNanos)
        val identities = identityEngine.associate(input.actorObservations, input.actorPredictions)

        val higherOrderIdentity =
            input.actorObservations
                .asSequence()
                .filter { it.actorHint != null }
                .groupBy { it.actorHint!! }
                .mapValues { (_, observations) ->
                    higherOrderIdentityEngine.evaluate(observations)
                }

        val crowd = runCatching {
            crowdEvidence.evaluate(
                ProductionCrowdMotionEvidenceEngine.Input(
                    timestampNanos = input.timestampNanos,
                    actors = identities.accepted.filter { !it.predicted },
                    runtimeHeadroom =
                        (1.0 - input.runtimePressure).coerceIn(0.0, 1.0),
                ),
            )
        }.getOrNull()
        val actorAllocation = scheduler.allocate(input.actorDemands, input.runtimePressure)
        val observedActorCount = identities.accepted
            .map { it.actorId }
            .distinct()
            .size
        val scheduledActorCount = input.actorDemands
            .map { it.actorId }
            .distinct()
            .size
        val authorityActorCount = maxOf(scheduledActorCount, observedActorCount)
        val identityDenominator = (
            identities.accepted.size +
                identities.rejectedDetectionIds.size +
                identities.ambiguousDetectionIds.size
            ).coerceAtLeast(1)
        val identityAmbiguityRisk =
            identities.ambiguousDetectionIds.size.toDouble() /
                identityDenominator.toDouble()
        val authority = evidenceAuthority.evaluate(
            proposals = proposals,
            denseTrackCount = denseTracks.tracks.size,
            actorCount = authorityActorCount,
            identityAmbiguityRisk = identityAmbiguityRisk.coerceIn(0.0, 1.0),
        )
        return Result(
            proposals = proposals,
            motion = motion,
            denseTracks = denseTracks,
            identities = identities,
            actorAllocation = actorAllocation,
            evidenceAuthority = authority,
            crowdEvidence = crowd,
            occlusionRecovery = occlusionResults,
            higherOrderIdentity = higherOrderIdentity,
        )
    }

    override fun close() {
        proposalEngine.close()
        motionEngine.close()
        correspondenceEngine.close()
        crowdEvidence.close()
        occlusionRecovery.reset()
        // Higher-order identity consensus is stateless; no retained per-frame state requires reset.
        identityEngine.clear()
    }
}
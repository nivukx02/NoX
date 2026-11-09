package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.ai.HierarchicalMotionInferenceEngine
import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame

/**
 * Converts frontier world-space motion hypotheses into the existing reconstruction contracts.
 * Multi-actor states remain separate skeletons; no actor is collapsed into a shared track.
 */
object FrontierMotionReconstructionAdapter {
    data class Result(
        val skeleton: SkeletonFrame,
        val skeletons: List<SkeletonFrame>,
        val originsByJointKey: Map<Long, EvidenceOrigin>,
        val predictedJointCount: Int,
        val observedJointCount: Int,
    )

    fun adapt(result: HierarchicalMotionInferenceEngine.Result): Result {
        val authoritative = result.states
            .groupBy { key(it.actorId, it.pointId) }
            .mapNotNull { (_, states) -> states.minByOrNull { it.hypothesisRank } }

        val originMap = authoritative.associate { state ->
            key(state.actorId, state.pointId) to state.origin
        }

        val skeletons = authoritative
            .groupBy { it.actorId }
            .toSortedMap()
            .map { (actorId, states) ->
                val joints = states
                    .sortedBy { it.pointId }
                    .map { state ->
                        ReconstructionJoint(
                            landmarkId = state.pointId,
                            position = state.positionMeters,
                            velocity = state.velocityMetersPerSecond,
                            acceleration = state.accelerationMetersPerSecond2,
                            confidence = state.confidence.coerceIn(0.0, 1.0),
                            observationCount = state.sourceObservationCount,
                            positionUncertaintyMeters = state.uncertaintyMeters.coerceAtLeast(0.0),
                        )
                    }
                SkeletonFrame(
                    timestampNanos = result.timestampNanos,
                    trackId = actorId.toLong(),
                    joints = joints,
                    globalConfidence = joints.map { it.confidence }.averageOrZero(),
                )
            }

        return Result(
            skeleton = skeletons.firstOrNull() ?: emptySkeleton(result.timestampNanos),
            skeletons = skeletons,
            originsByJointKey = originMap,
            predictedJointCount = authoritative.count { it.origin == EvidenceOrigin.TEMPORALLY_PROPAGATED },
            observedJointCount = authoritative.count {
                it.origin == EvidenceOrigin.OBSERVED ||
                    it.origin == EvidenceOrigin.GEOMETRICALLY_CONSTRAINED
            },
        )
    }

    private fun emptySkeleton(timestampNanos: Long) =
        SkeletonFrame(
            timestampNanos = timestampNanos,
            trackId = 0L,
            joints = emptyList(),
            globalConfidence = 0.0,
        )

    private fun key(actorId: Int, pointId: Int): Long =
        (actorId.toLong() shl 32) xor (pointId.toLong() and 0xffffffffL)
}

private fun List<Double>.averageOrZero(): Double =
    if (isEmpty()) 0.0 else average()
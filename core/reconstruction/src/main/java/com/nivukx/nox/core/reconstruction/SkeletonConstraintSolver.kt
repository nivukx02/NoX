package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Lightweight position-based dynamics for a 33-landmark human skeleton.
 *
 * Bone lengths are learned from confident early frames instead of assuming a body size in meters.
 * During motion, each constraint pulls a pair of joints toward its learned metric length. This
 * removes depth jitter and impossible limb stretching without locking the actor to a rigid pose.
 */
class SkeletonConstraintSolver(
    private val iterations: Int = 4,
    private val learningRate: Double = 0.18,
    private val minConfidence: Double = 0.35,
) {
    private data class Bone(val parent: Int, val child: Int)

    private val bones = listOf(
        Bone(11, 13), Bone(13, 15),
        Bone(12, 14), Bone(14, 16),
        Bone(11, 12),
        Bone(11, 23), Bone(12, 24),
        Bone(23, 24),
        Bone(23, 25), Bone(25, 27),
        Bone(24, 26), Bone(26, 28),
        Bone(27, 31), Bone(28, 32),
        Bone(0, 11), Bone(0, 12),
    )

    private val learned = HashMap<Bone, Double>()
    private val samples = HashMap<Bone, ArrayDeque<Double>>()
    private val maximumHistory = 24

    @Synchronized
    fun apply(input: List<ReconstructionJoint>): List<ReconstructionJoint> {
        val source = input.associateBy { it.landmarkId }.toMutableMap()
        learnLengths(source)

        if (learned.isEmpty()) return input
        repeat(iterations) {
            bones.forEach { bone ->
                val parent = source[bone.parent] ?: return@forEach
                val child = source[bone.child] ?: return@forEach

                val target = learned[bone] ?: return@forEach
                val relativeError =
                    abs(
                        child.position.distanceTo(parent.position) - target,
                    ) / target.coerceAtLeast(1e-6)
                if (relativeError > 0.42) return@forEach

                val delta = child.position - parent.position
                val distance = delta.norm()
                if (distance < 1e-7) return@forEach

                val error = distance - target
                val correction = delta * (error / distance * learningRate)
                val parentWeight = parent.confidence.coerceIn(0.05, 1.0)
                val childWeight = child.confidence.coerceIn(0.05, 1.0)
                val total = parentWeight + childWeight

                source[bone.parent] = parent.copy(
                    position = parent.position + correction * (parentWeight / total),
                )
                source[bone.child] = child.copy(
                    position = child.position - correction * (childWeight / total),
                )
            }
        }

        return input.mapNotNull { source[it.landmarkId] }
    }

    @Synchronized
    fun reset() {
        learned.clear()
        samples.clear()
    }

    private fun learnLengths(joints: Map<Int, ReconstructionJoint>) {
        bones.forEach { bone ->
            val a = joints[bone.parent] ?: return@forEach
            val b = joints[bone.child] ?: return@forEach
            if (a.confidence < minConfidence || b.confidence < minConfidence) return@forEach

            val length = a.position.distanceTo(b.position)
            if (!length.isFinite() || length < 1e-4) return@forEach

            val history = samples.getOrPut(bone) { ArrayDeque() }
            history.addLast(length)
            while (history.size > maximumHistory) {
                history.removeFirst()
            }

            val ordered = history.toList().sorted()
            val median =
                if (ordered.size % 2 == 0) {
                    (ordered[ordered.size / 2 - 1] + ordered[ordered.size / 2]) * 0.5
                } else {
                    ordered[ordered.size / 2]
                }

            val robustGate = (
                1.0 -
                    abs(length - median) /
                        median.coerceAtLeast(1e-6)
                ).coerceIn(0.0, 1.0)

            if (robustGate < 0.35) return@forEach

            val old = learned[bone]
            learned[bone] = if (old == null) {
                median
            } else {
                val target = (
                    old * 0.80 +
                        median * 0.20
                    )
                min(
                    max(target, old * 0.70),
                    old * 1.30,
                )
            }
        }
    }
}

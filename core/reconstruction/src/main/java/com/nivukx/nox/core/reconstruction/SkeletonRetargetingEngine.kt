package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.AvatarBonePose
import com.nivukx.nox.core.model.AvatarPose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec3
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min

/**
 * Converts reconstructed metric landmarks into an avatar-neutral bone pose graph.
 *
 * Rest lengths are learned once from the subject. Runtime frames are scaled against the learned
 * body height and each bone receives an orientation derived from the current parent/child vector.
 * The result is independent of the source phone layout and therefore suitable for later avatar
 * replacement or animation export.
 */
class SkeletonRetargetingEngine(
    private val skeletonHeightLandmarks: Pair<Int, Int> = 0 to 24,
    private val config: Config = Config(),
) {

    data class Config(
        val maximumJointUncertaintyMeters: Double = 0.18,
        val minimumBoneQuality: Double = 0.70,
    ) {
        init {
            require(maximumJointUncertaintyMeters > 0.0)
            require(minimumBoneQuality in 0.0..1.0)
        }
    }
    data class Bone(
        val name: String,
        val parentLandmark: Int,
        val childLandmark: Int,
        val restAxis: Vec3 = Vec3(0.0, 1.0, 0.0),
    )

    private val bones = listOf(
        Bone("hips", 23, 24),
        Bone("spine", 23, 11, Vec3(0.0, 1.0, 0.0)),
        Bone("chest", 11, 12),
        Bone("left_upper_arm", 11, 13, Vec3(1.0, 0.0, 0.0)),
        Bone("left_lower_arm", 13, 15, Vec3(1.0, 0.0, 0.0)),
        Bone("right_upper_arm", 12, 14, Vec3(-1.0, 0.0, 0.0)),
        Bone("right_lower_arm", 14, 16, Vec3(-1.0, 0.0, 0.0)),
        Bone("left_upper_leg", 23, 25, Vec3(0.0, -1.0, 0.0)),
        Bone("left_lower_leg", 25, 27, Vec3(0.0, -1.0, 0.0)),
        Bone("right_upper_leg", 24, 26, Vec3(0.0, -1.0, 0.0)),
        Bone("right_lower_leg", 26, 28, Vec3(0.0, -1.0, 0.0)),
        Bone("left_foot", 27, 31, Vec3(0.0, -1.0, 0.0)),
        Bone("right_foot", 28, 32, Vec3(0.0, -1.0, 0.0)),
    )

    private var referenceHeightMeters: Double? = null

    @Synchronized
    fun calibrate(frame: SkeletonFrame): Double? {
        val top = frame.joints.firstOrNull { it.landmarkId == skeletonHeightLandmarks.first }
            ?: return null
        val bottom = frame.joints.firstOrNull { it.landmarkId == skeletonHeightLandmarks.second }
            ?: return null

        val height = top.position.distanceTo(bottom.position)
        if (height < 0.5 || height > 3.0) return null

        referenceHeightMeters = height
        return height
    }

    @Synchronized
    fun retarget(frame: SkeletonFrame): AvatarPose? {
        val height = referenceHeightMeters ?: calibrate(frame) ?: return null
        val joints = frame.joints.associateBy { it.landmarkId }

        val root = joints[23]?.position ?: joints[24]?.position ?: return null
        val currentHeight = run {
            val top = joints[skeletonHeightLandmarks.first]?.position ?: return null
            val bottom = joints[skeletonHeightLandmarks.second]?.position ?: return null
            top.distanceTo(bottom)
        }

        val scale = (currentHeight / height).coerceIn(0.5, 2.0)
        val bonePoses = LinkedHashMap<String, AvatarBonePose>()

        bones.forEach { bone ->
            val parent = joints[bone.parentLandmark] ?: return@forEach
            val child = joints[bone.childLandmark] ?: return@forEach
            val parentQuality = jointQuality(parent)
            val childQuality = jointQuality(child)

            if (
                min(parentQuality, childQuality) <
                    config.minimumBoneQuality
            ) {
                return@forEach
            }

            val vector = (child.position - parent.position)
            if (vector.norm() < 1e-5) return@forEach

            val direction = vector.normalized()
            val rotation = rotationBetween(bone.restAxis.normalized(), direction)
            val localPosition = (parent.position - root) / scale

            bonePoses[bone.name] = AvatarBonePose(
                position = localPosition,
                rotation = rotation,
            )
        }

        val confidence = (
            frame.globalConfidence *
                bonePoses.size.toDouble() /
                    bones.size.coerceAtLeast(1) *
                averageBoneQuality(
                    joints = joints,
                )
            ).coerceIn(0.0, 1.0)

        return AvatarPose(
            timestampNanos = frame.timestampNanos,
            rootPositionMeters = root,
            scale = scale,
            bones = bonePoses,
            confidence = confidence.coerceIn(0.0, 1.0),
        )
    }

    @Synchronized
    fun reset() {
        referenceHeightMeters = null
    }

    private fun jointQuality(
        joint: com.nivukx.nox.core.model.ReconstructionJoint,
    ): Double {
        val uncertaintyScore =
            kotlin.math.exp(
                -joint.positionUncertaintyMeters /
                    config.maximumJointUncertaintyMeters
            ).coerceIn(0.0, 1.0)

        return (
            joint.confidence.coerceIn(0.0, 1.0) * 0.68 +
                uncertaintyScore * 0.32
            ).coerceIn(0.0, 1.0)
    }

    private fun averageBoneQuality(
        joints: Map<Int, com.nivukx.nox.core.model.ReconstructionJoint>,
    ): Double {
        val values = bones.mapNotNull { bone ->
            val parent = joints[bone.parentLandmark] ?: return@mapNotNull null
            val child = joints[bone.childLandmark] ?: return@mapNotNull null
            (jointQuality(parent) + jointQuality(child)) * 0.5
        }

        return if (values.isEmpty()) 0.0 else values.average()
    }

    private fun rotationBetween(from: Vec3, to: Vec3): Quat {
        val a = from.normalized()
        val b = to.normalized()
        val dot = a.dot(b).coerceIn(-1.0, 1.0)

        if (dot > 0.999999) return Quat.IDENTITY

        if (dot < -0.999999) {
            val axis = if (kotlin.math.abs(a.x) < 0.8) {
                a.cross(Vec3(1.0, 0.0, 0.0)).normalized()
            } else {
                a.cross(Vec3(0.0, 1.0, 0.0)).normalized()
            }

            return Quat(
                w = 0.0,
                x = axis.x,
                y = axis.y,
                z = axis.z,
            )
        }

        val cross = a.cross(b)
        val s = kotlin.math.sqrt((1.0 + dot) * 2.0)
        val invS = 1.0 / s

        return Quat(
            w = s * 0.5,
            x = cross.x * invS,
            y = cross.y * invS,
            z = cross.z * invS,
        ).normalized()
    }
}

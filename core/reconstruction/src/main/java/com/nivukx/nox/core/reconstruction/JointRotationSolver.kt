package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.AvatarBonePose
import com.nivukx.nox.core.model.AvatarPose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec3
import kotlin.math.acos
import kotlin.math.sqrt

/**
 * Geometry-driven bone orientation solver for avatar retargeting.
 *
 * The solver derives rotations from neighboring 3D joints instead of copying Euler angles from a
 * detector. It uses a stable look-at frame, optional pole vectors, quaternion continuity and
 * confidence-aware temporal blending. Ambiguous bones fall back to the previous orientation.
 */
class JointRotationSolver(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumBoneLengthMeters: Double = 0.015,
        val temporalBlend: Double = 0.72,
        val confidenceFloor: Double = 0.20,
        val maxAngularVelocityRadiansPerSecond: Double = 16.0,
        val preserveRootRotation: Boolean = true,
    )

    data class BoneDefinition(
        val name: String,
        val start: Int,
        val end: Int,
        val pole: Int? = null,
        val weight: Double = 1.0,
    )

    data class Result(
        val pose: AvatarPose,
        val angularVelocityByBone: Map<String, Vec3>,
        val confidenceByBone: Map<String, Double>,
    )

    private val defaultBones = listOf(
        BoneDefinition("spine", 23, 0, 11),
        BoneDefinition("leftUpperArm", 11, 13, 23),
        BoneDefinition("leftForearm", 13, 15, 11),
        BoneDefinition("rightUpperArm", 12, 14, 24),
        BoneDefinition("rightForearm", 14, 16, 12),
        BoneDefinition("leftUpperLeg", 23, 25, 27),
        BoneDefinition("leftLowerLeg", 25, 27, 23),
        BoneDefinition("rightUpperLeg", 24, 26, 28),
        BoneDefinition("rightLowerLeg", 26, 28, 24),
        BoneDefinition("head", 0, 11, 12),
    )

    fun solve(
        frame: SkeletonFrame,
        previous: AvatarPose? = null,
        definitions: List<BoneDefinition> = defaultBones,
    ): Result {
        val joints = frame.joints.associateBy { it.landmarkId }
        val bones = LinkedHashMap<String, AvatarBonePose>()
        val angularVelocity = LinkedHashMap<String, Vec3>()
        val confidence = LinkedHashMap<String, Double>()

        for (definition in definitions) {
            val start = joints[definition.start]
            val end = joints[definition.end]
            if (start == null || end == null) {
                previous?.bones?.get(definition.name)?.let {
                    bones[definition.name] = it
                }
                confidence[definition.name] = 0.0
                continue
            }

            val direction = end.position - start.position
            val length = direction.norm()
            val quality = minOf(
                start.confidence,
                end.confidence,
            ).coerceIn(0.0, 1.0) * definition.weight

            if (
                length < config.minimumBoneLengthMeters ||
                !direction.finite() ||
                quality < config.confidenceFloor
            ) {
                previous?.bones?.get(definition.name)?.let {
                    bones[definition.name] = it
                }
                confidence[definition.name] = quality
                continue
            }

            val pole = definition.pole
                ?.let { joints[it]?.position }
                ?: start.position + Vec3(0.0, 1.0, 0.0)

            val targetRotation = lookRotation(
                forward = direction.normalized(),
                upHint = (pole - start.position).normalized(),
            )

            val prior = previous?.bones?.get(definition.name)
            val rotation = if (prior == null) {
                targetRotation
            } else {
                val angle = quaternionAngle(prior.rotation, targetRotation)
                val maxAngle =
                    config.maxAngularVelocityRadiansPerSecond / 60.0
                val limited =
                    if (angle <= maxAngle || angle <= 1e-6) {
                        targetRotation
                    } else {
                        prior.rotation.slerp(
                            targetRotation,
                            (maxAngle / angle).coerceIn(0.0, 1.0),
                        )
                    }
                prior.rotation.slerp(
                    limited,
                    config.temporalBlend.coerceIn(0.0, 1.0),
                )
            }

            bones[definition.name] = AvatarBonePose(
                position = start.position,
                rotation = rotation.normalized(),
            )
            confidence[definition.name] = quality

            val previousRotation = prior?.rotation
            if (previousRotation != null) {
                angularVelocity[definition.name] =
                    angularVelocityFromDelta(
                        previousRotation,
                        rotation,
                        1.0 / 60.0,
                    )
            }
        }

        val root = frame.joints
            .filter { it.landmarkId == 23 || it.landmarkId == 24 }
            .map { it.position }
            .averageVectorOrZero()

        return Result(
            pose = AvatarPose(
                timestampNanos = frame.timestampNanos,
                rootPositionMeters = root,
                scale = estimateScale(frame),
                bones = bones,
                confidence = frame.globalConfidence.coerceIn(0.0, 1.0),
            ),
            angularVelocityByBone = angularVelocity,
            confidenceByBone = confidence,
        )
    }

    private fun lookRotation(
        forward: Vec3,
        upHint: Vec3,
    ): Quat {
        val f = forward.normalized()
        var r = upHint.cross(f).normalized()

        if (r.norm() < 1e-6) {
            r = Vec3(1.0, 0.0, 0.0).cross(f).normalized()
        }

        val u = f.cross(r).normalized()

        return rotationFromBasis(
            right = r,
            up = u,
            forward = f,
        )
    }

    private fun rotationFromBasis(
        right: Vec3,
        up: Vec3,
        forward: Vec3,
    ): Quat {
        val m00 = right.x
        val m01 = up.x
        val m02 = forward.x
        val m10 = right.y
        val m11 = up.y
        val m12 = forward.y
        val m20 = right.z
        val m21 = up.z
        val m22 = forward.z

        val trace = m00 + m11 + m22

        return when {
            trace > 0.0 -> {
                val s = sqrt(trace + 1.0) * 2.0
                Quat(
                    w = 0.25 * s,
                    x = (m21 - m12) / s,
                    y = (m02 - m20) / s,
                    z = (m10 - m01) / s,
                ).normalized()
            }

            m00 > m11 && m00 > m22 -> {
                val s = sqrt(1.0 + m00 - m11 - m22) * 2.0
                Quat(
                    w = (m21 - m12) / s,
                    x = 0.25 * s,
                    y = (m01 + m10) / s,
                    z = (m02 + m20) / s,
                ).normalized()
            }

            m11 > m22 -> {
                val s = sqrt(1.0 + m11 - m00 - m22) * 2.0
                Quat(
                    w = (m02 - m20) / s,
                    x = (m01 + m10) / s,
                    y = 0.25 * s,
                    z = (m12 + m21) / s,
                ).normalized()
            }

            else -> {
                val s = sqrt(1.0 + m22 - m00 - m11) * 2.0
                Quat(
                    w = (m10 - m01) / s,
                    x = (m02 + m20) / s,
                    y = (m12 + m21) / s,
                    z = 0.25 * s,
                ).normalized()
            }
        }
    }

    private fun quaternionAngle(
        a: Quat,
        b: Quat,
    ): Double {
        val dot =
            (a.w * b.w +
                a.x * b.x +
                a.y * b.y +
                a.z * b.z)
                .coerceIn(-1.0, 1.0)

        return 2.0 * acos(kotlin.math.abs(dot))
    }

    private fun angularVelocityFromDelta(
        previous: Quat,
        current: Quat,
        dtSeconds: Double,
    ): Vec3 {
        val relative =
            previous.normalized().conjugate() *
                current.normalized()

        val w = relative.w.coerceIn(-1.0, 1.0)
        val angle = 2.0 * acos(w)

        if (angle < 1e-6) {
            return Vec3(0.0, 0.0, 0.0)
        }

        val s = sqrt(
            (1.0 - w * w).coerceAtLeast(1e-12),
        )

        return Vec3(
            relative.x / s * angle / dtSeconds.coerceAtLeast(1e-4),
            relative.y / s * angle / dtSeconds.coerceAtLeast(1e-4),
            relative.z / s * angle / dtSeconds.coerceAtLeast(1e-4),
        )
    }

    private fun estimateScale(frame: SkeletonFrame): Double {
        val shoulderA = frame.joints.firstOrNull { it.landmarkId == 11 }
        val shoulderB = frame.joints.firstOrNull { it.landmarkId == 12 }
        val hipsA = frame.joints.firstOrNull { it.landmarkId == 23 }
        val hipsB = frame.joints.firstOrNull { it.landmarkId == 24 }

        val shoulderWidth =
            if (shoulderA != null && shoulderB != null) {
                shoulderA.position.distanceTo(shoulderB.position)
            } else {
                0.0
            }

        val hipWidth =
            if (hipsA != null && hipsB != null) {
                hipsA.position.distanceTo(hipsB.position)
            } else {
                0.0
            }

        return (shoulderWidth * 2.25 + hipWidth * 4.25)
            .coerceIn(0.25, 3.0)
    }

    private fun Vec3.finite(): Boolean =
        x.isFinite() && y.isFinite() && z.isFinite()

    private fun List<Vec3>.averageVectorOrZero(): Vec3 =
        if (isEmpty()) {
            Vec3(0.0, 0.0, 0.0)
        } else {
            Vec3(
                map { it.x }.average(),
                map { it.y }.average(),
                map { it.z }.average(),
            )
        }
}

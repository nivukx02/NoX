package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import kotlin.math.sqrt

object RigidTransform {
    fun alignAxes(forward: Vec3, upHint: Vec3): Quat {
        val f = forward.normalized()
        var r = upHint.cross(f)
        if (r.norm() < 1e-9) r = Vec3(1.0, 0.0, 0.0).cross(f)
        r = r.normalized()
        val u = f.cross(r).normalized()

        val m00 = r.x; val m01 = u.x; val m02 = f.x
        val m10 = r.y; val m11 = u.y; val m12 = f.y
        val m20 = r.z; val m21 = u.z; val m22 = f.z
        val trace = m00 + m11 + m22

        return when {
            trace > 0.0 -> {
                val s = sqrt(trace + 1.0) * 2.0
                Quat(0.25*s, (m21-m12)/s, (m02-m20)/s, (m10-m01)/s).normalized()
            }
            m00 > m11 && m00 > m22 -> {
                val s = sqrt(1.0+m00-m11-m22)*2.0
                Quat((m21-m12)/s, 0.25*s, (m01+m10)/s, (m02-m20)/s).normalized()
            }
            m11 > m22 -> {
                val s = sqrt(1.0+m11-m00-m22)*2.0
                Quat((m02-m20)/s, (m01+m10)/s, 0.25*s, (m12+m21)/s).normalized()
            }
            else -> {
                val s = sqrt(1.0+m22-m00-m11)*2.0
                Quat((m10-m01)/s, (m02+m20)/s, (m12+m21)/s, 0.25*s).normalized()
            }
        }
    }

    fun transform(pose: Pose, point: Vec3): Vec3 = pose.rotation.rotate(point) + pose.position

    fun inverse(pose: Pose): Pose {
        val r = pose.rotation.conjugate().normalized()
        return Pose(position = r.rotate(pose.position * -1.0), rotation = r)
    }
}

package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs

/**
 * Assigns semantic rig roles from calibrated optical-axis geometry.
 *
 * The assignment is rig-relative. The master establishes north/up/east; every camera is then
 * classified from its optical forward vector. Ambiguous cameras are left AUXILIARY rather than
 * being silently mislabeled.
 */
class DeviceRoleAssignmentEngine(
    private val verticalThreshold: Double = 0.72,
    private val minimumDirectionalScore: Double = 0.62,
) {
    data class Candidate(
        val id: DeviceId,
        val forward: Vec3,
        val up: Vec3,
    )

    data class Assignment(
        val id: DeviceId,
        val role: DeviceRole,
        val score: Double,
        val directionDegrees: Double?,
    )

    fun assign(
        candidates: List<Candidate>,
        north: Vec3,
        east: Vec3,
        up: Vec3,
    ): List<Assignment> {
        val n = north.normalized()
        val e = east.normalized()
        val u = up.normalized()

        return candidates.map { camera ->
            val f = camera.forward.normalized()
            val vertical = f.dot(u)

            if (abs(vertical) >= verticalThreshold) {
                val role = if (vertical < 0.0) DeviceRole.TOP else DeviceRole.BOTTOM
                return@map Assignment(
                    id = camera.id,
                    role = role,
                    score = abs(vertical),
                    directionDegrees = null,
                )
            }

            val horizontal = (f - u * vertical).normalized()
            val northScore = horizontal.dot(n)
            val eastScore = horizontal.dot(e)

            val candidatesByDirection = listOf(
                DeviceRole.FRONT to -northScore,
                DeviceRole.REAR to northScore,
                DeviceRole.RIGHT to -eastScore,
                DeviceRole.LEFT to eastScore,
            )

            val best = candidatesByDirection.maxByOrNull { it.second }
                ?: return@map Assignment(camera.id, DeviceRole.AUXILIARY, 0.0, null)

            val score = best.second.coerceIn(0.0, 1.0)
            if (score < minimumDirectionalScore) {
                Assignment(camera.id, DeviceRole.AUXILIARY, score, null)
            } else {
                val direction = Math.toDegrees(
                    kotlin.math.atan2(eastScore, northScore),
                )
                Assignment(
                    id = camera.id,
                    role = best.first,
                    score = score,
                    directionDegrees = direction,
                )
            }
        }
    }
}

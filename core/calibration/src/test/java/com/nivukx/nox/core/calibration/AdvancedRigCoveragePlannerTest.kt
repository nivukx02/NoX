package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.DeviceRole
import org.junit.Assert.assertTrue
import org.junit.Test

class AdvancedRigCoveragePlannerTest {

    private fun assignment(
        role: DeviceRole,
        score: Double = 0.90,
    ) =
        DeviceRoleAssignmentEngine.Assignment(
            id = DeviceId(role.name.lowercase()),
            role = role,
            score = score,
            directionDegrees = null,
        )

    @Test
    fun frontalLateralRigImprovesUpperBodyCoverage() {
        val report =
            AdvancedRigCoveragePlanner().evaluate(
                listOf(
                    assignment(DeviceRole.FRONT),
                    assignment(DeviceRole.LEFT),
                    assignment(DeviceRole.RIGHT),
                    assignment(DeviceRole.REAR),
                ),
            )

        assertTrue(report.globalScore > 0.55)
        assertTrue(report.upperBodyScore > 0.55)
        assertTrue(report.handScore > 0.40)
    }

    @Test
    fun topAndBottomViewsImproveSurfaceAndFeet() {
        val planner = AdvancedRigCoveragePlanner()
        val base =
            planner.evaluate(
                listOf(
                    assignment(DeviceRole.FRONT),
                    assignment(DeviceRole.REAR),
                    assignment(DeviceRole.LEFT),
                    assignment(DeviceRole.RIGHT),
                ),
            )
        val expanded =
            planner.evaluate(
                listOf(
                    assignment(DeviceRole.FRONT),
                    assignment(DeviceRole.REAR),
                    assignment(DeviceRole.LEFT),
                    assignment(DeviceRole.RIGHT),
                    assignment(DeviceRole.TOP),
                    assignment(DeviceRole.BOTTOM),
                ),
            )

        assertTrue(expanded.surfaceScore >= base.surfaceScore)
        assertTrue(expanded.footScore >= base.footScore)
    }
}

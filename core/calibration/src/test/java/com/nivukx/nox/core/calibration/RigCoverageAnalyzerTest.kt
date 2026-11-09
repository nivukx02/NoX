package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.DeviceRole
import org.junit.Assert.assertTrue
import org.junit.Test

class RigCoverageAnalyzerTest {
    @Test
    fun recommendsMissingVerticalCoverageForFullVolume() {
        val assignments = listOf(
            DeviceRoleAssignmentEngine.Assignment(
                id = DeviceId("master"),
                role = DeviceRole.MASTER,
                score = 1.0,
                directionDegrees = null,
            ),
            DeviceRoleAssignmentEngine.Assignment(
                id = DeviceId("front"),
                role = DeviceRole.FRONT,
                score = 0.95,
                directionDegrees = 0.0,
            ),
            DeviceRoleAssignmentEngine.Assignment(
                id = DeviceId("left"),
                role = DeviceRole.LEFT,
                score = 0.90,
                directionDegrees = -90.0,
            ),
        )

        val report = RigCoverageAnalyzer().evaluate(
            assignments,
            RigCoverageAnalyzer.CaptureProfile.FULL_VOLUME,
        )

        assertTrue(report.score >= 0.0)
        assertTrue(
            report.recommendations.any {
                it.contains("top", ignoreCase = true) ||
                    it.contains("low-angle", ignoreCase = true)
            },
        )
    }
}

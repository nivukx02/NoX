package com.nivukx.nox.core.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Contract tests that do not emulate a vendor Camera2 HAL.
 */
class CameraSessionNegotiationEngineTest {

    @Test
    fun sessionPlanAdmissibilityFollowsSupportFlag() {
        val accepted = CameraSessionNegotiationEngine.SessionPlan(
            cameraId = "0",
            outputs = emptyList(),
            verification =
                CameraSessionNegotiationEngine.Verification.HARDWARE_VERIFIED,
            supported = true,
            reason = "verified",
            readoutTimestampRequested = true,
        )
        val rejected = accepted.copy(
            verification =
                CameraSessionNegotiationEngine.Verification.REJECTED,
            supported = false,
        )

        assertTrue(accepted.admissible)
        assertTrue(!rejected.admissible)
    }

    @Test
    fun verificationStatesAreExplicitAndStable() {
        assertEquals(
            listOf(
                CameraSessionNegotiationEngine.Verification.HARDWARE_VERIFIED,
                CameraSessionNegotiationEngine.Verification.PLATFORM_DECLARED_ONLY,
                CameraSessionNegotiationEngine.Verification.UNKNOWN,
                CameraSessionNegotiationEngine.Verification.REJECTED,
            ),
            CameraSessionNegotiationEngine.Verification.entries,
        )
    }

    @Test
    fun high_speed_range_selection_requires_constrained_session() {
        val ranges =
            listOf(
                Range(30, 60),
                Range(60, 120),
                Range(120, 240),
            )

        val plan =
            engine().selectHighSpeedPlan(
                cameraId = "0",
                requestedFps = 118,
                availableRanges = ranges,
                outputCount = 2,
            )

        assertTrue(plan.supported)
        assertTrue(plan.requiresConstrainedSession)
        assertEquals(Range(60, 120), plan.selectedRange)
        assertTrue(plan.admissible)
    }

    @Test
    fun high_speed_range_selection_rejects_output_topology_above_two() {
        val plan =
            engine().selectHighSpeedPlan(
                cameraId = "0",
                requestedFps = 120,
                availableRanges = listOf(Range(60, 120)),
                outputCount = 3,
            )

        assertTrue(!plan.supported)
        assertEquals(
            CameraSessionNegotiationEngine.Verification.REJECTED,
            plan.verification,
        )
    }

    @Test
    fun concurrentPlanRequiresPositiveSupportForAdmission() {
        val plan = CameraSessionNegotiationEngine.ConcurrentPlan(
            cameraIds = setOf("0", "1"),
            verification =
                CameraSessionNegotiationEngine.Verification.REJECTED,
            supported = false,
            reasons = listOf("platform rejected"),
        )

        assertTrue(!plan.admissible)
        assertEquals(listOf("platform rejected"), plan.reasons)
    }

    @Test
    fun mixedUnsupportedReadoutRequestFailsClosed() {
        val surfaceA = mockSurface()
        val surfaceB = mockSurface()

        assertFailsWith<IllegalArgumentException> {
            engine().buildRegularSessionConfiguration(
                listOf(
                    CameraSessionNegotiationEngine.OutputSpec(
                        surface = surfaceA,
                        name = "rgb",
                        enableReadoutTimestamp = true,
                        hardwareReadoutSupported = true,
                    ),
                    CameraSessionNegotiationEngine.OutputSpec(
                        surface = surfaceB,
                        name = "depth",
                        enableReadoutTimestamp = true,
                        hardwareReadoutSupported = false,
                    ),
                ),
            )
        }
    }

    @Test
    fun conflictingExplicitTimestampBasesFailClosed() {
        assertFailsWith<IllegalArgumentException> {
            engine().buildRegularSessionConfiguration(
                listOf(
                    CameraSessionNegotiationEngine.OutputSpec(
                        surface = mockSurface(),
                        name = "rgb",
                        timestampBase =
                            OutputConfiguration.TIMESTAMP_BASE_SENSOR,
                    ),
                    CameraSessionNegotiationEngine.OutputSpec(
                        surface = mockSurface(),
                        name = "aux",
                        timestampBase =
                            OutputConfiguration.TIMESTAMP_BASE_REALTIME,
                    ),
                ),
            )
        }
    }

}

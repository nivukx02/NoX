package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.HighEndMotionCaptureSettings
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FrontierHighEndConfigurationAdapterTest {

    @Test
    fun mapsMaximumFederationAndRobustWorldSpacePolicy() {
        val settings =
            HighEndMotionCaptureSettings()
                .withFrontierWorldSpaceRobustness(
                    iterations = 7,
                    huberResidualMillimeters = 5.0,
                    maximumConditionEstimate = 5.0e8,
                )
                .withFrontierCameraFederationPolicy(
                    authorityBudget = 32,
                    witnessBudget = 96,
                    globalEnvelope = 128,
                )

        val bundle =
            FrontierHighEndConfigurationAdapter.from(settings)

        assertEquals(7, bundle.worldSpace.robustIterations)
        assertEquals(
            0.005,
            bundle.worldSpace.huberResidualMeters,
            1e-12,
        )
        assertEquals(
            5.0e8,
            bundle.worldSpace.maximumConditionEstimate,
            1.0,
        )
        assertEquals(128, bundle.federation.maximumCameras)
        assertEquals(32, bundle.federation.authorityCameras)
        assertEquals(96, bundle.federation.witnessCameras)
    }

    @Test
    fun mapsSurfaceCapacityAndCalibrationAuthority() {
        val settings =
            HighEndMotionCaptureSettings()
                .withFrontierSurfaceCapacity(
                    enabled = true,
                    patchCapacity = 700_000,
                    historyFrames = 24,
                )
                .withFrontierCalibrationPolicy(
                    graphEnabled = true,
                    maximumCycleResidualMillimeters = 6.0,
                )

        val bundle =
            FrontierHighEndConfigurationAdapter.from(settings)

        assertTrue(bundle.surface.maximumPatches == 700_000)
        assertEquals(24, bundle.surface.maximumHistoryFrames)
        assertTrue(
            bundle.calibration.maximumCycleResidualMeters <=
                0.006 + 1e-12,
        )
        assertTrue(bundle.calibration.maximumCameras >= 2)
    }

    @Test
    fun mapsPrecisionParallaxContractWithoutClaimingPhysicalAccuracy() {
        val settings =
            HighEndMotionCaptureSettings()
                .withFrontierPrecisionQualification(
                    requiresIndependentReference = true,
                    minimumViews = 6,
                    minimumParallaxDegrees = 16.0,
                )

        val bundle =
            FrontierHighEndConfigurationAdapter.from(settings)

        assertEquals(
            16.0,
            bundle.calibration.minimumParallaxDegrees,
            0.0,
        )
        assertTrue(
            bundle.calibration.preferredParallaxDegrees >=
                16.0,
        )
    }
}

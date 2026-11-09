package com.nivukx.nox.core.nativeengine.calibration

import org.junit.Assert.assertEquals
import org.junit.Test

class NativeCalibrationConfigTest {
    @Test
    fun defaultsAreDeterministic() {
        val config = NativeCalibrationConfig()
        assertEquals(2.0, config.huberDeltaPx, 0.0)
        assertEquals(16, config.maxIterations)
        assertEquals(12, config.minimumObservationsPerCamera)
        assertEquals(true, config.refineIntrinsics)
        assertEquals(true, config.refineDistortion)
        assertEquals(12, config.refinementIterations)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsWeakObservationRequirement() {
        NativeCalibrationConfig(minimumObservationsPerCamera = 5)
    }
    @Test
    fun refinementPriorsStayBounded() {
        val config =
            NativeCalibrationConfig(
                focalPriorSigmaRatio = 0.08,
                principalPointPriorSigmaRatio = 0.04,
                distortionPriorSigma = 0.03,
                refinementDamping = 1e-2,
                refinementIterations = 20,
            )

        assertEquals(0.08, config.focalPriorSigmaRatio, 0.0)
        assertEquals(0.04, config.principalPointPriorSigmaRatio, 0.0)
        assertEquals(0.03, config.distortionPriorSigma, 0.0)
        assertEquals(1e-2, config.refinementDamping, 0.0)
        assertEquals(20, config.refinementIterations)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsOverlyAggressiveFocalPrior() {
        NativeCalibrationConfig(focalPriorSigmaRatio = 0.80)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsOverlyAggressiveRefinementIterations() {
        NativeCalibrationConfig(refinementIterations = 65)
    }

}

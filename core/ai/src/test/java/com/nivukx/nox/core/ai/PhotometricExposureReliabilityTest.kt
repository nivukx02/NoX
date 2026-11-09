package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PhotometricExposureReliabilityTest {
    @Test
    fun small_exposure_changes_preserve_photometric_authority() {
        val engine = SparsePhotometricMotionFieldEngine()
        assertEquals(
            1.0,
            engine.exposureReliability(0.08),
            absoluteTolerance = 1e-9,
        )
    }

    @Test
    fun large_exposure_changes_reduce_photometric_authority_but_remain_finite() {
        val engine = SparsePhotometricMotionFieldEngine()
        val moderate = engine.exposureReliability(0.24)
        val severe = engine.exposureReliability(0.80)

        assertTrue(moderate < 1.0)
        assertTrue(severe <= moderate)
        assertTrue(severe >= 0.35)
        assertTrue(severe.isFinite())
    }

    @Test
    fun negative_and_positive_exposure_shifts_are_symmetric() {
        val engine = SparsePhotometricMotionFieldEngine()
        assertEquals(
            engine.exposureReliability(0.31),
            engine.exposureReliability(-0.31),
            absoluteTolerance = 1e-9,
        )
    }
}

package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class TriangulationPrecisionTuningTest {

    @Test
    fun precisionTuningAcceptsStrictTailEnvelope() {
        val engine = TriangulationEngine()

        val tuning =
            TriangulationEngine.PrecisionTuning(
                tailResidualQuantile = 0.99,
                residualInflation = 1.30,
                minimumPixelNoisePx = 0.15,
            )

        engine.configurePrecisionTuning(tuning)

        assertEquals(tuning, engine.precisionTuning())
    }

    @Test
    fun precisionTuningRejectsImpossibleTailQuantile() {
        assertFailsWith<IllegalArgumentException> {
            TriangulationEngine.PrecisionTuning(
                tailResidualQuantile = 1.0,
            )
        }
    }

    @Test
    fun precisionTuningRejectsSubPixelNoiseBelowSensorFloor() {
        assertFailsWith<IllegalArgumentException> {
            TriangulationEngine.PrecisionTuning(
                minimumPixelNoisePx = 0.01,
            )
        }
    }
}

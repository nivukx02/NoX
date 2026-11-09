package com.nivukx.nox

import kotlin.test.Test
import kotlin.test.assertEquals

class NativeDepthConfidencePolicyTest {
    @Test
    fun capsAdaptiveAmplificationAtNativeMaximumConfidence() {
        assertEquals(
            1.0,
            NativeDepthConfidencePolicy.resolve(
                frameConfidence = 0.8,
                adaptiveScale = 1.5,
            ),
            0.0,
        )
    }

    @Test
    fun preservesFiniteDownweightingBelowOne() {
        assertEquals(
            0.3,
            NativeDepthConfidencePolicy.resolve(
                frameConfidence = 0.6,
                adaptiveScale = 0.5,
            ),
            1e-12,
        )
    }

    @Test
    fun nonFiniteInputsNeverEscapeIntoNativeApi() {
        assertEquals(
            0.0,
            NativeDepthConfidencePolicy.resolve(
                frameConfidence = Double.NaN,
                adaptiveScale = 1.0,
            ),
            0.0,
        )
        assertEquals(
            0.0,
            NativeDepthConfidencePolicy.resolve(
                frameConfidence = 0.8,
                adaptiveScale = Double.POSITIVE_INFINITY,
            ),
            0.0,
        )
        assertEquals(
            0.0,
            NativeDepthConfidencePolicy.resolve(
                frameConfidence = 0.8,
                adaptiveScale = Double.NaN,
            ),
            0.0,
        )
    }
}

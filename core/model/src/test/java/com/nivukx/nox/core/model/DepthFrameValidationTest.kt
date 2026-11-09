package com.nivukx.nox.core.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class DepthFrameValidationTest {
    @Test
    fun rejectsDimensionProductsThatOverflowIntBeforeComparingArrays() {
        assertFailsWith<IllegalArgumentException> {
            DepthFrame(
                timestampNanos = 1L,
                source = DepthSource.DEPTH16,
                width = 65_536,
                height = 65_536,
                depthMeters = FloatArray(0),
            )
        }
    }

    @Test
    fun descriptorKeepsConfidenceFiniteAndClampedForMalformedSamples() {
        val frame =
            DepthFrame(
                timestampNanos = 2L,
                source = DepthSource.DEPTH16,
                width = 3,
                height = 1,
                depthMeters = floatArrayOf(1.0f, 2.0f, 3.0f),
                confidence = floatArrayOf(Float.NaN, 2.0f, -1.0f),
            )

        val descriptor = frame.descriptor()
        assertTrue(descriptor.meanConfidence.isFinite())
        assertEquals(1.0 / 3.0, descriptor.meanConfidence, 1e-12)
    }

    @Test
    fun nullConfidenceMapPreservesDefaultFullConfidence() {
        val frame =
            DepthFrame(
                timestampNanos = 3L,
                source = DepthSource.DEPTH16,
                width = 2,
                height = 1,
                depthMeters = floatArrayOf(1.0f, 2.0f),
            )

        assertEquals(1.0, frame.descriptor().meanConfidence, 0.0)
    }
}

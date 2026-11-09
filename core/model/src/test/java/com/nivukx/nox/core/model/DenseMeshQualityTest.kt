package com.nivukx.nox.core.model

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DenseMeshQualityTest {
    @Test
    fun qualityTracksEvidenceAvailabilitySeparatelyFromNumericFields() {
        val quality = DenseMeshQuality(
            timestampNanos = 1L,
            vertexCount = 4,
            triangleCount = 2,
            meanReprojectionErrorPx = 0.0,
            watertightnessScore = 0.5,
            temporalStability = 0.0,
            geometricConfidence = 0.7,
        )
        assertFalse(quality.reprojectionEvidenceAvailable)
        assertFalse(quality.watertightnessEvidenceAvailable)
        assertFalse(quality.temporalEvidenceAvailable)

        val measured = quality.copy(
            watertightnessEvidenceAvailable = true,
        )
        assertTrue(measured.watertightnessEvidenceAvailable)
    }
}

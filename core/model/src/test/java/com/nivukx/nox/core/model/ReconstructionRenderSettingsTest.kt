package com.nivukx.nox.core.model

import org.junit.Assert.assertEquals
import org.junit.Test

class ReconstructionRenderSettingsTest {
    @Test
    fun defaultsFavorDenseReconstructionVisibility() {
        val settings = ReconstructionRenderSettings()
        assertEquals(ReconstructionVisualizationMode.DENSE_SURFACE, settings.mode)
        assertEquals(true, settings.showUncertainty)
    }
}

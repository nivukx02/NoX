package com.nivukx.nox.ui.studio

import com.nivukx.nox.core.model.ReconstructionRenderSettings
import com.nivukx.nox.core.model.ReconstructionVisualizationMode
import org.junit.Assert.assertEquals
import org.junit.Test

class StudioOperatorSurfaceTest {
    @Test
    fun reconstructionInspectorStateRemainsPureData() {
        val base = ReconstructionRenderSettings()
        val updated = base.copy(mode = ReconstructionVisualizationMode.UNCERTAINTY, meshOpacity = 0.8)
        assertEquals(ReconstructionVisualizationMode.UNCERTAINTY, updated.mode)
        assertEquals(0.8, updated.meshOpacity, 0.0)
        assertEquals(base.showCameras, updated.showCameras)
    }
}
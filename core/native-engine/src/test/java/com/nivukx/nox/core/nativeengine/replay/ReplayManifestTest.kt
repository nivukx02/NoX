package com.nivukx.nox.core.nativeengine.replay

import org.junit.Assert.assertEquals
import org.junit.Test

class ReplayManifestTest {
    @Test
    fun preservesReplayFixtureFields() {
        val manifest=ReplayManifest(2,240,12.0,.8,.1,3,0,42)
        assertEquals(2,manifest.cameraCount)
        assertEquals(240L,manifest.frameCount)
        assertEquals(42,manifest.seed)
    }
}
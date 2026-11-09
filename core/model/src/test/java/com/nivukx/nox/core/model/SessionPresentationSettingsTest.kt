package com.nivukx.nox.core.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SessionPresentationSettingsTest {

    @Test
    fun sanitizesClipNameForFilesystemSafeMetadata() {
        val settings = SessionPresentationSettings(
            clipName = "  Hero / Take 01!?  ",
            countdownSeconds = 4,
        )

        assertEquals("Hero _ Take 01__", settings.sanitizedClipName())
        assertTrue(settings.countdownSeconds == 4)
    }

    @Test
    fun defaultSettingsRemainWithinProductionBounds() {
        val settings = SessionPresentationSettings()

        assertTrue(settings.clipName.isNotBlank())
        assertTrue(settings.countdownSeconds in 0..10)
    }
}

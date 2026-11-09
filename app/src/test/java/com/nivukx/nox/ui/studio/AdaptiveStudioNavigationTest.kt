package com.nivukx.nox.ui.studio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptiveStudioNavigationTest {
    @Test
    fun sectionCountRemainsStableForOperatorDeck() {
        assertEquals(12, StudioSection.entries.size)
        assertTrue(StudioSection.ADVANCED_SETTINGS in StudioSection.entries)
    }

    @Test
    fun everyDestinationExposesOperatorMetadata() {
        MotionStudioDestination.entries.forEach { destination ->
            assertTrue(destination.label.isNotBlank())
            assertTrue(destination.contentDescription.isNotBlank())
            assertTrue(destination.group.name.isNotBlank())
            assertTrue(destination.shortLabel.isNotBlank())
        }
    }

    @Test
    fun reconstructionAndCaptureRemainPrimaryOperatorDestinations() {
        assertEquals(StudioDestinationGroup.CAPTURE, MotionStudioDestination.CAPTURE.group)
        assertEquals(StudioDestinationGroup.SOLVE, MotionStudioDestination.RECONSTRUCTION.group)
        assertEquals("RECON", MotionStudioDestination.RECONSTRUCTION.shortLabel)
        assertEquals("CAP", MotionStudioDestination.CAPTURE.shortLabel)
    }
}

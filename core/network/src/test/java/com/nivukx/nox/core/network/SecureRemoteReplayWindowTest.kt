package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SecureRemoteReplayWindowTest {

    @Test
    fun firstSequenceIsAcceptedThenDuplicateIsRejected() {
        val window = SecureRemoteReplayWindow()

        assertTrue(
            window.accept(
                source = "cam-a",
                sequence = 100L,
            ),
        )

        assertFalse(
            window.accept(
                source = "cam-a",
                sequence = 100L,
            ),
        )
    }

    @Test
    fun newerWindowAcceptsForwardProgressAndOldPacketIsRejected() {
        val window =
            SecureRemoteReplayWindow(
                SecureRemoteReplayWindow.Config(
                    windowSize = 32,
                ),
            )

        assertTrue(window.accept("cam-b", 100L))
        assertTrue(window.accept("cam-b", 120L))
        assertFalse(window.accept("cam-b", 80L))
    }

    @Test
    fun differentSourcesHaveIndependentWindows() {
        val window = SecureRemoteReplayWindow()

        assertTrue(window.accept("cam-a", 1L))
        assertTrue(window.accept("cam-b", 1L))
        assertFalse(window.accept("cam-a", 1L))
        assertFalse(window.accept("cam-b", 1L))
    }
}

package com.nivukx.nox.ui.studio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class StudioNavigationModelTest {
    @Test
    fun selectionChangesWithoutMutatingOtherState() {
        val initial = StudioNavigationState(
            selected = MotionStudioDestination.RECONSTRUCTION,
            inspectorVisible = true,
        )
        val next = StudioNavigationReducer.select(
            initial,
            MotionStudioDestination.DIAGNOSTICS,
        )
        assertEquals(MotionStudioDestination.DIAGNOSTICS, next.selected)
        assertTrue(next.inspectorVisible)
    }

    @Test
    fun inspectorToggleIsDeterministic() {
        val initial = StudioNavigationState()
        val hidden = StudioNavigationReducer.toggleInspector(initial)
        assertFalse(hidden.inspectorVisible)
        assertTrue(
            StudioNavigationReducer.toggleInspector(hidden).inspectorVisible,
        )
    }
}
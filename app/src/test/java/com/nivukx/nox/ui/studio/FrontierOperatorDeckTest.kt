package com.nivukx.nox.ui.studio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierOperatorDeckTest {
    @Test fun keepsProductionCapacityVisible() {
        val state = FrontierOperatorState(FrontierOperationalState.LIVE,50,32,96,0.94,0.91,0.97,0.66,0.72,0.82,0.78,0.22)
        assertEquals(50,state.actorCount)
        assertTrue(state.authoritativeCameraCount <= 32)
        assertTrue(state.witnessCameraCount <= 96)
    }
    @Test(expected = IllegalArgumentException::class) fun rejectsInvalidConfidence() {
        FrontierOperatorState(FrontierOperationalState.DEGRADED,1,2,3,1.5,0.5,0.5,0.5,0.5,2.0,0.5,0.5)
    }
}
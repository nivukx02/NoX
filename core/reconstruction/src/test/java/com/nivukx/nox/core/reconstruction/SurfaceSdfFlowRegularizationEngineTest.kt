package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class SurfaceSdfFlowRegularizationEngineTest {
    @Test
    fun consistentNormalFlowRemainsNearOriginal() {
        val result = SurfaceSdfFlowRegularizationEngine().regularize(
            listOf(
                SurfaceSdfFlowRegularizationEngine.Constraint(
                    id=1L,
                    positionMeters=Vec3(0.0,0.0,0.0),
                    normal=Vec3(0.0,0.0,1.0),
                    flowDeltaMeters=Vec3(0.01,0.0,0.02),
                    signedDistanceDeltaMeters=0.02,
                    uncertaintyMeters=0.003,
                    confidence=0.95,
                )
            )
        ).single()
        assertTrue(result.normalResidualMeters < 0.005)
        assertTrue(result.regularizedFlowMeters.z > 0.019)
    }

    @Test
    fun contradictorySdfFlowIsDownweighted() {
        val result = SurfaceSdfFlowRegularizationEngine().regularize(
            listOf(
                SurfaceSdfFlowRegularizationEngine.Constraint(
                    id=2L,
                    positionMeters=Vec3(0.0,0.0,0.0),
                    normal=Vec3(0.0,0.0,1.0),
                    flowDeltaMeters=Vec3(0.0,0.0,0.10),
                    signedDistanceDeltaMeters=-0.10,
                    uncertaintyMeters=0.002,
                    confidence=0.50,
                )
            )
        ).single()
        assertTrue(result.constraintWeight < 0.20)
        assertTrue(result.warning != null)
    }
}

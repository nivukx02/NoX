package com.nivukx.nox.core.reconstruction

import kotlin.test.assertTrue
import org.junit.Test

class DepthUncertaintyModelEngineTest {

    @Test
    fun uncertaintyGrowsWithRange() {
        val engine = DepthUncertaintyModelEngine()
        val near = engine.estimate(
            DepthUncertaintyModelEngine.Input(
                depthMeters = 1.0,
                neighborDepthX = 1.0,
                neighborDepthY = 1.0,
                rayZ = 1.0,
                surfaceNormalObservability = 0.95,
                sourceClass =
                    DepthUncertaintyModelEngine.SourceClass.DEPTH_SENSOR,
            ),
        )
        val far = engine.estimate(
            DepthUncertaintyModelEngine.Input(
                depthMeters = 5.0,
                neighborDepthX = 5.0,
                neighborDepthY = 5.0,
                rayZ = 1.0,
                surfaceNormalObservability = 0.95,
                sourceClass =
                    DepthUncertaintyModelEngine.SourceClass.DEPTH_SENSOR,
            ),
        )

        assertTrue(far.sigmaMeters > near.sigmaMeters)
    }

    @Test
    fun discontinuityAndWeakTemporalEvidenceIncreaseUncertainty() {
        val engine = DepthUncertaintyModelEngine()
        val clean = engine.estimate(
            DepthUncertaintyModelEngine.Input(
                depthMeters = 2.0,
                neighborDepthX = 2.0,
                neighborDepthY = 2.0,
                rayZ = 1.0,
                surfaceNormalObservability = 0.95,
                temporalConsistency = 0.95,
                sourceConfidence = 0.95,
                sourceClass =
                    DepthUncertaintyModelEngine.SourceClass.DEPTH_SENSOR,
            ),
        )
        val degraded = engine.estimate(
            DepthUncertaintyModelEngine.Input(
                depthMeters = 2.0,
                neighborDepthX = 2.12,
                neighborDepthY = 1.88,
                rayZ = 0.55,
                surfaceNormalObservability = 0.35,
                temporalConsistency = 0.35,
                sourceConfidence = 0.45,
                sourceClass =
                    DepthUncertaintyModelEngine.SourceClass.DEPTH_SENSOR,
            ),
        )

        assertTrue(degraded.sigmaMeters > clean.sigmaMeters)
        assertTrue(degraded.confidence < clean.confidence)
    }

    @Test
    fun monocularPriorIsNeverTreatedAsMetricSensorQuality() {
        val engine = DepthUncertaintyModelEngine()
        val sensor = engine.estimate(
            DepthUncertaintyModelEngine.Input(
                depthMeters = 2.0,
                neighborDepthX = 2.0,
                neighborDepthY = 2.0,
                rayZ = 1.0,
                surfaceNormalObservability = 0.90,
                sourceClass =
                    DepthUncertaintyModelEngine.SourceClass.DEPTH_SENSOR,
            ),
        )
        val mono = engine.estimate(
            DepthUncertaintyModelEngine.Input(
                depthMeters = 2.0,
                neighborDepthX = 2.0,
                neighborDepthY = 2.0,
                rayZ = 1.0,
                surfaceNormalObservability = 0.90,
                sourceClass =
                    DepthUncertaintyModelEngine.SourceClass.MONOCULAR_PRIOR,
            ),
        )

        assertTrue(mono.sigmaMeters > sensor.sigmaMeters)
    }
}

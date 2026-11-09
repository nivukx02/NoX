package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TemporalReconstructionTuningTest {

    @Test
    fun presetsStayInsideRuntimeBounds() {
        listOf(
            TemporalReconstructionTuning.cinematic(),
            TemporalReconstructionTuning.liveAction(),
            TemporalReconstructionTuning.precision(),
        ).forEach { tuning ->
            assertTrue(tuning.predictionHorizonSeconds in 0.02..0.60)
            assertTrue(tuning.innovationGateScale in 0.65..1.80)
            assertTrue(tuning.uncertaintyGateScale in 0.50..1.80)
            assertTrue(tuning.reprojectionGateScale in 0.50..1.80)
            assertTrue(tuning.smoothingResponsiveness in 0.08..0.92)
            assertTrue(tuning.confidenceDecayScale in 0.50..2.00)
            assertTrue(tuning.outlierRetention in 0.0..0.35)
            assertTrue(tuning.topologyToleranceScale in 0.50..1.80)
        }
    }

    @Test
    fun stabilizerAcceptsRuntimePolicy() {
        val stabilizer = TemporalReconstructionStabilizer()
        val policy = TemporalReconstructionTuning.liveAction()

        stabilizer.configure(policy)

        assertEquals(
            policy,
            stabilizer.tuning(),
        )
    }
}

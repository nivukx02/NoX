package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ProductionCaptureProfile
import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptiveReconstructionDetailEngineTest {
    private val engine = AdaptiveReconstructionDetailEngine()

    @Test
    fun lowUncertaintyWithHealthyRigReachesReferenceTier() {
        val budget = engine.evaluate(
            AdaptiveReconstructionDetailEngine.Input(
                profile = ProductionCaptureProfile.NANO_METROLOGY_REFERENCE,
                p95UncertaintyMeters = 0.0006,
                geometricObservability = 0.97,
                thermalHeadroom = 0.92,
                memoryHeadroom = 0.90,
                gpuLatencyMs = 7.0,
                dynamicStress = 0.18,
                rigConfidence = 0.96,
                viewDiversity = 0.95,
                captureTimingConfidence = 0.98,
            ),
        )

        assertTrue(budget.tier.name == "REFERENCE")
        assertTrue(budget.preserveMetricEvidence)
        assertTrue(budget.effectiveVoxelSizeMeters >= 0.001)
        assertTrue(budget.bundleAdjustmentIterations > 1)
    }

    @Test
    fun thermalPressureReducesDetailWithoutDiscardingEvidence() {
        val budget = engine.evaluate(
            AdaptiveReconstructionDetailEngine.Input(
                profile = ProductionCaptureProfile.CINEMATIC_ULTRA,
                p95UncertaintyMeters = 0.004,
                geometricObservability = 0.84,
                thermalHeadroom = 0.38,
                memoryHeadroom = 0.72,
                gpuLatencyMs = 15.0,
                dynamicStress = 0.55,
                rigConfidence = 0.82,
                viewDiversity = 0.80,
                captureTimingConfidence = 0.90,
            ),
        )

        assertTrue(budget.meshVertexBudget < ProductionCaptureProfile.CINEMATIC_ULTRA.meshVertexBudget)
        assertTrue(budget.denseMeshHz < ProductionCaptureProfile.CINEMATIC_ULTRA.denseMeshHz)
        assertTrue(budget.holdOutput.not())
    }

    @Test
    fun unsafeRuntimeHoldsMetricOutput() {
        val budget = engine.evaluate(
            AdaptiveReconstructionDetailEngine.Input(
                profile = ProductionCaptureProfile.REFERENCE_BAKE,
                p95UncertaintyMeters = 0.30,
                geometricObservability = 0.28,
                thermalHeadroom = 0.12,
                memoryHeadroom = 0.15,
                gpuLatencyMs = 40.0,
                dynamicStress = 0.96,
                rigConfidence = 0.25,
                viewDiversity = 0.30,
                captureTimingConfidence = 0.22,
            ),
        )

        assertTrue(budget.holdOutput)
        assertTrue(budget.warnings.contains("RECONSTRUCTION_DETAIL_HELD_FOR_EVIDENCE_SAFETY"))
    }
    @Test
    fun spatialDetailFieldPrioritizesHighValueObservedGeometry() {
        val profile = ProductionCaptureProfile.default
        val engine = AdaptiveReconstructionDetailEngine()
        val globalInput = AdaptiveReconstructionDetailEngine.Input(
            profile = profile,
            p95UncertaintyMeters = 0.001,
            geometricObservability = 0.95,
            thermalHeadroom = 0.95,
            memoryHeadroom = 0.95,
            gpuLatencyMs = 8.0,
            dynamicStress = 0.20,
            rigConfidence = 0.95,
            viewDiversity = 0.90,
            captureTimingConfidence = 0.95,
        )
        val budgets = engine.evaluateRegions(
            globalInput,
            listOf(
                AdaptiveReconstructionDetailEngine.RegionInput(
                    regionId = 1L,
                    p95UncertaintyMeters = 0.001,
                    geometricObservability = 0.95,
                    dynamicStress = 0.20,
                    viewDiversity = 0.95,
                    semanticImportance = 1.0,
                    curvatureScore = 0.95,
                    actorPresent = true,
                ),
                AdaptiveReconstructionDetailEngine.RegionInput(
                    regionId = 2L,
                    p95UncertaintyMeters = 0.040,
                    geometricObservability = 0.30,
                    dynamicStress = 0.90,
                    viewDiversity = 0.25,
                    semanticImportance = 0.10,
                    curvatureScore = 0.10,
                    actorPresent = false,
                ),
            ),
        )
        val high = budgets.first { it.regionId == 1L }
        val low = budgets.first { it.regionId == 2L }
        assertTrue(high.detailWeight > low.detailWeight)
        assertTrue(high.relativeVertexAllocation > low.relativeVertexAllocation)
        assertTrue(high.tier.ordinal <= low.tier.ordinal)
    )
}
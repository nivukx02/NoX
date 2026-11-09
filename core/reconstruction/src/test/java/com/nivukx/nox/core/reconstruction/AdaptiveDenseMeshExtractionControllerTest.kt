package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptiveDenseMeshExtractionControllerTest {

    @Test
    fun firstEvaluation_alwaysRequestsAnExtraction() {
        val controller = AdaptiveDenseMeshExtractionController()

        val decision = controller.evaluate(
            AdaptiveDenseMeshExtractionController.Evidence(
                timestampNanos = 1_000_000L,
                motionEnergy = 0.10,
                uncertainty = 0.15,
                cameraCoverage = 0.90,
                reconstructionConfidence = 0.90,
                temporalStability = 0.92,
                thermalHeadroom = 0.90,
                gpuHeadroom = 0.90,
                requestedDetail = 0.90,
                voxelCount = 20_000,
                maxVoxels = 1_000_000,
            ),
        )

        assertTrue(decision.shouldExtract)
        assertTrue(decision.cellBudget > 0)
        assertTrue(decision.vertexBudget > 0)
    }

    @Test
    fun stableScene_defersFrequentExtraction_andReducesComputeDemand() {
        val controller = AdaptiveDenseMeshExtractionController()

        controller.evaluate(
            AdaptiveDenseMeshExtractionController.Evidence(
                timestampNanos = 1_000_000L,
                motionEnergy = 0.08,
                uncertainty = 0.10,
                cameraCoverage = 0.96,
                reconstructionConfidence = 0.94,
                temporalStability = 0.96,
                thermalHeadroom = 0.95,
                gpuHeadroom = 0.92,
                requestedDetail = 0.60,
                voxelCount = 60_000,
                maxVoxels = 1_000_000,
            ),
        )

        val decision = controller.evaluate(
            AdaptiveDenseMeshExtractionController.Evidence(
                timestampNanos = 41_000_000L,
                motionEnergy = 0.08,
                uncertainty = 0.10,
                cameraCoverage = 0.96,
                reconstructionConfidence = 0.94,
                temporalStability = 0.96,
                thermalHeadroom = 0.95,
                gpuHeadroom = 0.92,
                requestedDetail = 0.60,
                voxelCount = 60_000,
                maxVoxels = 1_000_000,
            ),
        )

        assertFalse(decision.shouldExtract)
        assertTrue(decision.minimumIntervalNanos >= 100_000_000L)
        assertTrue(decision.computePressure < 0.35)
    }

    @Test
    fun highMotionAndUncertainty_requestFastHighDetailExtraction() {
        val controller = AdaptiveDenseMeshExtractionController()

        controller.evaluate(
            AdaptiveDenseMeshExtractionController.Evidence(
                timestampNanos = 1_000_000L,
                motionEnergy = 0.10,
                uncertainty = 0.10,
                cameraCoverage = 0.90,
                reconstructionConfidence = 0.90,
                temporalStability = 0.90,
                thermalHeadroom = 0.90,
                gpuHeadroom = 0.90,
                requestedDetail = 0.80,
                voxelCount = 100_000,
                maxVoxels = 1_000_000,
            ),
        )

        val decision = controller.evaluate(
            AdaptiveDenseMeshExtractionController.Evidence(
                timestampNanos = 46_000_000L,
                motionEnergy = 0.95,
                uncertainty = 0.92,
                cameraCoverage = 0.82,
                reconstructionConfidence = 0.58,
                temporalStability = 0.48,
                thermalHeadroom = 0.86,
                gpuHeadroom = 0.88,
                requestedDetail = 0.98,
                voxelCount = 180_000,
                maxVoxels = 1_000_000,
            ),
        )

        assertTrue(decision.shouldExtract)
        assertTrue(decision.minimumIntervalNanos <= 60_000_000L)
        assertTrue(decision.vertexBudget > 250_000)
        assertTrue(decision.reason.contains("motion"))
    }

    @Test
    fun measuredExtractionLatency_becomesPartOfPressureFeedback() {
        val controller = AdaptiveDenseMeshExtractionController(
            AdaptiveDenseMeshExtractionController.Config(
                targetExtractionLatencyMillis = 12.0,
                maximumObservedLatencyMillis = 48.0,
            ),
        )

        controller.recordExtractionLatency(46.0)
        val decision = controller.evaluate(
            AdaptiveDenseMeshExtractionController.Evidence(
                timestampNanos = 1_000_000L,
                motionEnergy = 0.40,
                uncertainty = 0.35,
                cameraCoverage = 0.90,
                reconstructionConfidence = 0.80,
                temporalStability = 0.82,
                thermalHeadroom = 0.90,
                gpuHeadroom = 0.90,
                requestedDetail = 0.90,
                voxelCount = 100_000,
                maxVoxels = 1_000_000,
            ),
        )

        assertTrue(decision.observedExtractionLatencyMillis >= 40.0)
        assertTrue(decision.computePressure > 0.50)
    }

    @Test
    fun lowViewInformationAddsSurfaceComputePressure() {
        val controller = AdaptiveDenseMeshExtractionController()

        val strong =
            controller.evaluate(
                AdaptiveDenseMeshExtractionController.Evidence(
                    timestampNanos = 2_000_000L,
                    motionEnergy = 0.10,
                    uncertainty = 0.15,
                    cameraCoverage = 0.92,
                    cameraInformationGain = 0.96,
                    reconstructionConfidence = 0.92,
                    temporalStability = 0.92,
                    thermalHeadroom = 0.90,
                    gpuHeadroom = 0.90,
                    requestedDetail = 0.90,
                    voxelCount = 140_000,
                    maxVoxels = 1_000_000,
                ),
            )

        val weak =
            controller.evaluate(
                AdaptiveDenseMeshExtractionController.Evidence(
                    timestampNanos = 50_000_000L,
                    motionEnergy = 0.10,
                    uncertainty = 0.15,
                    cameraCoverage = 0.92,
                    cameraInformationGain = 0.20,
                    reconstructionConfidence = 0.92,
                    temporalStability = 0.92,
                    thermalHeadroom = 0.90,
                    gpuHeadroom = 0.90,
                    requestedDetail = 0.90,
                    voxelCount = 140_000,
                    maxVoxels = 1_000_000,
                ),
            )

        assertTrue(
            weak.computePressure >= strong.computePressure,
        )
        assertTrue(
            weak.detailLevel <= strong.detailLevel,
        )
        assertTrue(
            weak.reason == "view_information_limited" ||
                weak.reason == "balanced",
        )
    }

    @Test
    fun thermalPressure_reducesExtractionBudget_withoutDisablingCapturePolicy() {
        val controller = AdaptiveDenseMeshExtractionController()

        controller.evaluate(
            AdaptiveDenseMeshExtractionController.Evidence(
                timestampNanos = 1_000_000L,
                motionEnergy = 0.84,
                uncertainty = 0.72,
                cameraCoverage = 0.76,
                reconstructionConfidence = 0.62,
                temporalStability = 0.60,
                thermalHeadroom = 0.92,
                gpuHeadroom = 0.90,
                requestedDetail = 0.94,
                voxelCount = 120_000,
                maxVoxels = 1_000_000,
            ),
        )

        val decision = controller.evaluate(
            AdaptiveDenseMeshExtractionController.Evidence(
                timestampNanos = 48_000_000L,
                motionEnergy = 0.84,
                uncertainty = 0.72,
                cameraCoverage = 0.76,
                reconstructionConfidence = 0.62,
                temporalStability = 0.60,
                thermalHeadroom = 0.30,
                gpuHeadroom = 0.44,
                requestedDetail = 0.94,
                voxelCount = 120_000,
                maxVoxels = 1_000_000,
            ),
        )

        assertTrue(decision.shouldExtract)
        assertTrue(decision.cellBudget < 260_000)
        assertTrue(decision.computePressure > 0.50)
        assertTrue(decision.reason.contains("thermal"))
    }
    @Test
    fun high_quality_metric_evidence_allows_fine_mesh_budget() {
        val controller = AdaptiveDenseMeshExtractionController()

        val decision = controller.evaluate(
            AdaptiveDenseMeshExtractionController.Evidence(
                timestampNanos = 4_000_000_000L,
                motionEnergy = 0.30,
                uncertainty = 0.10,
                cameraCoverage = 0.95,
                cameraInformationGain = 0.92,
                reconstructionConfidence = 0.94,
                temporalStability = 0.94,
                thermalHeadroom = 0.95,
                gpuHeadroom = 0.92,
                requestedDetail = 0.98,
                voxelCount = 240_000,
                maxVoxels = 1_000_000,
                geometryConditioning = 0.93,
                viewDiversity = 0.88,
                metricUncertaintyP95Meters = 0.0012,
                contactConfidence = 0.94,
                identitySafety = 0.97,
                calibrationConfidence = 0.96,
            ),
        )

        assertTrue(decision.detailLevel > 0.70)
        assertTrue(decision.cellBudget > 150_000)
        assertTrue(decision.vertexBudget > 150_000)
        assertTrue(decision.computePressure < 0.45)
    }

    @Test
    fun weak_geometry_and_identity_reduce_extraction_priority() {
        val controller = AdaptiveDenseMeshExtractionController()

        val decision = controller.evaluate(
            AdaptiveDenseMeshExtractionController.Evidence(
                timestampNanos = 5_000_000_000L,
                motionEnergy = 0.42,
                uncertainty = 0.25,
                cameraCoverage = 0.82,
                cameraInformationGain = 0.74,
                reconstructionConfidence = 0.84,
                temporalStability = 0.80,
                thermalHeadroom = 0.95,
                gpuHeadroom = 0.90,
                requestedDetail = 0.95,
                voxelCount = 120_000,
                maxVoxels = 1_000_000,
                geometryConditioning = 0.24,
                viewDiversity = 0.22,
                metricUncertaintyP95Meters = 0.020,
                contactConfidence = 0.32,
                identitySafety = 0.28,
                calibrationConfidence = 0.38,
            ),
        )

        assertTrue(decision.computePressure > 0.25)
        assertTrue(
            decision.reason in setOf(
                "identity_safety_limited",
                "contact_evidence_limited",
                "calibration_limited",
                "metric_uncertainty_limited",
                "geometry_conditioning_limited",
                "view_diversity_limited",
            ),
        )
    }

    @Test
    fun missing_metric_evidence_is_bounded_and_never_nan() {
        val controller = AdaptiveDenseMeshExtractionController()

        val decision = controller.evaluate(
            AdaptiveDenseMeshExtractionController.Evidence(
                timestampNanos = 6_000_000_000L,
                motionEnergy = 0.10,
                uncertainty = 0.15,
                cameraCoverage = 0.90,
                reconstructionConfidence = 0.90,
                temporalStability = 0.90,
                thermalHeadroom = 0.90,
                gpuHeadroom = 0.90,
                requestedDetail = 0.70,
                voxelCount = 2_000,
                maxVoxels = 1_000_000,
                geometryConditioning = 0.85,
                viewDiversity = 0.85,
                metricUncertaintyP95Meters = 0.050,
                contactConfidence = 0.90,
                identitySafety = 0.95,
                calibrationConfidence = 0.90,
            ),
        )

        assertTrue(decision.detailLevel.isFinite())
        assertTrue(decision.computePressure.isFinite())
        assertTrue(decision.cellBudget > 0)
        assertTrue(decision.vertexBudget > 0)
    }

}

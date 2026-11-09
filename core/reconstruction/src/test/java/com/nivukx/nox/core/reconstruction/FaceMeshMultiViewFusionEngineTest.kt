package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceMeshMultiViewFusionEngineTest {

    @Test
    fun precisionQualificationRequiresSubMillimeterEvidenceAndStableTemporalSignal() {
        val qualified =
            FaceMeshMultiViewFusionEngine.Result(
                timestampNanos = 1_000_000L,
                points = emptyList(),
                triangles = emptyList(),
                confidence = 0.92,
                meanReprojectionErrorPx = 0.35,
                p95ReprojectionErrorPx = 0.80,
                meanUncertaintyMeters = 0.0004,
                acceptedPointRatio = 0.98,
                sourceDevices = emptyList(),
                sourceFrameCount = 4,
                meanTemporalStability = 0.91,
                meanPairwiseReprojectionConsensus = 0.96,
                temporalRejectedPointRatio = 0.01,
                meanSubMillimeterFeasibilityScore = 0.86,
                meanFirstOrderDepthErrorMillimeters = 0.62,
                warnings = emptyList(),
            )

        assertTrue(qualified.precisionQualified)

        val degraded =
            qualified.copy(
                meanSubMillimeterFeasibilityScore = 0.42,
            )

        assertFalse(degraded.precisionQualified)
    }

    @Test
    fun production_temporal_face_model_retains_acceleration_prediction() {
        val source =
            java.io.File(
                "src/main/java/com/nivukx/nox/core/reconstruction/FaceMeshMultiViewFusionEngine.kt",
            ).readText()

        assertTrue(source.contains("var acceleration: Vec3"))
        assertTrue(source.contains("measuredAcceleration"))
        assertTrue(source.contains("previous.acceleration"))
        assertTrue(source.contains("0.5 *"))
    }

    @Test
    fun precisionQualificationRejectsHighFirstOrderDepthError() {
        val result =
            FaceMeshMultiViewFusionEngine.Result(
                timestampNanos = 2_000_000L,
                points = emptyList(),
                triangles = emptyList(),
                confidence = 0.90,
                meanReprojectionErrorPx = 0.55,
                p95ReprojectionErrorPx = 1.20,
                meanUncertaintyMeters = 0.0007,
                acceptedPointRatio = 0.94,
                sourceDevices = emptyList(),
                sourceFrameCount = 3,
                meanTemporalStability = 0.88,
                meanPairwiseReprojectionConsensus = 0.91,
                temporalRejectedPointRatio = 0.03,
                meanSubMillimeterFeasibilityScore = 0.95,
                meanFirstOrderDepthErrorMillimeters = 1.40,
                warnings = emptyList(),
            )

        assertFalse(result.precisionQualified)
    }
}


class FaceMeshMultiViewFusionEngineTemporalGeometryContractTest {
    @Test
    fun productionPathContainsRollingShutterAndVelocityCompensation() {
        val source =
            java.io.File(
                "src/main/java/com/nivukx/nox/core/reconstruction/FaceMeshMultiViewFusionEngine.kt",
            ).readText()

        assertTrue(source.contains("rowTimeOffsetNanos ="))
        assertTrue(
            source.contains(
                "motionVelocityMetersPerSecond = previousVelocity",
            ),
        )
        assertTrue(source.contains("rollingShutterFraction ="))
        assertTrue(source.contains("calibrationUncertainty ="))
    }
}

package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class ActiveViewInformationGainEngineTest {

    @Test
    fun firstViewsPreferReliableAndGeometricallyDiverseCameras() {
        val engine = ActiveViewInformationGainEngine()

        val result = engine.order(
            point = Vec3(0.0, 0.0, 3.0),
            candidates = listOf(
                candidate("A", Vec3(-1.0, 0.0, 0.0), confidence = 0.96),
                candidate("B", Vec3(1.0, 0.0, 0.0), confidence = 0.94),
                candidate("C", Vec3(0.0, 0.05, 0.0), confidence = 0.98),
                candidate("D", Vec3(0.0, 1.5, 0.0), confidence = 0.88),
            ),
        )

        assertTrue(result.orderedIndices.isNotEmpty())
        assertEquals(4, result.selectedCameraIds.size)
        assertEquals(4, result.anchorIndices.size)
        assertTrue(result.informationGain > 0.0)
    }

    @Test
    fun lowQualityViewCanRemainAdmissibleButRanksBehindStrongViews() {
        val engine = ActiveViewInformationGainEngine()

        val result = engine.order(
            point = Vec3(0.0, 0.0, 2.5),
            candidates = listOf(
                candidate("strong", Vec3(-1.0, 0.0, 0.0), 0.95),
                candidate(
                    "weak",
                    Vec3(1.0, 0.0, 0.0),
                    confidence = 0.20,
                    calibration = 0.25,
                    uncertainty = 0.12,
                ),
                candidate("strong2", Vec3(0.0, 1.0, 0.0), 0.92),
            ),
        )

        assertTrue(result.rankedViews.first().cameraId != "weak")
        assertTrue(result.rankedViews.last().cameraId == "weak")
    }

    @Test
    fun boundedInputNeverExceedsConfiguredRankedViews() {
        val engine = ActiveViewInformationGainEngine(
            ActiveViewInformationGainEngine.Config(
                maximumRankedViews = 4,
                anchorViews = 2,
            ),
        )

        val candidates = (0 until 12).map { index ->
            candidate(
                cameraId = "cam-$index",
                position = Vec3(index.toDouble() * 0.1, 0.0, 0.0),
                confidence = 0.80,
            )
        }

        val result = engine.order(
            point = Vec3(0.0, 0.0, 3.0),
            candidates = candidates,
        )

        assertTrue(result.rankedViews.size <= 4)
        assertTrue(result.warnings.contains("ACTIVE_VIEW_INPUT_BOUNDED"))
    }

    private fun candidate(
        cameraId: String,
        position: Vec3,
        confidence: Double,
        calibration: Double = 0.95,
        uncertainty: Double = 0.004,
    ) =
        ActiveViewInformationGainEngine.Candidate(
            cameraId = cameraId,
            pose = Pose(position, Quat.IDENTITY),
            confidence = confidence,
            visibility = 0.95,
            calibrationConfidence = calibration,
            temporalAlignmentConfidence = 0.96,
            sourceQuality = 0.95,
            uncertaintyMeters = uncertainty,
        )
}

package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import kotlin.test.assertTrue
import org.junit.Test

class ActiveViewInformationGainPrecisionTest {

    @Test
    fun oppositeRayIsNotRankedAsMaximallyConditioned() {
        val engine =
            ActiveViewInformationGainEngine()
        val point =
            Vec3(0.0, 0.0, 0.0)
        val front =
            candidate(
                id = "front",
                position = Vec3(0.0, 0.0, 5.0),
            )
        val orthogonal =
            candidate(
                id = "orthogonal",
                position = Vec3(5.0, 0.0, 0.0),
            )
        val opposite =
            candidate(
                id = "opposite",
                position = Vec3(0.0, 0.0, -5.0),
            )

        val orthogonalScore =
            engine.order(
                point = point,
                candidates = listOf(front, orthogonal),
            ).rankedViews.first {
                it.cameraId == "orthogonal"
            }.angularNoveltyScore

        val oppositeScore =
            engine.order(
                point = point,
                candidates = listOf(front, opposite),
            ).rankedViews.first {
                it.cameraId == "opposite"
            }.angularNoveltyScore

        assertTrue(orthogonalScore > oppositeScore)
    }

    @Test
    fun calibrationAndTimingUncertaintyReduceViewAuthority() {
        val engine = ActiveViewInformationGainEngine()
        val result = engine.order(
            point = Vec3(0.0, 0.0, 3.0),
            candidates = listOf(
                candidate(
                    cameraId = "clean",
                    position = Vec3(-1.2, 0.0, 0.0),
                    calibrationTranslationUncertaintyMeters = 0.001,
                    calibrationRotationUncertaintyRadians = 0.001,
                    intrinsicRelativeUncertainty = 0.002,
                    timingUncertaintyMillis = 1.0,
                    rollingShutterRisk = 0.05,
                    motionBlurRisk = 0.05,
                ),
                candidate(
                    cameraId = "uncertain",
                    position = Vec3(1.2, 0.0, 0.0),
                    calibrationTranslationUncertaintyMeters = 0.040,
                    calibrationRotationUncertaintyRadians = 0.030,
                    intrinsicRelativeUncertainty = 0.08,
                    timingUncertaintyMillis = 32.0,
                    rollingShutterRisk = 0.75,
                    motionBlurRisk = 0.90,
                ),
            ),
        )

        val clean = result.rankedViews.first { it.cameraId == "clean" }
        val uncertain = result.rankedViews.first { it.cameraId == "uncertain" }

        assertTrue(clean.score > uncertain.score)
        assertTrue(clean.reliabilityScore > 0.0)
    }

    @Test
    fun opticalIntegrityNeverProducesOutOfRangeScores() {
        val engine = ActiveViewInformationGainEngine()
        val result = engine.order(
            point = Vec3(0.0, 0.0, 2.0),
            candidates = listOf(
                candidate(
                    cameraId = "a",
                    position = Vec3(-0.8, 0.0, 0.0),
                    motionBlurRisk = 1.0,
                ),
                candidate(
                    cameraId = "b",
                    position = Vec3(0.8, 0.0, 0.0),
                    motionBlurRisk = 0.0,
                ),
            ),
        )

        assertTrue(result.rankedViews.all { it.score in 0.0..1.0 })
        assertTrue(result.informationGain in 0.0..1.0)
        assertTrue(result.redundancyScore in 0.0..1.0)
    }

    private fun candidate(
        cameraId: String,
        position: Vec3,
        calibrationTranslationUncertaintyMeters: Double = 0.0,
        calibrationRotationUncertaintyRadians: Double = 0.0,
        intrinsicRelativeUncertainty: Double = 0.0,
        timingUncertaintyMillis: Double = 0.0,
        rollingShutterRisk: Double = 0.0,
        motionBlurRisk: Double = 0.0,
    ) =
        ActiveViewInformationGainEngine.Candidate(
            cameraId = cameraId,
            pose = Pose(position, Quat.IDENTITY),
            confidence = 0.97,
            visibility = 0.97,
            calibrationConfidence = 0.97,
            temporalAlignmentConfidence = 0.97,
            sourceQuality = 0.97,
            uncertaintyMeters = 0.003,
            calibrationTranslationUncertaintyMeters =
                calibrationTranslationUncertaintyMeters,
            calibrationRotationUncertaintyRadians =
                calibrationRotationUncertaintyRadians,
            intrinsicRelativeUncertainty =
                intrinsicRelativeUncertainty,
            timingUncertaintyMillis = timingUncertaintyMillis,
            rollingShutterRisk = rollingShutterRisk,
            motionBlurRisk = motionBlurRisk,
        )
}

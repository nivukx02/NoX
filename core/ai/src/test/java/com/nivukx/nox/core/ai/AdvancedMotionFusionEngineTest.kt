package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.EyeObservation
import com.nivukx.nox.core.model.EyeSide
import com.nivukx.nox.core.model.EyeTrackingFrame
import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.FaceMeshPoint
import com.nivukx.nox.core.model.HandTrackingFrame
import com.nivukx.nox.core.model.TrackedObject
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AdvancedMotionFusionEngineTest {

    @Test
    fun eyeCoverageUsesEyeEvidence() {
        val engine = AdvancedMotionFusionEngine()

        val result = engine.assess(
            timestampNanos = 100L,
            faces = listOf(
                face(confidence = 0.30),
            ),
            eyes = listOf(
                eyeFrame(confidence = 0.92),
            ),
            hands = emptyHands(),
            objects = emptyList(),
            motion = motionQuality(),
        )

        assertEquals(0.92, result.eyeCoverage, 0.0001)
        assertTrue(result.faceCoverage < result.eyeCoverage)
    }

    @Test
    fun animalDominatesWhenItIsTheStrongestTrackedSubject() {
        val engine = AdvancedMotionFusionEngine()

        val result = engine.assess(
            timestampNanos = 200L,
            faces = emptyList(),
            eyes = emptyList(),
            hands = emptyHands(),
            objects = listOf(
                TrackedObject(
                    trackingId = 7,
                    subjectType = SubjectType.ANIMAL,
                    left = 10f,
                    top = 10f,
                    right = 610f,
                    bottom = 610f,
                    confidence = 0.91,
                    timestampNanos = 200L,
                ),
                TrackedObject(
                    trackingId = 9,
                    subjectType = SubjectType.OBJECT,
                    left = 20f,
                    top = 20f,
                    right = 120f,
                    bottom = 120f,
                    confidence = 0.40,
                    timestampNanos = 200L,
                ),
            ),
            motion = motionQuality(),
        )

        assertEquals(
            SubjectType.ANIMAL,
            result.dominantSubject?.type,
        )
        assertEquals(0.91, result.animalConfidence, 0.0001)
    }

    private fun face(confidence: Double): FaceFrame =
        FaceFrame(
            timestampNanos = 100L,
            trackingId = 1,
            boundsLeft = 0f,
            boundsTop = 0f,
            boundsRight = 400f,
            boundsBottom = 400f,
            yawDegrees = 0f,
            rollDegrees = 0f,
            points = listOf(
                FaceMeshPoint(
                    index = 0,
                    point = Vec2(200.0, 200.0),
                    confidence = confidence,
                ),
            ),
        )

    private fun eyeFrame(confidence: Double): EyeTrackingFrame =
        EyeTrackingFrame(
            timestampNanos = 100L,
            trackingId = 1,
            left = EyeObservation(
                eye = EyeSide.LEFT,
                center = Vec2(190.0, 190.0),
                irisCenter = Vec2(192.0, 190.0),
                normalizedOffset = Vec2(0.0, 0.0),
                openness = 1.0,
                blink = false,
                confidence = confidence,
            ),
            right = null,
            combinedGaze = Vec3(0.0, 0.0, 1.0),
            fixationConfidence = confidence,
        )

    private fun emptyHands(): HandTrackingFrame =
        HandTrackingFrame(
            timestampNanos = 100L,
            imageWidth = 1,
            imageHeight = 1,
            hands = emptyList(),
            inferenceLatencyMillis = 0.0,
            modelReady = true,
        )

    private fun motionQuality(): MotionCaptureEngine.Quality =
        MotionCaptureEngine.Quality(
            timestampNanos = 100L,
            motionScore = 0.25,
            motionClass = MotionCaptureEngine.MotionClass.NORMAL,
            meanSpeedNormalizedPerSecond = 0.2,
            maxSpeedNormalizedPerSecond = 0.5,
            meanAccelerationNormalizedPerSecond2 = 1.0,
            maxAccelerationNormalizedPerSecond2 = 2.0,
            meanJerkNormalizedPerSecond3 = 0.5,
            jitterNormalized = 0.05,
            trackedLandmarks = 17,
            predictedLandmarks = 0,
            rejectedLandmarks = 0,
            meanConfidence = 0.9,
            occlusionRatio = 0.05,
            temporalStability = 0.9,
            spatialSpread = 0.20,
            stationaryRatio = 0.2,
            dynamicsConfidence = 0.92,
            meanInnovationNormalized = 0.01,
            rejectedMeasurementRatio = 0.0,
        )
}

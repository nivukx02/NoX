package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.EyeObservation
import com.nivukx.nox.core.model.EyeSide
import com.nivukx.nox.core.model.EyeTrackingFrame
import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.FocusTargetMode
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TrackedObject
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FocusTargetEngineTest {

    @Test
    fun eyeTargetTakesPriorityOverFaceAndSubject() {
        val engine = FocusTargetEngine()

        val face =
            face(
                trackingId = 7,
                left = 200f,
                top = 120f,
                right = 600f,
                bottom = 620f,
            )

        val eye =
            EyeTrackingFrame(
                timestampNanos = 1_000_000L,
                trackingId = 7,
                left = EyeObservation(
                    eye = EyeSide.LEFT,
                    center = Vec2(300.0, 260.0),
                    irisCenter = Vec2(305.0, 265.0),
                    normalizedOffset = Vec2(0.0, 0.0),
                    openness = 1.0,
                    blink = false,
                    confidence = 0.92,
                    gazeDirection = Vec3(0.0, 0.0, 1.0),
                ),
                right = null,
                combinedGaze = Vec3(0.0, 0.0, 1.0),
                fixationConfidence = 0.94,
            )

        val target =
            engine.select(
                timestampNanos = 1_000_000L,
                imageWidth = 1000,
                imageHeight = 1000,
                rotationDegrees = 0,
                faces = listOf(face),
                eyes = listOf(eye),
                objects = listOf(
                    TrackedObject(
                        trackingId = 3,
                        subjectType = SubjectType.ANIMAL,
                        left = 100f,
                        top = 100f,
                        right = 900f,
                        bottom = 900f,
                        confidence = 0.99,
                        timestampNanos = 1_000_000L,
                    ),
                ),
            )

        assertEquals(FocusTargetMode.EYE, target?.mode)
        assertTrue((target?.confidence ?: 0.0) > 0.70)
        assertEquals(0.305, target?.normalizedX ?: 0.0, 0.001)
        assertEquals(0.265, target?.normalizedY ?: 0.0, 0.001)
    }

    @Test
    fun animalObjectFallsBackToSubjectFocus() {
        val engine = FocusTargetEngine()

        val target =
            engine.select(
                timestampNanos = 2_000_000L,
                imageWidth = 1000,
                imageHeight = 500,
                rotationDegrees = 0,
                faces = emptyList(),
                eyes = emptyList(),
                objects = listOf(
                    TrackedObject(
                        trackingId = 8,
                        subjectType = SubjectType.ANIMAL,
                        left = 200f,
                        top = 100f,
                        right = 600f,
                        bottom = 300f,
                        confidence = 0.88,
                        timestampNanos = 2_000_000L,
                    ),
                ),
            )

        assertEquals(FocusTargetMode.SUBJECT, target?.mode)
        assertEquals(0.40, target?.normalizedX ?: 0.0, 0.001)
        assertEquals(0.40, target?.normalizedY ?: 0.0, 0.001)
    }

    @Test
    fun staleTargetExpires() {
        val engine = FocusTargetEngine(
            staleAfterNanos = 100L,
        )

        val first =
            engine.select(
                timestampNanos = 0L,
                imageWidth = 1000,
                imageHeight = 1000,
                rotationDegrees = 0,
                faces = emptyList(),
                eyes = emptyList(),
                objects = listOf(
                    TrackedObject(
                        trackingId = 8,
                        subjectType = SubjectType.OBJECT,
                        left = 300f,
                        top = 300f,
                        right = 700f,
                        bottom = 700f,
                        confidence = 0.9,
                        timestampNanos = 0L,
                    ),
                ),
            )

        assertTrue(first != null)

        val expired =
            engine.select(
                timestampNanos = 500L,
                imageWidth = 1000,
                imageHeight = 1000,
                rotationDegrees = 0,
                faces = emptyList(),
                eyes = emptyList(),
                objects = emptyList(),
            )

        assertEquals(null, expired)
    }

    private fun face(
        trackingId: Int,
        left: Float,
        top: Float,
        right: Float,
        bottom: Float,
    ): FaceFrame =
        FaceFrame(
            timestampNanos = 1_000_000L,
            trackingId = trackingId,
            boundsLeft = left,
            boundsTop = top,
            boundsRight = right,
            boundsBottom = bottom,
            yawDegrees = 0f,
            rollDegrees = 0f,
            points = emptyList(),
            leftEyeOpenProbability = 1.0,
            rightEyeOpenProbability = 1.0,
        )
}

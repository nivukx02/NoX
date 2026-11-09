package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TrackedObject
import org.junit.Assert.assertTrue
import org.junit.Test

class SemanticMotionEstimatorTest {

    @Test
    fun separatesCommonModeFromRelativeMotion() {
        val estimator = SemanticMotionEstimator()

        val first = listOf(
            track(
                id = 1L,
                subjectType = SubjectType.HUMAN,
                left = 100f,
                top = 100f,
                right = 300f,
                bottom = 400f,
                timestampNanos = 1_000_000_000L,
                detectorId = 1,
            ),
            track(
                id = 2L,
                subjectType = SubjectType.ANIMAL,
                left = 600f,
                top = 200f,
                right = 800f,
                bottom = 400f,
                timestampNanos = 1_000_000_000L,
                detectorId = 2,
            ),
        )

        estimator.update(
            tracks = first,
            timestampNanos = 1_000_000_000L,
            frameWidth = 1000,
            frameHeight = 1000,
        )

        val second = listOf(
            track(
                id = 1L,
                subjectType = SubjectType.HUMAN,
                left = 180f,
                top = 100f,
                right = 380f,
                bottom = 400f,
                timestampNanos = 1_100_000_000L,
                detectorId = 1,
            ),
            track(
                id = 2L,
                subjectType = SubjectType.ANIMAL,
                left = 680f,
                top = 200f,
                right = 880f,
                bottom = 400f,
                timestampNanos = 1_100_000_000L,
                detectorId = 2,
            ),
        )

        val result = estimator.update(
            tracks = second,
            timestampNanos = 1_100_000_000L,
            frameWidth = 1000,
            frameHeight = 1000,
        )

        assertTrue(result.activeSubjects == 2)
        assertTrue(result.commonModeEnergy > 0.0)
        assertTrue(result.cameraMotionRisk > 0.0)
    }

    @Test
    fun highAccelerationEscalatesMotionClass() {
        val estimator = SemanticMotionEstimator()

        estimator.update(
            tracks = listOf(
                track(
                    id = 10L,
                    subjectType = SubjectType.OBJECT,
                    left = 100f,
                    top = 100f,
                    right = 200f,
                    bottom = 200f,
                    timestampNanos = 2_000_000_000L,
                    detectorId = 10,
                ),
            ),
            timestampNanos = 2_000_000_000L,
            frameWidth = 1000,
            frameHeight = 1000,
        )

        val result = estimator.update(
            tracks = listOf(
                track(
                    id = 10L,
                    subjectType = SubjectType.OBJECT,
                    left = 800f,
                    top = 100f,
                    right = 900f,
                    bottom = 200f,
                    timestampNanos = 2_033_000_000L,
                    detectorId = 10,
                ),
            ),
            timestampNanos = 2_033_000_000L,
            frameWidth = 1000,
            frameHeight = 1000,
        )

        assertTrue(result.movingSubjects >= 1)
        assertTrue(result.globalEnergy > 0.0)
        assertTrue(
            result.subjects.first().classType !=
                SemanticMotionEstimator.MotionClass.STATIC,
        )
    }

    private fun track(
        id: Long,
        subjectType: SubjectType,
        left: Float,
        top: Float,
        right: Float,
        bottom: Float,
        timestampNanos: Long,
        detectorId: Int,
    ): SubjectTrackManager.Track =
        SubjectTrackManager.Track(
            id = id,
            subjectType = subjectType,
            left = left,
            top = top,
            right = right,
            bottom = bottom,
            velocityX = 0f,
            velocityY = 0f,
            confidence = 0.92,
            ageFrames = 1L,
            missedFrames = 0,
            lastTimestampNanos = timestampNanos,
            detectorTrackingId = detectorId,
        )
}

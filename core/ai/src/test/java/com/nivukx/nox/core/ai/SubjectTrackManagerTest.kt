package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TrackedObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SubjectTrackManagerTest {

    @Test
    fun keepsIdentityAcrossSmallMotion() {
        val tracker = SubjectTrackManager()

        val first = tracker.update(
            listOf(
                human(
                    trackingId = -1,
                    left = 100f,
                    top = 100f,
                    right = 180f,
                    bottom = 300f,
                    timestampNanos = 1_000_000_000L,
                ),
            ),
            timestampNanos = 1_000_000_000L,
        )

        val id = first.single().id

        val second = tracker.update(
            listOf(
                human(
                    trackingId = -1,
                    left = 106f,
                    top = 102f,
                    right = 186f,
                    bottom = 302f,
                    timestampNanos = 1_033_000_000L,
                ),
            ),
            timestampNanos = 1_033_000_000L,
        )

        assertEquals(id, second.single().id)
        assertTrue(second.single().confidence > 0.80)
    }

    @Test
    fun globalAssignmentKeepsTwoSubjectsDistinctDuringCrossing() {
        val tracker = SubjectTrackManager()

        val first =
            tracker.update(
                listOf(
                    human(
                        trackingId = 11,
                        left = 100f,
                        top = 100f,
                        right = 220f,
                        bottom = 340f,
                        timestampNanos = 0L,
                    ),
                    human(
                        trackingId = 12,
                        left = 700f,
                        top = 100f,
                        right = 820f,
                        bottom = 340f,
                        timestampNanos = 0L,
                    ),
                ),
                timestampNanos = 0L,
            )

        val firstIds =
            first
                .associateBy { it.detectorTrackingId }

        val second =
            tracker.update(
                listOf(
                    human(
                        trackingId = 11,
                        left = 410f,
                        top = 100f,
                        right = 530f,
                        bottom = 340f,
                        timestampNanos = 33_333_333L,
                    ),
                    human(
                        trackingId = 12,
                        left = 290f,
                        top = 100f,
                        right = 410f,
                        bottom = 340f,
                        timestampNanos = 33_333_333L,
                    ),
                ),
                timestampNanos = 33_333_333L,
            )

        assertEquals(
            firstIds[11]?.id,
            second.first {
                it.detectorTrackingId == 11
            }.id,
        )
        assertEquals(
            firstIds[12]?.id,
            second.first {
                it.detectorTrackingId == 12
            }.id,
        )
        assertEquals(
            2,
            second.size,
        )
    }

    @Test
    fun semanticClassPreventsHumanAnimalIdentityCollapse() {
        val tracker = SubjectTrackManager()

        val result =
            tracker.update(
                listOf(
                    human(
                        trackingId = 1,
                        left = 100f,
                        top = 100f,
                        right = 300f,
                        bottom = 500f,
                        timestampNanos = 0L,
                    ),
                    human(
                        trackingId = 2,
                        subjectType = SubjectType.ANIMAL,
                        left = 105f,
                        top = 105f,
                        right = 305f,
                        bottom = 305f,
                        timestampNanos = 0L,
                    ),
                ),
                timestampNanos = 0L,
            )

        assertEquals(
            2,
            result.size,
        )
        assertTrue(
            result.any {
                it.subjectType ==
                    SubjectType.HUMAN
            },
        )
        assertTrue(
            result.any {
                it.subjectType ==
                    SubjectType.ANIMAL
            },
        )
    }

    private fun human(
        trackingId: Int,
        subjectType: SubjectType = SubjectType.HUMAN,
        left: Float,
        top: Float,
        right: Float,
        bottom: Float,
        timestampNanos: Long,
    ): TrackedObject =
        TrackedObject(
            trackingId = trackingId,
            subjectType = subjectType,
            left = left,
            top = top,
            right = right,
            bottom = bottom,
            confidence = 0.95,
            timestampNanos = timestampNanos,
        )
    @Test
    fun globalAssociationScalesBeyondLegacyEightTrackWindow() {
        val tracker =
            SubjectTrackManager()

        val first =
            tracker.update(
                (0 until 20).map { index ->
                    human(
                        trackingId = 100 + index,
                        left = 40f + index * 55f,
                        top = 100f,
                        right = 85f + index * 55f,
                        bottom = 280f,
                        timestampNanos = 1_000_000_000L,
                    )
                },
                timestampNanos = 1_000_000_000L,
            )

        val idsByDetector =
            first.associate {
                it.detectorTrackingId to it.id
            }

        val second =
            tracker.update(
                (0 until 20).map { index ->
                    human(
                        trackingId = 100 + index,
                        left = 44f + index * 55f,
                        top = 102f,
                        right = 89f + index * 55f,
                        bottom = 282f,
                        timestampNanos = 1_033_333_333L,
                    )
                },
                timestampNanos = 1_033_333_333L,
            )

        assertEquals(20, second.size)
        second.forEach { track ->
            assertEquals(
                idsByDetector[track.detectorTrackingId],
                track.id,
            )
        }
    }

    @Test
    fun timestampRegressionResetsVelocityState() {
        val tracker = SubjectTrackManager()

        tracker.update(
            listOf(
                human(
                    trackingId = 1,
                    left = 100f,
                    top = 100f,
                    right = 180f,
                    bottom = 300f,
                    timestampNanos = 2_000_000_000L,
                ),
            ),
            timestampNanos = 2_000_000_000L,
        )

        val reset =
            tracker.update(
                listOf(
                    human(
                        trackingId = 2,
                        left = 400f,
                        top = 100f,
                        right = 480f,
                        bottom = 300f,
                        timestampNanos = 1_000_000_000L,
                    ),
                ),
                timestampNanos = 1_000_000_000L,
            )

        assertEquals(1, reset.size)
        assertEquals(2, reset.single().detectorTrackingId)
        assertEquals(0f, reset.single().velocityX)
    }

    @Test
    fun retainedTrackCountNeverExceedsProductionCeiling() {
        val tracker = SubjectTrackManager()

        val tracks =
            tracker.update(
                (0 until 80).map { index ->
                    human(
                        trackingId = 10_000 + index,
                        left = 10f + (index % 20) * 80f,
                        top = 10f + (index / 20) * 100f,
                        right = 60f + (index % 20) * 80f,
                        bottom = 80f + (index / 20) * 100f,
                        timestampNanos = 3_000_000_000L,
                    )
                },
                timestampNanos = 3_000_000_000L,
            )

        assertTrue(tracks.size <= 64)
    }

}

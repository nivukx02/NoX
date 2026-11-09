package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Vec2
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class TemporalPoseIdentityTrackerTest {

    @Test
    fun detectorOrderChangeDoesNotChangeLocalIdentity() {
        val tracker = TemporalPoseIdentityTracker()

        val first = tracker.update(
            timestampNanos = 1_000_000_000L,
            detections = listOf(
                detection(0.20, 0.50, 0),
                detection(0.80, 0.50, 1),
            ),
        )
        val firstIds =
            first.observations.associateBy {
                it.observation.center.x
            }

        val second = tracker.update(
            timestampNanos = 1_016_666_667L,
            detections = listOf(
                detection(0.795, 0.50, 7),
                detection(0.205, 0.50, 3),
            ),
        )
        val secondIds =
            second.observations.associateBy {
                it.observation.center.x
            }

        assertEquals(
            firstIds.getValue(0.20).localTrackId,
            secondIds.getValue(0.205).localTrackId,
        )
        assertEquals(
            firstIds.getValue(0.80).localTrackId,
            secondIds.getValue(0.795).localTrackId,
        )
    }

    @Test
    fun appearance_continuity_is_retained_across_detector_index_changes() {
        val tracker = TemporalPoseIdentityTracker(
            TemporalPoseIdentityTracker.Config(
                maximumTracks = 4,
            ),
        )

        val signature =
            List(64) { index ->
                when (index % 4) {
                    0 -> 0.80
                    1 -> 0.24
                    2 -> -0.15
                    else -> 0.08
                }
            }

        val first = tracker.update(
            timestampNanos = 3_000_000_000L,
            detections = listOf(
                detection(
                    x = 0.36,
                    y = 0.50,
                    detectorIndex = 4,
                ).copy(
                    appearanceSignature = signature,
                    appearanceQuality = 0.96,
                ),
            ),
        )
        val id =
            first.observations
                .single()
                .localTrackId

        val second = tracker.update(
            timestampNanos = 3_016_666_667L,
            detections = listOf(
                detection(
                    x = 0.365,
                    y = 0.50,
                    detectorIndex = 77,
                ).copy(
                    appearanceSignature = signature,
                    appearanceQuality = 0.96,
                ),
            ),
        )

        val observation =
            second.observations.single()

        assertEquals(
            id,
            observation.localTrackId,
        )
        assertTrue(
            observation.score > 0.50,
        )
    }

    @Test
    fun shortOcclusionUsesPrediction() {
        val tracker = TemporalPoseIdentityTracker()

        val initial = tracker.update(
            timestampNanos = 1_000_000_000L,
            detections = listOf(
                detection(0.45, 0.50, 0),
            ),
        )
        val id = initial.observations.single().localTrackId

        tracker.update(
            timestampNanos = 1_016_666_667L,
            detections = emptyList(),
        )

        val recovered = tracker.update(
            timestampNanos = 1_033_333_334L,
            detections = listOf(
                detection(0.451, 0.50, 0),
            ),
        )

        assertEquals(
            id,
            recovered.observations.single().localTrackId,
        )
        assertTrue(recovered.observations.single().score > 0.34)
    }

    @Test
    fun capacityIsBounded() {
        val tracker = TemporalPoseIdentityTracker(
            TemporalPoseIdentityTracker.Config(
                maximumTracks = 4,
            ),
        )

        val detections = (0 until 8).map { index ->
            detection(
                x = 0.08 + index * 0.12,
                y = 0.50,
                detectorIndex = index,
            )
        }

        val result = tracker.update(
            timestampNanos = 1_000_000_000L,
            detections = detections,
        )

        assertTrue(result.activeTrackCount <= 4)
        assertTrue(result.droppedObservationCount >= 4)
    }

    @Test
    fun resetRestartsIdentitySequence() {
        val tracker = TemporalPoseIdentityTracker()
        tracker.update(
            timestampNanos = 1_000_000_000L,
            detections = listOf(
                detection(0.50, 0.50, 0),
            ),
        )

        tracker.reset()

        val result = tracker.update(
            timestampNanos = 2_000_000_000L,
            detections = listOf(
                detection(0.50, 0.50, 0),
            ),
        )

        assertEquals(
            1L,
            result.observations.single().localTrackId,
        )
    }

    private fun detection(
        x: Double,
        y: Double,
        detectorIndex: Int,
    ) =
        TemporalPoseIdentityTracker.Observation(
            center = Vec2(x, y),
            width = 0.16,
            height = 0.42,
            confidence = 0.92,
            detectorIndex = detectorIndex,
        )
}

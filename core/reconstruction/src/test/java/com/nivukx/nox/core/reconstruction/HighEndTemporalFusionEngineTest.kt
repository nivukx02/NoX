package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class HighEndTemporalFusionEngineTest {

    private fun observation(
        trackId: Int,
        timestampNanos: Long,
        x: Double,
        cameraId: String,
        confidence: Double = 1.0,
    ) = HighEndTemporalFusionEngine.Observation(
        trackId = trackId,
        timestampNanos = timestampNanos,
        point = HighEndTemporalFusionEngine.Point3(x, 0.0, 1.0),
        confidence = confidence,
        cameraId = cameraId,
    )

    @Test
    fun rejects_out_of_order_frames() {
        val engine = HighEndTemporalFusionEngine()
        engine.update(1_000_000_000L, listOf(observation(1, 1_000_000_000L, 0.0, "a")))
        var rejected = false
        try {
            engine.update(900_000_000L, emptyList())
        } catch (_: IllegalArgumentException) {
            rejected = true
        }
        assertTrue(rejected)
    }

    @Test
    fun hard_outlier_does_not_teleport_track() {
        val engine = HighEndTemporalFusionEngine()
        engine.update(1_000_000_000L, listOf(observation(1, 1_000_000_000L, 0.0, "a")))
        val summary = engine.update(
            1_016_666_667L,
            listOf(
                observation(1, 1_016_666_667L, 0.02, "a"),
                observation(1, 1_016_666_667L, 8.0, "b", 0.99),
            ),
        )
        assertTrue(engine.snapshot(1)!!.position.x < 1.0)
        assertTrue(summary.rejectedObservations >= 1)
    }

    @Test
    fun prediction_is_time_bounded_and_confidence_decays() {
        val engine = HighEndTemporalFusionEngine(
            HighEndTemporalFusionEngine.Config(
                maxPredictionMilliseconds = 80.0,
                trackRetentionMilliseconds = 160.0,
            ),
        )
        engine.update(
            1_000_000_000L,
            listOf(observation(1, 1_000_000_000L, 0.0, "a")),
        )
        engine.update(
            1_016_666_667L,
            listOf(observation(1, 1_016_666_667L, 0.01, "a")),
        )
        engine.update(1_066_666_667L, emptyList())
        val occluded = engine.snapshot(1)!!
        assertTrue(
            occluded.state ==
                HighEndTemporalFusionEngine.TrackState.OCCLUDED,
        )
        assertTrue(occluded.predictedMilliseconds in 0.0..80.0)
        assertTrue(occluded.confidence < 1.0)

        engine.update(1_120_000_000L, emptyList())
        assertTrue(
            engine.snapshot(1)!!.state ==
                HighEndTemporalFusionEngine.TrackState.LOST,
        )
    }

    @Test
    fun recovery_returns_track_to_tracking() {
        val engine = HighEndTemporalFusionEngine()
        engine.update(
            1_000_000_000L,
            listOf(observation(4, 1_000_000_000L, 2.0, "a")),
        )
        engine.update(1_033_333_333L, emptyList())
        val recovered = engine.update(
            1_050_000_000L,
            listOf(observation(4, 1_050_000_000L, 2.03, "b")),
        )
        assertTrue(recovered.trackingTracks >= 1)
        assertTrue(
            engine.snapshot(4)!!.state ==
                HighEndTemporalFusionEngine.TrackState.TRACKING,
        )
    }

    @Test
    fun track_capacity_remains_bounded() {
        val engine = HighEndTemporalFusionEngine(
            HighEndTemporalFusionEngine.Config(
                maxTracks = 2,
                maxObservationsPerFrame = 4,
            ),
        )
        val observations = (1..6).map {
            observation(it, 1_000_000_000L, it.toDouble(), "a")
        }
        val summary = engine.update(1_000_000_000L, observations)
        assertTrue(summary.activeTracks <= 2)
        assertTrue(summary.acceptedObservations <= 4)
    }

    @Test
    fun reset_clears_live_temporal_state() {
        val engine = HighEndTemporalFusionEngine()
        engine.update(
            1_000_000_000L,
            listOf(observation(1, 1_000_000_000L, 1.0, "a")),
        )
        engine.reset()
        assertFalse(engine.snapshots().isNotEmpty())
        assertTrue(engine.lastSummary().temporalConsistency == 1.0)
    }

    @Test
    fun empty_tracker_reports_no_temporal_quality() {
        val engine = HighEndTemporalFusionEngine()
        val summary = engine.update(
            1_000_000_000L,
            emptyList(),
        )
        assertEquals(0.0, summary.temporalConsistency)
        assertEquals(0, summary.activeTracks)
    }


    @Test
    fun runtime_configuration_rebinds_bounds_without_losing_lifecycle_state() {
        val engine = HighEndTemporalFusionEngine()
        engine.update(
            1_000_000_000L,
            listOf(
                observation(1, 1_000_000_000L, 0.0, "a"),
            ),
        )

        engine.configure(
            HighEndTemporalFusionEngine.Config(
                maxTracks = 1,
                maxObservationsPerFrame = 2,
                maxPredictionMilliseconds = 24.0,
                trackRetentionMilliseconds = 500.0,
                hardOutlierDistance = 2.0,
                huberDelta = 1.0,
                confidenceHalfLifeMilliseconds = 100.0,
            ),
        )

        val summary = engine.update(
            1_016_666_667L,
            listOf(
                observation(1, 1_016_666_667L, 0.01, "a"),
                observation(2, 1_016_666_667L, 2.0, "b"),
                observation(3, 1_016_666_667L, 3.0, "c"),
            ),
        )

        assertTrue(summary.activeTracks <= 1)
        assertTrue(
            engine.snapshot(1)?.predictedMilliseconds ?: 0.0 <= 24.0,
        )
    }

}

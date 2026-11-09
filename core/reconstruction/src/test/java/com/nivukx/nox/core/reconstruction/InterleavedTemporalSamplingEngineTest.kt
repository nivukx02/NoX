package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class InterleavedTemporalSamplingEngineTest {

    private fun sample(
        cameraId: String,
        timestampMillis: Long,
        sequence: Long,
        sector: Int,
        residualMillis: Double = 0.2,
    ) =
        InterleavedTemporalSamplingEngine.CameraSample(
            cameraId = cameraId,
            timestampNanos = timestampMillis * 1_000_000L,
            sequence = sequence,
            sector = sector,
            confidence = 0.98,
            synchronized = true,
            timingResidualMillis = residualMillis,
            geometricQuality = 0.97,
            outlierScore = 0.0,
        )

    @Test
    fun four_measured_sectors_form_a_publishable_interleaved_window() {
        val engine =
            InterleavedTemporalSamplingEngine(
                InterleavedTemporalSamplingEngine.Config(
                    targetCameraCountPerWindow = 4,
                    maximumWindowMillis = 20.0,
                    maximumSampleSkewMillis = 12.0,
                    minimumCoverageRatio = 0.75,
                    minimumCameraDiversity = 0.70,
                    minimumTemporalHealth = 0.60,
                ),
            )

        var plan: InterleavedTemporalSamplingEngine.WindowPlan? = null
        repeat(4) { index ->
            plan =
                engine.ingest(
                    sample(
                        cameraId = "camera-$index",
                        timestampMillis = 100L + index,
                        sequence = index.toLong(),
                        sector = index,
                    ),
                )
        }

        requireNotNull(plan)
        assertTrue(plan.publishable)
        assertEquals(4, plan.cameraCount)
        assertTrue(plan.coverageRatio >= 0.75)
        assertTrue(plan.cameraDiversity >= 0.70)
        assertTrue(plan.effectiveSamplingMultiplier > 1.0)
        assertTrue(plan.warnings.isEmpty())
    }

    @Test
    fun excessive_sample_skew_blocks_publish_without_losing_samples() {
        val engine =
            InterleavedTemporalSamplingEngine(
                InterleavedTemporalSamplingEngine.Config(
                    targetCameraCountPerWindow = 3,
                    maximumWindowMillis = 30.0,
                    maximumSampleSkewMillis = 3.0,
                    minimumCoverageRatio = 0.70,
                    minimumCameraDiversity = 0.60,
                    minimumTemporalHealth = 0.60,
                ),
            )

        engine.ingest(
            sample(
                cameraId = "a",
                timestampMillis = 1_000L,
                sequence = 1L,
                sector = 0,
            ),
        )
        engine.ingest(
            sample(
                cameraId = "b",
                timestampMillis = 1_001L,
                sequence = 1L,
                sector = 1,
            ),
        )
        val plan =
            engine.ingest(
                sample(
                    cameraId = "c",
                    timestampMillis = 1_020L,
                    sequence = 1L,
                    sector = 4,
                ),
            )

        requireNotNull(plan)
        assertTrue(!plan.publishable)
        assertTrue(
            plan.warnings.any {
                it == "INTERLEAVED_SAMPLE_SKEW_EXCEEDED"
            },
        )
        assertEquals(3, plan.cameraCount)
        assertTrue(engine.summary().acceptedSamples == 3L)
    }

    @Test
    fun duplicate_or_out_of_order_samples_are_rejected_deterministically() {
        val engine =
            InterleavedTemporalSamplingEngine()

        engine.ingest(
            sample(
                cameraId = "camera-1",
                timestampMillis = 2_000L,
                sequence = 7L,
                sector = 2,
            ),
        )
        engine.ingest(
            sample(
                cameraId = "camera-1",
                timestampMillis = 1_999L,
                sequence = 6L,
                sector = 2,
            ),
        )

        val summary = engine.summary()
        assertEquals(1L, summary.acceptedSamples)
        assertEquals(1L, summary.droppedSamples)
    }

    @Test
    fun reset_clears_metadata_without_leaking_camera_history() {
        val engine =
            InterleavedTemporalSamplingEngine()

        repeat(6) { index ->
            engine.ingest(
                sample(
                    cameraId = "camera-$index",
                    timestampMillis = 3_000L + index,
                    sequence = index.toLong(),
                    sector = index,
                ),
            )
        }

        assertTrue(engine.summary().activeCameras > 0)
        engine.reset()

        val summary = engine.summary()
        assertEquals(0, summary.activeCameras)
        assertEquals(0L, summary.acceptedSamples)
        assertEquals(0L, summary.droppedSamples)
        assertEquals(1.0, summary.effectiveSamplingMultiplier)
        assertTrue(summary.latestPlan == null)
    }
}

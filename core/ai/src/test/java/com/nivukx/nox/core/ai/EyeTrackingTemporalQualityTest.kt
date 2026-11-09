package com.nivukx.nox.core.ai

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class EyeTrackingTemporalQualityTest {
    @Test
    fun clean_global_timing_keeps_gaze_quality_high() {
        val score = EyeTrackingEngine.temporalQualityScale(
            captureTimingConfidence = 0.98,
            rollingShutterSkewNanos = 0L,
            frameDurationNanos = 33_333_333L,
            exposureTimeNanos = 8_000_000L,
        )

        assertTrue(score > 0.90)
    }

    @Test
    fun degraded_capture_timing_reduces_eye_tracking_quality() {
        val strong = EyeTrackingEngine.temporalQualityScale(
            captureTimingConfidence = 0.98,
            rollingShutterSkewNanos = 8_000_000L,
            frameDurationNanos = 33_333_333L,
            exposureTimeNanos = 8_000_000L,
        )
        val weak = EyeTrackingEngine.temporalQualityScale(
            captureTimingConfidence = 0.45,
            rollingShutterSkewNanos = 24_000_000L,
            frameDurationNanos = 33_333_333L,
            exposureTimeNanos = 16_000_000L,
        )

        assertTrue(weak < strong)
        assertTrue(weak >= 0.05)
    }

    @Test
    fun missing_frame_duration_uses_safe_floor() {
        val score = EyeTrackingEngine.temporalQualityScale(
            captureTimingConfidence = 0.80,
            rollingShutterSkewNanos = 5_000_000L,
            frameDurationNanos = 0L,
            exposureTimeNanos = 0L,
        )

        assertTrue(score in 0.05..1.0)
        assertEquals(true, score.isFinite())
    }

    @Test
    fun iris_structure_score_rewards_boundary_supported_candidates() {
        val darkFlat =
            EyeTrackingEngine.irisStructureScore(
                darknessScore = 0.98,
                annulusContrast = 0.20,
                radialSymmetry = 0.35,
                gradientScore = 0.05,
                temporalPrior = 0.30,
            )

        val coherentBoundary =
            EyeTrackingEngine.irisStructureScore(
                darknessScore = 0.86,
                annulusContrast = 0.78,
                radialSymmetry = 0.82,
                gradientScore = 0.88,
                temporalPrior = 0.74,
            )

        assertTrue(coherentBoundary > darkFlat)
        assertTrue(coherentBoundary > 0.65)
    }

    @Test
    fun stale_timestamp_is_rejected_for_eye_temporal_state() {
        assertTrue(
            EyeTrackingEngine.isStaleTimestamp(
                previousTimestampNanos = 10_000_000_000L,
                currentTimestampNanos = 9_000_000_000L,
            ),
        )
        assertTrue(
            EyeTrackingEngine.isStaleTimestamp(
                previousTimestampNanos = 10_000_000_000L,
                currentTimestampNanos = 10_000_000_000L,
            ),
        )
        assertTrue(
            !EyeTrackingEngine.isStaleTimestamp(
                previousTimestampNanos = 10_000_000_000L,
                currentTimestampNanos = 10_000_000_001L,
            ),
        )
    }

    @Test
    fun normalized_iris_quality_is_not_collapsed_by_byte_scale() { {
        val confident =
            EyeTrackingEngine.irisCandidateConfidence(
                compactness = 0.0,
                roiHalfWidth = 20.0,
                candidateQuality = 0.90,
            )
        val diffuse =
            EyeTrackingEngine.irisCandidateConfidence(
                compactness = 400.0,
                roiHalfWidth = 20.0,
                candidateQuality = 0.90,
            )

        assertTrue(confident > 0.85)
        assertTrue(diffuse <= 0.90)
        assertTrue(diffuse >= 0.0)
    }

    @Test
    fun subpixel_centroid_stays_bounded_and_follows_dark_boundary() {
        val offset =
            EyeTrackingEngine.subpixelCentroid(
                listOf(
                    EyeTrackingEngine.SubpixelSample(-1, 0, 0.30, 0.40),
                    EyeTrackingEngine.SubpixelSample(0, 0, 0.70, 0.80),
                    EyeTrackingEngine.SubpixelSample(1, 0, 0.96, 0.90),
                ),
            )

        assertTrue(offset.x > 0.0)
        assertTrue(offset.x <= 0.75)
        assertTrue(kotlin.math.abs(offset.y) < 0.01)
    }


        assertTrue(
            EyeTrackingEngine.isStaleTimestamp(
                previousTimestampNanos = 10_000_000_000L,
                currentTimestampNanos = 9_000_000_000L,
            ),
        )

    @Test
    fun constructorRejectsNonFiniteFaceThresholds() {
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            EyeTrackingEngine(minimumFaceDimensionPx = Float.POSITIVE_INFINITY)
        }
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            EyeTrackingEngine(minimumOpenProbability = Double.NaN)
        }
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            EyeTrackingEngine(blinkCloseThreshold = Double.POSITIVE_INFINITY)
        }
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            EyeTrackingEngine(maximumTracks = Int.MAX_VALUE)
        }
    }

    @Test
    fun temporalQualityRejectsNonFiniteAndNegativeTimingInputs() {
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            EyeTrackingEngine.temporalQualityScale(
                captureTimingConfidence = Double.NaN,
                rollingShutterSkewNanos = 0L,
                frameDurationNanos = 33_000_000L,
                exposureTimeNanos = 5_000_000L,
            )
        }
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            EyeTrackingEngine.temporalQualityScale(
                captureTimingConfidence = 0.9,
                rollingShutterSkewNanos = -1L,
                frameDurationNanos = 33_000_000L,
                exposureTimeNanos = 5_000_000L,
            )
        }
    }

}
package com.nivukx.nox.core.ai

import kotlin.test.assertTrue
import org.junit.Test

class FaceMeshTimingQualityTest {
    @Test
    fun dense_mesh_timing_quality_is_bounded_and_monotonic() {
        val strong = FaceMeshGeometryEngine.temporalQualityScale(
            captureTimingConfidence = 0.98,
            rollingShutterSkewNanos = 4_000_000L,
            frameDurationNanos = 33_333_333L,
            exposureTimeNanos = 6_000_000L,
        )
        val weak = FaceMeshGeometryEngine.temporalQualityScale(
            captureTimingConfidence = 0.42,
            rollingShutterSkewNanos = 26_000_000L,
            frameDurationNanos = 33_333_333L,
            exposureTimeNanos = 18_000_000L,
        )

        assertTrue(strong in 0.05..1.0)
        assertTrue(weak in 0.05..1.0)
        assertTrue(weak < strong)
    }
}

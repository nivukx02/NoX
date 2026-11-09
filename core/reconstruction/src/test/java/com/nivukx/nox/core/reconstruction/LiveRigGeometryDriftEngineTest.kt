package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LiveRigGeometryDriftEngineTest {

    @Test
    fun stableRigIsAccepted() {
        val engine = LiveRigGeometryDriftEngine()
        val result = engine.update(input(1, 1.2, 1.0, 2.5, 0.95, 0.94))

        assertEquals(
            LiveRigGeometryDriftEngine.State.STABLE,
            result.state,
        )
        assertTrue(result.accepted)
    }

    @Test
    fun unstableCaptureIsAmbiguousInsteadOfImmediateDrift() {
        val engine = LiveRigGeometryDriftEngine()
        val result = engine.update(
            input(
                2,
                median = 7.0,
                p95 = 14.0,
                maximum = 28.0,
                inlier = 0.63,
                score = 0.58,
                cameraMotionRisk = 0.92,
            ),
        )

        assertEquals(
            LiveRigGeometryDriftEngine.State.AMBIGUOUS,
            result.state,
        )
        assertTrue(result.accepted)
    }

    @Test
    fun persistentResidualEventuallyRequiresRevalidation() {
        val engine = LiveRigGeometryDriftEngine(
            LiveRigGeometryDriftEngine.Config(
                persistentWatchFrames = 2,
                persistentRevalidateFrames = 3,
                persistentBlockFrames = 8,
            ),
        )

        var result = inputResult(engine, 3L, 7.0, 14.0, 18.0, 0.72, 0.58)
        repeat(4) {
            result = inputResult(
                engine,
                4L + it,
                7.0,
                14.0,
                18.0,
                0.72,
                0.58,
            )
        }

        assertEquals(
            LiveRigGeometryDriftEngine.State.REVALIDATE,
            result.state,
        )
        assertTrue(result.requiresRecalibration)
    }

    @Test
    fun hardResidualCanBlockAfterPersistence() {
        val engine = LiveRigGeometryDriftEngine(
            LiveRigGeometryDriftEngine.Config(
                persistentWatchFrames = 1,
                persistentRevalidateFrames = 2,
                persistentBlockFrames = 3,
            ),
        )

        var result: LiveRigGeometryDriftEngine.Result? = null
        repeat(6) { index ->
            result = inputResult(
                engine,
                4L + index,
                10.0,
                22.0,
                30.0,
                0.42,
                0.35,
            )
        }

        assertEquals(
            LiveRigGeometryDriftEngine.State.BLOCKED,
            result?.state,
        )
        assertTrue(result?.fatal == true)
        assertTrue(result?.warnings?.any { it.startsWith("fatal:") } == true)
    }

    @Test
    fun healthyRecoveryLeavesBlockedStateAfterStableWindow() {
        val engine = LiveRigGeometryDriftEngine(            LiveRigGeometryDriftEngine.Config(
                persistentWatchFrames = 1,
                persistentRevalidateFrames = 2,
                persistentBlockFrames = 3,
                recoveryFrames = 3,
            ),
        )

        repeat(6) { index ->
            inputResult(
                engine,
                5L + index,
                10.0,
                22.0,
                30.0,
                0.42,
                0.35,
            )
        }

        var result: LiveRigGeometryDriftEngine.Result? = null
        repeat(5) { index ->
            result = inputResult(
                engine,
                20L + index,
                1.0,
                2.0,
                3.0,
                0.96,
                0.95,
            )
        }

        assertTrue(
            result?.state ==
                LiveRigGeometryDriftEngine.State.STABLE ||
                result?.state ==
                LiveRigGeometryDriftEngine.State.WATCH,
        )
    }

    private fun inputResult(
        engine: LiveRigGeometryDriftEngine,
        timestamp: Long,
        median: Double,
        p95: Double,
        maximum: Double,
        inlier: Double,
        score: Double,
    ): LiveRigGeometryDriftEngine.Result =
        engine.update(
            input(
                timestamp,
                median,
                p95,
                maximum,
                inlier,
                score,
            ),
        )

    private fun input(
        timestamp: Long,
        median: Double,
        p95: Double,
        maximum: Double,
        inlier: Double,
        score: Double,
        cameraMotionRisk: Double = 0.0,
    ): LiveRigGeometryDriftEngine.Input =
        LiveRigGeometryDriftEngine.Input(
            timestampNanos = timestamp.toLong() * 1_000_000L,
            calibrationReprojectionErrorPx = 1.2,
            liveMedianResidualPx = median,
            liveP95ResidualPx = p95,
            liveMaximumResidualPx = maximum,
            liveInlierRatio = inlier,
            liveMultiViewScore = score,
            cameraCount = 4,
            clockSkewMillis = 1.0,
            cameraMotionRisk = cameraMotionRisk,
            synchronizationConfidence = 0.94,
            captureTimingConfidence = 0.96,
        )
}

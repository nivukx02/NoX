package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ThermalExtrinsicDriftEngineHysteresisTest {

    @Test
    fun transientThresholdCrossingDoesNotImmediatelyFlipStableCamera() {
        val engine =
            ThermalExtrinsicDriftEngine(
                ThermalExtrinsicDriftEngine.Config(
                    minimumSamples = 3,
                    stableThreshold = 0.82,
                    watchThreshold = 0.62,
                    recalibrateThreshold = 0.40,
                    stateEnterFrames = 2,
                    recoveryFrames = 3,
                ),
            )

        repeat(3) { index ->
            engine.observe(sample(index, driftMm = 0.05))
        }

        val firstWatch =
            engine.observe(
                sample(3, driftMm = 0.42),
            )

        assertEquals(
            ThermalExtrinsicDriftEngine.State.STABLE,
            firstWatch.state,
        )
    }

    @Test
    fun persistentDegradationEventuallyTransitionsToRecalibrate() {
        val engine =
            ThermalExtrinsicDriftEngine(
                ThermalExtrinsicDriftEngine.Config(
                    minimumSamples = 3,
                    recalibrateThreshold = 0.80,
                    watchThreshold = 0.90,
                    stableThreshold = 0.95,
                    stateEnterFrames = 2,
                    recoveryFrames = 3,
                ),
            )

        repeat(3) { index ->
            engine.observe(sample(index, driftMm = 0.05))
        }

        val degraded =
            (0 until 3).map { index ->
                engine.observe(
                    sample(
                        index + 3,
                        driftMm = 1.1,
                    ),
                )
            }.last()

        assertTrue(
            degraded.state ==
                ThermalExtrinsicDriftEngine.State.RECALIBRATE ||
                degraded.state ==
                ThermalExtrinsicDriftEngine.State.BLOCKED,
        )
    }

    private fun sample(
        index: Int,
        driftMm: Double,
    ): ThermalExtrinsicDriftEngine.Sample =
        ThermalExtrinsicDriftEngine.Sample(
            cameraId = DeviceId("cam-h"),
            timestampNanos = (index + 1L) * 1_000_000_000L,
            temperatureCelsius = 24.0 + index * 4.0,
            calibratedPose = Pose(
                position = Vec3(0.0, 0.0, 0.0),
                rotation = Quat.IDENTITY,
            ),
            observedPose = Pose(
                position = Vec3(driftMm / 1000.0, 0.0, 0.0),
                rotation = Quat.IDENTITY,
            ),
            calibrationConfidence = 0.99,
            mountIntegrity = 0.99,
        )
}
